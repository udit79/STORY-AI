from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone Nemotron OCR v2 full-page manga benchmark. "
            "Runs without CTD, PaddleOCR, Qwen3-VL, preprocessing, or Laya."
        )
    )
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="STORY-AI root. Auto-detected from /mnt/d/STORY-AI when omitted.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Local Nemotron checkpoint directory. Defaults to repo_root/nemotron-ocr-v2/v2_english.",
    )
    parser.add_argument(
        "--oracle",
        type=Path,
        default=None,
        help="Preprocessing oracle JSONL. Defaults to experiments/laya/data/preprocessing_oracle_<sequence>.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/nemotron_fullpage_gt/<sequence>.",
    )
    parser.add_argument(
        "--merge-level",
        choices=["word", "sentence", "paragraph"],
        default="word",
    )
    parser.add_argument(
        "--skip-relational",
        action="store_true",
        help="Use Nemotron's raw per-word path without relational grouping.",
    )
    parser.add_argument(
        "--include-all-regions",
        action="store_true",
        help="Also write unmatched predictions in the per-region output (default behavior already writes all regions).",
    )
    return parser.parse_args()


def find_repo_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    candidates = [
        Path("/mnt/d/STORY-AI"),
        Path.cwd(),
    ]
    for candidate in candidates:
        if (candidate / "dataset").exists() and (candidate / "experiments").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not detect STORY-AI root. Pass --repo-root /mnt/d/STORY-AI."
    )


def normalize_text(text: str) -> str:
    text = str(text or "").upper()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein(a: str, b: str) -> int:
    a = normalize_text(a)
    b = normalize_text(b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j, cb in enumerate(b, start=1):
        cur = [j]
        for i, ca in enumerate(a, start=1):
            cur.append(min(
                cur[-1] + 1,
                prev[i] + 1,
                prev[i - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(reference, hypothesis) / len(reference)


def bbox_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def intersection_area(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: list[float], b: list[float]) -> float:
    inter = intersection_area(a, b)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union else 0.0


def center_inside(point: tuple[float, float], box: list[float], margin: float = 0.0) -> bool:
    cx, cy = point
    return (
        box[0] - margin <= cx <= box[2] + margin
        and box[1] - margin <= cy <= box[3] + margin
    )


def overlap_over_pred(box: list[float], gt_box: list[float]) -> float:
    area = bbox_area(box)
    return intersection_area(box, gt_box) / area if area else 0.0


def load_oracle(path: Path, sequence_id: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Oracle file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("sequence_id") != sequence_id:
                continue
            features = row.get("features") or {}
            try:
                bbox = [
                    float(features["x1"]),
                    float(features["y1"]),
                    float(features["x2"]),
                    float(features["y2"]),
                ]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Missing/invalid oracle bbox at line {line_no}: {row}"
                ) from exc
            rows.append({
                "page_index": int(row["page_index"]),  # oracle convention: 0-based
                "block_index": int(row["block_index"]),
                "raw_block_id": row.get("raw_block_id"),
                "target_text": str(row.get("target_text", "")),
                "match_score": row.get("match_score"),
                "bbox": bbox,
                "baseline_text": row.get("baseline_text", ""),
                "baseline_cer": row.get("baseline_cer"),
            })
    if not rows:
        raise ValueError(f"No oracle rows found for sequence {sequence_id}: {path}")
    rows.sort(key=lambda r: (r["page_index"], r["block_index"]))
    return rows


def load_model(model_dir: Path):
    try:
        from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
    except ImportError as exc:
        raise RuntimeError(
            "Nemotron OCR is not installed in the active environment."
        ) from exc
    return NemotronOCRV2(model_dir=str(model_dir))


def normalize_prediction(pred: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    left = float(pred.get("left", 0.0))
    upper = float(pred.get("upper", 0.0))
    right = float(pred.get("right", 0.0))
    lower = float(pred.get("lower", 0.0))

    if max(abs(left), abs(upper), abs(right), abs(lower)) <= 1.5:
        box = [left * width, upper * height, right * width, lower * height]
    else:
        box = [left, upper, right, lower]

    # Nemotron's normalized output can arrive with the vertical coordinates
    # reversed relative to visual intuition; canonicalize the box before use.
    x1, x2 = sorted((box[0], box[2]))
    y1, y2 = sorted((box[1], box[3]))
    return {
        "text": str(pred.get("text", "")),
        "confidence": float(pred.get("confidence", 0.0)),
        "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
    }


def match_predictions_to_gt(
    predictions: list[dict[str, Any]],
    gt_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    GT-conditioned reconstruction:
    collect predicted word/region boxes whose centers are inside the GT block,
    with a small fallback for boxes having substantial overlap.

    This is intentionally NOT presented as end-to-end story extraction. It
    measures whether Nemotron can localize/recognize text covering the known
    story block when evaluated against the benchmark's GT geometry.
    """
    results: list[dict[str, Any]] = []
    for gt in gt_rows:
        candidates: list[dict[str, Any]] = []
        for idx, pred in enumerate(predictions):
            center = bbox_center(pred["bbox"])
            inside = center_inside(center, gt["bbox"], margin=2.0)
            overlap = overlap_over_pred(pred["bbox"], gt["bbox"])
            if inside or overlap >= 0.50:
                candidates.append({
                    "prediction_index": idx,
                    "text": pred["text"],
                    "confidence": pred["confidence"],
                    "bbox": pred["bbox"],
                    "iou": iou(pred["bbox"], gt["bbox"]),
                    "overlap_over_prediction": overlap,
                })

        # Reading order within a GT block: top-to-bottom, then left-to-right.
        candidates.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
        hypothesis = " ".join(c["text"] for c in candidates if c["text"].strip())
        results.append({
            "page_index": gt["page_index"],
            "block_index": gt["block_index"],
            "target_text": gt["target_text"],
            "prediction_text": hypothesis,
            "cer": cer(gt["target_text"], hypothesis),
            "exact_match": normalize_text(gt["target_text"]) == normalize_text(hypothesis),
            "matched_region_count": len(candidates),
            "mean_region_confidence": (
                sum(c["confidence"] for c in candidates) / len(candidates)
                if candidates else 0.0
            ),
            "regions": candidates,
        })
    return results


def region_story_membership(predictions: list[dict[str, Any]], gt_rows: list[dict[str, Any]]) -> dict[str, Any]:
    story_indices: set[int] = set()
    for idx, pred in enumerate(predictions):
        for gt in gt_rows:
            center = bbox_center(pred["bbox"])
            if center_inside(center, gt["bbox"], margin=2.0) or overlap_over_pred(pred["bbox"], gt["bbox"]) >= 0.50:
                story_indices.add(idx)
                break
    return {
        "total_predictions": len(predictions),
        "inside_story_gt_boxes": len(story_indices),
        "outside_story_gt_boxes": len(predictions) - len(story_indices),
        "story_box_membership_rate": (
            len(story_indices) / len(predictions) if predictions else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(args.repo_root)

    data_dir = repo_root / "dataset" / "development" / "images" / args.sequence_id
    pages = [data_dir / f"{i:02d}.png" for i in (1, 2, 3)]
    missing = [p for p in pages if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing sequence pages:\n" + "\n".join(map(str, missing)))

    model_dir = args.model_dir or (repo_root / "nemotron-ocr-v2" / "v2_english")
    oracle_path = args.oracle or (
        repo_root / "experiments" / "laya" / "data" / f"preprocessing_oracle_{args.sequence_id}.jsonl"
    )
    output_dir = args.output_dir or (
        repo_root / "outputs" / "nemotron_fullpage_gt" / args.sequence_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_rows = load_oracle(oracle_path, args.sequence_id)
    print(f"Sequence       : {args.sequence_id}")
    print(f"Model dir      : {model_dir}")
    print(f"Oracle         : {oracle_path}")
    print(f"GT story rows  : {len(gt_rows)}")
    print(f"Merge level    : {args.merge_level}")
    print(f"Skip relational: {args.skip_relational}")

    # Fail loudly before inference if the local checkpoint is incomplete.
    required = ["detector.pth", "recognizer.pth", "relational.pth", "charset.txt"]
    missing_weights = [name for name in required if not (model_dir / name).exists()]
    if missing_weights:
        raise FileNotFoundError(
            f"Local model directory is incomplete: {missing_weights}. Path: {model_dir}"
        )

    ocr = load_model(model_dir)
    all_records: list[dict[str, Any]] = []
    all_block_metrics: list[dict[str, Any]] = []

    for page_number, page_path in enumerate(pages, start=1):
        from PIL import Image

        with Image.open(page_path) as image:
            width, height = image.size

        if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
            pass

        start = time.perf_counter()
        if args.skip_relational:
            predictions = ocr(str(page_path), merge_level="word")
        else:
            predictions = ocr(str(page_path), merge_level=args.merge_level)
        runtime_ms = (time.perf_counter() - start) * 1000.0

        regions = [normalize_prediction(pred, width, height) for pred in predictions]
        gt_page = [g for g in gt_rows if g["page_index"] == page_number - 1]
        block_metrics = match_predictions_to_gt(regions, gt_page)
        membership = region_story_membership(regions, gt_page)

        covered = sum(1 for row in block_metrics if row["matched_region_count"] > 0)
        page_cer = (
            sum(row["cer"] for row in block_metrics) / len(block_metrics)
            if block_metrics else None
        )
        exact_rate = (
            sum(1 for row in block_metrics if row["exact_match"]) / len(block_metrics)
            if block_metrics else None
        )

        record = {
            "sequence_id": args.sequence_id,
            "page_index": page_number,
            "page_path": str(page_path),
            "image_size": {"width": width, "height": height},
            "model": "nvidia/nemotron-ocr-v2",
            "model_dir": str(model_dir),
            "merge_level": "word" if args.skip_relational else args.merge_level,
            "skip_relational": args.skip_relational,
            "runtime_ms": round(runtime_ms, 2),
            "region_count": len(regions),
            "regions": regions,
            "gt_blocks": block_metrics,
            "page_metrics": {
                "gt_blocks": len(gt_page),
                "gt_blocks_covered": covered,
                "gt_block_coverage": covered / len(gt_page) if gt_page else None,
                "gt_conditioned_mean_cer": page_cer,
                "gt_conditioned_exact_match_rate": exact_rate,
                **membership,
            },
        }
        all_records.append(record)
        all_block_metrics.extend(block_metrics)

        print("\n" + "=" * 88)
        print(f"PAGE {page_number}: {page_path}")
        print(f"regions={len(regions)} runtime_ms={runtime_ms:.1f}")
        print(
            f"GT coverage={record['page_metrics']['gt_block_coverage']} "
            f"GT-conditioned CER={page_cer}"
        )
        for row in block_metrics:
            print(
                f"  GT block {row['block_index']:02d} | "
                f"CER={row['cer']:.4f} | "
                f"regions={row['matched_region_count']:02d} | "
                f"pred={row['prediction_text']!r}"
            )

    total_gt = len(all_block_metrics)
    covered_gt = sum(1 for r in all_block_metrics if r["matched_region_count"] > 0)
    overall = {
        "sequence_id": args.sequence_id,
        "model": "nvidia/nemotron-ocr-v2",
        "merge_level": "word" if args.skip_relational else args.merge_level,
        "skip_relational": args.skip_relational,
        "pages": 3,
        "gt_story_blocks": total_gt,
        "gt_story_blocks_covered": covered_gt,
        "gt_story_block_coverage": covered_gt / total_gt if total_gt else None,
        "gt_conditioned_mean_cer": (
            sum(r["cer"] for r in all_block_metrics) / total_gt if total_gt else None
        ),
        "gt_conditioned_exact_match_rate": (
            sum(1 for r in all_block_metrics if r["exact_match"]) / total_gt if total_gt else None
        ),
        "mean_runtime_ms_per_page": (
            sum(r["runtime_ms"] for r in all_records) / len(all_records)
        ),
        "total_predicted_regions": sum(r["region_count"] for r in all_records),
        "total_regions_inside_story_gt_boxes": sum(
            r["page_metrics"]["inside_story_gt_boxes"] for r in all_records
        ),
        "total_regions_outside_story_gt_boxes": sum(
            r["page_metrics"]["outside_story_gt_boxes"] for r in all_records
        ),
        "note": (
            "Recognition CER is GT-conditioned: predictions are collected inside known story GT boxes. "
            "Do not interpret it as end-to-end story extraction performance."
        ),
    }

    raw_path = output_dir / f"nemotron_{'skiprel_word' if args.skip_relational else args.merge_level}_raw.jsonl"
    metrics_path = output_dir / f"nemotron_{'skiprel_word' if args.skip_relational else args.merge_level}_metrics.json"

    with raw_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("FINAL SUMMARY")
    print(json.dumps(overall, indent=2))
    print(f"Raw output    : {raw_path}")
    print(f"Metrics       : {metrics_path}")


if __name__ == "__main__":
    main()
