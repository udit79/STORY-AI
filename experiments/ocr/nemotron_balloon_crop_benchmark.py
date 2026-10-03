"""Benchmark Nemotron on production final balloon crops.

The oracle supplies labelled development text boxes and text only.  Crop
geometry comes from the production CTD/local-layout/grouping/crop path; an
unmatched oracle box is reported as a coverage failure, never as an OCR miss.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.layout_grouping import PageLayoutGrouper
from app.models.balloon_crops import BalloonCropStore
from app.models.ctd import CTDPageLocalizer, load_ctd_detector
from app.models.manga_layout import load_manga109_balloon_provider
from app.models.nemotron_wsl_bridge import NemotronWSLBridge
from app.schemas.page import BoundingBox, PageRepresentation, TextRegion


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sequences", nargs="+", default=None)
    p.add_argument("--model-dir", type=Path, default=ROOT / "nemotron-ocr-v2" / "v2_english")
    p.add_argument("--ctd-model", type=Path, default=ROOT / "vendor" / "comic-text-detector" / "data" / "comictextdetector.pt.onnx")
    p.add_argument("--layout-weights", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "nemotron_balloon_crop_benchmark")
    p.add_argument("--merge-level", choices=["word", "sentence", "paragraph"], default="paragraph")
    return p.parse_args()


def norm(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").upper()).strip()


def distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def token_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, token in enumerate(a, 1):
        cur = [i]
        for j, other in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (token != other)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = norm(reference), norm(hypothesis)
    return distance(ref, hyp) / len(ref) if ref else float(bool(hyp))


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = norm(reference).split(), norm(hypothesis).split()
    if not ref:
        return float(bool(hyp))
    return token_distance(ref, hyp) / len(ref)


def box(row: dict[str, Any]) -> BoundingBox:
    f = row["features"]
    return BoundingBox(x1=float(f["x1"]), y1=float(f["y1"]), x2=float(f["x2"]), y2=float(f["y2"]))


def area(b: BoundingBox) -> float:
    return max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)


def iou(a: BoundingBox, b: BoundingBox) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = area(a) + area(b) - inter
    return inter / union if union else 0.0


def load_oracle(path: Path, sequence_id: str) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("sequence_id") == sequence_id:
            rows.append(row)
    return rows


def wsl_path(path: Path) -> str:
    drive = path.drive.rstrip(":").lower()
    return f"/mnt/{drive}/{path.as_posix().split(':', 1)[-1].lstrip('/')}" if drive else path.as_posix()


def classify(row: dict[str, Any], balloon: Any | None, text: str) -> dict[str, bool]:
    features = row.get("features") or {}
    truth_punctuation = "".join(ch for ch in norm(row.get("target_text", "")) if not ch.isalnum() and not ch.isspace())
    output_punctuation = "".join(ch for ch in norm(text) if not ch.isalnum() and not ch.isspace())
    return {
        "empty_output": not bool(norm(text)),
        "punctuation_error": truth_punctuation != output_punctuation,
        "multi_line": int(features.get("line_count", 1)) > 1,
        "multi_fragment_balloon": bool(balloon and len(balloon.text_region_ids) > 1),
        "stylized_text": bool(features.get("orientation") == "vertical" or int(features.get("line_count", 1)) >= 6),
    }


def main() -> int:
    cfg = args()
    oracle_dir = ROOT / "experiments" / "laya" / "data"
    sequences = cfg.sequences or sorted(p.stem.removeprefix("preprocessing_oracle_") for p in oracle_dir.glob("preprocessing_oracle_*.jsonl"))
    layout_weights = cfg.layout_weights
    if layout_weights is None:
        layout_weights = next((p / "best.pt" for p in Path.home().glob(".cache/huggingface/hub/models--huyvux3005--manga109-segmentation-bubble/snapshots/*") if (p / "best.pt").exists()), None)
    if layout_weights is None:
        raise FileNotFoundError("Manga layout weights not found; pass --layout-weights")

    localizer = CTDPageLocalizer(load_ctd_detector(str(cfg.ctd_model), device="cpu"))
    layout_provider = load_manga109_balloon_provider(layout_weights)
    crop_store = BalloonCropStore(cfg.output_dir / "crops")
    bridge = NemotronWSLBridge.start(model_dir=wsl_path(cfg.model_dir), merge_level=cfg.merge_level)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for sequence_id in sequences:
            oracle_path = oracle_dir / f"preprocessing_oracle_{sequence_id}.jsonl"
            gt_rows = load_oracle(oracle_path, sequence_id)
            by_page: dict[int, list[dict[str, Any]]] = {}
            for row in gt_rows:
                by_page.setdefault(int(row["page_index"]), []).append(row)
            for page_index in range(3):
                image_path = ROOT / "dataset" / "development" / "images" / sequence_id / f"{page_index + 1:02d}.png"
                regions = localizer.localize(image_path)
                page = PageRepresentation(page_index=page_index, image_path=str(image_path), text_regions=regions)
                grouped = PageLayoutGrouper(panel_provider=layout_provider, balloon_provider=layout_provider).group_page(page, sequence_id, image_path)
                crop_paths: dict[str, Path] = {}
                predictions: dict[str, str] = {}
                for balloon in grouped.page.balloons:
                    crop_paths[balloon.id] = crop_store.create(image_path, sequence_id, page_index, balloon)
                for gt in by_page.get(page_index, []):
                    gt_box = box(gt)
                    matches = sorted(((iou(gt_box, b.bbox), b) for b in grouped.page.balloons), key=lambda item: item[0], reverse=True)
                    score, balloon = matches[0] if matches else (0.0, None)
                    matched = balloon if score > 0.05 else None
                    if matched is None:
                        rows.append({"sequence_id": sequence_id, "page_index": page_index, "balloon_id": None, "ground_truth_text": gt.get("target_text", ""), "nemotron_text": None, "CER": None, "WER": None, "exact_match": None, "coverage_failure": True, **classify(gt, None, "")})
                        continue
                    if matched.id not in predictions:
                        output = bridge(str(crop_paths[matched.id]), merge_level=cfg.merge_level)
                        predictions[matched.id] = " ".join(str(item.get("text", "")) for item in output).strip()
                    text = predictions[matched.id]
                    truth = str(gt.get("target_text", ""))
                    rows.append({"sequence_id": sequence_id, "page_index": page_index, "balloon_id": matched.id, "ground_truth_text": truth, "nemotron_text": text, "CER": cer(truth, text), "WER": wer(truth, text), "exact_match": norm(truth) == norm(text), "coverage_failure": False, "crop_path": str(crop_paths[matched.id]), **classify(gt, matched, text)})
    finally:
        bridge.shutdown()

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = cfg.output_dir / "nemotron_balloon_crop_results.jsonl"
    result_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    matched = [r for r in rows if not r["coverage_failure"]]
    summary = {"gt_balloons": len(rows), "matched_crops": len(matched), "coverage": len(matched) / len(rows) if rows else 0.0, "CER": sum(r["CER"] for r in matched) / len(matched) if matched else None, "WER": sum(r["WER"] for r in matched) / len(matched) if matched else None, "exact_match": sum(r["exact_match"] for r in matched), "exact_match_rate": sum(r["exact_match"] for r in matched) / len(matched) if matched else None, "empty_outputs": sum(r["empty_output"] for r in matched), "difficult_or_multi_fragment": sum(r["multi_line"] or r["multi_fragment_balloon"] or r["stylized_text"] for r in matched), "runtime_seconds": round(time.perf_counter() - started, 2), "results_path": str(result_path)}
    (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Results: {result_path}")
    print("\nExamples:")
    for row in (matched[:10]):
        print(f"GT: {row['ground_truth_text']}\nNemotron: {row['nemotron_text']}\nCER: {row['CER']:.4f}\nStatus: {'EXACT' if row['exact_match'] else 'OCR_FAILURE'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
