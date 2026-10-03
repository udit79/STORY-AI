"""Compare PaddleOCR and Nemotron on identical labelled development crops.

Paddle uses CTD line crops within the balloons; Nemotron receives the original
balloon crops. Startup, preprocessing, and inference wall times are reported
separately, with inference measured on the shared development examples.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paddleocr import TextRecognition

from app.models.ctd import load_ctd_detector
from app.models.nemotron_wsl_bridge import NemotronWSLBridge
from experiments.ocr.nemotron_balloon_crop_benchmark import cer, norm, wer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT
        / "outputs/nemotron_balloon_crop_benchmark/nemotron_balloon_crop_results.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/ocr_head_to_head",
    )
    parser.add_argument("--model", default="en_PP-OCRv5_mobile_rec")
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--ctd-model",
        type=Path,
        default=ROOT / "vendor/comic-text-detector/data/comictextdetector.pt.onnx",
    )
    parser.add_argument("--ctd-device", default="cuda")
    parser.add_argument("--nemotron-batch-size", type=int, default=8)
    return parser.parse_args()


def result_text(result: Any) -> tuple[str, float]:
    payload = result.json
    record = payload.get("res", payload)
    return str(record.get("rec_text", "")), float(record.get("rec_score", 0.0))


def order_quad(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()
    return np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(diffs)],
            points[np.argmax(sums)],
            points[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )


def rectify_line(image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    polygon = np.asarray(polygon, dtype=np.float32)
    if len(polygon) == 4:
        source = order_quad(polygon)
        width = max(
            1,
            round(
                max(
                    np.linalg.norm(source[2] - source[3]),
                    np.linalg.norm(source[1] - source[0]),
                )
            ),
        )
        height = max(
            1,
            round(
                max(
                    np.linalg.norm(source[1] - source[2]),
                    np.linalg.norm(source[0] - source[3]),
                )
            ),
        )
        target = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        crop = cv2.warpPerspective(
            image, cv2.getPerspectiveTransform(source, target), (width, height)
        )
    else:
        x1, y1 = np.floor(polygon.min(axis=0)).astype(int)
        x2, y2 = np.ceil(polygon.max(axis=0)).astype(int)
        crop = image[max(0, y1) : max(1, y2), max(0, x1) : max(1, x2)]
    if crop.size == 0:
        return crop
    height, width = crop.shape[:2]
    if height > width * 1.35:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(
        crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )


def main() -> int:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if not row.get("coverage_failure")]
    if not rows:
        raise SystemExit(f"No matched labelled crops in {args.input}")

    crop_paths = [Path(row["crop_path"]) for row in rows]
    missing = [str(path) for path in crop_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} benchmark crops are missing; first: {missing[0]}"
        )

    detector = load_ctd_detector(
        args.ctd_model, device=args.ctd_device, input_size=1024
    )
    prep_started = time.perf_counter()
    line_crops: list[np.ndarray] = []
    line_meta: list[tuple[int, float, float]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = args.output_dir / "paddle_line_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    for row_index, path in enumerate(crop_paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        _, _, blocks = detector(image)
        for block in blocks or []:
            for polygon in block.lines:
                points = np.asarray(polygon, dtype=np.float32)
                line_crop = rectify_line(image, points)
                if line_crop.size == 0:
                    continue
                ys, xs = points[:, 1], points[:, 0]
                line_crops.append(line_crop)
                line_meta.append((row_index, float(ys.mean()), float(xs.mean())))

    if not line_crops:
        raise RuntimeError("CTD found no text lines in any benchmark balloon")
    line_paths = []
    for line_index, crop in enumerate(line_crops):
        line_path = crop_dir / f"line_{line_index:05d}.png"
        cv2.imwrite(str(line_path), crop)
        line_paths.append(str(line_path))
    paddle_preprocessing_elapsed = time.perf_counter() - prep_started

    paddle_startup_started = time.perf_counter()
    recognizer = TextRecognition(
        model_name=args.model,
        engine="onnxruntime",
        device=args.device,
    )
    paddle_startup_elapsed = time.perf_counter() - paddle_startup_started
    started = time.perf_counter()
    results = list(recognizer.predict(input=line_paths, batch_size=args.batch_size))
    elapsed = time.perf_counter() - started
    if len(results) != len(line_crops):
        raise RuntimeError(
            f"Paddle returned {len(results)} predictions for {len(line_crops)} CTD line crops"
        )

    nemotron_startup_started = time.perf_counter()
    bridge = NemotronWSLBridge.start(merge_level="paragraph")
    nemotron_startup_elapsed = time.perf_counter() - nemotron_startup_started
    nemotron_started = time.perf_counter()
    nemotron_texts: list[str] = []
    for start in range(0, len(crop_paths), args.nemotron_batch_size):
        predictions = bridge.ocr_batch(
            crop_paths[start : start + args.nemotron_batch_size],
            merge_level="paragraph",
        )
        nemotron_texts.extend(
            " ".join(str(item.get("text", "")) for item in output).strip()
            for output in predictions
        )
    nemotron_inference_elapsed = time.perf_counter() - nemotron_started
    bridge.shutdown()
    if len(nemotron_texts) != len(rows):
        raise RuntimeError(
            f"Nemotron returned {len(nemotron_texts)} predictions for {len(rows)} balloon crops"
        )

    paddle_parts: list[list[tuple[float, float, str]]] = [[] for _ in rows]
    for meta, result in zip(line_meta, results, strict=True):
        row_index, y, x = meta
        text, _ = result_text(result)
        if norm(text):
            paddle_parts[row_index].append((y, x, text.strip()))

    paired: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        parts = sorted(paddle_parts[row_index], key=lambda item: (item[0], item[1]))
        text = " ".join(part[2] for part in parts)
        truth = str(row.get("ground_truth_text", ""))
        paired.append(
            {
                **{
                    key: row.get(key)
                    for key in ("sequence_id", "page_index", "balloon_id", "crop_path")
                },
                "ground_truth_text": truth,
                "nemotron_text": nemotron_texts[row_index],
                "paddle_text": text,
                "paddle_line_count": len(parts),
                "nemotron_CER": cer(truth, nemotron_texts[row_index]),
                "paddle_CER": cer(truth, text),
                "nemotron_WER": wer(truth, nemotron_texts[row_index]),
                "paddle_WER": wer(truth, text),
                "nemotron_exact": norm(truth) == norm(nemotron_texts[row_index]),
                "paddle_exact": norm(truth) == norm(text),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "paired_results.jsonl"
    result_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in paired) + "\n",
        encoding="utf-8",
    )
    n = len(paired)
    nemotron_cer = sum(row["nemotron_CER"] for row in paired) / n
    paddle_cer = sum(row["paddle_CER"] for row in paired) / n
    nemotron_wer = sum(row["nemotron_WER"] for row in paired) / n
    paddle_wer = sum(row["paddle_WER"] for row in paired) / n
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        by_sequence[str(row["sequence_id"])].append(row)
    summary = {
        "comparison": "paired OCR recognition on identical production balloon crops and development references",
        "reference_source": str(args.input),
        "examples": n,
        "paddle_model": args.model,
        "paddle_device": args.device,
        "batch_size": args.batch_size,
        "paddle_pipeline": "CTD text-line detection inside each identical balloon crop, perspective rectification, 4x upscale, PaddleOCR recognition, top-to-bottom then left-to-right reassembly",
        "paddle_inference_seconds": round(elapsed, 3),
        "paddle_preprocessing_seconds": round(paddle_preprocessing_elapsed, 3),
        "paddle_startup_seconds": round(paddle_startup_elapsed, 3),
        "paddle_total_seconds_excluding_startup": round(
            paddle_preprocessing_elapsed + elapsed, 3
        ),
        "nemotron_startup_seconds": round(nemotron_startup_elapsed, 3),
        "nemotron_batch_size": args.nemotron_batch_size,
        "nemotron_inference_seconds": round(nemotron_inference_elapsed, 3),
        "normalization": "uppercase, collapse whitespace, strip outer whitespace; punctuation retained",
        "nemotron": {
            "mean_CER": nemotron_cer,
            "mean_WER": nemotron_wer,
            "exact_match_rate": sum(row["nemotron_exact"] for row in paired) / n,
        },
        "paddle": {
            "mean_CER": paddle_cer,
            "mean_WER": paddle_wer,
            "exact_match_rate": sum(row["paddle_exact"] for row in paired) / n,
        },
        "paired_wins": {
            "nemotron_lower_CER": sum(
                row["nemotron_CER"] < row["paddle_CER"] for row in paired
            ),
            "paddle_lower_CER": sum(
                row["paddle_CER"] < row["nemotron_CER"] for row in paired
            ),
            "ties_CER": sum(row["paddle_CER"] == row["nemotron_CER"] for row in paired),
        },
        "per_sequence": {
            sequence_id: {
                "examples": len(sequence_rows),
                "nemotron_mean_CER": sum(row["nemotron_CER"] for row in sequence_rows)
                / len(sequence_rows),
                "paddle_mean_CER": sum(row["paddle_CER"] for row in sequence_rows)
                / len(sequence_rows),
                "nemotron_exact_match_rate": sum(
                    row["nemotron_exact"] for row in sequence_rows
                )
                / len(sequence_rows),
                "paddle_exact_match_rate": sum(
                    row["paddle_exact"] for row in sequence_rows
                )
                / len(sequence_rows),
            }
            for sequence_id, sequence_rows in sorted(by_sequence.items())
        },
        "results": str(result_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
