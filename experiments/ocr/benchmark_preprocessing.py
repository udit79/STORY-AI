import cv2
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, r"vendor\comic-text-detector")

from inference import TextDetector
from paddleocr import TextRecognition


PAGE = Path(r"dataset\development\images\seq_2032620aa4e4ac7f\01.png")
OUT = Path(r"outputs\ocr_preprocess_benchmark")
OUT.mkdir(parents=True, exist_ok=True)

IMG = cv2.imread(str(PAGE))
if IMG is None:
    raise FileNotFoundError(PAGE)


def order_quad(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)

    s = points.sum(axis=1)
    d = np.diff(points, axis=1).ravel()

    return np.array([
        points[np.argmin(s)],  # top-left
        points[np.argmin(d)],  # top-right
        points[np.argmax(s)],  # bottom-right
        points[np.argmax(d)],  # bottom-left
    ], dtype=np.float32)


def rectify_polygon(poly: np.ndarray, scale: int = 4, pad: int = 12):
    poly = np.asarray(poly, dtype=np.float32)

    # CTD lines are normally quadrilaterals.
    # For arbitrary polygons, use the minimum-area rectangle.
    if len(poly) == 4:
        src = order_quad(poly)

        width_a = np.linalg.norm(src[2] - src[3])
        width_b = np.linalg.norm(src[1] - src[0])
        height_a = np.linalg.norm(src[1] - src[2])
        height_b = np.linalg.norm(src[0] - src[3])

        w = max(1, int(round(max(width_a, width_b))))
        h = max(1, int(round(max(height_a, height_b))))

        dst = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        crop = cv2.warpPerspective(IMG, M, (w, h))
    else:
        rect = cv2.minAreaRect(poly)
        box = cv2.boxPoints(rect).astype(np.float32)
        box = order_quad(box)

        w = max(1, int(round(rect[1][0])))
        h = max(1, int(round(rect[1][1])))

        w = max(w, 1)
        h = max(h, 1)

        dst = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(box, dst)
        crop = cv2.warpPerspective(IMG, M, (w, h))

    # OCR wants normal horizontal text.
    h, w = crop.shape[:2]
    if h > w * 1.35:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

    # Upscale small manga text.
    crop = cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    # Give the recognizer some breathing room.
    crop = cv2.copyMakeBorder(
        crop,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    return crop


def extract_result(result):
    payload = result.json
    res = payload.get("res", payload)

    return (
        str(res.get("rec_text", "")),
        float(res.get("rec_score", 0.0)),
    )


# -----------------------------
# CTD
# -----------------------------
detector = TextDetector(
    model_path=r"vendor\comic-text-detector\data\comictextdetector.pt.onnx",
    input_size=1024,
    device="cuda",
)

_, _, blocks = detector(IMG)

print(f"Detected blocks: {len(blocks)}")


# -----------------------------
# OCR model
# -----------------------------
recognizer = TextRecognition(
    model_name="en_PP-OCRv5_mobile_rec",
    engine="onnxruntime",
    device="gpu:0",
)


items = []
improved_inputs = []
improved_meta = []


# -----------------------------
# Build improved crops
# -----------------------------
for block_id, block in enumerate(blocks):
    for line_id, line in enumerate(block.lines):

        poly = np.asarray(line, dtype=np.float32)

        crop = rectify_polygon(poly)

        crop_path = OUT / f"b{block_id:02d}_l{line_id:02d}_improved.png"
        cv2.imwrite(str(crop_path), crop)

        improved_inputs.append(str(crop_path))
        improved_meta.append(
            {
                "block_id": block_id,
                "line_id": line_id,
                "bbox": [
                    int(poly[:, 0].min()),
                    int(poly[:, 1].min()),
                    int(poly[:, 0].max()),
                    int(poly[:, 1].max()),
                ],
                "crop": str(crop_path),
            }
        )


# -----------------------------
# Batch OCR
# -----------------------------
results = list(
    recognizer.predict(
        input=improved_inputs,
        batch_size=8,
    )
)


for meta, result in zip(improved_meta, results):
    text, score = extract_result(result)

    record = {
        **meta,
        "text": text,
        "score": score,
    }

    items.append(record)

    print(
        f"B{meta['block_id']} "
        f"L{meta['line_id']} "
        f"score={score:.3f} "
        f"text={text!r}"
    )


with open(OUT / "results.json", "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

print()
print(f"Saved {len(items)} results to:")
print(OUT / "results.json")