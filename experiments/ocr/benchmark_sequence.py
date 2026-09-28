import cv2
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, r"vendor\comic-text-detector")

from inference import TextDetector
from paddleocr import TextRecognition


SEQUENCE_ID = "seq_2032620aa4e4ac7f"

PAGES = [
    Path(r"dataset\development\images\seq_2032620aa4e4ac7f\01.png"),
    Path(r"dataset\development\images\seq_2032620aa4e4ac7f\02.png"),
    Path(r"dataset\development\images\seq_2032620aa4e4ac7f\03.png"),
]

OUT = Path(r"outputs\ocr_sequence_benchmark") / SEQUENCE_ID
OUT.mkdir(parents=True, exist_ok=True)


def order_quad(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)

    s = points.sum(axis=1)
    d = np.diff(points, axis=1).ravel()

    return np.array([
        points[np.argmin(s)],
        points[np.argmin(d)],
        points[np.argmax(s)],
        points[np.argmax(d)],
    ], dtype=np.float32)


def rectify_line(img: np.ndarray, polygon: np.ndarray) -> tuple[np.ndarray, bool]:
    polygon = np.asarray(polygon, dtype=np.float32)

    # CTD normally returns quadrilateral line polygons.
    if len(polygon) == 4:
        src = order_quad(polygon)

        w = max(
            1,
            int(round(max(
                np.linalg.norm(src[2] - src[3]),
                np.linalg.norm(src[1] - src[0]),
            )))
        )

        h = max(
            1,
            int(round(max(
                np.linalg.norm(src[1] - src[2]),
                np.linalg.norm(src[0] - src[3]),
            )))
        )

        dst = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src, dst)
        crop = cv2.warpPerspective(img, matrix, (w, h))

    else:
        rect = cv2.minAreaRect(polygon)
        box = cv2.boxPoints(rect).astype(np.float32)
        box = order_quad(box)

        w = max(1, int(round(rect[1][0])))
        h = max(1, int(round(rect[1][1])))

        dst = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(box, dst)
        crop = cv2.warpPerspective(img, matrix, (w, h))

    rotated = False

    h, w = crop.shape[:2]

    if h > w * 1.35:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        rotated = True

    # Upscale small manga text.
    crop = cv2.resize(
        crop,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC,
    )

    # Padding helps the recognition model around tight CTD crops.
    crop = cv2.copyMakeBorder(
        crop,
        12,
        12,
        12,
        12,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    return crop, rotated


def parse_result(result):
    payload = result.json
    res = payload.get("res", payload)

    return (
        str(res.get("rec_text", "")),
        float(res.get("rec_score", 0.0)),
    )


detector = TextDetector(
    model_path=r"vendor\comic-text-detector\data\comictextdetector.pt.onnx",
    input_size=1024,
    device="cuda",
)

recognizer = TextRecognition(
    model_name="en_PP-OCRv5_mobile_rec",
    engine="onnxruntime",
    device="gpu:0",
)


all_results = []


for page_index, page_path in enumerate(PAGES):

    print()
    print("=" * 70)
    print(f"PAGE {page_index + 1}: {page_path.name}")
    print("=" * 70)

    img = cv2.imread(str(page_path))

    if img is None:
        raise FileNotFoundError(page_path)

    _, _, blocks = detector(img)

    print(f"Detected blocks: {len(blocks)}")

    page_out = OUT / f"page_{page_index + 1:02d}"
    page_out.mkdir(parents=True, exist_ok=True)

    page_items = []

    for block_id, block in enumerate(blocks):

        for line_id, line in enumerate(block.lines):

            polygon = np.asarray(line, dtype=np.float32)

            crop, rotated = rectify_line(img, polygon)

            crop_path = (
                page_out /
                f"b{block_id:02d}_l{line_id:02d}.png"
            )

            cv2.imwrite(str(crop_path), crop)

            results = list(
                recognizer.predict(
                    input=str(crop_path),
                    batch_size=1,
                )
            )

            if results:
                text, score = parse_result(results[0])
            else:
                text, score = "", 0.0

            x1 = int(polygon[:, 0].min())
            y1 = int(polygon[:, 1].min())
            x2 = int(polygon[:, 0].max())
            y2 = int(polygon[:, 1].max())

            item = {
                "sequence_id": SEQUENCE_ID,
                "page_index": page_index,
                "page": page_path.name,
                "block_id": block_id,
                "line_id": line_id,
                "bbox": [x1, y1, x2, y2],
                "rotated": rotated,
                "text": text,
                "score": score,
                "crop": str(crop_path),
            }

            page_items.append(item)
            all_results.append(item)

            print(
                f"B{block_id} L{line_id} "
                f"{'ROT' if rotated else 'HOR'} "
                f"score={score:.3f} "
                f"text={text!r}"
            )

    with open(page_out / "results.json", "w", encoding="utf-8") as f:
        json.dump(page_items, f, ensure_ascii=False, indent=2)


with open(OUT / "results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)


print()
print("=" * 70)
print(f"TOTAL OCR LINES: {len(all_results)}")
print(f"Saved: {OUT / 'results.json'}")
print("=" * 70)