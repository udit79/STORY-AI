import cv2
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, r"vendor\comic-text-detector")

from inference import TextDetector
from paddleocr import TextRecognition

PAGE = Path(r"dataset\development\images\seq_2032620aa4e4ac7f\01.png")
OUT = Path(r"outputs\ocr_all_lines")
OUT.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(PAGE))
if img is None:
    raise FileNotFoundError(PAGE)

detector = TextDetector(
    model_path=r"vendor\comic-text-detector\data\comictextdetector.pt.onnx",
    input_size=1024,
    device="cuda",
)

_, _, blocks = detector(img)

recognizer = TextRecognition(
    model_name="en_PP-OCRv5_mobile_rec",
    engine="onnxruntime",
    device="gpu:0",
)

print(f"Detected blocks: {len(blocks)}")

items = []

for block_id, block in enumerate(blocks):
    for line_id, line in enumerate(block.lines):
        poly = np.asarray(line, dtype=np.int32)
        x1 = max(0, int(poly[:, 0].min()) - 8)
        y1 = max(0, int(poly[:, 1].min()) - 8)
        x2 = min(img.shape[1], int(poly[:, 0].max()) + 8)
        y2 = min(img.shape[0], int(poly[:, 1].max()) + 8)

        crop = img[y1:y2, x1:x2]

        h, w = crop.shape[:2]
        rotated = False

        # Vertical CTD lines need to be rotated for horizontal English OCR.
        if h > w * 1.5:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            rotated = True

        crop_path = OUT / f"b{block_id:02d}_l{line_id:02d}.png"
        cv2.imwrite(str(crop_path), crop)

        result_list = list(recognizer.predict(
            input=str(crop_path),
            batch_size=1,
        ))

        if not result_list:
            text = ""
            score = 0.0
        else:
            result = result_list[0]

            payload = result.json
            res = payload.get("res", payload)

            text = res.get("rec_text", "")
            score = float(res.get("rec_score", 0.0))

        record = {
            "block_id": block_id,
            "line_id": line_id,
            "bbox": [x1, y1, x2, y2],
            "rotated": rotated,
            "text": text,
            "score": score,
            "crop": str(crop_path),
        }

        items.append(record)

        print(
            f"B{block_id} L{line_id} "
            f"{'ROT' if rotated else 'HOR'} "
            f"score={score:.3f} "
            f"text={text!r}"
        )

import json

with open(OUT / "results.json", "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=2)

print()
print(f"Saved {len(items)} line results to {OUT / 'results.json'}")
