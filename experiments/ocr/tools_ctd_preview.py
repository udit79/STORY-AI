import cv2
import sys
from pathlib import Path

sys.path.insert(0, r"vendor\comic-text-detector")

from inference import TextDetector

image_path = Path(r"dataset\development\images\seq_2032620aa4e4ac7f\01.png")
output_path = Path(r"outputs\ctd_preview.png")
output_path.parent.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(image_path))
if img is None:
    raise FileNotFoundError(image_path)

detector = TextDetector(
    model_path=r"vendor\comic-text-detector\data\comictextdetector.pt.onnx",
    input_size=1024,
    device="cuda",
)

_, _, blocks = detector(img)

preview = img.copy()

for i, block in enumerate(blocks):
    x1, y1, x2, y2 = map(int, block.xyxy)

    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        preview,
        f"{i}:{block.language}",
        (x1, max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
    )

    for line in block.lines:
        pts = cv2.UMat if False else None
        poly = __import__("numpy").array(line, dtype="int32")
        poly = poly.reshape((-1, 1, 2))
        cv2.polylines(preview, [poly], True, (255, 0, 0), 1)

cv2.imwrite(str(output_path), preview)

print(f"Detected blocks: {len(blocks)}")
print(f"Saved: {output_path}")
