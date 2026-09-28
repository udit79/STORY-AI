import cv2
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, r"vendor\comic-text-detector")

from inference import TextDetector
from paddleocr import TextRecognition

page = Path(r"dataset\development\images\seq_2032620aa4e4ac7f\01.png")
out = Path(r"outputs\ocr_test")
out.mkdir(parents=True, exist_ok=True)

img = cv2.imread(str(page))
if img is None:
    raise FileNotFoundError(page)

# CTD detection
detector = TextDetector(
    model_path=r"vendor\comic-text-detector\data\comictextdetector.pt.onnx",
    input_size=1024,
    device="cuda",
)

_, _, blocks = detector(img)

print("Detected blocks:", len(blocks))

# Use the first detected block and its first text line.
block = blocks[0]
line = np.asarray(block.lines[0], dtype=np.int32)

x1 = max(0, int(line[:, 0].min()) - 8)
y1 = max(0, int(line[:, 1].min()) - 8)
x2 = min(img.shape[1], int(line[:, 0].max()) + 8)
y2 = min(img.shape[0], int(line[:, 1].max()) + 8)

crop = img[y1:y2, x1:x2]

crop_path = out / "line0.png"
cv2.imwrite(str(crop_path), crop)

print("Crop:", crop.shape)
print("Crop saved:", crop_path)

# English recognition
recognizer = TextRecognition(
    model_name="en_PP-OCRv5_mobile_rec",
    engine="onnxruntime",
    device="gpu:0",
)

results = recognizer.predict(input=str(crop_path), batch_size=1)

for result in results:
    result.print()
