import torch
import laya

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

agent = laya.load(
    "convaiinnovations/laya",
    device="cuda",
)

state = {
    "width": 58,
    "height": 76,
    "aspect_ratio": 0.76,
    "vertical": True,
    "font_size": 58,
    "baseline_text": "WMM MEN ME",
    "baseline_confidence": 0.38,
}

questions = {
    "preprocess": {
        "type": "choice",
        "instructions": "Which preprocessing strategy should be tried next for this OCR region?",
        "criteria": {
            "rotate_cw": "rotate the region clockwise by 90 degrees before OCR",
            "rotate_ccw": "rotate the region counter-clockwise by 90 degrees before OCR",
            "upscale": "keep orientation and upscale the region before OCR",
            "contrast": "enhance contrast before OCR",
        },
    }
}

result = agent.predict(state, questions)

print("result:")
print(result)
