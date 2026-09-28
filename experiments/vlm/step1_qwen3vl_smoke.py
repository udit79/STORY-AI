from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        dtype=torch.float16,
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID)

    return model, processor


def build_messages(image_path: Path, ocr_candidate: str | None):
    if ocr_candidate:
        prompt = f"""
You are a visual OCR verifier for English manga.

Read only the story text in the supplied image.

Do not include:
- sound effects
- logos
- watermarks
- page numbers
- advertisements
- clothing/object text
- document/interface text
- other non-story text

OCR candidate:
{ocr_candidate}

Determine whether the OCR candidate is visually supported.

Return ONLY valid JSON:

{{
  "transcription": "best visual transcription",
  "candidate_supported": true,
  "visual_confidence": 0.0,
  "text_type": "dialogue|thought|narration|vocalisation|sound_effect|other",
  "notes": "brief reason"
}}

Rules:
- Preserve visible wording and punctuation.
- Do not invent missing words.
- visual_confidence is only a visual estimate.
- Lower confidence when characters are unclear.
"""
    else:
        prompt = """
You are a visual OCR model for English manga.

Read only the story text in the supplied image.

Exclude sound effects, logos, watermarks, page numbers,
advertisements, clothing/object text, document/interface text,
and other non-story text.

Return ONLY valid JSON:

{
  "transcription": "best visual transcription",
  "candidate_supported": true,
  "visual_confidence": 0.0,
  "text_type": "dialogue|thought|narration|vocalisation|sound_effect|other",
  "notes": "brief reason"
}

Rules:
- Preserve visible wording and punctuation.
- Do not invent missing words.
- Lower confidence when characters are unclear.
"""

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path.resolve().as_posix(),
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Path to manga image/crop",
    )

    parser.add_argument(
        "--ocr",
        default=None,
        help="Optional OCR candidate for verification",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        raise SystemExit(2)

    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception as exc:
        print(f"ERROR: invalid image: {exc}", file=sys.stderr)
        raise SystemExit(2)

    print(f"Loading {MODEL_ID}...")
    print("4-bit quantization enabled.")

    model, processor = load_model()

    messages = build_messages(image_path, args.ocr)

    # Qwen's official vision preprocessing path.
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos = list(videos)
        video_metadatas = list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        padding=True,
        do_resize=False,
        **video_kwargs,
    )

    # Avoid CPU/GPU mismatch with multimodal inputs.
    inputs = inputs.to(model.device)

    # Some Transformers versions include this field although the
    # Qwen3-VL model does not need it.
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    print("Running inference...")

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    print("\n========== RAW OUTPUT ==========")
    print(output_text)

    try:
        payload = json.loads(output_text)

        print("\n========== PARSED JSON ==========")
        print(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            )
        )

    except json.JSONDecodeError:
        print("\nWARNING: Model did not return valid JSON.")

    print("\n========== GPU ==========")

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    print(f"Allocated: {allocated:.2f} GiB")
    print(f"Reserved : {reserved:.2f} GiB")


if __name__ == "__main__":
    main()
