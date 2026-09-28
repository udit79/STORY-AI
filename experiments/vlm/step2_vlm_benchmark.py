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


def run_vlm(model, processor, image_path: Path, ocr_text: str):
    prompt = f"""
You are a visual OCR verifier for English manga.

Read ONLY the text contained in this specific crop.

OCR candidate:
{ocr_text}

Determine the best visually supported transcription.

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
- Do not invent text.
- Do not include text outside this crop.
- Do not merge this crop with other manga dialogue.
- visual_confidence is a model-generated visual estimate, not a calibrated probability.
"""

    messages = [
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

    video_metadatas = None

    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos = list(videos)
        video_metadatas = list(video_metadatas)

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

    inputs = inputs.to(model.device)

    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(inputs.input_ids, generated_ids)
    ]

    output = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = {
            "transcription": output,
            "candidate_supported": None,
            "visual_confidence": None,
            "text_type": "unknown",
            "notes": "Model output was not valid JSON.",
        }

    return parsed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=r"outputs\ocr_all_lines\results.json",
    )

    parser.add_argument(
        "--output",
        default=r"outputs\vlm_benchmark\results.jsonl",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"ERROR: missing input: {input_path}", file=sys.stderr)
        raise SystemExit(2)

    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer actual OCR regions over empty detections.
    records = [
        r for r in records
        if str(r.get("text", "")).strip()
        and str(r.get("crop", "")).strip()
    ]

    records = records[: args.limit]

    print(f"Selected {len(records)} OCR regions.")

    print("Loading Qwen3-VL...")
    model, processor = load_model()

    results = []

    for i, record in enumerate(records, start=1):
        crop = Path(record["crop"])

        if not crop.is_absolute():
            crop = Path(crop)

        if not crop.exists():
            print(f"[{i}] SKIP missing crop: {crop}")
            continue

        ocr_text = str(record.get("text", "")).strip()
        ocr_score = float(record.get("score", 0.0))

        print()
        print("=" * 70)
        print(f"[{i}/{len(records)}]")
        print(f"Crop : {crop}")
        print(f"OCR  : {ocr_text!r}")
        print(f"Score: {ocr_score:.4f}")

        try:
            vlm = run_vlm(
                model,
                processor,
                crop,
                ocr_text,
            )
        except Exception as exc:
            print(f"VLM ERROR: {exc}")
            vlm = {
                "transcription": "",
                "candidate_supported": None,
                "visual_confidence": None,
                "text_type": "error",
                "notes": str(exc),
            }

        vlm_text = str(vlm.get("transcription", "")).strip()

        exact_agreement = (
            vlm_text.casefold() == ocr_text.casefold()
            if vlm_text
            else False
        )

        result = {
            "block_id": record.get("block_id"),
            "line_id": record.get("line_id"),
            "crop": str(crop),
            "ocr_text": ocr_text,
            "ocr_score": ocr_score,
            "vlm_text": vlm_text,
            "vlm_confidence": vlm.get("visual_confidence"),
            "vlm_supported": vlm.get("candidate_supported"),
            "vlm_text_type": vlm.get("text_type"),
            "vlm_notes": vlm.get("notes"),
            "exact_agreement": exact_agreement,
        }

        results.append(result)

        print(f"VLM  : {vlm_text!r}")
        print(f"VLM confidence: {vlm.get('visual_confidence')}")
        print(f"Supported: {vlm.get('candidate_supported')}")
        print(f"Exact agreement: {exact_agreement}")

    with output_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    agreement_count = sum(
        bool(r["exact_agreement"])
        for r in results
    )

    print()
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print(f"Regions processed : {len(results)}")
    print(f"Exact OCR/VLM     : {agreement_count}")
    print(
        f"Agreement rate    : "
        f"{agreement_count / len(results):.3f}"
        if results
        else "Agreement rate    : N/A"
    )
    print(f"Saved             : {output_path}")


if __name__ == "__main__":
    main()
