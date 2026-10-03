"""Explicit local model initialization; importing this module loads no weights."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from app.models.candidate_adapters import (
    NemotronCandidateAdapter,
    Qwen3VLCandidateAdapter,
)


def load_paddle_recognizer(
    model_dir: str | Path,
    *,
    model_name: str = "en_PP-OCRv5_mobile_rec",
    engine: str = "onnxruntime",
    device: str = "cpu",
) -> Any:
    """Load PaddleOCR from an explicit local model directory (no auto-download)."""
    local_model_dir = Path(model_dir).resolve()
    if not local_model_dir.is_dir():
        raise FileNotFoundError(f"PaddleOCR model directory not found: {local_model_dir}")
    try:
        from paddleocr import TextRecognition
    except ImportError as exc:
        raise RuntimeError("PaddleOCR is not installed in the active environment") from exc
    return TextRecognition(
        model_name=model_name,
        model_dir=str(local_model_dir),
        engine=engine,
        device=device,
    )


class Qwen3VLLocalRunner:
    """Reuse one explicitly loaded Qwen3-VL model across crop/proposal calls."""

    def __init__(self, model: Any, processor: Any, max_new_tokens: int = 256) -> None:
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens

    def generate_json(
        self,
        image_paths: Sequence[str | Path],
        prompt: str,
    ) -> dict[str, Any]:
        """Run a caller-supplied prompt over ordered local images and parse JSON."""
        import torch
        from qwen_vl_utils import process_vision_info

        if not image_paths:
            raise ValueError("at least one local image is required for Qwen inference")
        messages = [
            {
                "role": "user",
                "content": [
                    *(
                        {"type": "image", "image": Path(image_path).resolve().as_posix()}
                        for image_path in image_paths
                    ),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt_text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated_ids = generated[:, inputs["input_ids"].shape[1] :]
        raw_output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "_raw_output": raw_output,
                "_json_parse_error": "Qwen output was not valid JSON.",
            }
        if not isinstance(parsed, Mapping):
            return {
                "_raw_output": raw_output,
                "_json_parse_error": "Qwen JSON output was not an object.",
            }
        return {**parsed, "_raw_output": raw_output}

    def __call__(self, image_path: Path) -> dict[str, Any]:
        prompt = """You are given ONE localized manga text block.
Read only text visible inside this crop. Do not infer from the page or invent text.
Return JSON with transcription, candidate_supported, visual_confidence, text_type,
and notes. Use text_type dialogue, thought, narration, vocalisation, sound_effect,
or unknown. Preserve visible wording and punctuation."""
        parsed = self.generate_json([image_path], prompt)
        if "_json_parse_error" in parsed:
            raw_output = str(parsed.get("_raw_output", "")).strip()
            return {
                "transcription": raw_output,
                "candidate_supported": None,
                "visual_confidence": None,
                "text_type": "unknown",
                "notes": str(parsed["_json_parse_error"]),
                "raw_output": raw_output,
            }
        return {
            **parsed,
            "transcription": str(parsed.get("transcription", "") or "").strip(),
            "candidate_supported": parsed.get("candidate_supported"),
            "visual_confidence": parsed.get("visual_confidence"),
            "text_type": parsed.get("text_type", "unknown"),
            "raw_output": parsed.get("_raw_output", ""),
        }


def load_qwen3vl_runner(
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    *,
    local_files_only: bool = True,
    device_map: str = "auto",
    quantize_4bit: bool = True,
    max_new_tokens: int = 128,
) -> Qwen3VLLocalRunner:
    """Load Qwen3-VL from the local Hugging Face cache only."""
    try:
        import torch
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen3VLForConditionalGeneration,
        )
    except ImportError as exc:
        raise RuntimeError("Qwen3-VL dependencies are not installed") from exc

    load_options: dict[str, Any] = {
        "device_map": device_map,
        "local_files_only": local_files_only,
    }
    if quantize_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit Qwen3-VL loading requires CUDA")
        load_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        load_options["dtype"] = torch.float16

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **load_options)
    processor = AutoProcessor.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    model.eval()
    return Qwen3VLLocalRunner(model, processor, max_new_tokens=max_new_tokens)


def load_nemotron_adapter(
    backend_factory: Callable[..., Callable[..., Any]],
    *,
    backend_options: Mapping[str, Any] | None = None,
    merge_level: str = "paragraph",
) -> NemotronCandidateAdapter:
    """Initialize a caller-owned local Nemotron backend without importing its package.

    ``backend_factory`` may wrap a WSL service, subprocess client, or an
    in-process Nemotron install. Its callable result accepts a crop path and
    ``merge_level`` and returns Nemotron's region-record format.
    """
    try:
        backend = backend_factory(**dict(backend_options or {}))
    except Exception as exc:
        raise RuntimeError(f"Nemotron backend initialization failed: {exc}") from exc
    return NemotronCandidateAdapter(backend, merge_level=merge_level)


def load_qwen_candidate_adapter(**options: Any) -> Qwen3VLCandidateAdapter:
    """Initialize the local Qwen runner and wrap it in the existing candidate adapter."""
    return Qwen3VLCandidateAdapter(load_qwen3vl_runner(**options))
