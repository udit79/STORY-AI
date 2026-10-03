"""Benchmark real Qwen3-VL OCR adjudication on one local balloon crop."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.models.local_runners import load_qwen3vl_runner


PROMPT = """Read only the visible text in this manga balloon crop. Return compact JSON with keys transcription, candidate_supported, visual_confidence, text_type, notes. Do not explain."""


def _memory() -> dict[str, int | None]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {"gpu_allocated_bytes": None, "gpu_reserved_bytes": None}
        return {
            "gpu_allocated_bytes": int(torch.cuda.memory_allocated()),
            "gpu_reserved_bytes": int(torch.cuda.memory_reserved()),
        }
    except Exception:  # noqa: BLE001
        return {"gpu_allocated_bytes": None, "gpu_reserved_bytes": None}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("crop", type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not args.crop.is_file():
        raise FileNotFoundError(args.crop)

    init_started = time.perf_counter()
    runner = load_qwen3vl_runner(
        local_files_only=True,
        quantize_4bit=True,
        max_new_tokens=args.max_new_tokens,
    )
    init_seconds = time.perf_counter() - init_started
    before = _memory()
    runs = []
    for index in range(max(1, args.runs)):
        started = time.perf_counter()
        result = runner.generate_json([args.crop], PROMPT)
        runs.append({
            "run": index + 1,
            "seconds": round(time.perf_counter() - started, 3),
            "memory_before": before if index == 0 else _memory(),
            "memory_after": _memory(),
            "json_parse_error": result.get("_json_parse_error"),
            "transcription": result.get("transcription"),
            "raw_output": result.get("_raw_output"),
        })
    report = {
        "crop": str(args.crop.resolve()),
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "quantization": "4-bit NF4 double-quantized, float16 compute",
        "max_new_tokens": args.max_new_tokens,
        "deterministic": True,
        "initialization_seconds": round(init_seconds, 3),
        "runs": runs,
    }
    output = args.output or args.crop.with_name("qwen_crop_benchmark.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
