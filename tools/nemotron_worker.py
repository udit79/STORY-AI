#!/usr/bin/env python3
"""Nemotron persistent worker — Linux/WSL side.

Protocol (newline-delimited JSON over stdin/stdout):

  Request  → {"id": "<str>", "image_path": "<abs-linux-path>", "merge_level": "paragraph"}
  Response → {"id": "<str>", "ok": true,  "predictions": [...]}
           | {"id": "<str>", "ok": false, "error": "<message>"}

Lifecycle:
  - Model is loaded ONCE at startup.
  - Worker reads requests line-by-line from stdin; writes responses to stdout.
  - All diagnostics go to stderr so stdout stays machine-readable.
  - Sending the request {"id": "shutdown", "image_path": ""} causes a clean exit.

Usage:
  python tools/nemotron_worker.py --model-dir /path/to/v2_english [--merge-level paragraph]
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# Ensure the nemotron-ocr package (which uses a .so extension module) is importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_NEMOTRON_SRC = _REPO_ROOT / "nemotron-ocr-v2" / "nemotron-ocr" / "src"
if str(_NEMOTRON_SRC) not in sys.path:
    sys.path.insert(0, str(_NEMOTRON_SRC))


def _log(msg: str) -> None:
    """Write a diagnostic message to stderr only."""
    print(f"[nemotron_worker] {msg}", file=sys.stderr, flush=True)


def _respond(response: dict) -> None:
    """Write one JSON response line to stdout."""
    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Local checkpoint directory (e.g. nemotron-ocr-v2/v2_english). "
        "If omitted, uses --lang to select a Hub checkpoint.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "multi", "v1"],
        default=None,
        help="Hub checkpoint alias when --model-dir is not supplied.",
    )
    parser.add_argument(
        "--merge-level",
        type=str,
        choices=["word", "sentence", "paragraph"],
        default="paragraph",
        help="Default merge level (can be overridden per-request).",
    )
    args = parser.parse_args()

    # ── Model loading ────────────────────────────────────────────────────────
    _log("Loading NemotronOCRV2 …")
    try:
        from nemotron_ocr.inference.pipeline_v2 import (  # type: ignore[import]
            NemotronOCRV2,
        )

        kwargs: dict = {}
        if args.model_dir is not None:
            kwargs["model_dir"] = args.model_dir
        elif args.lang is not None:
            kwargs["lang"] = args.lang
        # else: NemotronOCRV2 default (downloads Hub checkpoint if needed)

        ocr = NemotronOCRV2(**kwargs)
        _log("NemotronOCRV2 ready.")
    except Exception:  # noqa: BLE001
        err = traceback.format_exc()
        _log(f"FATAL: model load failed:\n{err}")
        # Signal to the Windows bridge that we failed before entering the loop.
        _respond({"id": "__startup_failed__", "ok": False, "error": err})
        sys.exit(1)

    # Signal ready to the Windows bridge.
    _respond({"id": "__ready__", "ok": True, "predictions": []})

    # ── Request loop ─────────────────────────────────────────────────────────
    default_merge_level = args.merge_level
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            req = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            _log(f"Malformed JSON request: {exc}")
            _respond({"id": "__parse_error__", "ok": False, "error": str(exc)})
            continue

        req_id = req.get("id", "")

        if req_id == "shutdown" or (
            not req.get("image_path") and not req.get("image_paths")
        ):
            _log("Shutdown request received.")
            _respond({"id": req_id, "ok": True, "predictions": []})
            break

        image_paths = req.get("image_paths")
        image_path = req.get("image_path", "")
        merge_level = req.get("merge_level") or default_merge_level

        if image_paths is not None:
            if not isinstance(image_paths, list) or not all(
                isinstance(path, str) for path in image_paths
            ):
                _respond(
                    {
                        "id": req_id,
                        "ok": False,
                        "error": "image_paths must be a list of paths",
                    }
                )
                continue
            missing = [path for path in image_paths if not Path(path).exists()]
            if missing:
                _respond(
                    {
                        "id": req_id,
                        "ok": False,
                        "error": f"image not found: {missing[0]}",
                    }
                )
                continue
            try:
                predictions = ocr(image_paths, merge_level=merge_level)
                safe_predictions = [
                    [
                        {
                            k: (float(v) if isinstance(v, float) else v)
                            for k, v in pred.items()
                        }
                        for pred in (result or [])
                    ]
                    for result in predictions
                ]
                _respond({"id": req_id, "ok": True, "predictions": safe_predictions})
            except Exception:  # noqa: BLE001
                err = traceback.format_exc()
                _log(f"Batch inference error for {len(image_paths)} images:\n{err}")
                _respond({"id": req_id, "ok": False, "error": err})
            continue

        if not Path(image_path).exists():
            _respond(
                {"id": req_id, "ok": False, "error": f"image not found: {image_path}"}
            )
            continue

        try:
            predictions = ocr(image_path, merge_level=merge_level)
            # Ensure JSON-safe output (all floats, no numpy types)
            safe_predictions = [
                {k: (float(v) if isinstance(v, float) else v) for k, v in pred.items()}
                for pred in (predictions or [])
            ]
            _respond({"id": req_id, "ok": True, "predictions": safe_predictions})
        except Exception:  # noqa: BLE001
            err = traceback.format_exc()
            _log(f"Inference error for {image_path}:\n{err}")
            _respond({"id": req_id, "ok": False, "error": err})


if __name__ == "__main__":
    main()
