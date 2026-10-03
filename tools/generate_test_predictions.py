"""Run the local production pipeline on every sequence in the test template."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class _Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _validate_prediction(row: dict, sequence_id: str) -> int:
    if (
        row.get("sequence_id") != sequence_id
        or not isinstance(row.get("pages"), list)
        or len(row["pages"]) != 3
    ):
        raise ValueError(
            f"pipeline output does not match the required schema for {sequence_id}"
        )
    item_count = 0
    for page in row["pages"]:
        if not isinstance(page, list):
            raise TypeError(f"invalid page list for {sequence_id}")
        for item in page:
            if (
                not isinstance(item, dict)
                or set(item) != {"speaker", "text"}
                or not isinstance(item["speaker"], str)
                or not item["speaker"].strip()
                or not isinstance(item["text"], str)
                or not item["text"].strip()
            ):
                raise ValueError(f"invalid story item in {sequence_id}: {item!r}")
            if item["text"].startswith("[balloon "):
                raise ValueError(f"placeholder text in {sequence_id}: {item['text']!r}")
            item_count += 1
    return item_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "test_predictions.jsonl"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse per-sequence outputs only when their validation log confirms the selected OCR backend",
    )
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / "test_runs")
    parser.add_argument(
        "--ocr-engine",
        choices=("nemotron", "paddle"),
        default="paddle",
        help="OCR evidence source selected for the quality/latency tradeoff (default: paddle)",
    )
    parser.add_argument(
        "--visual-speaker",
        action="store_true",
        help="Enable slow optional Qwen visual speaker grounding",
    )
    args = parser.parse_args()

    template = [
        json.loads(line)
        for line in (ROOT / "dataset" / "sample_submission.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    sequences = {
        row["sequence_id"]: row
        for row in json.loads(
            (ROOT / "dataset" / "sequences.json").read_text(encoding="utf-8")
        )
    }
    if len(template) != 15 or len({row["sequence_id"] for row in template}) != len(
        template
    ):
        raise ValueError("sample_submission.jsonl must contain 15 unique sequence IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)

    sys.path.insert(0, str(ROOT))
    from tools import pipeline_integration_smoke as pipeline

    predictions: list[dict] = []
    try:
        for number, expected in enumerate(template, start=1):
            sequence_id = expected["sequence_id"]
            sequence = sequences.get(sequence_id)
            if (
                sequence is None
                or sequence.get("split") != "test"
                or len(sequence.get("images", [])) != 3
            ):
                raise ValueError(f"invalid test sequence record: {sequence_id}")
            image_dir = ROOT / "dataset" / sequence["images"][0]
            image_dir = image_dir.parent
            sequence_dir = args.run_dir / sequence_id
            sequence_dir.mkdir(parents=True, exist_ok=True)
            generated_path = sequence_dir / "submission.jsonl"
            log_path = sequence_dir / "pipeline.log"
            if args.resume and generated_path.is_file() and log_path.is_file():
                log_text = log_path.read_text(
                    encoding="utf-8", errors="ignore"
                ).replace("\x00", "")
                engine_matches = (
                    "[OK] PaddleOCR (" in log_text
                    and "skipped via --skip-nemotron" in log_text
                    if args.ocr_engine == "paddle"
                    else "[OK] Nemotron WSL bridge" in log_text
                    and "skipped via --skip-paddle" in log_text
                )
                if (
                    engine_matches
                    and "VALIDATION:  0 error(s)" in log_text
                    and "RESULT: PASS" in log_text
                ):
                    lines = [
                        line
                        for line in generated_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                        if line.strip()
                    ]
                    if len(lines) == 1:
                        existing = json.loads(lines[0])
                        _validate_prediction(existing, sequence_id)
                        if (
                            existing.get("sequence_id") == sequence_id
                            and len(existing.get("pages", [])) == 3
                        ):
                            predictions.append(existing)
                            args.output.write_text(
                                "".join(
                                    json.dumps(row, ensure_ascii=False) + "\n"
                                    for row in predictions
                                ),
                                encoding="utf-8",
                            )
                            print(
                                f"[{number}/15] {sequence_id}: reused validated {args.ocr_engine} output",
                                flush=True,
                            )
                            continue
            command = [
                str(ROOT / "tools" / "pipeline_integration_smoke.py"),
                "--sequence-id",
                sequence_id,
                "--image-dir",
                str(image_dir),
                "--smoke-dir",
                str(sequence_dir / "artifacts"),
                "--submission-path",
                str(generated_path),
            ]
            command.append(
                "--skip-paddle" if args.ocr_engine == "nemotron" else "--skip-nemotron"
            )
            if not args.visual_speaker:
                command.append("--skip-visual-speaker")
            command.append("--keep-backends")
            if generated_path.exists():
                generated_path.unlink()
            print(f"[{number}/15] {sequence_id}: starting", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                tee = _Tee(sys.stdout, log)
                old_argv = sys.argv
                try:
                    sys.argv = command
                    with redirect_stdout(tee), redirect_stderr(tee):
                        return_code = pipeline.main()
                finally:
                    sys.argv = old_argv
            if return_code != 0 or not generated_path.is_file():
                raise RuntimeError(f"pipeline failed for {sequence_id}; see {log_path}")
            generated_lines = [
                line
                for line in generated_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(generated_lines) != 1:
                raise ValueError(
                    f"pipeline output must contain exactly one JSON object for {sequence_id}"
                )
            generated = json.loads(generated_lines[0])
            _validate_prediction(generated, sequence_id)
            predictions.append(generated)
            args.output.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in predictions
                ),
                encoding="utf-8",
            )
            print(
                f"[{number}/15] {sequence_id}: generated {sum(map(len, generated['pages']))} items",
                flush=True,
            )

        if [row["sequence_id"] for row in predictions] != [
            row["sequence_id"] for row in template
        ]:
            raise ValueError(
                "generated sequence IDs do not exactly match the test template order"
            )
        args.output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
            encoding="utf-8",
        )
        print(f"Wrote {len(predictions)} sequences to {args.output}", flush=True)
    finally:
        pipeline.shutdown_cached_backends()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
