"""Validate an automatically generated JSONL against the supplied template."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--template", type=Path, default=Path("dataset/sample_submission.jsonl"))
    args = parser.parse_args()

    template = read_jsonl(args.template)
    predictions = read_jsonl(args.predictions)
    expected_ids = [row.get("sequence_id") for row in template]
    actual_ids = [row.get("sequence_id") for row in predictions]
    errors: list[str] = []
    if len(expected_ids) != 15:
        errors.append(f"template has {len(expected_ids)} sequences, expected 15")
    if len(actual_ids) != len(set(actual_ids)):
        errors.append("prediction sequence IDs are not unique")
    if actual_ids != expected_ids:
        errors.append("prediction IDs/order do not exactly match the template")

    item_count = empty_pages = 0
    speaker_counts: Counter[str] = Counter()
    for row in predictions:
        sequence_id = row.get("sequence_id")
        pages = row.get("pages")
        if not isinstance(pages, list) or len(pages) != 3:
            errors.append(f"{sequence_id}: expected exactly three page lists")
            continue
        for page_index, page in enumerate(pages):
            if not isinstance(page, list):
                errors.append(f"{sequence_id}: page {page_index} is not a list")
                continue
            if not page:
                empty_pages += 1
            for item in page:
                item_count += 1
                if (
                    not isinstance(item, dict)
                    or set(item) != {"speaker", "text"}
                    or not isinstance(item["speaker"], str)
                    or not item["speaker"].strip()
                    or not isinstance(item["text"], str)
                    or not item["text"].strip()
                ):
                    errors.append(f"{sequence_id}: invalid item on page {page_index}: {item!r}")
                elif item["text"].startswith("[balloon "):
                    errors.append(f"{sequence_id}: placeholder item on page {page_index}")
                else:
                    speaker_counts[item["speaker"]] += 1

    print(json.dumps({
        "sequences": len(predictions),
        "items": item_count,
        "empty_pages": empty_pages,
        "speaker_labels": dict(speaker_counts),
        "errors": errors,
    }, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
