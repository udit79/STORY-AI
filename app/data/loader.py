import json
from pathlib import Path

from app.data.config import SEQUENCES_FILE
from app.schemas.sequence import SequenceRecord


def load_sequences(
    path: str | Path = SEQUENCES_FILE,
) -> list[SequenceRecord]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"sequences.json not found: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(
            "Expected sequences.json to contain a JSON list."
        )

    sequences: list[SequenceRecord] = []

    for index, item in enumerate(raw_data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Sequence record {index} is not an object."
            )

        sequence_id = item.get("sequence_id")
        split = item.get("split")
        images = item.get("images")

        if not sequence_id:
            raise ValueError(
                f"Record {index} missing sequence_id."
            )

        if split not in {"development", "test"}:
            raise ValueError(
                f"Invalid split for {sequence_id}: {split!r}"
            )

        if not isinstance(images, list) or len(images) != 3:
            raise ValueError(
                f"{sequence_id} must contain exactly 3 images."
            )

        sequences.append(
            SequenceRecord(
                sequence_id=sequence_id,
                split=split,
                page_paths=images,
            )
        )

    return sequences


def load_development_sequences() -> list[SequenceRecord]:
    return [
        seq
        for seq in load_sequences()
        if seq.split == "development"
    ]


def load_test_sequences() -> list[SequenceRecord]:
    return [
        seq
        for seq in load_sequences()
        if seq.split == "test"
    ]


if __name__ == "__main__":
    sequences = load_sequences()

    development = [
        seq for seq in sequences
        if seq.split == "development"
    ]

    test = [
        seq for seq in sequences
        if seq.split == "test"
    ]

    print(f"Total sequences : {len(sequences)}")
    print(f"Development     : {len(development)}")
    print(f"Test            : {len(test)}")

    print("\nFirst sequence:")
    print(development[0].model_dump())