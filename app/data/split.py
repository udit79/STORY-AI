import random
from pathlib import Path

from app.schemas.sequence import SequenceRecord


def split_sequences(
    sequences: list[SequenceRecord],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[SequenceRecord], list[SequenceRecord]]:

    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    items = sequences.copy()

    rng = random.Random(seed)
    rng.shuffle(items)

    val_size = max(1, round(len(items) * val_ratio))

    val = items[:val_size]
    train = items[val_size:]

    return train, val
def validate_split(
    train: list[SequenceRecord],
    val: list[SequenceRecord],
) -> None:

    train_ids = {x.sequence_id for x in train}
    val_ids = {x.sequence_id for x in val}

    overlap = train_ids & val_ids

    if overlap:
        raise ValueError(
            f"Sequence leakage detected: {sorted(overlap)}"
        )