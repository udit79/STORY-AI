import random

from app.schemas.sequence import SequenceRecord


def split_development_sequences(
    sequences: list[SequenceRecord],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[SequenceRecord], list[SequenceRecord]]:

    if not 0 < val_ratio < 1:
        raise ValueError(
            "val_ratio must be between 0 and 1."
        )

    development = [
        sequence
        for sequence in sequences
        if sequence.split == "development"
    ]

    if len(development) < 2:
        raise ValueError(
            "Need at least 2 development sequences."
        )

    shuffled = development.copy()

    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_size = max(
        1,
        round(len(shuffled) * val_ratio),
    )

    val = shuffled[:val_size]
    train = shuffled[val_size:]

    validate_no_sequence_leakage(train, val)

    return train, val


def validate_no_sequence_leakage(
    train: list[SequenceRecord],
    val: list[SequenceRecord],
) -> None:

    train_ids = {
        sequence.sequence_id
        for sequence in train
    }

    val_ids = {
        sequence.sequence_id
        for sequence in val
    }

    overlap = train_ids & val_ids

    if overlap:
        raise ValueError(
            "Sequence leakage detected: "
            f"{sorted(overlap)}"
        )