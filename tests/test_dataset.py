from app.data.loader import (
    load_sequences,
    load_development_sequences,
    load_test_sequences,
)
from app.data.paths import resolve_page_paths
from app.data.split import split_development_sequences


def test_sequences_load():

    sequences = load_sequences()

    assert sequences

    for sequence in sequences:
        assert sequence.sequence_id
        assert sequence.split in {
            "development",
            "test",
        }
        assert len(sequence.page_paths) == 3


def test_development_and_test_are_disjoint():

    development = load_development_sequences()
    test = load_test_sequences()

    development_ids = {
        sequence.sequence_id
        for sequence in development
    }

    test_ids = {
        sequence.sequence_id
        for sequence in test
    }

    assert development_ids.isdisjoint(test_ids)


def test_image_paths_exist():

    development = load_development_sequences()

    for sequence in development[:5]:
        paths = resolve_page_paths(sequence)

        assert len(paths) == 3

        for path in paths:
            assert path.is_file()


def test_train_validation_split():

    development = load_development_sequences()

    train, val = split_development_sequences(
        development,
        val_ratio=0.2,
        seed=42,
    )

    assert train
    assert val

    train_ids = {
        sequence.sequence_id
        for sequence in train
    }

    val_ids = {
        sequence.sequence_id
        for sequence in val
    }

    assert train_ids.isdisjoint(val_ids)