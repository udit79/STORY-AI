from pathlib import Path

from app.data.config import DATASET_ROOT
from app.schemas.sequence import SequenceRecord


def resolve_page_paths(
    sequence: SequenceRecord,
    dataset_root: Path = DATASET_ROOT,
) -> list[Path]:

    resolved = [
        dataset_root / relative_path
        for relative_path in sequence.page_paths
    ]

    missing = [
        path
        for path in resolved
        if not path.is_file()
    ]

    if missing:
        missing_text = "\n".join(
            str(path) for path in missing
        )

        raise FileNotFoundError(
            f"Missing image files for {sequence.sequence_id}:\n"
            f"{missing_text}"
        )

    return resolved