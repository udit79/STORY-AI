from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "dataset"

SEQUENCES_FILE = DATASET_ROOT / "sequences.json"
LABELS_FILE = DATASET_ROOT / "development" / "labels.jsonl"
SAMPLE_SUBMISSION_FILE = DATASET_ROOT / "sample_submission.jsonl"
SCORE_FILE = DATASET_ROOT / "score.py"