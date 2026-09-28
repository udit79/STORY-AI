import json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )
    from sklearn.model_selection import GroupKFold
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is not installed.\n"
        "Run:\n"
        "  uv add scikit-learn\n"
        "then rerun this script."
    ) from exc


DATA = Path("experiments") / "laya" / "data"
GAIN_THRESHOLD = 0.02


FEATURE_NAMES = [
    "baseline_recognition_score",
    "width",
    "height",
    "aspect_ratio",
    "area_ratio",
    "line_count",
    "mean_line_height",
    "ocr_char_count",
    "ocr_word_count",
    "ocr_punctuation_ratio",
    "ocr_digit_ratio",
    "is_vertical",
]


def load_rows():
    rows = []

    files = sorted(
        DATA.glob("preprocessing_oracle_seq_*.jsonl")
    )

    if not files:
        raise FileNotFoundError(
            f"No oracle files found in {DATA.resolve()}"
        )

    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                rows.append(json.loads(line))

    return rows


def extract_features(row):
    f = row["features"]
    text = row["baseline_text"] or ""

    characters = len(text)
    words = text.split()

    punctuation = sum(
        ch in ".,!?;:'\"-()"
        for ch in text
    )

    digits = sum(
        ch.isdigit()
        for ch in text
    )

    return {
        "baseline_recognition_score": float(
            row["candidates"]["NONE"]["recognition_score"]
        ),
        "width": float(f["width"]),
        "height": float(f["height"]),
        "aspect_ratio": float(f["aspect_ratio"]),
        "area_ratio": float(f["area_ratio"]),
        "line_count": float(f["line_count"]),
        "mean_line_height": float(
            f["mean_line_height"]
        ),
        "ocr_char_count": float(characters),
        "ocr_word_count": float(len(words)),
        "ocr_punctuation_ratio": (
            punctuation / max(1, characters)
        ),
        "ocr_digit_ratio": (
            digits / max(1, characters)
        ),
        "is_vertical": float(
            f["orientation"] == "vertical"
        ),
    }


def prepare(rows):
    X = np.array(
        [
            [
                extract_features(row)[name]
                for name in FEATURE_NAMES
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    y = np.array(
        [
            int(
                (
                    min(
                        row["candidates"],
                        key=lambda action:
                            row["candidates"][action]["cer"],
                    )
                    != "NONE"
                )
                and (
                    row["candidates"]["NONE"]["cer"]
                    - min(
                        result["cer"]
                        for result in row["candidates"].values()
                    )
                    >= GAIN_THRESHOLD
                )
            )
            for row in rows
        ],
        dtype=np.int64,
    )

    groups = np.array(
        [
            row["sequence_id"]
            for row in rows
        ]
    )

    return X, y, groups


def print_metrics(name, y_true, y_pred):
    print(
        f"{name:28} "
        f"Acc={accuracy_score(y_true, y_pred):.3f} "
        f"BA={balanced_accuracy_score(y_true, y_pred):.3f} "
        f"Prec={precision_score(y_true, y_pred, zero_division=0):.3f} "
        f"Recall={recall_score(y_true, y_pred, zero_division=0):.3f} "
        f"F1={f1_score(y_true, y_pred, zero_division=0):.3f}"
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    print(
        f"  confusion matrix [[TN, FP], [FN, TP]] = "
        f"{cm.tolist()}"
    )


def confidence_stump_threshold(x_train, y_train):
    """
    Find a threshold on baseline recognition score.

    We test both:
        predict helpful when confidence <= threshold
        predict helpful when confidence >= threshold

    The threshold is learned ONLY on the training sequences.
    """

    values = np.unique(x_train)

    if len(values) <= 1:
        return float(values[0]) if len(values) else 0.0, "le"

    thresholds = (
        values[:-1] + values[1:]
    ) / 2.0

    best = None

    for threshold in thresholds:
        for direction in ("le", "ge"):
            if direction == "le":
                pred = (
                    x_train <= threshold
                ).astype(int)
            else:
                pred = (
                    x_train >= threshold
                ).astype(int)

            score = balanced_accuracy_score(
                y_train,
                pred,
            )

            candidate = (
                score,
                threshold,
                direction,
            )

            if best is None or candidate > best:
                best = candidate

    return best[1], best[2]


def confidence_stump_predict(
    x,
    threshold,
    direction,
):
    if direction == "le":
        return (
            x <= threshold
        ).astype(int)

    return (
        x >= threshold
    ).astype(int)


def evaluate():
    rows = load_rows()
    X, y, groups = prepare(rows)

    print(f"Matched regions: {len(rows)}")
    print(f"Helpful: {int(y.sum())}")
    print(
        f"Not helpful: {int((y == 0).sum())}"
    )

    print()
    print("=" * 78)
    print("GROUPED OUT-OF-FOLD COMPARISON")
    print("=" * 78)

    n_groups = len(np.unique(groups))

    if n_groups < 3:
        raise RuntimeError(
            "Need at least 3 sequences for grouped evaluation."
        )

    cv = GroupKFold(
        n_splits=n_groups
    )

    models = {
        "LOGISTIC_REGRESSION": Pipeline(
            [
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        C=1.0,
                    ),
                ),
            ]
        ),
        "SHALLOW_TREE": DecisionTreeClassifier(
            max_depth=2,
            class_weight="balanced",
            random_state=42,
        ),
    }

    oof = {
        name: np.zeros_like(y)
        for name in models
    }

    confidence_oof = np.zeros_like(y)

    fold = 0

    for train_idx, test_idx in cv.split(
        X,
        y,
        groups,
    ):
        fold += 1

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        # -----------------------------
        # Confidence-only model
        # -----------------------------

        confidence_col = (
            FEATURE_NAMES.index(
                "baseline_recognition_score"
            )
        )

        threshold, direction = (
            confidence_stump_threshold(
                X_train[:, confidence_col],
                y_train,
            )
        )

        confidence_oof[test_idx] = (
            confidence_stump_predict(
                X_test[:, confidence_col],
                threshold,
                direction,
            )
        )

        # -----------------------------
        # Multi-feature models
        # -----------------------------

        for name, model in models.items():
            model.fit(
                X_train,
                y_train,
            )

            oof[name][test_idx] = (
                model.predict(X_test)
            )

        print()
        print(
            f"Fold {fold}: "
            f"held-out sequence(s)="
            f"{sorted(set(groups[test_idx]))}"
        )

        print(
            f"  confidence rule: "
            f"threshold={threshold:.4f} "
            f"direction={direction}"
        )

    print()
    print("-" * 78)

    # Reference model.
    always_none = np.zeros_like(y)

    print_metrics(
        "ALWAYS_NOT_HELPFUL",
        y,
        always_none,
    )

    print_metrics(
        "CONFIDENCE_ONLY",
        y,
        confidence_oof,
    )

    for name, predictions in oof.items():
        print_metrics(
            name,
            y,
            predictions,
        )

    print()
    print("=" * 78)
    print("LOGISTIC REGRESSION COEFFICIENTS")
    print("=" * 78)

    final_model = models[
        "LOGISTIC_REGRESSION"
    ]

    final_model.fit(
        X,
        y,
    )

    classifier = final_model.named_steps[
        "model"
    ]

    coefficients = classifier.coef_[0]

    ranked = sorted(
        zip(
            FEATURE_NAMES,
            coefficients,
        ),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    for name, coefficient in ranked:
        print(
            f"{name:32} "
            f"{coefficient:+.4f}"
        )

    print()
    print("=" * 78)
    print("DECISION")
    print("=" * 78)

    print(
        "This is still a diagnostic, not the final policy model."
    )
    print(
        "The key comparison is whether a small multi-feature model "
        "generalizes better than the confidence-only baseline under "
        "sequence-level cross-validation."
    )


if __name__ == "__main__":
    evaluate()
