import json
from collections import Counter
from pathlib import Path

import numpy as np


DATA = Path("experiments") / "laya" / "data"

# This is an exploratory label, not a final training label.
# It asks whether the best tested preprocessing action improves
# character error rate by at least 0.02 over the raw baseline.
GAIN_THRESHOLD = 0.02


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
                row = json.loads(line)
                row["_oracle_file"] = path.name
                rows.append(row)

    return rows


def raw_best_action(row):
    candidates = row["candidates"]

    action = min(
        candidates,
        key=lambda name: candidates[name]["cer"],
    )

    baseline_cer = candidates["NONE"]["cer"]
    best_cer = candidates[action]["cer"]
    gain = baseline_cer - best_cer

    return action, gain


def extract_features(row):
    """
    Only use information that is available at inference time.

    IMPORTANT:
    baseline_cer and oracle_action are deliberately NOT features.
    They depend on ground truth and would leak the answer.
    """

    features = row["features"]
    text = row["baseline_text"]

    if not isinstance(text, str):
        text = ""

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
        "width": float(features["width"]),
        "height": float(features["height"]),
        "aspect_ratio": float(features["aspect_ratio"]),
        "area_ratio": float(features["area_ratio"]),
        "line_count": float(features["line_count"]),
        "mean_line_height": float(
            features["mean_line_height"]
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
            features["orientation"] == "vertical"
        ),
    }


def make_dataset(rows):
    data = []

    for row in rows:
        action, gain = raw_best_action(row)

        helpful = int(
            action != "NONE"
            and gain >= GAIN_THRESHOLD
        )

        data.append(
            {
                "sequence_id": row["sequence_id"],
                "features": extract_features(row),
                "helpful": helpful,
                "best_action": action,
                "gain": gain,
            }
        )

    return data


def balanced_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    recalls = []

    for cls in (0, 1):
        mask = y_true == cls

        if mask.sum() == 0:
            continue

        recalls.append(
            float(
                (y_pred[mask] == cls).mean()
            )
        )

    if not recalls:
        return 0.0

    return sum(recalls) / len(recalls)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (
        (tp + tn) / len(y_true)
        if len(y_true)
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy(
            y_true,
            y_pred,
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def fit_stump(x, y):
    """
    Fit an interpretable one-feature decision stump.

    We consider both:
        x <= threshold
        x >= threshold

    and select the rule with the best balanced accuracy.

    Returns:
        {
            "threshold": ...,
            "direction": "le" or "ge",
            "positive_class": ...
        }
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    unique = np.unique(x)

    # Constant feature.
    if len(unique) <= 1:
        majority = int(
            np.mean(y) >= 0.5
        )

        return {
            "threshold": float(unique[0])
            if len(unique)
            else 0.0,
            "direction": "constant",
            "constant_prediction": majority,
        }

    thresholds = (
        unique[:-1] + unique[1:]
    ) / 2.0

    best = None

    for threshold in thresholds:
        for direction in ("le", "ge"):
            if direction == "le":
                pred = (
                    x <= threshold
                ).astype(int)
            else:
                pred = (
                    x >= threshold
                ).astype(int)

            score = balanced_accuracy(
                y,
                pred,
            )

            candidate = (
                score,
                threshold,
                direction,
            )

            if (
                best is None
                or candidate > best
            ):
                best = candidate

    return {
        "threshold": float(best[1]),
        "direction": best[2],
    }


def predict_stump(model, x):
    x = np.asarray(x, dtype=float)

    if model["direction"] == "constant":
        return np.full(
            len(x),
            model["constant_prediction"],
            dtype=int,
        )

    if model["direction"] == "le":
        return (
            x <= model["threshold"]
        ).astype(int)

    return (
        x >= model["threshold"]
    ).astype(int)


def leave_one_sequence_out(data, feature_name):
    """
    Grouped evaluation:
    all regions from one sequence stay in the held-out fold.
    """

    sequence_ids = sorted(
        {
            item["sequence_id"]
            for item in data
        }
    )

    all_true = []
    all_pred = []
    fold_results = []

    for held_out in sequence_ids:
        train = [
            item
            for item in data
            if item["sequence_id"] != held_out
        ]

        test = [
            item
            for item in data
            if item["sequence_id"] == held_out
        ]

        x_train = [
            item["features"][feature_name]
            for item in train
        ]

        y_train = [
            item["helpful"]
            for item in train
        ]

        x_test = [
            item["features"][feature_name]
            for item in test
        ]

        y_test = [
            item["helpful"]
            for item in test
        ]

        model = fit_stump(
            x_train,
            y_train,
        )

        predictions = predict_stump(
            model,
            x_test,
        )

        all_true.extend(y_test)
        all_pred.extend(
            predictions.tolist()
        )

        fold_results.append(
            (
                held_out,
                metrics(
                    y_test,
                    predictions,
                ),
            )
        )

    return metrics(
        all_true,
        all_pred,
    ), fold_results


def main():
    rows = load_rows()
    data = make_dataset(rows)

    print(
        f"Oracle files: "
        f"{len({r['_oracle_file'] for r in rows})}"
    )
    print(
        f"Matched regions: {len(data)}"
    )
    print(
        f"Helpful threshold: "
        f"gain >= {GAIN_THRESHOLD:.2f}"
    )

    helpful_count = sum(
        item["helpful"]
        for item in data
    )

    print(
        f"Helpful: {helpful_count}"
    )
    print(
        f"Not helpful: "
        f"{len(data) - helpful_count}"
    )

    print()
    print("=" * 72)
    print("BEST ACTIONS AMONG HELPFUL CASES")
    print("=" * 72)

    action_counts = Counter(
        item["best_action"]
        for item in data
        if item["helpful"]
    )

    for action in (
        "NONE",
        "UPSCALE",
        "CONTRAST",
        "THRESHOLD",
        "ROTATE_CW",
        "ROTATE_CCW",
    ):
        print(
            f"{action:12} "
            f"{action_counts[action]:4}"
        )

    print()
    print("=" * 72)
    print("FEATURE SUMMARY: HELPFUL VS NOT HELPFUL")
    print("=" * 72)

    feature_names = list(
        data[0]["features"].keys()
    )

    for name in feature_names:
        helpful = [
            item["features"][name]
            for item in data
            if item["helpful"]
        ]

        not_helpful = [
            item["features"][name]
            for item in data
            if not item["helpful"]
        ]

        print()
        print(name)

        if helpful:
            print(
                f"  helpful     "
                f"mean={np.mean(helpful):.4f} "
                f"median={np.median(helpful):.4f}"
            )
        else:
            print(
                "  helpful     no examples"
            )

        print(
            f"  not helpful "
            f"mean={np.mean(not_helpful):.4f} "
            f"median={np.median(not_helpful):.4f}"
        )

    print()
    print("=" * 72)
    print("LEAVE-ONE-SEQUENCE-OUT SINGLE-FEATURE TEST")
    print("=" * 72)

    # This is deliberately simple. We are testing whether an
    # interpretable signal exists, not building the final model.

    results = []

    for feature_name in feature_names:
        result, folds = (
            leave_one_sequence_out(
                data,
                feature_name,
            )
        )

        results.append(
            (
                result["balanced_accuracy"],
                feature_name,
                result,
            )
        )

    results.sort(
        reverse=True,
        key=lambda item: item[0],
    )

    # Reference baseline:
    # always predict NOT_HELPFUL.
    y = [
        item["helpful"]
        for item in data
    ]

    baseline_pred = np.zeros(
        len(y),
        dtype=int,
    )

    baseline = metrics(
        y,
        baseline_pred,
    )

    print()
    print(
        f"{'ALWAYS NONE':32} "
        f"BA={baseline['balanced_accuracy']:.3f} "
        f"F1={baseline['f1']:.3f} "
        f"Recall={baseline['recall']:.3f}"
    )

    for score, feature_name, result in results:
        print(
            f"{feature_name:32} "
            f"BA={result['balanced_accuracy']:.3f} "
            f"F1={result['f1']:.3f} "
            f"Recall={result['recall']:.3f}"
        )

    print()
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)

    best_score, best_feature, best_result = (
        results[0]
    )

    print(
        f"Best single feature: "
        f"{best_feature}"
    )

    print(
        f"Grouped balanced accuracy: "
        f"{best_result['balanced_accuracy']:.3f}"
    )

    print(
        "\nThis is only a diagnostic. "
        "Four sequences are too few to establish "
        "a reliable learned policy."
    )


if __name__ == "__main__":
    main()
