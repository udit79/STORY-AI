import json
from pathlib import Path
from collections import Counter

DATA = Path("experiments") / "laya" / "data"

files = sorted(
    DATA.glob("preprocessing_oracle_seq_*.jsonl")
)

rows = []

for path in files:
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)

print(f"Files: {len(files)}")
print(f"Matched regions: {len(rows)}")

print()
print("=" * 70)
print("RAW BEST ACTION DISTRIBUTION")
print("=" * 70)

raw_counts = Counter()

for row in rows:
    candidates = row["candidates"]

    best_action = min(
        candidates,
        key=lambda action: candidates[action]["cer"],
    )

    raw_counts[best_action] += 1

for action in [
    "NONE",
    "UPSCALE",
    "CONTRAST",
    "THRESHOLD",
    "ROTATE_CW",
    "ROTATE_CCW",
]:
    print(f"{action:12} {raw_counts[action]:4}")

print()
print("=" * 70)
print("THRESHOLD SENSITIVITY")
print("=" * 70)

for threshold in [
    0.00,
    0.01,
    0.02,
    0.03,
    0.05,
]:
    counts = Counter()
    gains = []
    selected_cers = []
    baseline_cers = []

    for row in rows:
        candidates = row["candidates"]

        baseline_cer = candidates["NONE"]["cer"]

        best_action = min(
            candidates,
            key=lambda action: candidates[action]["cer"],
        )

        best_cer = candidates[
            best_action
        ]["cer"]

        gain = baseline_cer - best_cer

        if (
            best_action != "NONE"
            and gain >= threshold
        ):
            selected_action = best_action
            selected_cer = best_cer
        else:
            selected_action = "NONE"
            selected_cer = baseline_cer

        counts[selected_action] += 1
        gains.append(
            baseline_cer - selected_cer
        )
        baseline_cers.append(baseline_cer)
        selected_cers.append(selected_cer)

    mean_gain = (
        sum(gains) / len(gains)
        if gains
        else 0.0
    )

    mean_baseline = (
        sum(baseline_cers)
        / len(baseline_cers)
    )

    mean_selected = (
        sum(selected_cers)
        / len(selected_cers)
    )

    print()
    print(
        f"Threshold >= {threshold:.2f}"
    )
    print(
        f"  NONE       : {counts['NONE']}"
    )
    print(
        f"  UPSCALE    : {counts['UPSCALE']}"
    )
    print(
        f"  CONTRAST   : {counts['CONTRAST']}"
    )
    print(
        f"  THRESHOLD  : {counts['THRESHOLD']}"
    )
    print(
        f"  ROTATE_CW  : {counts['ROTATE_CW']}"
    )
    print(
        f"  ROTATE_CCW : {counts['ROTATE_CCW']}"
    )
    print(
        f"  mean CER    : {mean_baseline:.4f}"
        f" -> {mean_selected:.4f}"
    )
    print(
        f"  mean gain   : {mean_gain:.4f}"
    )


print()
print("=" * 70)
print("LARGEST PREPROCESSING IMPROVEMENTS")
print("=" * 70)

improvements = []

for row in rows:
    candidates = row["candidates"]

    baseline_cer = candidates["NONE"]["cer"]

    best_action = min(
        candidates,
        key=lambda action: candidates[action]["cer"],
    )

    best_cer = candidates[
        best_action
    ]["cer"]

    gain = baseline_cer - best_cer

    if best_action != "NONE":
        improvements.append(
            (
                gain,
                row["sequence_id"],
                row["page_index"],
                row["block_index"],
                best_action,
                baseline_cer,
                best_cer,
                row["target_text"],
            )
        )

improvements.sort(
    reverse=True,
    key=lambda x: x[0],
)

for item in improvements[:15]:
    (
        gain,
        sequence_id,
        page_index,
        block_index,
        action,
        baseline,
        best,
        target,
    ) = item

    print()
    print(
        f"gain={gain:.4f} "
        f"action={action} "
        f"baseline={baseline:.4f} "
        f"best={best:.4f}"
    )
    print(
        f"{sequence_id} "
        f"page={page_index + 1} "
        f"block={block_index}"
    )
    print(
        f"target={target!r}"
    )
