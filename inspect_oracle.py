import json

path = r"experiments\laya\data\preprocessing_oracle_seq_2032620aa4e4ac7f.jsonl"

with open(path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]

for row in rows:
    print()
    print("=" * 80)
    print("BLOCK:", row["block_index"])
    print("TARGET:", row["target_text"])
    print("BASELINE:", row["baseline_text"])
    print("BASELINE CER:", row["baseline_cer"])
    print()

    for action, result in row["candidates"].items():
        print(
            f"{action:12} "
            f"CER={result['cer']:.3f} "
            f"SCORE={result['recognition_score']:.3f} "
            f"TEXT={result['text']!r}"
        )
