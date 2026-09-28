#!/usr/bin/env python3
"""Score three-page manga transcripts with sequence-wide speaker-label matching."""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

PUNCT = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "…": "...", "–": "-", "—": "-", "−": "-"})
TOKEN = re.compile(r"[^\W_]+(?:['-][^\W_]+)*|[^\w\s]", re.UNICODE)


def norm(text):
    text = unicodedata.normalize("NFKC", text).translate(PUNCT).casefold()
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    text = re.sub(r"\.{3,}", "...", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,!?;:%)\]}])", r"\1", text)
    return re.sub(r"([(\[{])\s+", r"\1", text)


def distance(a, b):
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


@lru_cache(maxsize=65536)
def token_match(a, b):
    if a == b:
        return 1.0
    if min(len(a), len(b)) < 3:
        return 0.0
    score = 1 - distance(a, b) / max(len(a), len(b))
    return score if score >= 0.6 else 0.0


def tokens(rows):
    return [(token, row["speaker"]) for row in rows for token in TOKEN.findall(norm(row["text"]))]


def align_tokens(gold, predicted):
    """Maximum-weight monotonic alignment, using text alone to break ties."""
    n, m = len(gold), len(predicted)
    # ponytail: O(n*m) per page; use a linear-memory aligner only if pages become much larger.
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [bytearray(m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = token_match(gold[i - 1][0], predicted[j - 1][0])
            choices = [(dp[i - 1][j], 1), (dp[i][j - 1], 2)]
            if score:
                choices.append((dp[i - 1][j - 1] + score, 3))
            dp[i][j], back[i][j] = max(choices, key=lambda x: x[0])
    pairs = []
    i, j = n, m
    while i and j:
        move = back[i][j]
        if move == 3:
            pairs.append((gold[i - 1], predicted[j - 1], token_match(gold[i - 1][0], predicted[j - 1][0])))
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return list(reversed(pairs))


def best_mapping(pairs):
    """One-to-one maximum-weight assignment (Hungarian algorithm)."""
    weights = defaultdict(float)
    for gold, predicted, weight in pairs:
        if gold is not None and gold != "NARRATION" and predicted != "NARRATION":
            weights[predicted, gold] += weight
    predicted_labels = sorted({predicted for predicted, _ in weights})
    gold_labels = sorted({gold for _, gold in weights})
    size = max(len(predicted_labels), len(gold_labels))
    if not size:
        return {"NARRATION": "NARRATION"}
    cost = [[-weights[predicted, gold] for gold in gold_labels] + [0.0] * (size - len(gold_labels)) for predicted in predicted_labels]
    cost += [[0.0] * size for _ in range(size - len(predicted_labels))]
    u = v = None
    u, v, matched, way = [0.0] * (size + 1), [0.0] * (size + 1), [0] * (size + 1), [0] * (size + 1)
    for i in range(1, size + 1):
        matched[0], column = i, 0
        minimum, used = [float("inf")] * (size + 1), [False] * (size + 1)
        while True:
            used[column] = True
            row, delta, next_column = matched[column], float("inf"), 0
            for candidate in range(1, size + 1):
                if not used[candidate]:
                    current = cost[row - 1][candidate - 1] - u[row] - v[candidate]
                    if current < minimum[candidate]:
                        minimum[candidate], way[candidate] = current, column
                    if minimum[candidate] < delta:
                        delta, next_column = minimum[candidate], candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    u[matched[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched[column] == 0:
                break
        while column:
            next_column = way[column]
            matched[column], column = matched[next_column], next_column
    mapping = {"NARRATION": "NARRATION"}
    for column in range(1, len(gold_labels) + 1):
        row = matched[column]
        if 0 < row <= len(predicted_labels) and weights[predicted_labels[row - 1], gold_labels[column - 1]] > 0:
            mapping[predicted_labels[row - 1]] = gold_labels[column - 1]
    return mapping


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_jsonl(path, *, reference=False, pages=3):
    result = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=no_duplicate_keys)
            if not isinstance(row, dict) or set(row) != {"sequence_id", "pages"}:
                raise ValueError("Expected exactly sequence_id and pages")
            sequence_id = row["sequence_id"]
            if not isinstance(sequence_id, str) or not sequence_id.strip() or sequence_id != sequence_id.strip():
                raise ValueError("sequence_id must be a nonempty, trimmed string")
            if sequence_id in result:
                raise ValueError(f"Duplicate sequence_id: {sequence_id}")
            if not isinstance(row["pages"], list) or len(row["pages"]) != pages:
                raise ValueError(f"pages must be a list of exactly {pages} lists")
            for page in row["pages"]:
                if not isinstance(page, list):
                    raise ValueError("Each page must be a list")
                for item in page:
                    if not isinstance(item, dict) or set(item) != {"speaker", "text"}:
                        raise ValueError("Each line needs exactly speaker and text")
                    speaker, text = item["speaker"], item["text"]
                    if reference and speaker == "UNKNOWN":
                        item["speaker"] = speaker = None
                    if not (reference and speaker is None):
                        if not isinstance(speaker, str) or not speaker.strip() or speaker != speaker.strip():
                            raise ValueError("speaker must be a nonempty, trimmed string")
                    if not isinstance(text, str) or not norm(text):
                        raise ValueError("text must be nonempty; use an empty list for an empty page")
                    if any(unicodedata.category(char) == "Cc" and not char.isspace() for char in text):
                        raise ValueError("Text contains a control character")
            result[sequence_id] = row["pages"]
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    return result


def score_sequence(gold_pages, predicted_pages):
    pairs, page_scores = [], {}
    active_page_scores = []
    gold_speakers, predicted_speakers = Counter(), Counter()
    gold_count = predicted_count = known_gold = 0
    for page_number, (gold, predicted) in enumerate(zip(gold_pages, predicted_pages), 1):
        gold_text, predicted_text = (norm(" ".join(row["text"] for row in page)) for page in (gold, predicted))
        page_scores[str(page_number)] = max(0.0, 1 - distance(gold_text, predicted_text) / max(len(gold_text), len(predicted_text), 1))
        if gold_text or predicted_text:
            active_page_scores.append(page_scores[str(page_number)])
        gold_tokens, predicted_tokens = tokens(gold), tokens(predicted)
        gold_speakers.update(speaker for _, speaker in gold_tokens if speaker is not None)
        predicted_speakers.update(speaker for _, speaker in predicted_tokens)
        gold_count += len(gold_tokens)
        predicted_count += len(predicted_tokens)
        known_gold += sum(speaker is not None for _, speaker in gold_tokens)
        pairs.extend(align_tokens(gold_tokens, predicted_tokens))
    labels = [(gold[1], predicted[1], weight) for gold, predicted, weight in pairs if gold[1] is not None]
    mapping = best_mapping(labels)
    correct = sum(weight for gold, predicted, weight in labels if mapping.get(predicted) == gold)
    aligned = sum(weight for _, _, weight in labels)
    masked_predicted = sum(gold[1] is None for gold, _, _ in pairs)
    speaker_predicted = predicted_count - masked_predicted
    # Ignore predictions aligned to unresolved reference identities.
    for gold, predicted, _ in pairs:
        if gold[1] is None:
            predicted_speakers[predicted[1]] -= 1
    predicted_speakers = +predicted_speakers
    # Each identity contributes its own text-and-speaker F1, so a long
    # monologue cannot outweigh several missed characters. Match names using
    # this same objective; NARRATION remains a fixed identity.
    balanced_pairs = [
        (gold, predicted, 2 * weight / (gold_speakers[gold] + predicted_speakers[predicted]))
        for gold, predicted, weight in labels
    ]
    balanced_mapping = best_mapping(balanced_pairs)
    balanced_sum = sum(weight for gold, predicted, weight in balanced_pairs
                       if balanced_mapping.get(predicted) == gold)
    extra_speakers = sum(balanced_mapping.get(label) not in gold_speakers
                         for label in predicted_speakers)
    balanced_f1 = min(1.0, balanced_sum / (len(gold_speakers) + extra_speakers)) if known_gold else None
    return {
        "text_order_score": sum(active_page_scores) / len(active_page_scores) if active_page_scores else 1.0,
        "active_text_pages": len(active_page_scores),
        "balanced_joint_f1": balanced_f1,
        "balanced_speaker_mapping": balanced_mapping,
        "speaker_accuracy_on_matched": correct / aligned if aligned else (0.0 if known_gold else None),
        "joint_f1": 2 * correct / (known_gold + speaker_predicted) if known_gold else None,
        "matched_token_coverage": sum(weight for _, _, weight in pairs) / gold_count if gold_count else (1.0 if not predicted_count else 0.0),
        "gold_tokens": gold_count,
        "predicted_tokens": predicted_count,
        "scored_gold_speaker_tokens": known_gold,
        "scored_predicted_speaker_tokens": speaker_predicted,
        "matched_scored_token_weight": aligned,
        "correct_speaker_token_weight": correct,
        "speaker_mapping": mapping,
        "page_text_scores": page_scores,
    }


def evaluate(references, predictions, pages=3):
    if not references:
        raise ValueError("Reference file has no sequences")
    extra = sorted(predictions.keys() - references.keys())
    if extra:
        raise ValueError(f"Unknown extra sequence_id(s): {', '.join(extra)}")
    missing = sorted(references.keys() - predictions.keys())
    scores = {}
    for sequence_id, gold in references.items():
        predicted = predictions.get(sequence_id, [[] for _ in range(pages)])
        row = score_sequence(gold, predicted)
        row["missing_sequence"] = sequence_id in missing
        if row["missing_sequence"]:
            row["text_order_score"] = row["matched_token_coverage"] = 0.0
            row["page_text_scores"] = {str(page): 0.0 for page in range(1, pages + 1)}
        scores[sequence_id] = row
    macro = {}
    for key in ("text_order_score", "balanced_joint_f1", "speaker_accuracy_on_matched", "joint_f1", "matched_token_coverage"):
        eligible = list(scores.values())
        if key in ("text_order_score", "matched_token_coverage"):
            active = [row for row in eligible if row["active_text_pages"] or row["missing_sequence"]]
            eligible = active or eligible
        values = [row[key] for row in eligible if row[key] is not None]
        macro[key] = sum(values) / len(values) if values else None
    return {"version": "2.1", "pages_per_sequence": pages, "macro": macro, "missing_sequences": missing, "sequences": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pages-per-sequence", type=int, default=3)
    args = parser.parse_args()
    if args.pages_per_sequence < 1:
        parser.error("pages-per-sequence must be positive")
    try:
        references = load_jsonl(args.references, reference=True, pages=args.pages_per_sequence)
        predictions = load_jsonl(args.predictions, pages=args.pages_per_sequence)
        report = evaluate(references, predictions, args.pages_per_sequence)
    except ValueError as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["macro"], indent=2))


if __name__ == "__main__":
    main()
