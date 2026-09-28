import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from rapidfuzz import fuzz

sys.path.insert(0, r"vendor\comic-text-detector")
from inference import TextDetector
from paddleocr import TextRecognition


import argparse


# ---------------------------------------------------------
# Sequence configuration
# ---------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Build preprocessing oracle data for one manga sequence."
)

parser.add_argument(
    "--sequence-id",
    required=True,
    help="Development sequence ID, e.g. seq_2032620aa4e4ac7f",
)

args = parser.parse_args()

SEQUENCE_ID = args.sequence_id

DATASET = Path("dataset")
LABEL_FILE = DATASET / "development" / "labels.jsonl"

PAGES = [
    DATASET
    / "development"
    / "images"
    / SEQUENCE_ID
    / f"{page_index:02d}.png"
    for page_index in range(1, 4)
]

OUT = Path("experiments") / "laya" / "data"
OUT.mkdir(parents=True, exist_ok=True)

TMP_OCR = OUT / "_tmp_ocr.png"


# ---------------------------------------------------------
# Oracle configuration
# ---------------------------------------------------------

# Each action is an explicit preprocessing recipe.
#
# NONE:
#   rectify only
#
# UPSCALE:
#   4x upscale
#
# CONTRAST:
#   4x upscale + histogram equalization
#
# THRESHOLD:
#   4x upscale + Otsu threshold
#
# ROTATE_CW:
#   rotate clockwise + 4x upscale
#
# ROTATE_CCW:
#   rotate counter-clockwise + 4x upscale
#
# IMPORTANT:
# NONE must remain a true no-op after rectification.
ACTIONS = [
    "NONE",
    "UPSCALE",
    "CONTRAST",
    "THRESHOLD",
    "ROTATE_CW",
    "ROTATE_CCW",
]

MATCH_THRESHOLD = 50.0
MIN_CER_GAIN = 0.03
DUPLICATE_IOU_THRESHOLD = 0.85
PADDING = 12


# ---------------------------------------------------------
# Text utilities
# ---------------------------------------------------------

def normalize(text: str) -> str:
    text = text.upper()

    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("—", "-")
        .replace("–", "-")
        .replace("…", "...")
    )

    text = re.sub(
        r"[^A-Z0-9'!?.,:;()\-\"]",
        " ",
        text,
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def cer(pred: str, target: str) -> float:
    pred = normalize(pred)
    target = normalize(target)

    if not target:
        return 0.0 if not pred else 1.0

    # Simple Levenshtein distance.
    prev = list(range(len(target) + 1))

    for i, pc in enumerate(pred, start=1):
        curr = [i]

        for j, tc in enumerate(target, start=1):
            insert = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (pc != tc)

            curr.append(
                min(insert, delete, replace)
            )

        prev = curr

    return prev[-1] / max(1, len(target))


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

def load_labels():
    with open(LABEL_FILE, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            if row["sequence_id"] == SEQUENCE_ID:
                return row

    raise RuntimeError(
        f"Labels not found: {SEQUENCE_ID}"
    )


# ---------------------------------------------------------
# Geometry
# ---------------------------------------------------------

def order_quad(points):
    points = points.astype(np.float32)

    s = points.sum(axis=1)
    d = np.diff(points, axis=1).ravel()

    return np.array(
        [
            points[np.argmin(s)],
            points[np.argmin(d)],
            points[np.argmax(s)],
            points[np.argmax(d)],
        ],
        dtype=np.float32,
    )


def rectify(poly, image):
    poly = np.asarray(
        poly,
        dtype=np.float32,
    )

    if len(poly) == 4:
        src = order_quad(poly)

        w = max(
            1,
            int(
                round(
                    max(
                        np.linalg.norm(
                            src[2] - src[3]
                        ),
                        np.linalg.norm(
                            src[1] - src[0]
                        ),
                    )
                )
            ),
        )

        h = max(
            1,
            int(
                round(
                    max(
                        np.linalg.norm(
                            src[1] - src[2]
                        ),
                        np.linalg.norm(
                            src[0] - src[3]
                        ),
                    )
                )
            ),
        )

        dst = np.array(
            [
                [0, 0],
                [w - 1, 0],
                [w - 1, h - 1],
                [0, h - 1],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(
            src,
            dst,
        )

        return cv2.warpPerspective(
            image,
            matrix,
            (w, h),
        )

    rect = cv2.minAreaRect(poly)

    box = cv2.boxPoints(rect).astype(
        np.float32
    )

    box = order_quad(box)

    w = max(
        1,
        int(round(rect[1][0])),
    )

    h = max(
        1,
        int(round(rect[1][1])),
    )

    dst = np.array(
        [
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(
        box,
        dst,
    )

    return cv2.warpPerspective(
        image,
        matrix,
        (w, h),
    )


# ---------------------------------------------------------
# Preprocessing recipes
# ---------------------------------------------------------

def apply_recipe(poly, image, action):
    """
    Apply exactly one named preprocessing recipe.

    NONE really means:
        rectify only

    There is deliberately no automatic vertical rotation here.
    Rotation must be an explicit action so the oracle can learn it.
    """

    crop = rectify(poly, image)

    # Explicit rotation.
    if action == "ROTATE_CW":
        crop = cv2.rotate(
            crop,
            cv2.ROTATE_90_CLOCKWISE,
        )

    elif action == "ROTATE_CCW":
        crop = cv2.rotate(
            crop,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )

    # Every non-NONE recipe uses 4x upscale.
    if action != "NONE":
        crop = cv2.resize(
            crop,
            None,
            fx=4,
            fy=4,
            interpolation=cv2.INTER_CUBIC,
        )

    # Contrast recipe.
    if action == "CONTRAST":
        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.equalizeHist(gray)

        crop = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR,
        )

    # Threshold recipe.
    elif action == "THRESHOLD":
        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1]

        crop = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR,
        )

    # Padding is also part of preprocessing.
    # Therefore NONE does not receive it.
    if action != "NONE":
        crop = cv2.copyMakeBorder(
            crop,
            PADDING,
            PADDING,
            PADDING,
            PADDING,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    return crop


# ---------------------------------------------------------
# OCR
# ---------------------------------------------------------

def recognize(recognizer, crop):
    """
    OCR one crop.

    A single temporary file is reused so that generating
    oracle data does not leave thousands of PNG files behind.
    """

    ok = cv2.imwrite(
        str(TMP_OCR),
        crop,
    )

    if not ok:
        raise RuntimeError(
            f"Failed to write temporary OCR image: {TMP_OCR}"
        )

    results = list(
        recognizer.predict(
            input=str(TMP_OCR),
            batch_size=1,
        )
    )

    if not results:
        return "", 0.0

    payload = results[0].json
    res = payload.get(
        "res",
        payload,
    )

    return (
        str(
            res.get(
                "rec_text",
                "",
            )
        ),
        float(
            res.get(
                "rec_score",
                0.0,
            )
        ),
    )


# ---------------------------------------------------------
# Block geometry
# ---------------------------------------------------------

def polygon_bbox(poly):
    points = np.asarray(
        poly,
        dtype=np.float32,
    ).reshape(-1, 2)

    x1 = float(points[:, 0].min())
    y1 = float(points[:, 1].min())
    x2 = float(points[:, 0].max())
    y2 = float(points[:, 1].max())

    return (
        x1,
        y1,
        x2,
        y2,
    )


def block_bbox(block):
    boxes = [
        polygon_bbox(line)
        for line in block.lines
        if np.asarray(line).size
    ]

    if not boxes:
        return None

    x1 = min(
        box[0]
        for box in boxes
    )

    y1 = min(
        box[1]
        for box in boxes
    )

    x2 = max(
        box[2]
        for box in boxes
    )

    y2 = max(
        box[3]
        for box in boxes
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


def bbox_area(box):
    if box is None:
        return 0.0

    x1, y1, x2, y2 = box

    return (
        max(0.0, x2 - x1)
        * max(0.0, y2 - y1)
    )


def bbox_iou(a, b):
    if a is None or b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    inter = iw * ih

    union = (
        bbox_area(a)
        + bbox_area(b)
        - inter
    )

    if union <= 0:
        return 0.0

    return inter / union


def deduplicate_blocks(blocks):
    """
    Remove near-identical CTD duplicate blocks.

    CTD has previously produced pairs of almost-identical
    detections for the same text region. We keep the larger
    candidate and discard its duplicate.
    """

    candidates = []

    for raw_id, block in enumerate(blocks):
        if not block.lines:
            continue

        box = block_bbox(block)

        candidates.append(
            {
                "raw_block_id": raw_id,
                "block": block,
                "bbox": box,
                "area": bbox_area(box),
            }
        )

    # Larger boxes first.
    candidates.sort(
        key=lambda item: item["area"],
        reverse=True,
    )

    kept = []

    for candidate in candidates:
        is_duplicate = any(
            bbox_iou(
                candidate["bbox"],
                existing["bbox"],
            )
            >= DUPLICATE_IOU_THRESHOLD
            for existing in kept
        )

        if not is_duplicate:
            kept.append(candidate)

    # Preserve original CTD order.
    kept.sort(
        key=lambda item: item["raw_block_id"]
    )

    return kept


# ---------------------------------------------------------
# Features for Laya
# ---------------------------------------------------------

def block_features(block, image_shape):
    box = block_bbox(block)

    page_h, page_w = image_shape[:2]

    width = max(
        1.0,
        box[2] - box[0],
    )

    height = max(
        1.0,
        box[3] - box[1],
    )

    orientation = (
        "vertical"
        if height > width * 1.35
        else "horizontal"
    )

    line_heights = []

    for line in block.lines:
        line_box = polygon_bbox(line)

        line_heights.append(
            max(
                1.0,
                line_box[3] - line_box[1],
            )
        )

    return {
        "x1": round(box[0], 2),
        "y1": round(box[1], 2),
        "x2": round(box[2], 2),
        "y2": round(box[3], 2),
        "width": round(width, 2),
        "height": round(height, 2),
        "aspect_ratio": round(
            width / height,
            4,
        ),
        "area_ratio": round(
            bbox_area(box)
            / (page_w * page_h),
            6,
        ),
        "orientation": orientation,
        "line_count": len(block.lines),
        "mean_line_height": round(
            float(np.mean(line_heights)),
            2,
        ),
    }


# ---------------------------------------------------------
# Evaluate every preprocessing action
# ---------------------------------------------------------

def evaluate_block(
    recognizer,
    block,
    image,
):
    candidate_results = {}

    for action in ACTIONS:
        line_texts = []
        line_scores = []

        for line in block.lines:
            crop = apply_recipe(
                line,
                image,
                action,
            )

            text, score = recognize(
                recognizer,
                crop,
            )

            line_texts.append(text)
            line_scores.append(score)

        predicted = normalize(
            " ".join(line_texts)
        )

        candidate_results[action] = {
            "text": predicted,
            "recognition_score": round(
                float(
                    np.mean(line_scores)
                )
                if line_scores
                else 0.0,
                6,
            ),
        }

    return candidate_results


# ---------------------------------------------------------
# GT matching
# ---------------------------------------------------------

def best_block_gt_match(
    block_info,
    gt_texts,
):
    """
    For oracle construction only:

    Compare all preprocessing outputs against all GT
    utterances and find the best textual correspondence.

    The test system will NEVER have access to these GT texts.
    """

    best = None

    for gt_index, gt in enumerate(gt_texts):
        target = normalize(gt)

        for action, result in block_info[
            "candidates"
        ].items():

            predicted = result["text"]

            score = fuzz.ratio(
                predicted,
                target,
            )

            if (
                best is None
                or score > best["match_score"]
            ):
                best = {
                    "gt_index": gt_index,
                    "match_score": score,
                    "match_action": action,
                }

    return best


def make_one_to_one_matches(
    block_infos,
    gt_texts,
):
    """
    Global one-to-one matching.

    A CTD block can match at most one GT utterance.
    A GT utterance can match at most one CTD block.
    """

    pairs = []

    for block_info in block_infos:
        match = best_block_gt_match(
            block_info,
            gt_texts,
        )

        if match is None:
            continue

        pairs.append(
            (
                match["match_score"],
                block_info["block_index"],
                match["gt_index"],
                match["match_action"],
            )
        )

    # Highest-confidence pairs first.
    pairs.sort(reverse=True)

    used_blocks = set()
    used_gt = set()

    matches = {}

    for (
        score,
        block_index,
        gt_index,
        match_action,
    ) in pairs:

        if score < MATCH_THRESHOLD:
            continue

        if block_index in used_blocks:
            continue

        if gt_index in used_gt:
            continue

        used_blocks.add(block_index)
        used_gt.add(gt_index)

        matches[block_index] = {
            "gt_index": gt_index,
            "match_score": score,
            "match_action": match_action,
        }

    return matches


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

labels = load_labels()

detector = TextDetector(
    model_path=(
        r"vendor\comic-text-detector"
        r"\data\comictextdetector.pt.onnx"
    ),
    input_size=1024,
    device="cuda",
)

recognizer = TextRecognition(
    model_name="en_PP-OCRv5_mobile_rec",
    engine="onnxruntime",
    device="gpu:0",
)


# ---------------------------------------------------------
# Main oracle generation
# ---------------------------------------------------------

oracle_rows = []
unmatched_rows = []

for page_index, page_path in enumerate(PAGES):

    page_labels = labels["pages"][page_index]

    image = cv2.imread(
        str(page_path)
    )

    if image is None:
        raise FileNotFoundError(
            page_path
        )

    _, _, raw_blocks = detector(
        image
    )

    blocks = deduplicate_blocks(
        raw_blocks
    )

    print()
    print(
        f"PAGE {page_index + 1}"
    )
    print(
        f"Raw CTD blocks: {len(raw_blocks)}"
    )
    print(
        f"After deduplication: {len(blocks)}"
    )
    print(
        f"GT utterances: {len(page_labels)}"
    )

    gt_texts = [
        item["text"]
        for item in page_labels
    ]

    block_infos = []

    # -----------------------------------------------------
    # OCR every block under every candidate recipe
    # -----------------------------------------------------

    for block_index, item in enumerate(
        blocks
    ):

        block = item["block"]

        candidates = evaluate_block(
            recognizer,
            block,
            image,
        )

        info = {
            "block_index": block_index,
            "raw_block_id": item[
                "raw_block_id"
            ],
            "block": block,
            "features": block_features(
                block,
                image.shape,
            ),
            "candidates": candidates,
        }

        block_infos.append(info)

    # -----------------------------------------------------
    # Global one-to-one matching
    # -----------------------------------------------------

    matches = make_one_to_one_matches(
        block_infos,
        gt_texts,
    )

    # -----------------------------------------------------
    # Build oracle rows
    # -----------------------------------------------------

    for info in block_infos:

        block_index = info[
            "block_index"
        ]

        candidates = info[
            "candidates"
        ]

        baseline = candidates[
            "NONE"
        ]

        match = matches.get(
            block_index
        )

        # ---------------------------------------------
        # No reliable GT match
        # ---------------------------------------------

        if match is None:

            best = best_block_gt_match(
                info,
                gt_texts,
            )

            unmatched_rows.append(
                {
                    "sequence_id": SEQUENCE_ID,
                    "page_index": page_index,
                    "block_index": block_index,
                    "raw_block_id": info[
                        "raw_block_id"
                    ],
                    "features": info[
                        "features"
                    ],
                    "baseline_text": baseline[
                        "text"
                    ],
                    "baseline_recognition_score": (
                        baseline[
                            "recognition_score"
                        ]
                    ),
                    "best_gt_match_score": (
                        best["match_score"]
                        if best
                        else 0.0
                    ),
                    "best_gt_index": (
                        best["gt_index"]
                        if best
                        else None
                    ),
                    "best_gt_match_action": (
                        best[
                            "match_action"
                        ]
                        if best
                        else None
                    ),
                }
            )

            if best:
                print(
                    f"B{block_index}: unmatched "
                    f"(best_similarity="
                    f"{best['match_score']:.1f} "
                    f"baseline="
                    f"{baseline['text']!r})"
                )
            else:
                print(
                    f"B{block_index}: unmatched "
                    f"baseline="
                    f"{baseline['text']!r}"
                )

            continue

        # ---------------------------------------------
        # Matched block
        # ---------------------------------------------

        target = gt_texts[
            match["gt_index"]
        ]

        # Calculate CER for every action.
        for action, result in candidates.items():
            result["cer"] = cer(
                result["text"],
                target,
            )

        best_raw_action = min(
            ACTIONS,
            key=lambda action:
                candidates[action]["cer"],
        )

        baseline_cer = candidates[
            "NONE"
        ]["cer"]

        best_cer = candidates[
            best_raw_action
        ]["cer"]

        gain = (
            baseline_cer
            - best_cer
        )

        # ---------------------------------------------
        # Oracle decision
        #
        # If the retry only improves CER slightly,
        # teach Laya to do nothing.
        # ---------------------------------------------

        if (
            best_raw_action == "NONE"
            or gain < MIN_CER_GAIN
        ):
            oracle_action = "NONE"
            oracle_gain = 0.0
        else:
            oracle_action = (
                best_raw_action
            )
            oracle_gain = gain

        row = {
            "sequence_id": SEQUENCE_ID,
            "page_index": page_index,
            "block_index": block_index,
            "raw_block_id": info[
                "raw_block_id"
            ],
            "target_text": target,
            "match_score": round(
                match["match_score"],
                4,
            ),
            "match_action": match[
                "match_action"
            ],
            "features": info[
                "features"
            ],
            "baseline_text": baseline[
                "text"
            ],
            "baseline_cer": round(
                baseline_cer,
                6,
            ),
            "candidates": candidates,
            "best_raw_action": best_raw_action,
            "oracle_action": oracle_action,
            "oracle_cer": round(
                candidates[
                    oracle_action
                ]["cer"],
                6,
            ),
            "oracle_gain": round(
                oracle_gain,
                6,
            ),
        }

        oracle_rows.append(row)

        print(
            f"B{block_index}: "
            f"GT#{match['gt_index']} "
            f"match="
            f"{match['match_score']:.1f} "
            f"target={target!r}"
        )

        print(
            f"  baseline CER="
            f"{baseline_cer:.3f} "
            f"best="
            f"{best_raw_action} "
            f"best CER="
            f"{best_cer:.3f} "
            f"oracle="
            f"{oracle_action} "
            f"gain="
            f"{oracle_gain:.3f}"
        )


# ---------------------------------------------------------
# Save output
# ---------------------------------------------------------

oracle_output = (
    OUT
    / f"preprocessing_oracle_{SEQUENCE_ID}.jsonl"
)

unmatched_output = (
    OUT
    / f"preprocessing_unmatched_{SEQUENCE_ID}.jsonl"
)

with open(
    oracle_output,
    "w",
    encoding="utf-8",
) as f:

    for row in oracle_rows:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


with open(
    unmatched_output,
    "w",
    encoding="utf-8",
) as f:

    for row in unmatched_rows:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False,
            )
            + "\n"
        )


try:
    TMP_OCR.unlink()
except FileNotFoundError:
    pass


print()
print(
    f"Oracle rows: {len(oracle_rows)}"
)
print(
    f"Unmatched rows: {len(unmatched_rows)}"
)
print(
    f"Saved oracle: {oracle_output}"
)
print(
    f"Saved unmatched diagnostics: "
    f"{unmatched_output}"
)