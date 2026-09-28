from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# IMPORTANT:
# Import torch before onnxruntime/PaddleOCR so the CUDA runtime DLLs
# used by the existing Windows environment are available.
import torch  # noqa: F401
import onnxruntime as ort

# Existing project setup.
sys.path.insert(0, r"vendor\comic-text-detector")

from inference import TextDetector
from paddleocr import TextRecognition

from rapidfuzz.fuzz import ratio


# ============================================================
# CONSTANTS
# ============================================================

ROOT = Path.cwd()

CTD_MODEL = (
    ROOT
    / r"vendor\comic-text-detector\data\comictextdetector.pt.onnx"
)

ORACLE_SCRIPT = (
    ROOT
    / r"experiments\laya\oracle\build_oracle.py"
)

ORACLE_DIR = (
    ROOT
    / r"experiments\laya\data"
)

OUTPUT_ROOT = (
    ROOT
    / r"outputs\vlm_block_benchmark"
)

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


# ============================================================
# HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    """Normalize text only for comparison."""
    if not text:
        return ""

    text = str(text).upper()
    text = re.sub(r"\s+", " ", text).strip()

    return text


def cer(reference: str, hypothesis: str) -> float:
    """
    Character Error Rate using Levenshtein distance.
    """
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)

    if not reference:
        return 0.0 if not hypothesis else 1.0

    # Standard DP Levenshtein.
    prev = list(range(len(hypothesis) + 1))

    for i, rc in enumerate(reference, start=1):
        curr = [i]

        for j, hc in enumerate(hypothesis, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (rc != hc)

            curr.append(
                min(
                    insert_cost,
                    delete_cost,
                    replace_cost,
                )
            )

        prev = curr

    return prev[-1] / len(reference)


def bbox_iou(
    a: list[float] | tuple[float, ...],
    b: list[float] | tuple[float, ...],
) -> float:
    """
    IoU for [x1,y1,x2,y2].
    """
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    denom = area_a + area_b - inter

    if denom <= 0:
        return 0.0

    return inter / denom




def get_block_rect(block: Any) -> list[float]:
    """
    CTD TextBlock exposes bounding_rect() as a method.

    Returns:
        [x1, y1, x2, y2]
    """
    rect_attr = getattr(block, "bounding_rect", None)

    if rect_attr is None:
        raise AttributeError(
            f"CTD TextBlock has no bounding_rect: {type(block)!r}"
        )

    # bounding_rect is a METHOD in this CTD implementation.
    if callable(rect_attr):
        rect = rect_attr()
    else:
        rect = rect_attr

    arr = np.asarray(
        rect,
        dtype=np.float32,
    ).reshape(-1)

    if arr.size != 4:
        raise ValueError(
            f"Unexpected bounding_rect(): {rect!r}"
        )

    x, y, w, h = map(float, arr)

    return [
        x,
        y,
        x + w,
        y + h,
    ]


def dedup_blocks(
    blocks: list[Any],
    iou_threshold: float = 0.85,
) -> list[Any]:
    """
    Remove near-duplicate CTD blocks using bounding_rect.
    """
    kept: list[Any] = []

    for block in blocks:
        current = get_block_rect(block)

        duplicate = False

        for old in kept:
            old_box = get_block_rect(old)

            if bbox_iou(current, old_box) >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(block)

    return kept


def block_bbox(block: Any) -> list[int]:
    """
    Convert CTD bounding_rect [x,y,w,h]
    into [x1,y1,x2,y2].
    """
    x1, y1, x2, y2 = get_block_rect(block)

    return [
        max(0, int(np.floor(x1))),
        max(0, int(np.floor(y1))),
        int(np.ceil(x2)),
        int(np.ceil(y2)),
    ]

def line_bbox(line: Any) -> list[int]:
    """
    Convert CTD line polygon into [x1,y1,x2,y2].
    """
    poly = np.asarray(line, dtype=np.float32)

    x1 = int(np.floor(poly[:, 0].min()))
    y1 = int(np.floor(poly[:, 1].min()))
    x2 = int(np.ceil(poly[:, 0].max()))
    y2 = int(np.ceil(poly[:, 1].max()))

    return [x1, y1, x2, y2]


# ============================================================
# PADDLE OCR
# ============================================================

def extract_paddle_result(result: Any) -> tuple[str, float, dict[str, Any]]:
    """
    Robustly extract TextRecognition output.

    PaddleOCR TextRecognition documents:
        rec_text
        rec_score

    Depending on PaddleOCR version, result.json can be a callable
    or an already-materialized object.
    """

    payload: Any = None

    json_attr = getattr(result, "json", None)

    if callable(json_attr):
        payload = json_attr()
    elif json_attr is not None:
        payload = json_attr

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = None

    if payload is None:
        # Last-resort introspection.
        data_attr = getattr(result, "data", None)

        if isinstance(data_attr, dict):
            payload = data_attr

    if payload is None:
        raise RuntimeError(
            "Unable to extract PaddleOCR result. "
            f"Result type={type(result)!r}"
        )

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Unexpected PaddleOCR result payload type: "
            f"{type(payload)!r}"
        )

    res = payload.get("res", payload)

    if not isinstance(res, dict):
        raise RuntimeError(
            "Unexpected PaddleOCR 'res' payload: "
            f"{type(res)!r}"
        )

    text = str(res.get("rec_text", "") or "")
    score = float(res.get("rec_score", 0.0) or 0.0)

    return text, score, res


def recognize_line(
    recognizer: TextRecognition,
    image: np.ndarray,
    out_path: Path,
) -> tuple[str, float, dict[str, Any]]:
    """
    Run PaddleOCR TextRecognition on one CTD line.

    This intentionally uses a line crop, not a whole CTD block.
    """

    if image is None or image.size == 0:
        return "", 0.0, {}

    crop = image

    h, w = crop.shape[:2]
    rotated = False

    # English recognition is horizontal.
    # Rotate clearly vertical lines.
    if h > w * 1.5:
        crop = cv2.rotate(
            crop,
            cv2.ROTATE_90_CLOCKWISE,
        )
        rotated = True

    # Keep the crop exactly as produced here.
    # Do not secretly upscale/threshold/etc.
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ok = cv2.imwrite(str(out_path), crop)

    if not ok:
        raise IOError(
            f"Failed to save OCR crop: {out_path}"
        )

    results = list(
        recognizer.predict(
            input=str(out_path),
            batch_size=1,
        )
    )

    if not results:
        return "", 0.0, {
            "rotated": rotated,
            "raw_result": None,
        }

    text, score, raw = extract_paddle_result(
        results[0]
    )

    raw = dict(raw)
    raw["rotated"] = rotated

    return text, score, raw


def recognize_block_lines(
    recognizer: TextRecognition,
    image: np.ndarray,
    block: Any,
    block_dir: Path,
) -> tuple[str, float, list[dict[str, Any]]]:
    """
    Recognize all CTD lines belonging to one CTD block.

    Lines are spatially ordered before being joined.
    """

    line_records: list[dict[str, Any]] = []

    lines = list(getattr(block, "lines", []))

    for line_id, line in enumerate(lines):
        bbox = line_bbox(line)

        x1, y1, x2, y2 = bbox

        # Small safety padding.
        px = 8
        py = 8

        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)

        x2 = min(image.shape[1], x2 + px)
        y2 = min(image.shape[0], y2 + py)

        crop = image[y1:y2, x1:x2]

        crop_path = (
            block_dir
            / f"line_{line_id:02d}.png"
        )

        text, score, raw = recognize_line(
            recognizer,
            crop,
            crop_path,
        )

        # Center used only for line ordering.
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        line_records.append(
            {
                "line_id": line_id,
                "bbox": [x1, y1, x2, y2],
                "cx": cx,
                "cy": cy,
                "text": text,
                "score": score,
                "crop": str(crop_path),
                "raw": raw,
            }
        )

    # Manga CTD lines within a block are normally read
    # top-to-bottom for horizontal English.
    line_records.sort(
        key=lambda item: (
            item["cy"],
            item["cx"],
        )
    )

    nonempty = [
        item
        for item in line_records
        if normalize_text(item["text"])
    ]

    if not nonempty:
        return "", 0.0, line_records

    text_parts = [
        item["text"].strip()
        for item in nonempty
    ]

    block_text = " ".join(text_parts)

    scores = [
        float(item["score"])
        for item in nonempty
    ]

    block_score = float(np.mean(scores))

    return block_text, block_score, line_records


# ============================================================
# ORACLE
# ============================================================

def oracle_path(sequence_id: str) -> Path:
    return (
        ORACLE_DIR
        / f"preprocessing_oracle_{sequence_id}.jsonl"
    )


def unmatched_path(sequence_id: str) -> Path:
    return (
        ORACLE_DIR
        / f"preprocessing_unmatched_{sequence_id}.jsonl"
    )


def run_oracle_builder(sequence_id: str) -> None:
    """
    Rebuild the existing preprocessing oracle.

    We reuse this oracle for GT block correspondence.
    """

    print("Running existing oracle builder...")

    subprocess.run(
        [
            sys.executable,
            str(ORACLE_SCRIPT),
            "--sequence-id",
            sequence_id,
        ],
        check=True,
    )


def load_oracle(
    sequence_id: str,
) -> dict[tuple[int, int], dict[str, Any]]:
    """
    Load oracle rows keyed by:
        (page_index, block_index)

    The oracle builder stores the CTD deduplicated block position
    as `block_index`.
    """

    path = oracle_path(sequence_id)

    if not path.exists():
        raise FileNotFoundError(
            f"Oracle file not found: {path}"
        )

    mapping: dict[tuple[int, int], dict[str, Any]] = {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            # These are the actual oracle schema fields.
            page_index = int(row["page_index"])
            block_index = int(row["block_index"])

            key = (page_index, block_index)

            if key in mapping:
                raise ValueError(
                    f"Duplicate oracle key {key} "
                    f"at line {line_number}"
                )

            mapping[key] = row

    return mapping

def get_oracle_gt(
    oracle: dict[tuple[int, int], dict[str, Any]],
    page_index: int,
    block_index: int,
) -> tuple[str | None, float | None, dict[str, Any] | None]:

    row = oracle.get(
        (page_index, block_index)
    )

    if row is None:
        return None, None, None

    target_text = row.get("target_text")

    similarity = row.get(
        "match_score",
        row.get("match_similarity"),
    )

    if similarity is not None:
        similarity = float(similarity)

    return target_text, similarity, row


# ============================================================
# QWEN3-VL
# ============================================================

def load_qwen():
    """
    Load Qwen3-VL locally.

    Requires the existing cached model.
    """

    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
    )

    print(
        f"Loading Qwen3-VL: {MODEL_ID}"
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=quant_config,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    return model, processor


def qwen_block(
    model,
    processor,
    image_path: Path,
) -> dict[str, Any]:
    """
    Run Qwen3-VL on one localized CTD block.

    Output must remain structured and must not silently
    become a page-level transcription.
    """

    from qwen_vl_utils import process_vision_info

    prompt = """
You are given ONE localized manga text block.

Read ONLY the text physically visible inside this image crop.

Do not reconstruct text from the surrounding page.
Do not invent missing words.
Do not merge text from outside the crop.

Return JSON only:

{
  "transcription": "...",
  "candidate_supported": true,
  "text_type": "dialogue|thought|narration|vocalisation|sound_effect|unknown"
}

Rules:
- Preserve the visible wording as closely as possible.
- Preserve punctuation.
- For English manga, output uppercase when visually clear.
- If the crop is not story text, still classify it appropriately.
- Do not add explanations.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": str(image_path),
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(
        messages
    )

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        if hasattr(value, "to")
        else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    generated_ids = generated[:, inputs["input_ids"].shape[1]:]

    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    # Parse JSON, including fenced JSON if Qwen emits it.
    cleaned = output.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
        )
        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Keep the transcription rather than inventing fields.
        parsed = {
            "transcription": cleaned,
            "candidate_supported": True,
            "text_type": "unknown",
            "_raw_output": output,
        }

    transcription = str(
        parsed.get(
            "transcription",
            "",
        )
        or ""
    ).strip()

    candidate_supported = bool(
        parsed.get(
            "candidate_supported",
            True,
        )
    )

    text_type = str(
        parsed.get(
            "text_type",
            "unknown",
        )
        or "unknown"
    ).strip()

    return {
        "transcription": transcription,
        "candidate_supported": candidate_supported,
        "text_type": text_type,
        "raw_output": output,
    }


# ============================================================
# PAGE PROCESSING
# ============================================================

def process_page(
    sequence_id: str,
    page_index: int,
    page_path: Path,
    detector: TextDetector,
    recognizer: TextRecognition,
    model,
    processor,
    oracle: dict[tuple[int, int], dict[str, Any]],
    limit_per_page: int | None,
) -> list[dict[str, Any]]:

    print()
    print("=" * 90)
    print(f"PAGE {page_index}")
    print(f"IMAGE: {page_path}")
    print("=" * 90)

    image = cv2.imread(str(page_path))

    if image is None:
        raise FileNotFoundError(page_path)

    _, _, raw_blocks = detector(image)

    blocks = dedup_blocks(
        raw_blocks,
        iou_threshold=0.85,
    )

    print(
        f"CTD raw blocks   : {len(raw_blocks)}"
    )
    print(
        f"CTD unique blocks: {len(blocks)}"
    )

    if limit_per_page is not None:
        blocks = blocks[:limit_per_page]

    page_out = (
        OUTPUT_ROOT
        / sequence_id
        / f"page_{page_index}"
    )

    page_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    records: list[dict[str, Any]] = []

    for block_id, block in enumerate(blocks):

        b = block_bbox(block)

        x1, y1, x2, y2 = b

        # Padding around the block for VLM visual context.
        px = 12
        py = 12

        cx1 = max(0, x1 - px)
        cy1 = max(0, y1 - py)
        cx2 = min(image.shape[1], x2 + px)
        cy2 = min(image.shape[0], y2 + py)

        crop = image[
            cy1:cy2,
            cx1:cx2,
        ]

        block_dir = (
            page_out
            / f"block_{block_id:02d}"
        )

        block_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        block_path = (
            block_dir
            / "block.png"
        )

        if not cv2.imwrite(
            str(block_path),
            crop,
        ):
            raise IOError(
                f"Could not save {block_path}"
            )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        ocr_text, ocr_score, line_items = (
            recognize_block_lines(
                recognizer=recognizer,
                image=image,
                block=block,
                block_dir=block_dir,
            )
        )

        # ----------------------------------------------------
        # VLM
        # ----------------------------------------------------

        vlm = qwen_block(
            model=model,
            processor=processor,
            image_path=block_path,
        )

        vlm_text = vlm["transcription"]

        # ----------------------------------------------------
        # GT
        # ----------------------------------------------------

        # Runtime pages are 1-based:
        #   1 -> 01.png
        #   2 -> 02.png
        #   3 -> 03.png
        #
        # Oracle page_index is 0-based:
        #   0 -> 01.png
        #   1 -> 02.png
        #   2 -> 03.png

        oracle_page_index = page_index - 1

        gt_text, oracle_similarity, oracle_row = get_oracle_gt(
            oracle=oracle,
            page_index=oracle_page_index,
            block_index=block_id,
        )

        if oracle_row is not None:
            oracle_bbox = [
                float(oracle_row["features"]["x1"]),
                float(oracle_row["features"]["y1"]),
                float(oracle_row["features"]["x2"]),
                float(oracle_row["features"]["y2"]),
            ]

            runtime_bbox = [
                float(b[0]),
                float(b[1]),
                float(b[2]),
                float(b[3]),
            ]

            bbox_iou_score = bbox_iou(
                runtime_bbox,
                oracle_bbox,
            )

            if bbox_iou_score < 0.85:
                print(
                    "WARNING: Oracle/runtime bbox drift: "
                    f"page={page_index}, "
                    f"block={block_id}, "
                    f"IoU={bbox_iou_score:.3f}"
                )

        # ----------------------------------------------------
        # DEBUG / TRACEABILITY
        # ----------------------------------------------------

        print()
        print("-" * 90)
        print(
            f"SEQUENCE : {sequence_id}"
        )
        print(
            f"PAGE     : {page_index}"
        )
        print(
            f"PAGE PATH: {page_path}"
        )
        print(
            f"BLOCK    : {block_id}"
        )
        print(
            f"BBOX     : {tuple(b)}"
        )
        print(
            f"CROP     : {block_path}"
        )

        print()
        print(
            f"OCR      : {ocr_text!r}"
        )
        print(
            f"OCR CONF : {ocr_score:.4f}"
        )

        print()
        print(
            f"VLM      : {vlm_text!r}"
        )
        print(
            f"VLM TYPE : {vlm['text_type']}"
        )
        print(
            f"VLM SUPP : {vlm['candidate_supported']}"
        )

        print()
        print(
            f"GT       : {gt_text!r}"
        )

        if oracle_similarity is not None:
            print(
                f"ORACLE MATCH: {oracle_similarity:.2f}"
            )

        if gt_text:
            ocr_cer = cer(
                gt_text,
                ocr_text,
            )

            vlm_cer = cer(
                gt_text,
                vlm_text,
            )

            if ocr_cer < vlm_cer:
                better = "OCR"
            elif vlm_cer < ocr_cer:
                better = "VLM"
            else:
                better = "EQUAL"

            print()
            print(
                f"OCR CER  : {ocr_cer:.4f}"
            )
            print(
                f"VLM CER  : {vlm_cer:.4f}"
            )
            print(
                f"BETTER   : {better}"
            )

        else:
            ocr_cer = None
            vlm_cer = None
            better = "UNMATCHED"

            print()
            print(
                "GT       : UNMATCHED"
            )

        record = {
            "sequence_id": sequence_id,
            "page_index": page_index,
            "page_path": str(page_path),
            "block_id": block_id,
            "bbox": b,
            "crop": str(block_path),

            "ocr": {
                "text": ocr_text,
                "score": ocr_score,
                "lines": line_items,
            },

            "vlm": vlm,

            "ground_truth": {
                "text": gt_text,
                "oracle_similarity": oracle_similarity,
                "oracle_row": oracle_row,
            },

            "metrics": {
                "ocr_cer": ocr_cer,
                "vlm_cer": vlm_cer,
                "better": better,
            },
        }

        records.append(record)

    return records


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-id",
        required=True,
    )

    parser.add_argument(
        "--limit-per-page",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    sequence_id = args.sequence_id

    # --------------------------------------------------------
    # Environment checks
    # --------------------------------------------------------

    print()
    print("ENVIRONMENT")
    print("-" * 60)
    print(
        f"PyTorch           : {torch.__version__}"
    )
    print(
        f"ONNX Runtime      : {ort.__version__}"
    )
    print(
        f"ORT providers     : {ort.get_available_providers()}"
    )
    print(
        f"CUDA available    : {torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():
        print(
            f"CUDA device       : {torch.cuda.get_device_name(0)}"
        )

    # Explicitly preload ORT CUDA DLLs when supported.
    preload_dlls = getattr(
        ort,
        "preload_dlls",
        None,
    )

    if callable(preload_dlls):
        try:
            preload_dlls()
            print(
                "ORT DLL preload    : OK"
            )
        except Exception as exc:
            print(
                f"ORT DLL preload    : failed ({exc})"
            )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    sequence_dir = (
        ROOT
        / "dataset"
        / "development"
        / "images"
        / sequence_id
    )

    if not sequence_dir.exists():
        raise FileNotFoundError(
            f"Sequence directory not found: "
            f"{sequence_dir}"
        )

    page_paths = [
        sequence_dir / "01.png",
        sequence_dir / "02.png",
        sequence_dir / "03.png",
    ]

    for path in page_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    # --------------------------------------------------------
    # Oracle
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Oracle
    # --------------------------------------------------------

    run_oracle_builder(sequence_id)

    oracle = load_oracle(
        sequence_id
    )

    print(
        f"Loaded oracle rows: {len(oracle)}"
    )

    if len(oracle) == 0:
        raise RuntimeError(
            "Oracle loader returned zero rows."
        )

    # --------------------------------------------------------
    # CTD
    # --------------------------------------------------------

    detector = TextDetector(
        model_path=str(CTD_MODEL),
        input_size=1024,
        device="cuda",
    )

    # --------------------------------------------------------
    # PaddleOCR
    # --------------------------------------------------------

    recognizer = TextRecognition(
        model_name="en_PP-OCRv5_mobile_rec",
        engine="onnxruntime",
        device="gpu:0",
    )

    # --------------------------------------------------------
    # Paddle sanity test
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("PADDLE OCR SANITY CHECK")
    print("=" * 90)

    sanity_image = None
    sanity_block = None

    first_image = cv2.imread(
        str(page_paths[0])
    )

    _, _, sanity_blocks = detector(
        first_image
    )

    if sanity_blocks:
        sanity_block = sanity_blocks[0]

        lines = list(
            getattr(
                sanity_block,
                "lines",
                [],
            )
        )

        if lines:
            bbox = line_bbox(lines[0])

            x1, y1, x2, y2 = bbox

            sanity_image = first_image[
                max(0, y1 - 8):min(
                    first_image.shape[0],
                    y2 + 8,
                ),
                max(0, x1 - 8):min(
                    first_image.shape[1],
                    x2 + 8,
                ),
            ]

    if sanity_image is None:
        raise RuntimeError(
            "Could not obtain a CTD line for PaddleOCR sanity check."
        )

    sanity_out = (
        OUTPUT_ROOT
        / sequence_id
        / "sanity_ocr"
        / "line.png"
    )

    sanity_text, sanity_score, sanity_raw = (
        recognize_line(
            recognizer,
            sanity_image,
            sanity_out,
        )
    )

    print(
        f"Sanity OCR text : {sanity_text!r}"
    )
    print(
        f"Sanity OCR score: {sanity_score:.4f}"
    )

    if not sanity_text.strip():
        print()
        print(
            "FATAL: PaddleOCR returned EMPTY TEXT."
        )
        print(
            "Stopping benchmark because the OCR branch "
            "is not functioning."
        )
        print()
        print(
            "Raw Paddle result:"
        )
        print(
            json.dumps(
                sanity_raw,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        raise SystemExit(2)

    print(
        "PaddleOCR sanity check: PASS"
    )

    # --------------------------------------------------------
    # Qwen
    # --------------------------------------------------------

    model, processor = load_qwen()

    # --------------------------------------------------------
    # Process pages
    # --------------------------------------------------------

    all_records: list[dict[str, Any]] = []

    for page_index, page_path in enumerate(
        page_paths,
        start=1,
    ):
        records = process_page(
            sequence_id=sequence_id,
            page_index=page_index,
            page_path=page_path,
            detector=detector,
            recognizer=recognizer,
            model=model,
            processor=processor,
            oracle=oracle,
            limit_per_page=args.limit_per_page,
        )

        all_records.extend(records)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    matched = [
        record
        for record in all_records
        if record["ground_truth"]["text"]
    ]

    ocr_better = 0
    vlm_better = 0
    equal = 0

    ocr_cers: list[float] = []
    vlm_cers: list[float] = []

    for record in matched:
        oc = record["metrics"]["ocr_cer"]
        vc = record["metrics"]["vlm_cer"]

        if oc is None or vc is None:
            continue

        ocr_cers.append(float(oc))
        vlm_cers.append(float(vc))

        if oc < vc:
            ocr_better += 1
        elif vc < oc:
            vlm_better += 1
        else:
            equal += 1

    print()
    print("=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)

    print(
        f"Sequence          : {sequence_id}"
    )
    print(
        f"Blocks processed  : {len(all_records)}"
    )
    print(
        f"Reliable matches  : {len(matched)}"
    )
    print(
        f"OCR better        : {ocr_better}"
    )
    print(
        f"VLM better        : {vlm_better}"
    )
    print(
        f"Equal             : {equal}"
    )

    if ocr_cers:
        print(
            f"Mean OCR CER      : "
            f"{np.mean(ocr_cers):.4f}"
        )
    else:
        print(
            "Mean OCR CER      : N/A"
        )

    if vlm_cers:
        print(
            f"Mean VLM CER      : "
            f"{np.mean(vlm_cers):.4f}"
        )
    else:
        print(
            "Mean VLM CER      : N/A"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = (
        OUTPUT_ROOT
        / sequence_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        output_dir
        / "results.jsonl"
    )

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for record in all_records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    print()
    print(
        f"Saved results     : {results_path}"
    )


if __name__ == "__main__":
    main()