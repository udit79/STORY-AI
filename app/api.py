"""Local FastAPI wrapper for pipeline runs and their artifacts."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, RedirectResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / ".smoke" / "fastapi-runs"
UPLOADS_ROOT = ROOT / ".smoke" / "fastapi-uploads"
PIPELINE_SCRIPT = ROOT / "tools" / "pipeline_integration_smoke.py"
SEQUENCES_PATH = ROOT / "dataset" / "sequences.json"

app = FastAPI(
    title="STORY-AI Local API",
    description=(
        "Run the three-page local pipeline and inspect its console log, "
        "submission, and character identity graph."
    ),
    version="0.1.0",
)


@app.get("/", include_in_schema=False)
def api_root() -> RedirectResponse:
    """Send visitors who open the bare API address to interactive docs."""
    return RedirectResponse(url="/docs", status_code=307)


class RunOptions(BaseModel):
    ocr_engine: Literal["paddle", "nemotron"] = Field(
        default="paddle", description="OCR backend; Nemotron requires the local WSL worker."
    )
    visual_speaker: bool = Field(
        default=False, description="Enable the slower Qwen visual speaker check."
    )
    ctd_model: str | None = Field(
        default=None, description="Override the local CTD ONNX model path."
    )
    layout_weights: str | None = Field(
        default=None, description="Override the local YOLO balloon checkpoint path."
    )
    paddle_model_dir: str | None = Field(
        default=None, description="Override the local PaddleOCR model directory."
    )
    rtdetr_model: str | None = Field(
        default=None, description="Override the local RT-DETRv4 ONNX model path."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"ocr_engine": "paddle", "visual_speaker": False},
                {"ocr_engine": "nemotron", "visual_speaker": False},
            ]
        }
    }


class RunState(BaseModel):
    sequence_id: str
    status: Literal["queued", "running", "completed", "failed"]
    started_at: str | None = None
    finished_at: str | None = None
    exit_code: int | None = None
    log_url: str
    graph_url: str
    submission_url: str


class CharacterGraph(BaseModel):
    sequence_id: str
    nodes: list[dict[str, Any]]
    identity_edges: list[dict[str, Any]]
    speaker_links: list[dict[str, Any]]


class UploadReceipt(BaseModel):
    upload_id: str
    page_files: list[str]
    run_url: str


_RUN_LOCK = Lock()
_RUN_STATES: dict[str, RunState] = {}


@lru_cache(maxsize=1)
def _sequence_index() -> dict[str, dict[str, Any]]:
    records = json.loads(SEQUENCES_PATH.read_text(encoding="utf-8"))
    return {record["sequence_id"]: record for record in records}


def _upload_directory(upload_id: str) -> Path | None:
    if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
        return None
    return UPLOADS_ROOT / upload_id


def _is_known_run_id(sequence_id: str) -> bool:
    if sequence_id in _sequence_index():
        return True
    if not sequence_id.startswith("upload-"):
        return False
    upload_directory = _upload_directory(sequence_id.removeprefix("upload-"))
    return upload_directory is not None and upload_directory.is_dir()


def _pipeline_defaults() -> dict[str, Path]:
    from tools.pipeline_integration_smoke import (
        DEFAULT_CTD_MODEL,
        DEFAULT_LAYOUT_WEIGHTS,
        DEFAULT_PADDLE_MODEL_DIR,
        DEFAULT_RTDETR_MODEL,
    )

    return {
        "ctd": DEFAULT_CTD_MODEL,
        "layout": DEFAULT_LAYOUT_WEIGHTS,
        "paddle": DEFAULT_PADDLE_MODEL_DIR,
        "rtdetr": DEFAULT_RTDETR_MODEL,
    }


def _qwen_cache_root() -> Path:
    return Path(
        os.environ.get(
            "HF_HOME",
            os.environ.get(
                "TRANSFORMERS_CACHE",
                str(Path.home() / ".cache" / "huggingface" / "hub"),
            ),
        )
    ).expanduser()


def _effective_model_paths(options: RunOptions | None = None) -> dict[str, Path]:
    defaults = _pipeline_defaults()
    overrides = {
        "ctd": options.ctd_model if options else None,
        "layout": options.layout_weights if options else None,
        "paddle": options.paddle_model_dir if options else None,
        "rtdetr": options.rtdetr_model if options else None,
    }
    return {
        name: Path(overrides[name]).expanduser().resolve()
        if overrides[name]
        else defaults[name].expanduser().resolve()
        for name in defaults
    }


def _model_config() -> dict[str, Any]:
    paths = _effective_model_paths()
    qwen_cache = _qwen_cache_root()
    qwen_model_cache = qwen_cache / "models--Qwen--Qwen3-VL-4B-Instruct"
    return {
        "ctd": {"path": str(paths["ctd"]), "available": paths["ctd"].is_file()},
        "paddleocr": {
            "path": str(paths["paddle"]),
            "available": paths["paddle"].is_dir(),
            "default_engine": "onnxruntime",
            "device": "cpu",
        },
        "layout": {
            "path": str(paths["layout"]),
            "available": paths["layout"].is_file(),
            "model_id": "huyvux3005/manga109-segmentation-bubble",
        },
        "qwen": {
            "model_id": "Qwen/Qwen3-VL-4B-Instruct",
            "cache_root": str(qwen_cache),
            "available": qwen_model_cache.is_dir(),
            "local_files_only": True,
            "load_in_4bit": True,
        },
        "rtdetr": {
            "path": str(paths["rtdetr"]),
            "available": paths["rtdetr"].is_file(),
            "model_id": "tori29umai/rtdetrv4-x-manga109s_v2",
        },
        "mobilenetv3": {
            "weights": "torchvision IMAGENET1K_V1",
            "device": "cpu",
            "may_download_if_uncached": True,
        },
        "nemotron": {
            "optional": True,
            "wsl_worker": str(ROOT / "tools" / "nemotron_worker.py"),
            "local_model_directory": str(ROOT / "nemotron-ocr-v2"),
        },
    }


def _state_path(sequence_id: str) -> Path:
    return RUNS_ROOT / sequence_id / "run.json"


def _persist_state(state: RunState) -> None:
    state_path = _state_path(state.sequence_id)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _RUN_STATES[state.sequence_id] = state


def _read_state(sequence_id: str) -> RunState | None:
    if sequence_id in _RUN_STATES:
        return _RUN_STATES[sequence_id]
    path = _state_path(sequence_id)
    if not path.is_file():
        return None
    return RunState.model_validate_json(path.read_text(encoding="utf-8"))


def _run_pipeline(
    sequence_id: str,
    options: RunOptions,
    image_dir_override: Path | None = None,
) -> None:
    run_dir = RUNS_ROOT / sequence_id
    log_path = run_dir / "pipeline.log"
    submission_path = run_dir / "submission.jsonl"
    artifacts_dir = run_dir / "artifacts"
    paths = _effective_model_paths(options)
    state = _read_state(sequence_id)
    started_at = datetime.now(UTC).isoformat()
    state = RunState(
        sequence_id=sequence_id,
        status="running",
        started_at=started_at,
        log_url=f"/runs/{sequence_id}/log",
        graph_url=f"/runs/{sequence_id}/character-graph",
        submission_url=f"/runs/{sequence_id}/submission",
    )
    _persist_state(state)

    if image_dir_override is None:
        sequence = _sequence_index()[sequence_id]
        image_dir = (ROOT / "dataset" / sequence["images"][0]).parent
    else:
        image_dir = image_dir_override
    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--sequence-id",
        sequence_id,
        "--image-dir",
        str(image_dir),
        "--ctd-model",
        str(paths["ctd"]),
        "--layout-weights",
        str(paths["layout"]),
        "--paddle-model-dir",
        str(paths["paddle"]),
        "--rtdetr-model",
        str(paths["rtdetr"]),
        "--smoke-dir",
        str(artifacts_dir),
        "--submission-path",
        str(submission_path),
    ]
    command.append("--skip-paddle" if options.ocr_engine == "nemotron" else "--skip-nemotron")
    if not options.visual_speaker:
        command.append("--skip-visual-speaker")

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        completed = RunState(
            sequence_id=sequence_id,
            status="completed" if result.returncode == 0 else "failed",
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
            exit_code=result.returncode,
            log_url=f"/runs/{sequence_id}/log",
            graph_url=f"/runs/{sequence_id}/character-graph",
            submission_url=f"/runs/{sequence_id}/submission",
        )
    except Exception as exc:  # noqa: BLE001 - preserve failure detail for the API log
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\nAPI runner error: {type(exc).__name__}: {exc}\n")
        completed = RunState(
            sequence_id=sequence_id,
            status="failed",
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
            exit_code=-1,
            log_url=f"/runs/{sequence_id}/log",
            graph_url=f"/runs/{sequence_id}/character-graph",
            submission_url=f"/runs/{sequence_id}/submission",
        )
    _persist_state(completed)


def _queue_run(
    sequence_id: str,
    image_dir: Path,
    background_tasks: BackgroundTasks,
    options: RunOptions,
) -> RunState:
    image_dir = image_dir.resolve()
    if not image_dir.is_dir() or len(list(image_dir.glob("*.png"))) != 3:
        raise HTTPException(status_code=409, detail="Exactly three local PNG pages are required")

    paths = _effective_model_paths(options)
    models = _model_config()
    missing = []
    for name, path in paths.items():
        if name in {"ctd", "layout", "rtdetr"} and not path.is_file():
            missing.append(f"{name}:{path}")
        if name == "paddle" and options.ocr_engine == "paddle" and not path.is_dir():
            missing.append(f"paddleocr:{path}")
    if not models["qwen"]["available"]:
        missing.append("qwen local cache")
    if missing:
        raise HTTPException(
            status_code=409,
            detail={"message": "Required local models are not ready", "missing": missing},
        )

    with _RUN_LOCK:
        existing = _read_state(sequence_id)
        if existing is not None and existing.status in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="This sequence is already running")
        state = RunState(
            sequence_id=sequence_id,
            status="queued",
            log_url=f"/runs/{sequence_id}/log",
            graph_url=f"/runs/{sequence_id}/character-graph",
            submission_url=f"/runs/{sequence_id}/submission",
        )
        _persist_state(state)
    background_tasks.add_task(_run_pipeline, sequence_id, options, image_dir)
    return state


def build_character_graph(trace: dict[str, Any]) -> dict[str, Any]:
    """Convert a saved speaker/identity trace into API-friendly graph data."""
    nodes = [
        {
            "id": character["character_id"],
            "page_index": character["page_index"],
            "label": character.get("label"),
            "identity_cluster_id": character.get("identity_cluster_id"),
            "bbox": character.get("bbox"),
        }
        for character in trace.get("characters", [])
    ]
    identity_edges = list(trace.get("identity_edges", []))
    speaker_links = [
        {
            "balloon_id": balloon["balloon_id"],
            "character_id": balloon["selected_character_id"],
            "speaker_label": balloon.get("final_speaker"),
            "grounding_method": balloon.get("speaker_grounding_method"),
            "confidence": balloon.get("speaker_grounding_confidence"),
        }
        for balloon in trace.get("balloons", [])
        if balloon.get("selected_character_id")
    ]
    return {
        "sequence_id": trace["sequence_id"],
        "nodes": nodes,
        "identity_edges": identity_edges,
        "speaker_links": speaker_links,
    }


@app.get("/health", summary="Health check", description="Confirm the local API process is responding.")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get(
    "/config",
    summary="Inspect local model configuration",
    description="Show effective model paths, cache availability, and whether required model files are present.",
)
def system_config() -> dict[str, Any]:
    """Report effective local model paths and whether the expected files exist."""
    models = _model_config()
    required = ("ctd", "paddleocr", "layout", "qwen", "rtdetr")
    missing = [name for name in required if not models[name]["available"]]
    return {
        "repository_root": str(ROOT),
        "python_required": "3.11",
        "default_ocr_engine": "paddle",
        "visual_speaker_default": False,
        "models_ready": not missing,
        "missing_required_models": missing,
        "models": models,
    }


@app.get(
    "/runs",
    response_model=list[RunState],
    summary="List recorded runs",
    description="Return queued, running, completed, or failed run records persisted by this API process.",
)
def list_runs() -> list[RunState]:
    if not RUNS_ROOT.is_dir():
        return []
    states = []
    for path in sorted(RUNS_ROOT.glob("*/run.json")):
        states.append(RunState.model_validate_json(path.read_text(encoding="utf-8")))
    return states


@app.post(
    "/runs/{sequence_id}",
    response_model=RunState,
    status_code=202,
    summary="Run a dataset sequence",
    description="Queue the three local images for a sequence from dataset/sequences.json.",
)
def start_run(
    sequence_id: str,
    background_tasks: BackgroundTasks,
    options: RunOptions | None = None,
) -> RunState:
    if sequence_id not in _sequence_index():
        raise HTTPException(status_code=404, detail="Unknown sequence_id")
    sequence = _sequence_index()[sequence_id]
    image_dir = (ROOT / "dataset" / sequence["images"][0]).parent
    if not image_dir.is_dir() or len(sequence.get("images", [])) != 3:
        raise HTTPException(status_code=409, detail="Three local page images are required")
    return _queue_run(sequence_id, image_dir, background_tasks, options or RunOptions())


@app.post(
    "/uploads",
    response_model=UploadReceipt,
    status_code=201,
    summary="Upload a custom three-page sequence",
    description="Attach one image to each page field. Images are stored locally in page order and normalized to PNG.",
)
async def upload_sequence(
    page1: UploadFile = File(..., description="Image for page 1"),  # noqa: B008
    page2: UploadFile = File(..., description="Image for page 2"),  # noqa: B008
    page3: UploadFile = File(..., description="Image for page 3"),  # noqa: B008
) -> UploadReceipt:
    """Upload exactly three page images in reading-sequence order."""
    import cv2
    import numpy as np

    encoded_pages: list[bytes] = []
    for upload in (page1, page2, page3):
        data = await upload.read(20 * 1024 * 1024 + 1)
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Each page image must be at most 20 MiB")
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=415, detail="All three uploads must be readable images")
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise HTTPException(status_code=422, detail="Could not normalize uploaded page image")
        encoded_pages.append(encoded.tobytes())

    upload_id = uuid4().hex
    upload_directory = UPLOADS_ROOT / upload_id
    upload_directory.mkdir(parents=True, exist_ok=False)
    page_files = []
    for page_number, data in enumerate(encoded_pages, start=1):
        filename = f"{page_number:02d}.png"
        (upload_directory / filename).write_bytes(data)
        page_files.append(filename)
    return UploadReceipt(
        upload_id=upload_id,
        page_files=page_files,
        run_url=f"/uploads/{upload_id}/runs",
    )


@app.post(
    "/uploads/{upload_id}/runs",
    response_model=RunState,
    status_code=202,
    summary="Run an uploaded sequence",
    description="Queue a previously uploaded three-page sequence with the selected local model configuration.",
)
def start_uploaded_run(
    upload_id: str,
    background_tasks: BackgroundTasks,
    options: RunOptions | None = None,
) -> RunState:
    upload_directory = _upload_directory(upload_id)
    if upload_directory is None or not upload_directory.is_dir():
        raise HTTPException(status_code=404, detail="Uploaded sequence not found")
    return _queue_run(
        f"upload-{upload_id}",
        upload_directory,
        background_tasks,
        options or RunOptions(),
    )


@app.get(
    "/runs/{sequence_id}",
    response_model=RunState,
    summary="Get run status",
    description="Return status and artifact URLs for a dataset or uploaded-sequence run.",
)
def get_run(sequence_id: str) -> RunState:
    if not _is_known_run_id(sequence_id):
        raise HTTPException(status_code=404, detail="Unknown sequence_id")
    state = _read_state(sequence_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return state


@app.get(
    "/runs/{sequence_id}/log",
    response_class=PlainTextResponse,
    summary="Read the pipeline log",
    description="Return the captured stdout and stderr for this run as plain text.",
)
def get_run_log(sequence_id: str) -> str:
    log_path = RUNS_ROOT / sequence_id / "pipeline.log"
    if not _is_known_run_id(sequence_id) or not log_path.is_file():
        raise HTTPException(status_code=404, detail="Run log not found")
    return log_path.read_text(encoding="utf-8", errors="replace")


@app.get(
    "/runs/{sequence_id}/character-graph",
    response_model=CharacterGraph,
    summary="Get the character identity graph",
    description="Return character nodes, pairwise cross-page identity edges, and balloon-to-speaker links.",
)
def get_character_graph(sequence_id: str) -> dict[str, Any]:
    trace_path = (
        RUNS_ROOT
        / sequence_id
        / "artifacts"
        / sequence_id
        / "speaker_identity_trace.json"
    )
    if not _is_known_run_id(sequence_id) or not trace_path.is_file():
        raise HTTPException(status_code=404, detail="Character identity trace not found")
    return build_character_graph(json.loads(trace_path.read_text(encoding="utf-8")))


@app.get(
    "/runs/{sequence_id}/submission",
    summary="Get the run submission",
    description="Return the generated three-page submission object as JSON.",
)
def get_submission(sequence_id: str) -> dict[str, Any]:
    path = RUNS_ROOT / sequence_id / "submission.jsonl"
    if not _is_known_run_id(sequence_id) or not path.is_file():
        raise HTTPException(status_code=404, detail="Run submission not found")
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])