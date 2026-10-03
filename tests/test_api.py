import importlib
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx2")

TestClient = importlib.import_module("fastapi.testclient").TestClient
api = importlib.import_module("app.api")


def test_character_graph_keeps_identity_edges_and_speaker_links():
    graph = api.build_character_graph(
        {
            "sequence_id": "seq-test",
            "characters": [
                {
                    "character_id": "character-p0-a",
                    "page_index": 0,
                    "label": "Character A",
                    "identity_cluster_id": "identity-001",
                    "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                },
                {
                    "character_id": "character-p1-a",
                    "page_index": 1,
                    "label": "Character A",
                    "identity_cluster_id": "identity-001",
                    "bbox": {"x1": 1, "y1": 1, "x2": 11, "y2": 11},
                },
            ],
            "identity_edges": [
                {"source": "character-p0-a", "target": "character-p1-a", "score": 0.9}
            ],
            "balloons": [
                {
                    "balloon_id": "balloon-1",
                    "selected_character_id": "character-p0-a",
                    "final_speaker": "Character A",
                    "speaker_grounding_method": "geometry",
                    "speaker_grounding_confidence": 0.8,
                }
            ],
        }
    )

    assert graph["sequence_id"] == "seq-test"
    assert len(graph["nodes"]) == 2
    assert graph["identity_edges"][0]["score"] == 0.9
    assert graph["speaker_links"][0]["character_id"] == "character-p0-a"


def test_config_endpoint_reports_model_readiness():
    response = TestClient(api.app).get("/config")

    assert response.status_code == 200
    config = response.json()
    assert config["default_ocr_engine"] == "paddle"
    assert "qwen" in config["models"]
    assert isinstance(config["models_ready"], bool)


def test_start_run_queues_pipeline_for_known_sequence(tmp_path, monkeypatch):
    sequence_id = "seq_952f154fb1505883"
    model_paths = {
        "ctd": tmp_path / "ctd.onnx",
        "layout": tmp_path / "layout.pt",
        "paddle": tmp_path / "paddle-model",
        "rtdetr": tmp_path / "rtdetr.onnx",
    }
    for name, path in model_paths.items():
        if name == "paddle":
            path.mkdir()
        else:
            path.write_text("local model", encoding="utf-8")
    queued = []
    monkeypatch.setattr(api, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(api, "_RUN_STATES", {})
    monkeypatch.setattr(api, "_effective_model_paths", lambda _options: model_paths)
    monkeypatch.setattr(api, "_model_config", lambda: {"qwen": {"available": True}})
    monkeypatch.setattr(
        api,
        "_run_pipeline",
        lambda run_id, options, image_dir: queued.append(
            (run_id, options.ocr_engine, image_dir)
        ),
    )

    response = TestClient(api.app).post(f"/runs/{sequence_id}", json={})

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert queued == [(sequence_id, "paddle", api.ROOT / "dataset/development/images" / sequence_id)]


def _png_bytes():
    import cv2
    import numpy as np

    success, encoded = cv2.imencode(".png", np.zeros((16, 16, 3), dtype=np.uint8))
    assert success
    return encoded.tobytes()


def test_upload_sequence_stores_three_ordered_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "UPLOADS_ROOT", tmp_path)
    png = _png_bytes()
    response = TestClient(api.app).post(
        "/uploads",
        files={
            "page1": ("first.png", png, "image/png"),
            "page2": ("second.png", png, "image/png"),
            "page3": ("third.png", png, "image/png"),
        },
    )

    assert response.status_code == 201
    receipt = response.json()
    assert receipt["page_files"] == ["01.png", "02.png", "03.png"]
    assert receipt["run_url"] == f"/uploads/{receipt['upload_id']}/runs"
    assert all((tmp_path / receipt["upload_id"] / name).is_file() for name in receipt["page_files"])


def test_uploaded_sequence_can_be_queued(tmp_path, monkeypatch):
    upload_id = "a" * 32
    upload_directory = tmp_path / "uploads" / upload_id
    upload_directory.mkdir(parents=True)
    png = _png_bytes()
    for page_number in range(1, 4):
        (upload_directory / f"{page_number:02d}.png").write_bytes(png)
    model_paths = {
        "ctd": tmp_path / "ctd.onnx",
        "layout": tmp_path / "layout.pt",
        "paddle": tmp_path / "paddle-model",
        "rtdetr": tmp_path / "rtdetr.onnx",
    }
    for name, path in model_paths.items():
        if name == "paddle":
            path.mkdir()
        else:
            path.write_text("local model", encoding="utf-8")
    queued = []
    monkeypatch.setattr(api, "UPLOADS_ROOT", tmp_path / "uploads")
    monkeypatch.setattr(api, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(api, "_RUN_STATES", {})
    monkeypatch.setattr(api, "_effective_model_paths", lambda _options: model_paths)
    monkeypatch.setattr(api, "_model_config", lambda: {"qwen": {"available": True}})
    monkeypatch.setattr(api, "_run_pipeline", lambda *args: queued.append(args))

    response = TestClient(api.app).post(f"/uploads/{upload_id}/runs", json={})

    assert response.status_code == 202
    assert response.json()["sequence_id"] == f"upload-{upload_id}"
    assert queued[0][0] == f"upload-{upload_id}"
    assert queued[0][2] == upload_directory


def test_log_endpoint_returns_saved_pipeline_log(tmp_path, monkeypatch):
    sequence_id = "seq_952f154fb1505883"
    run_dir = tmp_path / sequence_id
    run_dir.mkdir()
    (run_dir / "pipeline.log").write_text("run log line\n", encoding="utf-8")
    monkeypatch.setattr(api, "RUNS_ROOT", tmp_path)

    response = TestClient(api.app).get(f"/runs/{sequence_id}/log")

    assert response.status_code == 200
    assert response.text == "run log line\n"


def test_character_graph_endpoint_reads_generated_trace(tmp_path, monkeypatch):
    sequence_id = "seq_952f154fb1505883"
    trace_dir = tmp_path / sequence_id / "artifacts" / sequence_id
    trace_dir.mkdir(parents=True)
    trace = {
        "sequence_id": sequence_id,
        "characters": [],
        "identity_edges": [],
        "balloons": [],
    }
    (trace_dir / "speaker_identity_trace.json").write_text(
        json.dumps(trace), encoding="utf-8"
    )
    monkeypatch.setattr(api, "RUNS_ROOT", tmp_path)

    response = TestClient(api.app).get(
        f"/runs/{sequence_id}/character-graph"
    )

    assert response.status_code == 200
    assert response.json() == {
        "sequence_id": sequence_id,
        "nodes": [],
        "identity_edges": [],
        "speaker_links": [],
    }