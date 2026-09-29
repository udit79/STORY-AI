"""Structural tests for the Nemotron WSL bridge adapter.

These tests verify the Windows-side bridge behavior without requiring WSL or
the actual model to be available. A subprocess.Popen mock simulates the worker.
"""

from __future__ import annotations

import io
import json
from unittest.mock import MagicMock, patch

import pytest

from app.models.nemotron_wsl_bridge import (
    NemotronWSLBridge,
    NemotronWSLBridgeError,
    _win_to_wsl,
)

# ---------------------------------------------------------------------------
# Path conversion
# ---------------------------------------------------------------------------

class TestWinToWsl:
    def test_basic_d_drive(self) -> None:
        result = _win_to_wsl(r"D:\STORY-AI\foo\bar.png")
        assert result == "/mnt/d/STORY-AI/foo/bar.png"

    def test_c_drive(self) -> None:
        result = _win_to_wsl(r"C:\Users\test\image.png")
        assert result == "/mnt/c/Users/test/image.png"

    def test_lowercase_drive(self) -> None:
        result = _win_to_wsl(r"d:\foo\bar.png")
        assert result == "/mnt/d/foo/bar.png"

    def test_already_posix_like(self) -> None:
        # Paths without a Windows drive letter pass through unchanged
        result = _win_to_wsl("/mnt/d/already/posix.png")
        assert result.startswith("/")


# ---------------------------------------------------------------------------
# Mock worker helpers
# ---------------------------------------------------------------------------

def _make_mock_proc(responses: list[dict]) -> MagicMock:
    """Return a Popen mock that emits newline-delimited JSON responses."""
    response_lines = [json.dumps(r) + "\n" for r in responses]
    stdout_iter = iter(response_lines)

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.returncode = None

    stdin_buf: list[str] = []
    mock_proc.stdin = MagicMock()
    mock_proc.stdin.write = lambda s: stdin_buf.append(s)
    mock_proc.stdin.flush = MagicMock()
    mock_proc.stdin.close = MagicMock()

    # stdout.readline() pops from the prepared responses list
    def _readline() -> str:
        try:
            return next(stdout_iter)
        except StopIteration:
            return ""

    mock_proc.stdout = MagicMock()
    mock_proc.stdout.readline = _readline
    mock_proc.stderr = io.StringIO("")  # empty stderr

    return mock_proc, stdin_buf


# ---------------------------------------------------------------------------
# Bridge construction
# ---------------------------------------------------------------------------

class TestNemotronWSLBridgeConstruction:
    def test_direct_construction(self) -> None:
        """NemotronWSLBridge can be constructed directly (skipping start())."""
        mock_proc, _ = _make_mock_proc([])
        bridge = NemotronWSLBridge(mock_proc, [], merge_level="paragraph")
        assert bridge._merge_level == "paragraph"

    def test_start_succeeds_with_ready_signal(self) -> None:
        """start() returns once the worker emits the __ready__ response."""
        mock_proc, _sent = _make_mock_proc([
            {"id": "__ready__", "ok": True, "predictions": []},
        ])
        with patch("subprocess.Popen", return_value=mock_proc):
            bridge = NemotronWSLBridge.start(startup_timeout=5.0)
        assert bridge is not None

    def test_start_raises_on_startup_failed(self) -> None:
        """start() raises NemotronWSLBridgeError when the worker reports __startup_failed__."""
        mock_proc, _ = _make_mock_proc([
            {"id": "__startup_failed__", "ok": False, "error": "CUDA not found"},
        ])
        with patch("subprocess.Popen", return_value=mock_proc), pytest.raises(NemotronWSLBridgeError, match="CUDA not found"):
            NemotronWSLBridge.start(startup_timeout=5.0)

    def test_start_raises_when_wsl_not_found(self) -> None:
        """start() raises NemotronWSLBridgeError when WSL is not available."""
        with patch("subprocess.Popen", side_effect=FileNotFoundError("wsl not found")), pytest.raises(NemotronWSLBridgeError, match="WSL"):
            NemotronWSLBridge.start(startup_timeout=5.0)


# ---------------------------------------------------------------------------
# OCR requests
# ---------------------------------------------------------------------------

class TestNemotronWSLBridgeOCR:
    def _bridge_with_responses(self, responses: list[dict]) -> tuple[NemotronWSLBridge, list[str]]:
        mock_proc, sent = _make_mock_proc(responses)
        bridge = NemotronWSLBridge(mock_proc, [], merge_level="paragraph")
        return bridge, sent

    def test_ocr_returns_predictions(self) -> None:
        expected = [{"text": "Hello", "confidence": 0.95, "left": 0.1, "upper": 0.1, "right": 0.5, "lower": 0.3}]
        # We need to know the req_id ahead of time — but we can't without mocking uuid.
        # Instead, patch uuid4 to a fixed value.
        with patch("app.models.nemotron_wsl_bridge.uuid.uuid4", return_value="test-uuid-1234"):
            bridge, _sent = self._bridge_with_responses([
                {"id": "test-uuid-1234", "ok": True, "predictions": expected},
            ])
            result = bridge.ocr(r"D:\STORY-AI\crop.png")
        assert result == expected

    def test_ocr_raises_on_worker_error(self) -> None:
        with patch("app.models.nemotron_wsl_bridge.uuid.uuid4", return_value="err-uuid"):
            bridge, _ = self._bridge_with_responses([
                {"id": "err-uuid", "ok": False, "error": "Inference failed"},
            ])
            with pytest.raises(NemotronWSLBridgeError, match="Inference failed"):
                bridge.ocr(r"D:\STORY-AI\crop.png")

    def test_ocr_converts_windows_path(self) -> None:
        """The bridge converts the Windows path to a WSL path in the request."""
        sent_requests: list[str] = []
        mock_proc, _ = _make_mock_proc([
            {"id": "path-uuid", "ok": True, "predictions": []},
        ])
        original_write = mock_proc.stdin.write

        def capturing_write(s: str) -> None:
            sent_requests.append(s)
            original_write(s)

        mock_proc.stdin.write = capturing_write

        with patch("app.models.nemotron_wsl_bridge.uuid.uuid4", return_value="path-uuid"):
            bridge = NemotronWSLBridge(mock_proc, [], merge_level="paragraph")
            bridge.ocr(r"D:\STORY-AI\crop.png")

        assert sent_requests, "no request was written"
        req = json.loads(sent_requests[0].strip())
        assert req["image_path"] == "/mnt/d/STORY-AI/crop.png"

    def test_ocr_skips_non_matching_responses(self) -> None:
        """The bridge skips responses for other req_ids before finding its own."""
        with patch("app.models.nemotron_wsl_bridge.uuid.uuid4", return_value="my-uuid"):
            bridge, _ = self._bridge_with_responses([
                {"id": "other-uuid", "ok": True, "predictions": [{"text": "wrong"}]},
                {"id": "my-uuid",    "ok": True, "predictions": [{"text": "correct"}]},
            ])
            result = bridge.ocr(r"D:\STORY-AI\crop.png")
        assert result == [{"text": "correct"}]

    def test_ocr_raises_when_worker_exited(self) -> None:
        mock_proc, _ = _make_mock_proc([])
        mock_proc.poll.return_value = 1  # worker is dead
        mock_proc.returncode = 1
        bridge = NemotronWSLBridge(mock_proc, [], merge_level="paragraph")
        with pytest.raises(NemotronWSLBridgeError, match="exited"):
            bridge.ocr(r"D:\STORY-AI\crop.png")

    def test_callable_interface(self) -> None:
        """Bridge is callable with (image_path, merge_level) like NemotronOCRV2."""
        with patch("app.models.nemotron_wsl_bridge.uuid.uuid4", return_value="call-uuid"):
            bridge, _ = self._bridge_with_responses([
                {"id": "call-uuid", "ok": True, "predictions": [{"text": "hi"}]},
            ])
            result = bridge(r"D:\STORY-AI\crop.png", merge_level="word")
        assert result == [{"text": "hi"}]


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

class TestNemotronWSLBridgeShutdown:
    def test_shutdown_sends_request(self) -> None:
        mock_proc, _sent = _make_mock_proc([
            {"id": "shutdown", "ok": True, "predictions": []},
        ])
        mock_proc.wait = MagicMock()
        bridge = NemotronWSLBridge(mock_proc, [], merge_level="paragraph")
        bridge.shutdown()
        mock_proc.stdin.close.assert_called_once()
        mock_proc.wait.assert_called_once()

    def test_shutdown_tolerates_dead_process(self) -> None:
        mock_proc, _ = _make_mock_proc([])
        mock_proc.poll.return_value = 0  # already done
        bridge = NemotronWSLBridge(mock_proc, [], merge_level="paragraph")
        bridge.shutdown()  # must not raise

    def test_context_manager(self) -> None:
        mock_proc, _ = _make_mock_proc([
            {"id": "shutdown", "ok": True, "predictions": []},
        ])
        mock_proc.wait = MagicMock()
        bridge = NemotronWSLBridge(mock_proc, [], merge_level="paragraph")
        with bridge:
            pass
        mock_proc.stdin.close.assert_called_once()
