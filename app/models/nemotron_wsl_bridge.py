"""Windows → WSL Nemotron bridge adapter.

Starts the ``tools/nemotron_worker.py`` script inside WSL under the nemotron
virtual environment and communicates with it over stdin/stdout using newline-
delimited JSON.

The worker process is started once and kept alive for the lifetime of this
object. Model loading happens inside WSL before the first request is accepted.

Usage::

    from app.models.nemotron_wsl_bridge import NemotronWSLBridge

    bridge = NemotronWSLBridge.start()
    predictions = bridge.ocr("/mnt/d/STORY-AI/.smoke/.../crop.png")
    bridge.shutdown()

or as a context manager::

    with NemotronWSLBridge.start() as bridge:
        predictions = bridge.ocr(image_path)

Windows→WSL path conversion:
    ``D:\\STORY-AI\\foo\\bar.png`` →  ``/mnt/d/STORY-AI/foo/bar.png``

The bridge transparently converts Windows paths before sending them to the
worker.

All diagnostics from the worker are written to stderr and captured in
``bridge.worker_stderr``.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Self

_ROOT = Path(__file__).resolve().parents[2]


def _win_to_wsl(path: str | Path) -> str:
    """Convert a Windows absolute path to a WSL /mnt/<drive>/... path."""
    p = Path(path).resolve()
    drive = p.drive  # e.g. "D:"
    if not drive:
        return str(p).replace("\\", "/")
    drive_letter = drive.rstrip(":").lower()
    rest = str(p)[len(drive):].replace("\\", "/").lstrip("/")
    return f"/mnt/{drive_letter}/{rest}"


class NemotronWSLBridgeError(RuntimeError):
    """Raised when the WSL worker fails to start or returns an error response."""


class NemotronWSLBridge:
    """Persistent Windows→WSL bridge for the Nemotron OCR worker."""

    # Paths used to start the worker inside WSL.
    _VENV_PYTHON     = "/mnt/d/STORY-AI/.venv-nemotron/bin/python"
    _WORKER_SCRIPT   = "/mnt/d/STORY-AI/tools/nemotron_worker.py"
    _CUDA_HOME       = "/mnt/d/STORY-AI/.cuda/cuda-12.8"
    _TORCH_HOME      = "/mnt/d/STORY-AI/.torch"
    _NEMOTRON_SRC    = "/mnt/d/STORY-AI/nemotron-ocr-v2/nemotron-ocr/src"

    def __init__(
        self,
        proc: subprocess.Popen,  # type: ignore[type-arg]
        stderr_lines: list[str],
        merge_level: str = "paragraph",
    ) -> None:
        self._proc = proc
        self._merge_level = merge_level
        self.worker_stderr = stderr_lines  # shared list appended by background thread

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def start(
        cls,
        model_dir: str | None = None,
        lang: str | None = None,
        merge_level: str = "paragraph",
        startup_timeout: float = 180.0,
    ) -> NemotronWSLBridge:
        """Spawn the WSL worker and wait until it signals ready.

        Args:
            model_dir:       Linux-side path to the checkpoint directory.
                             Defaults to the v2_english checkpoint inside the repo.
            lang:            Hub alias ("en", "multi", "v1"). Ignored if model_dir given.
            merge_level:     Default merge level for all requests.
            startup_timeout: Seconds to wait for the model to finish loading.

        Raises:
            NemotronWSLBridgeError: If the process fails to start or the model fails to load.
        """
        if model_dir is None:
            model_dir = "/mnt/d/STORY-AI/nemotron-ocr-v2/v2_english"

        cmd = ["wsl", "-e", cls._VENV_PYTHON, cls._WORKER_SCRIPT,
               "--model-dir", model_dir, "--merge-level", merge_level]
        if lang is not None and model_dir is None:
            cmd = ["wsl", "-e", cls._VENV_PYTHON, cls._WORKER_SCRIPT,
                   "--lang", lang, "--merge-level", merge_level]

        env = dict(os.environ)
        env["PYTHONPATH"]       = cls._NEMOTRON_SRC
        env["CUDA_HOME"]        = cls._CUDA_HOME
        env["TORCH_HOME"]       = cls._TORCH_HOME
        env["PATH"]             = cls._CUDA_HOME + "/bin:" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"]  = cls._CUDA_HOME + "/lib64:" + env.get("LD_LIBRARY_PATH", "")

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                bufsize=1,       # line-buffered
                text=True,
                encoding="utf-8",
            )
        except FileNotFoundError as exc:
            raise NemotronWSLBridgeError(
                "Could not launch WSL. Ensure WSL is installed and on PATH."
            ) from exc

        stderr_lines: list[str] = []

        def _drain_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_lines.append(line.rstrip())

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        bridge = cls(proc, stderr_lines, merge_level=merge_level)

        # Wait for __ready__ signal.
        try:
            bridge._wait_for_ready(startup_timeout)
        except NemotronWSLBridgeError:
            proc.terminate()
            raise

        return bridge

    def _wait_for_ready(self, timeout: float) -> None:
        """Block until the worker emits its __ready__ response."""
        import time

        assert self._proc.stdout is not None
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise NemotronWSLBridgeError(
                    f"Nemotron worker did not become ready within {timeout:.0f}s. "
                    f"Stderr tail: {self.worker_stderr[-5:]}"
                )
            # On Windows, select() does not work on pipes; use a short read loop.
            line = self._proc.stdout.readline()
            if not line:
                rc = self._proc.poll()
                if rc is not None:
                    raise NemotronWSLBridgeError(
                        f"Nemotron worker exited with code {rc} before becoming ready. "
                        f"Stderr: {self.worker_stderr[-10:]}"
                    )
                time.sleep(0.1)
                continue
            try:
                msg = json.loads(line.strip())
            except json.JSONDecodeError:
                continue  # skip non-JSON startup noise
            if msg.get("id") == "__startup_failed__":
                raise NemotronWSLBridgeError(
                    f"Nemotron worker failed at startup: {msg.get('error', '?')}"
                )
            if msg.get("id") == "__ready__":
                return  # worker is alive and model is loaded

    # ── Public API ────────────────────────────────────────────────────────────

    def ocr(
        self,
        image_path: str | Path,
        merge_level: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run Nemotron OCR on one image crop.

        Args:
            image_path:  Windows or WSL absolute path to the crop image.
            merge_level: Override the bridge-level default for this request.

        Returns:
            List of prediction dicts: {"text": str, "confidence": float,
                                       "left": float, "upper": float,
                                       "right": float, "lower": float}.

        Raises:
            NemotronWSLBridgeError: If the worker returns an error response
                                     or the process has exited unexpectedly.
        """
        if self._proc.poll() is not None:
            raise NemotronWSLBridgeError(
                f"Nemotron worker has exited (returncode={self._proc.returncode}). "
                "Cannot process request."
            )

        # Convert Windows path to WSL path if needed
        wsl_path = _win_to_wsl(image_path)
        req_id = str(uuid.uuid4())
        request = {
            "id": req_id,
            "image_path": wsl_path,
            "merge_level": merge_level or self._merge_level,
        }

        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()

        # Read response lines until we see our req_id
        assert self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                rc = self._proc.poll()
                raise NemotronWSLBridgeError(
                    f"Nemotron worker stdout closed unexpectedly (rc={rc}). "
                    f"Stderr tail: {self.worker_stderr[-5:]}"
                )
            try:
                resp = json.loads(line.strip())
            except json.JSONDecodeError:
                continue  # non-JSON line (shouldn't happen, but skip safely)
            if resp.get("id") != req_id:
                continue  # not our response
            if not resp.get("ok"):
                raise NemotronWSLBridgeError(
                    f"Nemotron worker error for {wsl_path}: {resp.get('error', '?')}"
                )
            return resp.get("predictions") or []

    def shutdown(self) -> None:
        """Send a shutdown request and wait for the worker to exit."""
        if self._proc.poll() is not None:
            return  # already gone
        try:
            assert self._proc.stdin is not None
            req = json.dumps({"id": "shutdown", "image_path": ""})
            self._proc.stdin.write(req + "\n")
            self._proc.stdin.flush()
            self._proc.stdin.close()
            self._proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self._proc.terminate()

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()

    # ── Callable (matches NemotronOCRV2 call signature for adapters) ──────────

    def __call__(
        self,
        image_path: str | Path,
        merge_level: str | None = None,
    ) -> list[dict[str, Any]]:
        """Make the bridge callable like NemotronOCRV2 for use with NemotronCandidateAdapter."""
        return self.ocr(image_path, merge_level=merge_level)
