#!/usr/bin/env python3
"""
STORY-AI — Project Setup and Environment Verification
======================================================

ONE COMMAND → reproducible project setup + environment verification.

Usage:
    python scripts/setup_project.py                 # check + create runtime dirs
    python scripts/setup_project.py --check-only    # report only, no side-effects
    python scripts/setup_project.py --verbose       # extended diagnostics

This script NEVER:
    - downloads large model weights automatically
    - modifies the dataset
    - silently overwrites configuration
    - crash on missing optional models (it reports them clearly)

Exit codes:
    0   All required checks passed
    1   One or more required checks failed (see FAIL entries in report)
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

REQUIRED_DIRS = [
    "app",
    "app/models",
    "app/schemas",
    "app/data",
    "config",
    "dataset",
    "dataset/development",
    "dataset/test",
    "tests",
    "tools",
    "vendor",
    "vendor/comic-text-detector",
    "scripts",
]

RUNTIME_DIRS_TO_CREATE = [
    ".smoke",
    ".smoke/pipeline-integration",
    "outputs",
]

REQUIRED_SOURCE_FILES = [
    "pyproject.toml",
    "uv.lock",
    "config/project.yaml",
    "app/__init__.py",
    "app/balloon_adjudication.py",
    "app/candidate_bank.py",
    "app/character_identity.py",
    "app/character_perception.py",
    "app/layout_grouping.py",
    "app/page_perception.py",
    "app/reading_order.py",
    "app/sequence_resolver.py",
    "app/sequence_validator.py",
    "app/speaker_grounding.py",
    "app/submission_serializer.py",
    "app/data/config.py",
    "app/data/loader.py",
    "app/data/paths.py",
    "app/data/split.py",
    "app/models/ctd.py",
    "app/models/local_runners.py",
    "app/models/manga_layout.py",
    "app/models/nemotron_wsl_bridge.py",
    "app/models/rtdetrv4_character_provider.py",
    "app/schemas/__init__.py",
    "app/schemas/candidates.py",
    "app/schemas/page.py",
    "app/schemas/sequence.py",
    "tools/pipeline_integration_smoke.py",
    "tools/nemotron_worker.py",
    "vendor/comic-text-detector",
]

CTD_MODEL = REPO_ROOT / "vendor" / "comic-text-detector" / "data" / "comictextdetector.pt.onnx"

HOME = Path.home()
PADDLE_MODEL_DIR = Path(
    os.environ.get(
        "STORYAI_PADDLE_MODEL_DIR",
        str(HOME / ".paddlex" / "official_models" / "en_PP-OCRv5_mobile_rec_onnx"),
    )
)

HF_CACHE = Path(
    os.environ.get(
        "STORYAI_QWEN_CACHE_DIR",
        os.environ.get("HF_HOME", str(HOME / ".cache" / "huggingface" / "hub")),
    )
)

LAYOUT_WEIGHTS = Path(
    os.environ.get(
        "STORYAI_LAYOUT_WEIGHTS",
        str(
            HF_CACHE
            / "models--huyvux3005--manga109-segmentation-bubble"
            / "snapshots"
            / "f9a4108c4955136a810e5e92207972f3fb3a65fd"
            / "best.pt"
        ),
    )
)

RTDETR_MODEL = Path(
    os.environ.get(
        "STORYAI_RTDETR_MODEL",
        str(
            HF_CACHE
            / "models--tori29umai--rtdetrv4-x-manga109s_v2"
            / "snapshots"
            / "864c3bfb837a03ecc62557d5152a5ade5566489b"
            / "model.onnx"
        ),
    )
)

QWEN_CACHE_PATTERN = HF_CACHE / "models--Qwen--Qwen3-VL-4B-Instruct"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Report:
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    INFO = "INFO"

    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose
        self._entries: list[tuple[str, str, str]] = []  # (status, category, message)
        self.fail_count = 0

    def _record(self, status: str, category: str, message: str) -> None:
        self._entries.append((status, category, message))
        if status == self.FAIL:
            self.fail_count += 1

    def ok(self, category: str, message: str) -> None:
        self._record(self.PASS, category, message)

    def fail(self, category: str, message: str) -> None:
        self._record(self.FAIL, category, message)

    def warn(self, category: str, message: str) -> None:
        self._record(self.WARN, category, message)

    def info(self, category: str, message: str) -> None:
        if self.verbose:
            self._record(self.INFO, category, message)

    def print_summary(self) -> None:
        width = 72
        print("=" * width)
        print("  STORY-AI  PROJECT SETUP REPORT")
        print("=" * width)
        for status, category, message in self._entries:
            # Keep output compatible with the default CP1252 Windows console.
            icon = {"PASS": "OK", "FAIL": "!!", "WARN": "!", "INFO": "-"}.get(status, "?")
            safe_message = message.encode("ascii", "replace").decode("ascii")
            print(f"  {icon}  [{status:<4}] {category:<28} {safe_message}")
        print("-" * width)
        if self.fail_count == 0:
            print("  OVERALL: PASS — environment is ready.")
        else:
            print(f"  OVERALL: FAIL — {self.fail_count} required check(s) failed.")
            print()
            print("  Remediation:")
            for status, category, message in self._entries:
                if status == self.FAIL:
                    safe_message = message.encode("ascii", "replace").decode("ascii")
                    print(f"    - [{category}] {safe_message}")
        print("=" * width)


def _check_python_version(report: _Report) -> None:
    required_major, required_minor = 3, 11
    major, minor = sys.version_info.major, sys.version_info.minor
    version_str = f"{major}.{minor}.{sys.version_info.micro}"
    if major == required_major and minor == required_minor:
        report.ok("Python", f"{version_str} (required 3.11)")
    elif major == required_major and minor > required_minor:
        report.warn("Python", f"{version_str} — pyproject.toml requires <3.12; test carefully")
    else:
        report.fail("Python", f"{version_str} — required 3.11.x; use: uv python install 3.11")


def _check_venv(report: _Report) -> None:
    venv = REPO_ROOT / ".venv"
    if venv.is_dir():
        report.ok("Virtual env", ".venv present")
        python_bin = venv / "Scripts" / "python.exe"
        if not python_bin.exists():
            python_bin = venv / "bin" / "python"
        if python_bin.exists():
            report.ok("Venv python", str(python_bin.relative_to(REPO_ROOT)))
        else:
            report.warn("Venv python", "python executable not found in .venv — run: uv sync")
    else:
        report.fail("Virtual env", ".venv missing — run: uv sync")


def _check_uv(report: _Report) -> None:
    try:
        result = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            report.ok("uv", result.stdout.strip())
        else:
            report.fail("uv", "uv not working — install from https://docs.astral.sh/uv/")
    except FileNotFoundError:
        report.fail("uv", "uv not found — install from https://docs.astral.sh/uv/")
    except Exception as exc:
        report.warn("uv", f"could not check: {exc}")


def _check_required_dirs(report: _Report) -> None:
    for d in REQUIRED_DIRS:
        path = REPO_ROOT / d
        if path.is_dir():
            report.info("Dir", f"{d} ✓")
        else:
            report.fail("Dir", f"missing: {d}")


def _check_required_source_files(report: _Report) -> None:
    for f in REQUIRED_SOURCE_FILES:
        path = REPO_ROOT / f
        if path.exists():
            report.info("File", f"{f} ✓")
        else:
            report.fail("Source file", f"missing: {f}")


def _check_ctd_model(report: _Report) -> None:
    if CTD_MODEL.is_file():
        size_mb = CTD_MODEL.stat().st_size / 1024 / 1024
        report.ok("CTD model", f"{CTD_MODEL.relative_to(REPO_ROOT)} ({size_mb:.1f} MB)")
    else:
        report.fail(
            "CTD model",
            f"missing: {CTD_MODEL.relative_to(REPO_ROOT)}\n"
            "    → Copy comictextdetector.pt.onnx into vendor/comic-text-detector/data/",
        )


def _check_paddle_model(report: _Report) -> None:
    if PADDLE_MODEL_DIR.is_dir():
        report.ok("PaddleOCR", f"{PADDLE_MODEL_DIR}")
    else:
        report.fail(
            "PaddleOCR",
            f"model dir missing: {PADDLE_MODEL_DIR}\n"
            "    → Override: STORYAI_PADDLE_MODEL_DIR\n"
            "    → Or run PaddleOCR once to trigger auto-download (requires internet)",
        )


def _check_qwen_cache(report: _Report) -> None:
    snapshots = QWEN_CACHE_PATTERN / "snapshots"
    if QWEN_CACHE_PATTERN.is_dir() and snapshots.is_dir():
        report.ok("Qwen3-VL cache", str(QWEN_CACHE_PATTERN))
    else:
        report.warn(
            "Qwen3-VL cache",
            f"not found at {QWEN_CACHE_PATTERN}\n"
            "    → Override: STORYAI_QWEN_CACHE_DIR\n"
            "    → Or let transformers auto-download on first run (large — ~4 GB)",
        )


def _check_layout_weights(report: _Report) -> None:
    if LAYOUT_WEIGHTS.is_file():
        report.ok("Layout YOLO", str(LAYOUT_WEIGHTS))
    else:
        report.warn(
            "Layout YOLO",
            f"not found at {LAYOUT_WEIGHTS}\n"
            "    → Override: STORYAI_LAYOUT_WEIGHTS\n"
            "    → Or let huggingface_hub auto-download on first run",
        )


def _check_rtdetr_model(report: _Report) -> None:
    if RTDETR_MODEL.is_file():
        report.ok("RT-DETRv4", str(RTDETR_MODEL))
    else:
        report.warn(
            "RT-DETRv4",
            f"not found at {RTDETR_MODEL}\n"
            "    → Override: STORYAI_RTDETR_MODEL\n"
            "    → Or let huggingface_hub auto-download on first run",
        )


def _check_cuda(report: _Report) -> None:
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0) if count > 0 else "unknown"
            report.ok("CUDA", f"available — {count} device(s): {name}")
        else:
            report.warn(
                "CUDA",
                "not available — Qwen will run on CPU (very slow)\n"
                "    → CUDA is required for practical Qwen3-VL inference",
            )
    except ImportError:
        report.fail(
            "CUDA / torch",
            "torch not importable — run: uv sync\n"
            "    → Ensure the virtual environment is active",
        )


def _check_wsl(report: _Report) -> None:
    if platform.system() != "Windows":
        report.info("WSL", "not on Windows — WSL bridge check skipped")
        return
    try:
        result = subprocess.run(
            ["wsl", "--status"], capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            report.ok("WSL", "wsl --status returned 0")
        else:
            report.warn(
                "WSL",
                "wsl --status returned non-zero — Nemotron bridge may not work\n"
                "    → Enable WSL2 in Windows Features",
            )
    except FileNotFoundError:
        report.warn(
            "WSL",
            "wsl command not found — Nemotron bridge unavailable\n"
            "    → Nemotron is optional; pipeline can run without it",
        )
    except Exception as exc:
        report.warn("WSL", f"could not check: {exc}")


def _create_runtime_dirs(report: _Report, check_only: bool) -> None:
    for d in RUNTIME_DIRS_TO_CREATE:
        path = REPO_ROOT / d
        if path.is_dir():
            report.info("Runtime dir", f"{d} already exists")
        else:
            if check_only:
                report.info("Runtime dir", f"would create: {d}")
            else:
                path.mkdir(parents=True, exist_ok=True)
                report.ok("Runtime dir", f"created: {d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Report only — do not create directories or modify anything",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print INFO-level entries (file-by-file checks)",
    )
    args = parser.parse_args()

    report = _Report(verbose=args.verbose)

    print(f"\n  Repository root: {REPO_ROOT}")
    print(f"  Python:          {sys.executable}")
    print(f"  Platform:        {platform.system()} {platform.release()}")
    print()

    # Required checks
    _check_python_version(report)
    _check_uv(report)
    _check_venv(report)
    _check_required_dirs(report)
    _check_required_source_files(report)
    _check_ctd_model(report)

    # Model checks (warnings, not hard failures — models can be missing until first run)
    _check_paddle_model(report)
    _check_qwen_cache(report)
    _check_layout_weights(report)
    _check_rtdetr_model(report)
    _check_cuda(report)
    _check_wsl(report)

    # Create runtime dirs
    _create_runtime_dirs(report, check_only=args.check_only)

    report.print_summary()
    return 1 if report.fail_count > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
