from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

import torch


class CliError(Exception):
    """User-facing exception with an exit code and optional hint."""

    def __init__(self, code: int, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


@dataclass(frozen=True)
class DoctorReport:
    python_version: str
    torch_version: str
    cuda_available: bool
    mps_available: bool
    ffmpeg_available: bool


def ensure_audio_file(audio_path: str) -> Path:
    path = Path(audio_path).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise CliError(5, f"Audio file not found: {audio_path}", "Check the input path and try again.")
    return path


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None:
        raise CliError(
            3,
            "ffmpeg is required for audio preprocessing but was not found in PATH.",
            "Install ffmpeg and ensure it is available in PATH.",
        )


def _mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    is_available = getattr(backend, "is_available", None)
    if callable(is_available):
        return bool(is_available())
    return False


def resolve_device(device: str) -> str:
    requested = device.lower().strip()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise CliError(2, f"Unsupported --device value: {device}", "Use one of: auto, cuda, mps, cpu.")

    cuda_available = bool(torch.cuda.is_available())
    mps_available = _mps_available()

    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"

    if requested == "cuda" and not cuda_available:
        raise CliError(3, "CUDA was requested but no CUDA device is available.", "Use --device auto or --device cpu.")

    if requested == "mps" and not mps_available:
        raise CliError(3, "MPS was requested but Apple Metal backend is unavailable.", "Use --device auto or --device cpu.")

    return requested


def resolve_torch_dtype(dtype: str, device: str) -> Any | None:
    requested = dtype.lower().strip()
    if requested not in {"auto", "float16", "bfloat16", "float32"}:
        raise CliError(2, f"Unsupported --dtype value: {dtype}", "Use one of: auto, float16, bfloat16, float32.")

    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float32":
        return torch.float32

    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def doctor_report() -> DoctorReport:
    import platform

    return DoctorReport(
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=bool(torch.cuda.is_available()),
        mps_available=_mps_available(),
        ffmpeg_available=shutil.which("ffmpeg") is not None,
    )
