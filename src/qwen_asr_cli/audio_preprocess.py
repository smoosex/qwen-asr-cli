from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

from qwen_asr_cli.runtime import CliError, ensure_ffmpeg_available


def preprocess_to_16k_mono(audio_path: Path) -> Path:
    ensure_ffmpeg_available()

    with tempfile.NamedTemporaryFile(prefix="qwen-asr-", suffix=".wav", delete=False) as handle:
        output_path = Path(handle.name)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        error_tail = result.stderr.strip().splitlines()[-1] if result.stderr else "unknown ffmpeg error"
        raise CliError(5, f"Failed to preprocess audio: {error_tail}", "Check audio format or ffmpeg availability.")

    return output_path


def cleanup_temp_file(path: Path, keep: bool) -> None:
    if keep:
        return
    path.unlink(missing_ok=True)
