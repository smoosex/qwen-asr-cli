from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
import io
import logging
import os
from pathlib import Path
from typing import Any

from qwen_asr import Qwen3ASRModel

from qwen_asr_cli.audio_preprocess import cleanup_temp_file, preprocess_to_16k_mono
from qwen_asr_cli.runtime import CliError


@dataclass(frozen=True)
class TranscribeOptions:
    audio_path: Path
    model_ref: str
    device: str
    dtype: Any
    language: str
    prompt: str
    max_new_tokens: int
    keep_preprocessed: bool
    suppress_backend_output: bool = True


@dataclass(frozen=True)
class TranscribeResult:
    text: str
    language: str


@contextmanager
def _suppress_backend_output(enabled: bool):
    if not enabled:
        yield
        return

    env_keys = ("TRANSFORMERS_VERBOSITY", "HF_HUB_DISABLE_PROGRESS_BARS", "TOKENIZERS_PARALLELISM")
    original_env = {key: os.environ.get(key) for key in env_keys}
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    logger_names = ("transformers", "huggingface_hub", "tokenizers")
    original_levels = {name: logging.getLogger(name).level for name in logger_names}
    for name in logger_names:
        logging.getLogger(name).setLevel(logging.ERROR)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            yield
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for name, level in original_levels.items():
            logging.getLogger(name).setLevel(level)


def _normalize_language(language: str) -> str | None:
    normalized = language.strip().lower()
    if normalized in {"", "auto"}:
        return None
    if normalized == "zh":
        return "Chinese"
    if normalized == "en":
        return "English"
    return language


def load_model(model_ref: str, device: str, dtype: Any, max_new_tokens: int) -> Qwen3ASRModel:
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype

    try:
        model = Qwen3ASRModel.from_pretrained(model_ref, max_new_tokens=max_new_tokens, **kwargs)
        # Transformers backend may initialize on CPU; move to target device when needed.
        if device in {"cuda", "mps"}:
            target_device = "cuda:0" if device == "cuda" else "mps"
            model.model = model.model.to(target_device)
        return model
    except Exception as exc:
        raise CliError(4, f"Failed to load model `{model_ref}`: {exc}") from exc


def run_transcription(options: TranscribeOptions) -> TranscribeResult:
    preprocessed = preprocess_to_16k_mono(options.audio_path)
    language = _normalize_language(options.language)

    try:
        with _suppress_backend_output(options.suppress_backend_output):
            model = load_model(
                model_ref=options.model_ref,
                device=options.device,
                dtype=options.dtype,
                max_new_tokens=options.max_new_tokens,
            )

            outputs = model.transcribe(
                audio=str(preprocessed),
                context=options.prompt,
                language=language,
            )
        if not outputs:
            raise CliError(6, "Model returned no transcription output.")

        first = outputs[0]
        return TranscribeResult(text=(first.text or "").strip(), language=first.language)
    except CliError:
        raise
    except Exception as exc:
        raise CliError(6, f"Transcription failed: {exc}") from exc
    finally:
        cleanup_temp_file(preprocessed, keep=options.keep_preprocessed)
