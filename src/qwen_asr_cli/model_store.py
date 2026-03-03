from __future__ import annotations

import os
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download
from platformdirs import user_cache_dir

from qwen_asr_cli.runtime import CliError


DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
DEFAULT_SOURCE = "huggingface"
MODEL_DIR_ENV = "QWEN_ASR_MODEL_DIR"
SUPPORTED_MODEL_IDS = (
    "Qwen/Qwen3-ASR-0.6B",
    "Qwen/Qwen3-ASR-1.7B",
)


def default_models_root() -> Path:
    return Path(user_cache_dir("qwen-asr-cli")).expanduser() / "models"


def _model_leaf_name(model_id: str) -> str:
    leaf = model_id.split("/")[-1].strip()
    return leaf or DEFAULT_MODEL_ID


def default_model_dir(model_id: str = DEFAULT_MODEL_ID) -> Path:
    return default_models_root() / _model_leaf_name(model_id)


def list_supported_models() -> tuple[str, ...]:
    return SUPPORTED_MODEL_IDS


def resolve_model_ref(model: str | None) -> str:
    if model:
        return model

    env_model_dir = os.getenv(MODEL_DIR_ENV)
    if env_model_dir:
        env_path = Path(env_model_dir).expanduser().resolve()
        if env_path.exists():
            return str(env_path)
        raise CliError(4, f"{MODEL_DIR_ENV} points to a missing path: {env_path}")

    default_path = default_model_dir()
    if default_path.exists():
        return str(default_path)

    raise CliError(
        4,
        f"Default model not found at: {default_path}",
        "Run `qwen-asr install-model` or pass --model <local-path-or-model-id>.",
    )


def _download_from_huggingface(model_id: str, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=str(destination))
    return destination


def _download_from_modelscope(model_id: str, destination: Path) -> Path:
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    except ImportError as exc:
        raise CliError(
            4,
            "ModelScope source was requested but modelscope is not installed.",
            "Install dependencies again with `uv sync`.",
        ) from exc

    downloaded = Path(ms_snapshot_download(model_id=model_id, cache_dir=str(destination.parent))).resolve()
    if downloaded == destination.resolve():
        return destination

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(downloaded, destination)
    return destination


def install_model(
    model_id: str = DEFAULT_MODEL_ID,
    target_dir: str | None = None,
    source: str = DEFAULT_SOURCE,
    force: bool = False,
) -> Path:
    source_norm = source.lower().strip()
    if source_norm not in {"huggingface", "modelscope"}:
        raise CliError(2, f"Unsupported --source value: {source}", "Use one of: huggingface, modelscope.")

    base_dir = Path(target_dir).expanduser().resolve() if target_dir else default_models_root().resolve()
    destination = base_dir / _model_leaf_name(model_id)

    if destination.exists() and not force:
        return destination

    if destination.exists() and force:
        shutil.rmtree(destination)

    try:
        if source_norm == "huggingface":
            return _download_from_huggingface(model_id=model_id, destination=destination)
        return _download_from_modelscope(model_id=model_id, destination=destination)
    except CliError:
        raise
    except Exception as exc:  # pragma: no cover - exercised by integration/manual runs
        raise CliError(4, f"Failed to install model `{model_id}` from `{source_norm}`: {exc}") from exc
