from pathlib import Path

import pytest

from qwen_asr_cli.model_store import MODEL_DIR_ENV, resolve_model_ref
from qwen_asr_cli.runtime import CliError


def test_resolve_model_ref_prefers_explicit() -> None:
    assert resolve_model_ref("/tmp/custom-model") == "/tmp/custom-model"


def test_resolve_model_ref_uses_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "env-model"
    model_dir.mkdir()
    monkeypatch.setenv(MODEL_DIR_ENV, str(model_dir))

    assert resolve_model_ref(None) == str(model_dir.resolve())


def test_resolve_model_ref_errors_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(MODEL_DIR_ENV, raising=False)
    monkeypatch.setattr("qwen_asr_cli.model_store.default_model_dir", lambda: tmp_path / "missing")

    with pytest.raises(CliError) as exc_info:
        resolve_model_ref(None)

    assert exc_info.value.code == 4
