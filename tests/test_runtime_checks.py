from pathlib import Path

import pytest

from qwen_asr_cli.runtime import CliError, ensure_audio_file, resolve_device, resolve_torch_dtype


def test_resolve_device_auto_prefers_cpu_when_accelerators_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)

    assert resolve_device("auto") == "cpu"


def test_resolve_device_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    assert resolve_device("auto") == "cuda"


def test_ensure_audio_file_raises_for_missing(tmp_path: Path) -> None:
    with pytest.raises(CliError) as exc_info:
        ensure_audio_file(str(tmp_path / "missing.wav"))
    assert exc_info.value.code == 5


def test_resolve_torch_dtype_auto_cpu() -> None:
    dtype = resolve_torch_dtype("auto", "cpu")
    assert str(dtype).endswith("float32")
