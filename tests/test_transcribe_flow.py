from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from qwen_asr_cli.transcribe import TranscribeOptions, _normalize_language, load_model, run_transcription


class _FakeModel:
    def __init__(self) -> None:
        self.calls = []

    def transcribe(self, audio: str, context: str, language: str | None):
        self.calls.append({"audio": audio, "context": context, "language": language})
        return [SimpleNamespace(text="hello world", language="en", time_stamps=None)]


def test_run_transcription_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"fake")

    preprocessed = tmp_path / "preprocessed.wav"
    preprocessed.write_bytes(b"fake-processed")

    fake_model = _FakeModel()
    cleaned = {"called": False}

    monkeypatch.setattr("qwen_asr_cli.transcribe.preprocess_to_16k_mono", lambda _: preprocessed)
    monkeypatch.setattr("qwen_asr_cli.transcribe.load_model", lambda **_: fake_model)

    def _cleanup(path: Path, keep: bool) -> None:
        cleaned["called"] = True
        assert path == preprocessed
        assert keep is False

    monkeypatch.setattr("qwen_asr_cli.transcribe.cleanup_temp_file", _cleanup)

    result = run_transcription(
        TranscribeOptions(
            audio_path=input_audio,
            model_ref="Qwen/Qwen3-ASR-0.6B",
            device="cpu",
            dtype=None,
            language="auto",
            prompt="",
            max_new_tokens=4096,
            keep_preprocessed=False,
        )
    )

    assert result.text == "hello world"
    assert result.language == "en"
    assert fake_model.calls[0]["language"] is None
    assert cleaned["called"] is True


def test_language_mapping() -> None:
    assert _normalize_language("auto") is None
    assert _normalize_language("zh") == "Chinese"
    assert _normalize_language("en") == "English"


def test_load_model_uses_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class _InnerModel:
        def to(self, _):
            return self

    class _Wrapper:
        def __init__(self) -> None:
            self.model = _InnerModel()

    def _fake_from_pretrained(model_ref: str, max_new_tokens: int, **kwargs):
        captured["model_ref"] = model_ref
        captured["max_new_tokens"] = max_new_tokens
        captured["kwargs"] = kwargs
        return _Wrapper()

    monkeypatch.setattr("qwen_asr_cli.transcribe.Qwen3ASRModel.from_pretrained", _fake_from_pretrained)

    _ = load_model(
        model_ref="Qwen/Qwen3-ASR-0.6B",
        device="cpu",
        dtype="fake-dtype",
        max_new_tokens=4096,
    )

    assert captured["model_ref"] == "Qwen/Qwen3-ASR-0.6B"
    assert captured["max_new_tokens"] == 4096
    assert "dtype" in captured["kwargs"]
    assert "torch_dtype" not in captured["kwargs"]


def test_run_transcription_suppresses_backend_noise(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys,
) -> None:
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"fake")

    preprocessed = tmp_path / "preprocessed.wav"
    preprocessed.write_bytes(b"fake-processed")

    class _NoisyModel:
        def transcribe(self, audio: str, context: str, language: str | None):
            print("backend-stdout-noise")
            print("backend-stderr-noise", file=sys.stderr)
            return [SimpleNamespace(text="clean result", language="Chinese", time_stamps=None)]

    monkeypatch.setattr("qwen_asr_cli.transcribe.preprocess_to_16k_mono", lambda _: preprocessed)
    monkeypatch.setattr("qwen_asr_cli.transcribe.load_model", lambda **_: _NoisyModel())
    monkeypatch.setattr("qwen_asr_cli.transcribe.cleanup_temp_file", lambda *_, **__: None)

    result = run_transcription(
        TranscribeOptions(
            audio_path=input_audio,
            model_ref="Qwen/Qwen3-ASR-0.6B",
            device="cpu",
            dtype=None,
            language="auto",
            prompt="",
            max_new_tokens=4096,
            keep_preprocessed=False,
            suppress_backend_output=True,
        )
    )

    captured = capsys.readouterr()
    assert result.text == "clean result"
    assert "backend-stdout-noise" not in captured.out
    assert "backend-stderr-noise" not in captured.err
