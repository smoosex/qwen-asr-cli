from pathlib import Path

from qwen_asr_cli.commands import build_parser, command_install_model
from qwen_asr_cli.model_store import DEFAULT_MODEL_ID, DEFAULT_SOURCE


def test_transcribe_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["transcribe", "demo.wav"])

    assert args.audio_path == "demo.wav"
    assert args.model is None
    assert args.device == "auto"
    assert args.dtype == "auto"
    assert args.language == "auto"
    assert args.max_new_tokens == 4096
    assert args.keep_preprocessed is False


def test_install_model_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["install-model"])

    assert args.model == DEFAULT_MODEL_ID
    assert args.source == DEFAULT_SOURCE
    assert args.target_dir is None
    assert args.force is False


def test_install_model_prints_available_models(mocker, capsys) -> None:
    parser = build_parser()
    args = parser.parse_args(["install-model"])
    mocker.patch("qwen_asr_cli.commands.install_model", return_value=Path("/tmp/model"))

    code = command_install_model(args)
    output = capsys.readouterr().out

    assert code == 0
    assert "Available official models:" in output
    assert DEFAULT_MODEL_ID in output
