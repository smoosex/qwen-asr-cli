from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from qwen_asr_cli.model_store import (
    DEFAULT_MODEL_ID,
    DEFAULT_SOURCE,
    install_model,
    list_supported_models,
    resolve_model_ref,
)
from qwen_asr_cli.runtime import CliError, doctor_report, ensure_audio_file, resolve_device, resolve_torch_dtype
from qwen_asr_cli.transcribe import TranscribeOptions, run_transcription


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-asr", description="CLI tool for Qwen3-ASR transcription")
    subparsers = parser.add_subparsers(dest="command", required=True)
    available_models_text = ", ".join(list_supported_models())

    install = subparsers.add_parser("install-model", help="Install model into local cache")
    install.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Model ID to download. Official options: {available_models_text}",
    )
    install.add_argument("--target-dir", default=None, help="Custom target directory for model files")
    install.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        choices=["huggingface", "modelscope"],
        help="Model source provider",
    )
    install.add_argument("--force", action="store_true", help="Re-download even if target exists")

    transcribe = subparsers.add_parser("transcribe", help="Transcribe a single audio file")
    transcribe.add_argument("audio_path", help="Input audio file path")
    transcribe.add_argument("--output", default=None, help="Optional path to save output text")
    transcribe.add_argument("--model", default=None, help="Model ID or local model path")
    transcribe.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    transcribe.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    transcribe.add_argument("--language", default="auto", choices=["auto", "zh", "en"])
    transcribe.add_argument("--prompt", default="", help="Prompt/context text")
    transcribe.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum generated tokens (default: 4096)")
    transcribe.add_argument("--keep-preprocessed", action="store_true")
    transcribe.add_argument("--verbose", action="store_true")

    subparsers.add_parser("doctor", help="Check runtime dependencies and device availability")
    return parser


def _print_verbose(message: str, enabled: bool) -> None:
    if enabled:
        print(message, file=sys.stderr)


def command_install_model(args: argparse.Namespace) -> int:
    supported_models = list_supported_models()
    print("Available official models:")
    for model_id in supported_models:
        print(f"- {model_id}")
    print(f"Default model: {DEFAULT_MODEL_ID}")
    if args.model not in supported_models:
        print(f"Note: `{args.model}` is not in the official list, continuing anyway.")

    start = time.perf_counter()
    path = install_model(model_id=args.model, target_dir=args.target_dir, source=args.source, force=args.force)
    duration = time.perf_counter() - start
    print(f"Installed model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Path: {path}")
    print(f"Elapsed: {duration:.2f}s")
    return 0


def command_transcribe(args: argparse.Namespace) -> int:
    audio_path = ensure_audio_file(args.audio_path)
    model_ref = resolve_model_ref(args.model)
    device = resolve_device(args.device)
    dtype_value = resolve_torch_dtype(args.dtype, device)

    _print_verbose(f"Audio: {audio_path}", args.verbose)
    _print_verbose(f"Model: {model_ref}", args.verbose)
    _print_verbose(f"Device: {device}", args.verbose)
    _print_verbose(f"DType: {dtype_value}", args.verbose)

    start = time.perf_counter()
    result = run_transcription(
        TranscribeOptions(
            audio_path=audio_path,
            model_ref=model_ref,
            device=device,
            dtype=dtype_value,
            language=args.language,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            keep_preprocessed=args.keep_preprocessed,
            suppress_backend_output=not args.verbose,
        )
    )
    elapsed = time.perf_counter() - start

    print(result.text)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.text + "\n", encoding="utf-8")
        _print_verbose(f"Saved: {output_path}", args.verbose)

    _print_verbose(f"Detected language: {result.language}", args.verbose)
    _print_verbose(f"Elapsed: {elapsed:.2f}s", args.verbose)
    return 0


def command_doctor() -> int:
    report = doctor_report()
    print(f"python_version={report.python_version}")
    print(f"torch_version={report.torch_version}")
    print(f"cuda_available={str(report.cuda_available).lower()}")
    print(f"mps_available={str(report.mps_available).lower()}")
    print(f"ffmpeg_available={str(report.ffmpeg_available).lower()}")
    return 0


def dispatch(args: argparse.Namespace) -> int:
    if args.command == "install-model":
        return command_install_model(args)
    if args.command == "transcribe":
        return command_transcribe(args)
    if args.command == "doctor":
        return command_doctor()
    raise CliError(2, f"Unknown command: {args.command}")
