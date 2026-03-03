# qwen-asr CLI

A lightweight CLI wrapper around `qwen-asr` for single-file speech transcription.

## Features

- Local transcription with Qwen3-ASR (transformers backend)
- Default model install: `Qwen/Qwen3-ASR-0.6B`
- Default source: Hugging Face (`--source huggingface`)
- Audio preprocessing with `ffmpeg` to `16kHz mono`
- Device auto-selection: `cuda > mps > cpu`
- Default `max_new_tokens`: `4096`
- Default transcribe output: final text only (backend logs are suppressed)

## Prerequisites

- Python 3.10+
- `uv`
- `ffmpeg` in `PATH`

## Setup

```bash
uv sync --dev
```

## Quick Start

Install default model:

```bash
uv run qwen-asr install-model
```

Install the larger official model:

```bash
uv run qwen-asr install-model --model Qwen/Qwen3-ASR-1.7B
```

Transcribe an audio file:

```bash
uv run qwen-asr transcribe ./demo.wav
```

Force Chinese/English language:

```bash
uv run qwen-asr transcribe ./demo.wav --language zh
uv run qwen-asr transcribe ./demo.wav --language en
```

Save output text:

```bash
uv run qwen-asr transcribe ./demo.wav --output ./demo.txt
```

Use a custom model path:

```bash
uv run qwen-asr transcribe ./demo.wav --model /path/to/model
```

Check runtime:

```bash
uv run qwen-asr doctor
```

## Model Resolution Priority

If `--model` is omitted in `transcribe`, the CLI resolves model in this order:

1. `QWEN_ASR_MODEL_DIR`
2. Default model path (`platformdirs` cache + `Qwen3-ASR-0.6B`)
3. Error with hint to run `qwen-asr install-model`

## Official Installable Models

- `Qwen/Qwen3-ASR-0.6B` (default)
- `Qwen/Qwen3-ASR-1.7B`

## Commands

```bash
uv run qwen-asr install-model --help
uv run qwen-asr transcribe --help
uv run qwen-asr doctor --help
```

`--language` mapping:

- `auto` -> auto detect
- `zh` -> `Chinese` (qwen-asr language name)
- `en` -> `English` (qwen-asr language name)

## Test

```bash
uv run pytest
```
