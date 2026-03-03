#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re


def normalize_version(version: str) -> str:
    return version[1:] if version.startswith("v") else version


def validate_sha256(value: str) -> None:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError("sha256 must be a 64-character hex string")


def build_formula(version: str, sha256: str) -> str:
    return f'''class QwenAsrCli < Formula
  desc "CLI wrapper for Qwen3-ASR single-file transcription"
  homepage "https://github.com/smoosex/qwen-asr-cli"
  url "https://files.pythonhosted.org/packages/source/q/qwen-asr-cli/qwen_asr_cli-{{version}}.tar.gz"
  sha256 "{{sha256}}"
  license "MIT"

  depends_on "python@3.11"
  depends_on "ffmpeg"

  def install
    venv = libexec/"venv"
    system Formula["python@3.11"].opt_bin/"python3", "-m", "venv", venv
    system venv/"bin/pip", "install", "--upgrade", "pip", "setuptools", "wheel"
    system venv/"bin/pip", "install", buildpath
    bin.install_symlink venv/"bin/qwen-asr"
  end

  test do
    assert_match "CLI tool for Qwen3-ASR transcription", shell_output("#{{{{bin}}}}/qwen-asr --help")
  end
end
'''.format(version=version, sha256=sha256)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Homebrew formula for qwen-asr-cli")
    parser.add_argument("--version", required=True, help="Version string, e.g. v0.1.1 or 0.1.1")
    parser.add_argument("--sha256", required=True, help="SHA256 of sdist tarball")
    parser.add_argument("--output", default="Formula/qwen-asr-cli.rb", help="Output path")
    args = parser.parse_args()

    version = normalize_version(args.version)
    validate_sha256(args.sha256)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_formula(version=version, sha256=args.sha256.lower()), encoding="utf-8")


if __name__ == "__main__":
    main()
