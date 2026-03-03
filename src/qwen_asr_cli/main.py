from __future__ import annotations

import sys

from qwen_asr_cli.commands import build_parser, dispatch
from qwen_asr_cli.runtime import CliError


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        code = dispatch(args)
    except CliError as err:
        print(f"Error: {err.message}", file=sys.stderr)
        if err.hint:
            print(f"Hint: {err.hint}", file=sys.stderr)
        raise SystemExit(err.code)

    raise SystemExit(code)


if __name__ == "__main__":
    main()
