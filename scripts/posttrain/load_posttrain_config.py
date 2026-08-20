"""Load a flat post-training YAML mapping as validated shell variables."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import yaml

_VARIABLE = re.compile(r"[A-Z_][A-Z0-9_]*\Z")


def _scalar(value: object, name: str) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str | int | float):
        return str(value)
    raise TypeError(f"Config value {name} must be a string, number, boolean, or null.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    args = parser.parse_args()

    path = Path(args.config).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Post-training config must be a YAML mapping.")
    for raw_name, value in payload.items():
        name = str(raw_name)
        if not _VARIABLE.fullmatch(name):
            raise ValueError(
                f"Invalid config key {name!r}; use an uppercase environment-variable name."
            )
        if value is None:
            continue
        sys.stdout.buffer.write(name.encode() + b"\0" + _scalar(value, name).encode() + b"\0")


if __name__ == "__main__":
    main()
