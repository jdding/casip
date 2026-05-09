#!/usr/bin/env python3
"""Compile Python source text without writing __pycache__ files."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    paths = []
    for directory in ["scripts", "src", "figures", "tools"]:
        paths.extend((ROOT / directory).rglob("*.py"))
    for path in sorted(paths):
        source = path.read_text(encoding="utf-8")
        compile(source, str(path.relative_to(ROOT)), "exec")
    print(f"OK: syntax check passed for {len(paths)} Python files.")


if __name__ == "__main__":
    main()
