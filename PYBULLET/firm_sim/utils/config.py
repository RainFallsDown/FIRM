"""Small config loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_yaml_like(path: str | Path) -> Dict[str, Any]:
    """Load a tiny YAML-like config without external dependencies.

    This helper supports the simple `key: value` and one-level nested maps used
    by the scaffold configs. It is intentionally minimal so the skeleton stays
    lightweight before dependency choices are finalized.
    """

    text = Path(path).read_text(encoding="utf-8")
    root: Dict[str, Any] = {}
    stack = [(0, root)]

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        value = value.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        if value == "":
            node: Dict[str, Any] = {}
            current[key] = node
            stack.append((indent, node))
        else:
            current[key] = _coerce_scalar(value)

    return root


def _coerce_scalar(value: str) -> Any:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")
