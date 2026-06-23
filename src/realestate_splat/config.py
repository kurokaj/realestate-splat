"""Small config loader for pipeline scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from realestate_splat.cli import parse_scalar


def normalize_key(key: str) -> str:
    return key.replace("-", "_")


def normalize_keys(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in mapping.items():
        new_key = normalize_key(str(key))
        if isinstance(value, dict):
            normalized[new_key] = normalize_keys(value)
        else:
            normalized[new_key] = value
    return normalized


def load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise SystemExit(f"Config file does not exist: {path}")

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        loaded = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        loaded = load_yaml_like(text, path)
    else:
        raise SystemExit(f"Unsupported config extension for {path}; use .json, .yaml, or .yml.")

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config root must be a mapping: {path}")
    return normalize_keys(loaded)


def load_yaml_like(text: str, path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return parse_simple_yaml(text, path)

    loaded = yaml.safe_load(text)
    return loaded or {}


def parse_simple_yaml(text: str, path: Path) -> Dict[str, Any]:
    """Parse mapping-only YAML used by the pipeline configs."""

    root: Dict[str, Any] = {}
    stack = [(-1, root)]

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if "\t" in raw_line:
            raise SystemExit(f"{path}:{line_number}: tabs are not supported by the fallback YAML parser.")
        if ":" not in line:
            raise SystemExit(f"{path}:{line_number}: expected KEY: VALUE.")

        indent = len(line) - len(line.lstrip(" "))
        key, value = line.strip().split(":", 1)
        key = normalize_key(key.strip())
        value = value.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise SystemExit(f"{path}:{line_number}: invalid indentation.")

        parent = stack[-1][1]
        if not value:
            child: Dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)

    return root

