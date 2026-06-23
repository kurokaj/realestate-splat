"""Placeholder for a future raw gsplat training backend."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def build_commands(_settings: Mapping[str, Any]) -> Dict[str, Any]:
    raise SystemExit(
        "backend=raw_gsplat is reserved for a later custom trainer. "
        "Use backend=splatfacto for milestone 2."
    )

