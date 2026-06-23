"""Training backend registry."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from . import raw_gsplat, splatfacto


SUPPORTED_BACKENDS = {
    "splatfacto": splatfacto,
    "raw_gsplat": raw_gsplat,
}


def validate_backend_name(backend: str) -> None:
    if backend not in SUPPORTED_BACKENDS:
        choices = ", ".join(sorted(SUPPORTED_BACKENDS))
        raise SystemExit(f"Unsupported training backend '{backend}'. Choose one of: {choices}")


def build_backend_commands(backend: str, settings: Mapping[str, Any]) -> Dict[str, Any]:
    validate_backend_name(backend)
    return SUPPORTED_BACKENDS[backend].build_commands(settings)
