"""Small RunPod REST adapter for controller GPU stages."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from controller_common.config import runpod_api_key


RUNPOD_REST_BASE_URL = "https://api.runpod.io/v2"
RUNPOD_REST_API_VERSION = "v2"
RUNPOD_USER_AGENT = "Buildvision3D-Controller/1.0"


@dataclass(frozen=True)
class RunpodPod:
    id: str
    raw: dict[str, Any]


class RunpodClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = RUNPOD_REST_BASE_URL) -> None:
        self.api_key = api_key or runpod_api_key()
        if not self.api_key:
            raise ValueError("RUNPOD_API_KEY is required for runpod_colmap")
        self.base_url = base_url.rstrip("/")

    def create_pod(self, payload: dict[str, Any]) -> RunpodPod:
        response = self._request("POST", "/pods", payload)
        pod_id = response.get("id")
        if not pod_id:
            raise RuntimeError(f"RunPod create pod response did not include id: {response}")
        return RunpodPod(id=str(pod_id), raw=response)

    def get_pod(self, pod_id: str) -> dict[str, Any]:
        return self._request("GET", f"/pods/{pod_id}")

    def delete_pod(self, pod_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/pods/{pod_id}")

    def _request(self, method: str, path: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": RUNPOD_USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Runpod REST API {RUNPOD_REST_API_VERSION} {method} {path} "
                f"failed with HTTP {exc.code}: {format_error_body(error_body)}"
            ) from exc
        if not raw_body.strip():
            return {}
        return json.loads(raw_body)


def build_gpu_pod_payload(
    *,
    name: str,
    image: str,
    gpu_type_id: str,
    gpu_count: int,
    cloud: str,
    disk_gb: int,
    min_vcpu_per_gpu: int,
    min_ram_per_gpu: int,
    remote_command: str,
    env: Mapping[str, str],
) -> dict[str, Any]:
    """Build the Runpod REST API v2 request for a disposable GPU Pod."""
    if not gpu_type_id.strip():
        raise ValueError("Runpod REST API v2 requires one GPU type id")
    return {
        "name": name,
        "image": image,
        "cloud": cloud,
        "gpu": {
            "id": gpu_type_id,
            "count": int(gpu_count),
            "minVcpuCountPerGpu": int(min_vcpu_per_gpu),
            "minRamPerGpu": int(min_ram_per_gpu),
        },
        "disk": int(disk_gb),
        "entrypoint": ["bash", "-lc"],
        "cmd": [remote_command],
        "env": dict(env),
        "ports": [],
        "globalNetworking": False,
        "startJupyter": False,
        "startSsh": False,
    }


def format_error_body(raw_body: str) -> str:
    """Compact RFC 9457 v2 errors while retaining non-JSON edge errors."""
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body
    if not isinstance(payload, dict):
        return raw_body
    parts = [str(payload.get(key)) for key in ("title", "detail") if payload.get(key)]
    errors = payload.get("errors")
    if isinstance(errors, list):
        parts.extend(str(error) for error in errors)
    return "; ".join(parts) or raw_body
