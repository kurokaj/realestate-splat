"""Small RunPod REST adapter for controller GPU stages."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from controller_common.config import runpod_api_key


RUNPOD_REST_BASE_URL = "https://rest.runpod.io/v1"


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
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"RunPod API {method} {path} failed with HTTP {exc.code}: {error_body}") from exc
        if not raw_body.strip():
            return {}
        return json.loads(raw_body)
