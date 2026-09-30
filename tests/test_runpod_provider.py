from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from controller_common.runpod_provider import (
    RUNPOD_REST_BASE_URL,
    RUNPOD_USER_AGENT,
    RunpodClient,
    build_gpu_pod_payload,
)


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class RunpodClientTests(unittest.TestCase):
    def test_requests_identify_the_controller_as_an_api_client(self) -> None:
        response = FakeResponse({"id": "pod-123"})
        with patch("controller_common.runpod_provider.urllib.request.urlopen", return_value=response) as urlopen:
            pod = RunpodClient(api_key="test-key").create_pod({"name": "test-pod"})

        self.assertEqual(pod.id, "pod-123")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.runpod.io/v2/pods")
        self.assertEqual(RUNPOD_REST_BASE_URL, "https://api.runpod.io/v2")
        self.assertEqual(request.get_header("User-agent"), RUNPOD_USER_AGENT)
        self.assertEqual(request.get_header("Accept"), "application/json")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(request.get_method(), "POST")

    def test_gpu_pod_payload_uses_v2_shape(self) -> None:
        payload = build_gpu_pod_payload(
            name="buildvision3d-colmap-test",
            image="example/colmap:latest",
            gpu_type_id="NVIDIA RTX A4500",
            gpu_count=1,
            cloud="COMMUNITY",
            disk_gb=20,
            min_vcpu_per_gpu=4,
            min_ram_per_gpu=16,
            remote_command="python3 scripts/run_colmap_stage.py",
            env={"R2_BUCKET": "test"},
        )

        self.assertEqual(payload["image"], "example/colmap:latest")
        self.assertEqual(payload["disk"], 20)
        self.assertEqual(
            payload["gpu"],
            {
                "id": "NVIDIA RTX A4500",
                "count": 1,
                "minVcpuCountPerGpu": 4,
                "minRamPerGpu": 16,
            },
        )
        self.assertEqual(payload["entrypoint"], ["bash", "-lc"])
        self.assertEqual(payload["cmd"], ["python3 scripts/run_colmap_stage.py"])
        self.assertNotIn("imageName", payload)
        self.assertNotIn("gpuTypeIds", payload)
        self.assertNotIn("containerDiskInGb", payload)

    def test_get_and_delete_use_v2_pod_resource(self) -> None:
        responses = [
            FakeResponse({"id": "pod-123", "status": "RUNNING"}),
            FakeResponse({}),
        ]
        with patch("controller_common.runpod_provider.urllib.request.urlopen", side_effect=responses) as urlopen:
            client = RunpodClient(api_key="test-key")
            pod = client.get_pod("pod-123")
            deleted = client.delete_pod("pod-123")

        self.assertEqual(pod["status"], "RUNNING")
        self.assertEqual(deleted, {})
        get_request = urlopen.call_args_list[0].args[0]
        delete_request = urlopen.call_args_list[1].args[0]
        self.assertEqual(get_request.full_url, "https://api.runpod.io/v2/pods/pod-123")
        self.assertEqual(get_request.get_method(), "GET")
        self.assertEqual(delete_request.full_url, "https://api.runpod.io/v2/pods/pod-123")
        self.assertEqual(delete_request.get_method(), "DELETE")


if __name__ == "__main__":
    unittest.main()
