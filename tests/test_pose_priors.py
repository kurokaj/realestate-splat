from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for candidate in (ROOT_DIR, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from realestate_splat.pose_priors import (
    extract_arkit_position_priors,
    write_arkit_position_priors,
)


def manifest_entry(name: str, frame_index: int, position: tuple[float, float, float], capture_id: str = "capture-1") -> dict:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = position
    return {
        "image_name": name,
        "source_kind": "skanea_rgbd_frame",
        "rgbd": {
            "capture_id": capture_id,
            "frame_index": frame_index,
            "camera_transform": transform.reshape(16, order="F").tolist(),
            "camera_transform_layout": "4x4 column-major camera-to-world",
        },
    }


class FakeDatabase:
    def __init__(self, names: list[str]) -> None:
        self.images = [SimpleNamespace(name=name, data_id=f"data:{index}") for index, name in enumerate(names)]
        self.priors: list[object] = []

    def __enter__(self) -> "FakeDatabase":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read_all_images(self) -> list[object]:
        return self.images

    def clear_pose_priors(self) -> None:
        self.priors.clear()

    def write_pose_prior(self, prior: object) -> int:
        self.priors.append(prior)
        return len(self.priors)

    def num_pose_priors(self) -> int:
        return len(self.priors)


@unittest.skipIf(np is None, "NumPy pose-prior dependency unavailable")
class PosePriorTests(unittest.TestCase):
    def test_extracts_column_major_camera_positions(self) -> None:
        manifest = {
            "images": [
                manifest_entry("000001.jpg", 1, (1.0, 2.0, 3.0)),
                manifest_entry("000000.jpg", 0, (-1.0, 0.5, 4.0)),
            ]
        }

        priors = extract_arkit_position_priors(manifest)

        self.assertEqual([item.image_name for item in priors], ["000000.jpg", "000001.jpg"])
        np.testing.assert_allclose(priors[0].position_meters, [-1.0, 0.5, 4.0])

    def test_rejects_unrelated_capture_coordinate_systems(self) -> None:
        manifest = {
            "images": [
                manifest_entry("a.jpg", 0, (0.0, 0.0, 0.0), capture_id="capture-a"),
                manifest_entry("b.jpg", 0, (0.0, 0.0, 0.0), capture_id="capture-b"),
            ]
        }

        with self.assertRaisesRegex(ValueError, "unrelated ARKit coordinate systems"):
            extract_arkit_position_priors(manifest)

    def test_writes_conservative_cartesian_priors_and_leaves_other_images_unconstrained(self) -> None:
        names = ["000000.jpg", "000001.jpg"]
        manifest = {
            "images": [
                manifest_entry(names[0], 0, (0.0, 0.0, 0.0)),
                manifest_entry(names[1], 1, (0.2, 0.0, -0.1)),
            ]
        }
        database = FakeDatabase(names + ["drone_hero.jpg"])

        class FakePosePrior:
            def __init__(self, **kwargs: object) -> None:
                self.__dict__.update(kwargs)

        fake_pycolmap = SimpleNamespace(
            Database=SimpleNamespace(open=lambda _path: database),
            PosePrior=FakePosePrior,
            PosePriorCoordinateSystem=SimpleNamespace(CARTESIAN="cartesian"),
        )

        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(sys.modules, {"pycolmap": fake_pycolmap}):
            summary = write_arkit_position_priors(Path(temp_dir) / "database.db", manifest)

        self.assertEqual(summary["prior_count"], 2)
        self.assertEqual(summary["database_image_count"], 3)
        self.assertEqual(summary["unconstrained_image_count"], 1)
        self.assertEqual(summary["position_standard_deviation_meters"], 0.10)
        self.assertEqual(database.priors[0].coordinate_system, "cartesian")
        np.testing.assert_allclose(database.priors[1].position, [0.2, 0.0, -0.1])
        np.testing.assert_allclose(database.priors[1].position_covariance, np.eye(3) * 0.01)


if __name__ == "__main__":
    unittest.main()
