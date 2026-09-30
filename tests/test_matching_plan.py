from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from controller_common.matching_executor import resolve_group_image_names
from controller_common.matching_plan import build_hybrid_matching_plan, build_source_groups


def skanea_frame(capture: str, frame_index: int, location: str = "room") -> dict:
    return {
        "image_name": f"{capture}_{frame_index:06d}.jpg",
        "role": "coverage_image",
        "source_kind": "skanea_rgbd_frame",
        "source_id": f"skanea_{capture}_{frame_index:06d}",
        "camera_group": f"skanea_{capture}",
        "location": location,
    }


class MatchingPlanTests(unittest.TestCase):
    def test_skanea_frames_form_one_ordered_capture_group(self) -> None:
        manifest = {"images": [skanea_frame("capture_a", 0), skanea_frame("capture_a", 1)]}

        groups = build_source_groups(manifest)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["image_count"], 2)
        self.assertTrue(groups[0]["ordered"])
        self.assertEqual(groups[0]["locations"], ["room"])
        self.assertEqual(len(groups[0]["source_ids"]), 2)

        names = resolve_group_image_names({groups[0]["id"]: groups[0]}, manifest)
        self.assertEqual(names[groups[0]["id"]], ["capture_a_000000.jpg", "capture_a_000001.jpg"])

    def test_hybrid_plan_adds_sequential_stage_for_each_skanea_capture(self) -> None:
        manifest = {
            "images": [
                skanea_frame("capture_a", 0, "room_a"),
                skanea_frame("capture_a", 1, "room_a"),
                skanea_frame("capture_b", 0, "room_b"),
            ]
        }
        groups = build_source_groups(manifest)

        plan = build_hybrid_matching_plan(
            manifest,
            {"processing_strategy": "multiple_videos", "video_bridge_matching_style": "exhaustive"},
            [{"from": groups[0]["id"], "to": groups[1]["id"]}],
        )

        sequential_stages = [stage for stage in plan["matching_stages"] if stage["matching_style"] == "sequential"]
        bridge_stages = [stage for stage in plan["matching_stages"] if stage["kind"] == "bridge"]
        self.assertEqual(len(sequential_stages), 2)
        self.assertEqual(len(bridge_stages), 1)


if __name__ == "__main__":
    unittest.main()
