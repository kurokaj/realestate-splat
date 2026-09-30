from __future__ import annotations

import json
import unittest

from controller_common.preprocess_assembly import parse_group_output_specs


class PreprocessGroupOutputTests(unittest.TestCase):
    def test_parser_preserves_approved_stage_run_id_for_provenance(self) -> None:
        parsed = parse_group_output_specs(
            [
                json.dumps(
                    {
                        "group_key": "location:kitchen",
                        "output_uri": "r2://bucket/preprocess/groups/kitchen/current/",
                        "stage_run_id": "preprocess-run-123",
                    }
                )
            ]
        )

        self.assertEqual(
            parsed,
            [
                {
                    "group_key": "location:kitchen",
                    "output_uri": "r2://bucket/preprocess/groups/kitchen/current",
                    "stage_run_id": "preprocess-run-123",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
