from __future__ import annotations

import unittest

from scripts.run_colmap import effective_mapper_options


class MapperOptionCompatibilityTests(unittest.TestCase):
    def test_global_gpu_defaults_are_limited_to_runtime_supported_options(self) -> None:
        help_text = """
          --GlobalMapper.gp_use_gpu arg
          --GlobalMapper.gp_gpu_index arg
          --GlobalMapper.ba_ceres_use_gpu arg
          --GlobalMapper.ba_ceres_gpu_index arg
        """

        options = effective_mapper_options(
            {"mode": "global", "use_gpu": True, "mapper_options": {}},
            mapper_help=help_text,
        )

        self.assertEqual(
            options,
            {
                "GlobalMapper.gp_use_gpu": 1,
                "GlobalMapper.gp_gpu_index": 0,
                "GlobalMapper.ba_ceres_use_gpu": 1,
                "GlobalMapper.ba_ceres_gpu_index": 0,
            },
        )
        self.assertNotIn("GlobalMapper.ba_gpu_index", options)

    def test_newer_global_bundle_adjustment_gpu_selector_is_supported(self) -> None:
        help_text = """
          --GlobalMapper.ba_ceres_use_gpu arg
          --GlobalMapper.ba_gpu_index arg
        """

        options = effective_mapper_options(
            {"mode": "global", "use_gpu": True, "mapper_options": {}},
            mapper_help=help_text,
        )

        self.assertEqual(options["GlobalMapper.ba_ceres_use_gpu"], 1)
        self.assertEqual(options["GlobalMapper.ba_gpu_index"], 0)
        self.assertNotIn("GlobalMapper.ba_ceres_gpu_index", options)

    def test_unsupported_automatic_default_is_skipped(self) -> None:
        options = effective_mapper_options(
            {"mode": "global", "use_gpu": True, "mapper_options": {}},
            mapper_help="--GlobalMapper.ba_ceres_use_gpu arg",
        )

        self.assertEqual(options, {"GlobalMapper.ba_ceres_use_gpu": 1})

    def test_explicit_mapper_option_is_not_silently_removed(self) -> None:
        options = effective_mapper_options(
            {
                "mode": "global",
                "use_gpu": True,
                "mapper_options": {"GlobalMapper.custom_option": 7},
            },
            mapper_help="--GlobalMapper.ba_ceres_use_gpu arg",
        )

        self.assertEqual(options["GlobalMapper.custom_option"], 7)

if __name__ == "__main__":
    unittest.main()
