from __future__ import annotations

import unittest

from scripts.run_colmap import effective_mapper_options, mapper_name_for_mode


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

    def test_pose_prior_mode_uses_robust_database_covariance(self) -> None:
        help_text = """
          --Mapper.ba_use_gpu arg
          --Mapper.ba_gpu_index arg
          --overwrite_priors_covariance arg
          --use_robust_loss_on_prior_position arg
          --prior_position_loss_scale arg
        """

        options = effective_mapper_options(
            {"mode": "pose_prior_incremental", "use_gpu": True, "mapper_options": {}},
            mapper_help=help_text,
        )

        self.assertEqual(mapper_name_for_mode("pose_prior_incremental"), "pose_prior_mapper")
        self.assertEqual(options["overwrite_priors_covariance"], 0)
        self.assertEqual(options["use_robust_loss_on_prior_position"], 1)
        self.assertEqual(options["prior_position_loss_scale"], 7.815)
        self.assertEqual(options["Mapper.ba_use_gpu"], 1)


if __name__ == "__main__":
    unittest.main()
