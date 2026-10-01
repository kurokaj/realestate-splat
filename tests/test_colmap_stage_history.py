from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_colmap_stage import copy_colmap_outputs, uploaded_objects


class ColmapHistoryPayloadTests(unittest.TestCase):
    def test_full_colmap_payload_can_be_copied_to_history(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "run"
            colmap_dir = run_dir / "colmap"
            (colmap_dir / "sparse" / "0").mkdir(parents=True)
            (colmap_dir / "sparse_txt").mkdir(parents=True)
            (colmap_dir / "database.db").write_bytes(b"features")
            (colmap_dir / "database_global.db").write_bytes(b"calibrated")
            (colmap_dir / "sparse" / "0" / "points3D.bin").write_bytes(b"points")
            (colmap_dir / "sparse_txt" / "points3D.txt").write_text("# points\n", encoding="utf-8")
            history_dir = root / "history"

            copy_colmap_outputs(run_dir, history_dir)

            self.assertEqual((history_dir / "database.db").read_bytes(), b"features")
            self.assertEqual((history_dir / "database_global.db").read_bytes(), b"calibrated")
            self.assertTrue((history_dir / "sparse" / "0" / "points3D.bin").is_file())
            self.assertTrue((history_dir / "sparse_txt" / "points3D.txt").is_file())

    def test_upload_marker_inventory_lists_nested_full_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sparse" / "0").mkdir(parents=True)
            (root / "logs").mkdir()
            (root / "database.db").write_bytes(b"database")
            (root / "sparse" / "0" / "points3D.bin").write_bytes(b"points")
            (root / "logs" / "mapper.log").write_text("done\n", encoding="utf-8")
            (root / "upload_complete.json").write_text("{}\n", encoding="utf-8")

            inventory = uploaded_objects(root)

            self.assertEqual(
                inventory,
                ["database.db", "logs/mapper.log", "sparse/0/points3D.bin"],
            )


if __name__ == "__main__":
    unittest.main()
