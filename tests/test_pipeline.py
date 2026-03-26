from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.customer_segmentation_analysis.analysis import run_analysis


class PipelineTestCase(unittest.TestCase):
    def test_pipeline_creates_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            project_root = Path(tmp_dir)
            (project_root / "data" / "processed").mkdir(parents=True, exist_ok=True)
            (project_root / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
            (project_root / "docs").mkdir(parents=True, exist_ok=True)

            metrics = run_analysis(project_root=project_root, customer_count=120, seed=7, as_of_date="2026-03-26")

            self.assertGreater(metrics["customer_count"], 0)
            self.assertGreater(metrics["order_count"], 0)
            self.assertIn(
                int(metrics["best_k"]),
                {4, 5, 6, 7},
            )

            expected_files = [
                project_root / "data" / "processed" / "customer_base.csv",
                project_root / "data" / "processed" / "transactions.csv",
                project_root / "outputs" / "customer_segments.csv",
                project_root / "outputs" / "segment_profiles.csv",
                project_root / "outputs" / "retention_playbook.csv",
                project_root / "outputs" / "executive_summary.md",
                project_root / "docs" / "project_story.md",
                project_root / "outputs" / "figures" / "cluster_projection.png",
            ]

            for path in expected_files:
                self.assertTrue(path.exists(), msg=f"Expected output missing: {path}")


if __name__ == "__main__":
    unittest.main()
