"""Local fake provider used by Milestone 8A worker development."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from controller_common.config import fake_stage_duration_seconds


class FakeProvider:
    """Pretends to run a disposable stage without starting external compute."""

    name = "local_fake"

    def run_stage(self, stage_run: dict[str, Any], progress: Optional[Callable[[int, str], None]] = None) -> dict[str, Any]:
        duration = fake_stage_duration_seconds()
        steps = [
            (25, "Fake provider preparing inputs"),
            (50, "Fake provider running stage command"),
            (75, "Fake provider collecting outputs"),
            (100, "Fake provider complete"),
        ]
        for percent, message in steps:
            if progress is not None:
                progress(percent, message)
            if duration:
                time.sleep(duration / len(steps))
        stage = stage_run["stage"]
        return {
            "provider": self.name,
            "stage": stage,
            "fake": True,
            "message": f"Fake {stage} stage completed",
        }
