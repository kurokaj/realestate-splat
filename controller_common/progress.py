"""Small progress reporters shared by local and remote stage runners."""

from __future__ import annotations

import json
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from src.realestate_splat.storage import copy_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class R2ProgressReporter:
    """Publish throttled, human-readable progress for a disposable stage pod."""

    def __init__(
        self,
        *,
        stage: str,
        stage_run_id: str,
        output_uri: str,
        endpoint_url: Optional[str] = None,
        min_interval_seconds: float = 5.0,
    ) -> None:
        self.stage = stage
        self.stage_run_id = stage_run_id
        self.progress_uri = f"{output_uri.rstrip('/')}/current/progress.json"
        self.endpoint_url = endpoint_url
        self.min_interval_seconds = min_interval_seconds
        self._last_write = 0.0
        self._last_key: Optional[tuple[Any, ...]] = None

    def update(
        self,
        percent: int,
        phase: str,
        message: str,
        *,
        phase_percent: Optional[int] = None,
        details: Optional[dict[str, Any]] = None,
        force: bool = False,
    ) -> None:
        snapshot = {
            "schema_version": 1,
            "stage": self.stage,
            "stage_run_id": self.stage_run_id,
            "percent": max(0, min(100, int(percent))),
            "phase": phase,
            "phase_percent": (
                max(0, min(100, int(phase_percent)))
                if phase_percent is not None
                else None
            ),
            "message": message,
            "details": details or {},
            "updated_at": utc_now(),
        }
        key = (
            snapshot["percent"],
            snapshot["phase"],
            snapshot["phase_percent"],
            snapshot["message"],
            json.dumps(snapshot["details"], sort_keys=True, default=str),
        )
        now = time.monotonic()
        if not force and key == self._last_key:
            return
        if not force and now - self._last_write < self.min_interval_seconds:
            return
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2)
            handle.write("\n")
            handle.flush()
            copy_file(handle.name, self.progress_uri, endpoint_url=self.endpoint_url)
        self._last_write = now
        self._last_key = key


class LineProgressParser:
    """Translate common COLMAP/Nerfstudio output lines into stage progress."""

    def __init__(self, reporter: Callable[..., None]) -> None:
        self.reporter = reporter
        self.last_phase = "starting"

    def feed(self, line: str) -> None:
        text = line.strip()
        lowered = text.lower()
        if not text:
            return

        if "feature_extractor" in lowered or "feature extraction" in lowered:
            self.last_phase = "feature_extraction"
            self.reporter(15, self.last_phase, "Extracting image features")
        elif "processed file" in lowered:
            match = re.search(r"\[(\d+)\s*/\s*(\d+)\]", text)
            if match:
                current, total = (int(value) for value in match.groups())
                phase_percent = round(current / max(1, total) * 100)
                self.reporter(
                    10 + round(15 * current / max(1, total)),
                    "feature_extraction",
                    f"Extracting image features ({current}/{total})",
                    phase_percent=phase_percent,
                    details={"processed_images": current, "image_count": total},
                )
        elif "matching stage" in lowered:
            self.last_phase = "matching"
            match = re.search(r"\[(\d+)/(\d+)\]", text)
            if match:
                current, total = (int(value) for value in match.groups())
                phase_percent = round(current / max(1, total) * 100)
                overall = 25 + round(45 * current / max(1, total))
                self.reporter(
                    overall,
                    "matching",
                    text,
                    phase_percent=phase_percent,
                    details={"stage_index": current, "stage_count": total},
                )
            else:
                self.reporter(35, self.last_phase, text)
        elif "sequential" in lowered and "matcher" in lowered:
            self.last_phase = "matching_videos"
            self.reporter(35, self.last_phase, "Matching video frames sequentially")
        elif "vocab" in lowered and "matcher" in lowered:
            self.last_phase = "matching_bridge"
            self.reporter(55, self.last_phase, "Running vocabulary-tree bridge matching")
        elif "exhaustive" in lowered and "matcher" in lowered:
            self.last_phase = "matching_bridge"
            self.reporter(55, self.last_phase, "Running exhaustive bridge matching")
        elif "global_mapper" in lowered or "global mapping" in lowered:
            self.last_phase = "global_mapping"
            self.reporter(75, self.last_phase, "Building global reconstruction")
        elif " mapper" in lowered or lowered.startswith("$ mapper"):
            self.last_phase = "mapping"
            self.reporter(75, self.last_phase, "Building reconstruction")
        elif "view_graph_calibrator" in lowered or "view graph" in lowered:
            self.last_phase = "view_graph_calibration"
            self.reporter(68, self.last_phase, "Calibrating the view graph")
        elif "model_converter" in lowered or "export" in lowered:
            self.last_phase = "exporting"
            self.reporter(88, self.last_phase, "Exporting reconstruction artifacts")
