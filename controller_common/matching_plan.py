"""Normalized COLMAP matching-plan models.

The plan is independent from COLMAP command construction so future hybrid
strategies can use the same structure as the current single-mode workflow.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


SUPPORTED_STRATEGIES = {
    "single",
    "video_plus_heroes",
    "multiple_videos",
    "multiple_videos_plus_heroes",
}
SUPPORTED_MATCHING_STYLES = {"sequential", "exhaustive", "vocab_tree"}


def build_single_matching_plan(
    image_manifest: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the compatibility plan for the existing one-matcher workflow."""
    groups = build_source_groups(image_manifest)
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "strategy": "single",
        "groups": groups,
        "connections": [],
        "bridge_sources": [],
        "matching_stages": [
            {
                "id": "single_matcher",
                "kind": "all_groups",
                "groups": [group["id"] for group in groups],
                "matching_style": str(settings.get("matcher", "exhaustive")),
                "feature_extractor": settings.get("feature_extractor"),
                "matching_type": settings.get("matching_type"),
            }
        ],
    }


def build_source_groups(image_manifest: Mapping[str, Any]) -> list[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for entry in image_manifest.get("images") or []:
        if not isinstance(entry, Mapping):
            continue
        source_id = str(entry.get("source_id") or "unassigned")
        camera_group = str(entry.get("camera_group") or "default")
        group_id = f"{source_id}:{camera_group}"
        group = grouped.setdefault(
            group_id,
            {
                "id": group_id,
                "kind": "hero" if entry.get("role") == "hero" else "coverage",
                "source_ids": [source_id],
                "locations": [],
                "camera_groups": [camera_group],
                "ordered": False,
                "image_count": 0,
            },
        )
        group["image_count"] += 1
        location = entry.get("location")
        if location and location not in group["locations"]:
            group["locations"].append(location)
        if entry.get("role") != "hero":
            group["ordered"] = True
    return sorted(grouped.values(), key=lambda group: group["id"])


def matching_plan_summary(plan: Mapping[str, Any]) -> Dict[str, Any]:
    stages = plan.get("matching_stages") or []
    return {
        "strategy": plan.get("strategy"),
        "group_count": len(plan.get("groups") or []),
        "connection_count": len(plan.get("connections") or []),
        "bridge_source_count": len(plan.get("bridge_sources") or []),
        "matching_stage_count": len(stages),
        "matching_styles": [stage.get("matching_style") for stage in stages if isinstance(stage, Mapping)],
    }


def validate_matching_plan(plan: Mapping[str, Any]) -> None:
    strategy = plan.get("strategy")
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unsupported matching strategy: {strategy}")
    groups = plan.get("groups")
    stages = plan.get("matching_stages")
    if not isinstance(groups, list):
        raise ValueError("Matching plan groups must be a list")
    if not isinstance(stages, list) or not stages:
        raise ValueError("Matching plan must contain at least one matching stage")
    group_ids = {group.get("id") for group in groups if isinstance(group, Mapping)}
    if len(group_ids) != len(groups) or None in group_ids:
        raise ValueError("Matching plan groups must have unique ids")
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise ValueError("Matching plan stages must be objects")
        style = stage.get("matching_style")
        if style not in SUPPORTED_MATCHING_STYLES:
            raise ValueError(f"Unsupported matching style in plan: {style}")
        if not stage.get("groups"):
            raise ValueError("Matching plan stages must reference at least one group")
        for group_id in stage.get("groups") or []:
            if group_id not in group_ids:
                raise ValueError(f"Matching stage references unknown group: {group_id}")
