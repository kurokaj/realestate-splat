"""Normalized COLMAP matching-plan models.

The plan is independent from COLMAP command construction so future hybrid
strategies can use the same structure as the current single-mode workflow.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


SUPPORTED_STRATEGIES = {
    "single",
    "video_plus_heroes",
    "multiple_videos",
    "multiple_videos_plus_heroes",
}
SUPPORTED_MATCHING_STYLES = {"sequential", "exhaustive", "vocab_tree"}
HYBRID_STRATEGIES = {"video_plus_heroes", "multiple_videos", "multiple_videos_plus_heroes"}


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
        "sequential_loop_detection": bool(settings.get("sequential_loop_detection", True)),
        "single_matching_style": str(settings.get("matcher", "exhaustive")),
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
                "kind": (
                    "hero"
                    if entry.get("role") == "hero"
                    else "video"
                    if entry.get("role") == "coverage_video"
                    else "coverage_images"
                ),
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
        if entry.get("role") == "coverage_video":
            group["ordered"] = True
    return sorted(grouped.values(), key=lambda group: group["id"])


def build_hybrid_matching_plan(
    image_manifest: Mapping[str, Any],
    settings: Mapping[str, Any],
    connections: Optional[list[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build the first manual hybrid plan on one shared COLMAP database.

    Ordered coverage groups are matched sequentially. Explicit connections
    then add targeted cross-group matching, keeping heroes out of the large
    all-to-all path unless the user connects them.
    """
    strategy = str(settings.get("processing_strategy") or "single")
    if strategy not in HYBRID_STRATEGIES:
        raise ValueError(f"Unsupported hybrid strategy: {strategy}")
    groups = build_source_groups(image_manifest)
    group_by_id = {group["id"]: group for group in groups}
    normalized_connections: list[Dict[str, Any]] = []
    stages: list[Dict[str, Any]] = []
    for group in groups:
        if group.get("ordered"):
            stages.append({
                "id": f"sequential_{safe_plan_id(group['id'])}",
                "kind": "source_group",
                "groups": [group["id"]],
                "matching_style": "sequential",
                "matching_type": settings.get("matching_type"),
            })
    requested_connections = list(connections or [])
    hero_style = str(settings.get("hero_matching_style") or "exhaustive")
    bridge_style = str(settings.get("video_bridge_matching_style") or "exhaustive")
    if bridge_style not in {"exhaustive", "vocab_tree"}:
        raise ValueError("Video bridge matching must use exhaustive or vocab_tree")
    if hero_style not in {"exhaustive", "vocab_tree"}:
        raise ValueError("Hero matching must use exhaustive or vocab_tree")
    coverage_groups = [group for group in groups if group.get("kind") in {"video", "coverage_images"}]
    for hero in [group for group in groups if group.get("kind") == "hero"]:
        for coverage in coverage_groups:
            if set(hero.get("locations") or []) & set(coverage.get("locations") or []):
                requested_connections.append({
                    "from": hero["id"],
                    "to": coverage["id"],
                    "matching_style": hero_style,
                    "kind": "hero_location",
                })
    seen_connections: set[tuple[str, str]] = set()
    for connection in requested_connections:
        left = str(connection.get("from") or connection.get("source") or "")
        right = str(connection.get("to") or connection.get("target") or "")
        if not left or not right or left == right:
            continue
        left = resolve_group_reference(left, group_by_id)
        right = resolve_group_reference(right, group_by_id)
        if left is None or right is None:
            raise ValueError(f"Connection references unknown source group: {connection.get('from')} -> {connection.get('to')}")
        if (left, right) in seen_connections:
            continue
        seen_connections.add((left, right))
        style = str(connection.get("matching_style") or bridge_style)
        if style not in {"exhaustive", "vocab_tree"}:
            raise ValueError("Hybrid bridge matching must use exhaustive or vocab_tree")
        normalized_connections.append({
            "from": left,
            "to": right,
            "kind": str(connection.get("kind") or "bridge"),
            "matching_style": style,
        })
        stages.append({
            "id": f"bridge_{safe_plan_id(left)}_{safe_plan_id(right)}",
            "kind": "bridge",
            "groups": [left, right],
            "matching_style": style,
            "matching_type": settings.get("matching_type"),
        })
    if not stages:
        raise ValueError("Hybrid matching plan has no source groups to match")
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "strategy": strategy,
        "groups": groups,
        "connections": normalized_connections,
        "bridge_sources": [
            {"source_id": connection["from"], "connects": [connection["to"]]}
            for connection in normalized_connections
        ],
        "sequential_loop_detection": bool(settings.get("sequential_loop_detection", True)),
        "hero_matching_style": hero_style,
        "video_bridge_matching_style": bridge_style,
        "matching_stages": stages,
    }


def safe_plan_id(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_") or "group"


def resolve_group_reference(
    reference: str,
    groups: Mapping[str, Mapping[str, Any]],
) -> Optional[str]:
    """Resolve UI/raw source IDs to the canonical approved-manifest group ID."""
    if reference in groups:
        return reference
    source_ref, separator, camera_ref = reference.partition(":")
    for group_id, group in groups.items():
        source_ids = {str(value) for value in group.get("source_ids", [])}
        camera_groups = {str(value) for value in group.get("camera_groups", [])}
        if source_ref in source_ids or (separator and camera_ref in camera_groups):
            return group_id
    return None


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
    for connection in plan.get("connections") or []:
        if not isinstance(connection, Mapping):
            raise ValueError("Matching plan connections must be objects")
        source = connection.get("from") or connection.get("source")
        targets = connection.get("connects") or ([connection.get("to")] if connection.get("to") else [])
        if source not in group_ids or not isinstance(targets, list) or not targets:
            raise ValueError("Matching plan connections must identify a source and targets")
        if any(target not in group_ids for target in targets):
            raise ValueError("Matching plan connection references an unknown group")
