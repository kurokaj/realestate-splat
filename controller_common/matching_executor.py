"""Execute scoped matching stages through the PyCOLMAP API."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


def execute_matching_plan(
    *,
    plan: Mapping[str, Any],
    image_manifest: Mapping[str, Any],
    database_path: Path,
    work_dir: Path,
    matching_type: str,
    use_gpu: bool,
    sequential_overlap: int = 10,
) -> list[Dict[str, Any]]:
    """Run each planned stage against one database.

    Pair generation is deliberately small and explicit. PyCOLMAP still owns
    descriptor matching and geometric verification through match_image_pairs.
    """
    try:
        import pycolmap  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Scoped matching requires PyCOLMAP in the COLMAP runtime image."
        ) from exc

    groups = {
        str(group["id"]): group
        for group in plan.get("groups", [])
        if isinstance(group, Mapping) and group.get("id")
    }
    names_by_group = resolve_group_image_names(groups, image_manifest)
    work_dir.mkdir(parents=True, exist_ok=True)
    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu
    matcher_type = getattr(pycolmap.FeatureMatcherType, matching_type)
    results: list[Dict[str, Any]] = []

    for index, stage in enumerate(plan.get("matching_stages", []), start=1):
        if not isinstance(stage, Mapping):
            raise ValueError("Matching plan stages must be objects")
        stage_id = str(stage.get("id") or f"stage_{index:03d}")
        style = str(stage.get("matching_style") or "exhaustive")
        print(
            f"Matching stage [{index}/{len(plan.get('matching_stages', []))}]: "
            f"{stage_id} ({style})",
            flush=True,
        )
        stage_matching_type = str(stage.get("matching_type") or matching_type)
        stage_matcher_type = getattr(pycolmap.FeatureMatcherType, stage_matching_type)
        matching_options = pycolmap.FeatureMatchingOptions(type=stage_matcher_type)
        stage_groups = [str(group_id) for group_id in stage.get("groups", [])]
        image_names = [name for group_id in stage_groups for name in names_by_group.get(group_id, [])]
        if not image_names:
            raise ValueError(f"Matching stage '{stage_id}' resolved to no images")

        pair_path = work_dir / f"{stage_id}.pairs.txt"
        if style == "vocab_tree":
            query_path = work_dir / f"{stage_id}.queries.txt"
            query_path.write_text("\n".join(image_names) + "\n", encoding="utf-8")
            options = pycolmap.VocabTreePairingOptions(match_list_path=str(query_path))
            pycolmap.match_vocabtree(
                str(database_path),
                matching_options=matching_options,
                pairing_options=options,
                device=device,
            )
            results.append(
                {
                    "id": stage_id,
                    "matching_style": style,
                    "query_image_count": len(image_names),
                    "pair_list": None,
                    "matching_type": stage_matching_type,
                }
            )
            print(f"Matching stage complete: {stage_id}", flush=True)
            continue

        pairs = build_pairs(style, stage_groups, names_by_group, sequential_overlap)
        pair_path.write_text("\n".join(f"{left} {right}" for left, right in pairs) + "\n", encoding="utf-8")
        pairing_options = pycolmap.ImportedPairingOptions(match_list_path=str(pair_path))
        pycolmap.match_image_pairs(
            str(database_path),
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=device,
        )
        results.append(
            {
                "id": stage_id,
                "matching_style": style,
                "image_count": len(image_names),
                "pair_count": len(pairs),
                "pair_list": str(pair_path),
                "matching_type": stage_matching_type,
            }
        )
        print(f"Matching stage complete: {stage_id}", flush=True)

    (work_dir / "matching_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    return results


def resolve_group_image_names(
    groups: Mapping[str, Mapping[str, Any]],
    image_manifest: Mapping[str, Any],
) -> Dict[str, list[str]]:
    names: Dict[str, list[str]] = {group_id: [] for group_id in groups}
    for entry in image_manifest.get("images") or []:
        if not isinstance(entry, Mapping):
            continue
        source_id = str(entry.get("source_id") or "unassigned")
        camera_group = str(entry.get("camera_group") or "default")
        group_id = f"{source_id}:{camera_group}"
        image_name = entry.get("image_name")
        if group_id in names and image_name:
            names[group_id].append(str(image_name))
    return names


def build_pairs(
    style: str,
    group_ids: Sequence[str],
    names_by_group: Mapping[str, Sequence[str]],
    overlap: int,
) -> list[tuple[str, str]]:
    if style == "sequential":
        if len(group_ids) != 1:
            raise ValueError("Sequential scoped stages must contain exactly one group")
        names = list(names_by_group.get(group_ids[0], []))
        return [
            (left, right)
            for index, left in enumerate(names)
            for right in names[index + 1 : index + 1 + max(1, int(overlap))]
        ]
    if style != "exhaustive":
        raise ValueError(f"Unsupported scoped matching style: {style}")
    if len(group_ids) == 1:
        return list(itertools.combinations(names_by_group.get(group_ids[0], []), 2))
    return [
        (left, right)
        for left_group, right_group in itertools.combinations(group_ids, 2)
        for left in names_by_group.get(left_group, [])
        for right in names_by_group.get(right_group, [])
    ]
