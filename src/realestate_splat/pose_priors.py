"""Write Skanea ARKit camera positions as COLMAP pose priors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


POSE_PRIOR_STANDARD_DEVIATIONS_METERS = {
    "strong": 0.05,
    "conservative": 0.10,
    "relaxed": 0.25,
}


@dataclass(frozen=True)
class ArkitPositionPrior:
    image_name: str
    capture_id: str
    frame_index: int
    position_meters: Any


def extract_arkit_position_priors(
    image_manifest: Mapping[str, Any],
) -> list[ArkitPositionPrior]:
    """Extract one gravity-aligned ARKit camera position per manifest image.

    A pose-prior reconstruction currently accepts one Skanea capture because
    separate captures have unrelated ARKit world origins. Multi-room sessions
    remain valid when they were recorded as one continuous capture.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Pose-prior incremental mode requires NumPy in the COLMAP runtime image") from exc

    priors: list[ArkitPositionPrior] = []
    capture_ids: set[str] = set()

    for entry in image_manifest.get("images") or []:
        if not isinstance(entry, Mapping):
            continue
        rgbd = entry.get("rgbd")
        if not isinstance(rgbd, Mapping):
            continue
        image_name = str(entry.get("image_name") or "").strip()
        capture_id = str(rgbd.get("capture_id") or "").strip()
        values = rgbd.get("camera_transform")
        layout = str(rgbd.get("camera_transform_layout") or "").lower()
        if not image_name or not capture_id or not isinstance(values, list) or len(values) != 16:
            raise ValueError(f"Skanea manifest entry {image_name or '<unnamed>'} has incomplete ARKit pose metadata")
        if "column-major" not in layout or "camera-to-world" not in layout:
            raise ValueError(f"Unsupported ARKit transform convention for {image_name}: {layout!r}")

        transform = np.asarray(values, dtype=np.float64).reshape((4, 4), order="F")
        if not np.all(np.isfinite(transform)):
            raise ValueError(f"ARKit camera transform contains non-finite values for {image_name}")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4):
            raise ValueError(f"ARKit camera transform has an invalid homogeneous row for {image_name}")

        capture_ids.add(capture_id)
        priors.append(
            ArkitPositionPrior(
                image_name=image_name,
                capture_id=capture_id,
                frame_index=int(rgbd.get("frame_index") or 0),
                position_meters=transform[:3, 3].copy(),
            )
        )

    if not priors:
        raise ValueError("Pose-prior incremental mode requires Skanea ARKit camera transforms")
    if len(capture_ids) != 1:
        raise ValueError(
            "Pose-prior incremental mode currently requires one continuous Skanea capture; "
            f"found {len(capture_ids)} unrelated ARKit coordinate systems"
        )
    return sorted(priors, key=lambda item: (item.frame_index, item.image_name))


def write_arkit_position_priors(
    database_path: Path,
    image_manifest: Mapping[str, Any],
    *,
    uncertainty_preset: str = "conservative",
) -> dict[str, Any]:
    """Replace database pose priors with metric ARKit camera positions."""
    try:
        import numpy as np
        import pycolmap  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Pose-prior incremental mode requires NumPy and PyCOLMAP in the COLMAP runtime image") from exc

    if uncertainty_preset not in POSE_PRIOR_STANDARD_DEVIATIONS_METERS:
        raise ValueError(f"Unknown pose-prior uncertainty preset: {uncertainty_preset}")
    standard_deviation = POSE_PRIOR_STANDARD_DEVIATIONS_METERS[uncertainty_preset]
    covariance = np.eye(3, dtype=np.float64) * standard_deviation**2
    priors = extract_arkit_position_priors(image_manifest)
    prior_by_name = {item.image_name: item for item in priors}

    with pycolmap.Database.open(database_path) as database:
        images = database.read_all_images()
        database_names = {image.name for image in images}
        manifest_names = set(prior_by_name)
        missing_images = sorted(manifest_names - database_names)
        if missing_images:
            raise ValueError(
                "Skanea pose-prior images are missing from the COLMAP database: "
                f"{missing_images[:5]}"
            )

        database.clear_pose_priors()
        for image in images:
            if image.name not in prior_by_name:
                continue
            item = prior_by_name[image.name]
            database.write_pose_prior(
                pycolmap.PosePrior(
                    corr_data_id=image.data_id,
                    position=item.position_meters,
                    position_covariance=covariance,
                    coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN,
                )
            )
        written_count = int(database.num_pose_priors())

    if written_count != len(priors):
        raise RuntimeError(f"Expected {len(priors)} pose priors but database contains {written_count}")
    return {
        "source": "Skanea ARKit camera_transform translation",
        "constraint": "position_only",
        "coordinate_system": "CARTESIAN",
        "capture_id": priors[0].capture_id,
        "database_image_count": len(database_names),
        "prior_count": written_count,
        "unconstrained_image_count": len(database_names - manifest_names),
        "uncertainty_preset": uncertainty_preset,
        "position_standard_deviation_meters": standard_deviation,
        "position_covariance_diagonal_meters_squared": [standard_deviation**2] * 3,
    }
