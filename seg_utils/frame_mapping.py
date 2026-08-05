"""Persist and consume the SAM-frame-to-camera relationship.

Iterative segmentation may remove empty rendered frames and then renumber the
remaining frames for SAM.  The manifest produced here preserves the original
camera identity so later stages do not have to infer it from the mask filename.
"""

import json
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple


FRAME_MAPPING_VERSION = 1
LATEST_FRAME_MAPPING_FILENAME = "sam_frame_mapping.json"


def _stage_filename(stage: str) -> str:
    safe_stage = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in str(stage)
    ).strip("_")
    if not safe_stage:
        safe_stage = "segmentation"
    return f"sam_frame_mapping_{safe_stage}.json"


def build_frame_camera_mapping(
    source_frame_indices: Sequence[int],
    total_source_frames: int,
    num_train_cameras: int,
    stage: str,
) -> Dict[str, Any]:
    """Build a serializable mapping from renumbered SAM frames to cameras."""
    total_source_frames = int(total_source_frames)
    num_train_cameras = int(num_train_cameras)
    if total_source_frames < 0:
        raise ValueError("total_source_frames must be non-negative")
    if not 0 <= num_train_cameras <= total_source_frames:
        raise ValueError(
            "num_train_cameras must be between zero and total_source_frames"
        )

    kept_indices = [int(idx) for idx in source_frame_indices]
    if len(set(kept_indices)) != len(kept_indices):
        raise ValueError("source_frame_indices contains duplicates")
    if any(idx < 0 or idx >= total_source_frames for idx in kept_indices):
        raise ValueError("source_frame_indices contains an out-of-range index")

    frames = []
    for sam_frame_idx, source_frame_idx in enumerate(kept_indices):
        if source_frame_idx < num_train_cameras:
            camera_kind = "train"
            camera_idx = source_frame_idx
        else:
            camera_kind = "sphere"
            camera_idx = source_frame_idx - num_train_cameras
        frames.append({
            "sam_frame_idx": sam_frame_idx,
            "source_frame_idx": source_frame_idx,
            "camera_kind": camera_kind,
            "camera_idx": camera_idx,
        })

    kept_set = set(kept_indices)
    filtered_source_indices = [
        idx for idx in range(total_source_frames) if idx not in kept_set
    ]
    filtered_train_indices = [
        idx for idx in filtered_source_indices if idx < num_train_cameras
    ]
    filtered_sphere_indices = [
        idx - num_train_cameras
        for idx in filtered_source_indices
        if idx >= num_train_cameras
    ]

    return {
        "version": FRAME_MAPPING_VERSION,
        "stage": str(stage),
        "total_source_frames": total_source_frames,
        "num_train_cameras": num_train_cameras,
        "num_sphere_cameras": total_source_frames - num_train_cameras,
        "kept_source_frame_indices": kept_indices,
        "filtered_source_frame_indices": filtered_source_indices,
        "filtered_train_camera_indices": filtered_train_indices,
        "filtered_sphere_camera_indices": filtered_sphere_indices,
        "frames": frames,
    }


def save_frame_camera_mapping(
    output_dir: str,
    mapping: Mapping[str, Any],
) -> str:
    """Save a stage-specific manifest and refresh the latest-manifest alias."""
    os.makedirs(output_dir, exist_ok=True)
    stage_path = os.path.join(output_dir, _stage_filename(mapping.get("stage", "")))
    latest_path = os.path.join(output_dir, LATEST_FRAME_MAPPING_FILENAME)

    for path in (stage_path, latest_path):
        with open(path, "w", encoding="utf-8") as file:
            json.dump(mapping, file, indent=2, ensure_ascii=False)
    return stage_path


def load_frame_camera_mapping(path: str) -> Dict[str, Any]:
    """Load and minimally validate a saved frame-camera manifest."""
    with open(path, "r", encoding="utf-8") as file:
        mapping = json.load(file)
    if not isinstance(mapping, dict) or not isinstance(mapping.get("frames"), list):
        raise ValueError(f"Invalid SAM frame-camera mapping: {path}")
    return mapping


def collect_mapped_train_masks(
    train_cameras: Sequence[Any],
    masks_by_sam_frame: Mapping[int, Any],
    frame_camera_mapping: Mapping[str, Any],
) -> Tuple[List[Any], List[Any]]:
    """Pair available SAM masks with their original training cameras.

    Missing mask indices are intentionally ignored.  This preserves alignment
    when low-quality masks are deleted and leave holes in the numbering.
    Supplemental spherical masks are ignored because ABR currently operates on
    the cached training-camera set.
    """
    frames = frame_camera_mapping.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Frame-camera mapping has no valid 'frames' list")

    masks = []
    cameras = []
    seen_sam_indices = set()
    for entry in sorted(frames, key=lambda item: int(item["sam_frame_idx"])):
        sam_frame_idx = int(entry["sam_frame_idx"])
        if sam_frame_idx in seen_sam_indices:
            raise ValueError(
                f"Duplicate SAM frame index in frame-camera mapping: {sam_frame_idx}"
            )
        seen_sam_indices.add(sam_frame_idx)

        if entry.get("camera_kind") != "train":
            continue
        camera_idx = int(entry["camera_idx"])
        if camera_idx < 0 or camera_idx >= len(train_cameras):
            raise ValueError(
                f"Training camera index {camera_idx} is out of range for "
                f"{len(train_cameras)} cached cameras"
            )
        if sam_frame_idx not in masks_by_sam_frame:
            continue
        masks.append(masks_by_sam_frame[sam_frame_idx])
        cameras.append(train_cameras[camera_idx])

    return masks, cameras
