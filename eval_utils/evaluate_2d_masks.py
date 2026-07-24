"""
Core 2D mask evaluation for LERF-OVS.

Called by run_lerf_ovs.py to render a segmented 3DGS as alpha masks, match the
evaluation views in the curated GT, and calculate per-view and mean IoU and
pixel accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Calculate intersection over union for two binary masks."""
    pred = pred_mask > 0
    gt = gt_mask > 0
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def calculate_pixel_accuracy(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> float:
    """Calculate binary foreground/background pixel accuracy."""
    pred = pred_mask > 0
    gt = gt_mask > 0
    return float(np.equal(pred, gt).sum()) / float(gt.size)


def _camera_index(cameras: list[Any]) -> dict[str, Any]:
    """Index cameras using frame_00041, 00041, and 41 aliases."""
    index: dict[str, Any] = {}
    for camera in cameras:
        image_name = Path(str(camera.image_name)).stem
        index.setdefault(image_name, camera)
        try:
            frame_number = int(image_name.split("_")[-1])
        except ValueError:
            continue
        index.setdefault(str(frame_number), camera)
        index.setdefault(f"{frame_number:05d}", camera)
        index.setdefault(f"frame_{frame_number:05d}", camera)
    return index


def _load_gt_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read GT mask: {path}")
    return (mask > 128).astype(np.uint8)


def evaluate_segmentation(
    segment_ply_path: str | Path,
    gt_mask_dir: str | Path,
    object_name: str,
    cameras: list[Any],
    background: Any,
    sh_degree: int,
    alpha_threshold: float = 0.1,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Render one segmentation result and compare it with LERF-OVS GT masks."""
    import torch

    from gaussiansplatting.scene.gaussian_model import GaussianModel
    from seg_utils.render_utils import render_gsplat_camera

    segment_ply_path = Path(segment_ply_path)
    gt_mask_dir = Path(gt_mask_dir)
    if not segment_ply_path.is_file():
        raise FileNotFoundError(
            f"Segmentation PLY does not exist: {segment_ply_path}"
        )
    if not gt_mask_dir.is_dir():
        raise FileNotFoundError(
            f"GT mask directory does not exist: {gt_mask_dir}"
        )

    gt_paths = sorted(gt_mask_dir.glob(f"{object_name}_frame_*_mask.jpg"))
    if not gt_paths:
        raise FileNotFoundError(
            f"No GT masks found for object {object_name!r}: {gt_mask_dir}"
        )

    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(segment_ply_path))
    if len(gaussians.get_xyz) == 0:
        del gaussians
        torch.cuda.empty_cache()
        raise ValueError(
            f"Segmentation PLY contains no Gaussians: {segment_ply_path}"
        )

    camera_by_name = _camera_index(cameras)
    per_frame_results: list[dict[str, Any]] = []
    try:
        for gt_path in gt_paths:
            frame_name = gt_path.name.split("_frame_", maxsplit=1)[1]
            frame_name = frame_name.removesuffix("_mask.jpg")
            camera = camera_by_name.get(frame_name)
            if camera is None:
                camera = camera_by_name.get(f"frame_{frame_name}")
            if camera is None:
                print(
                    f"[WARN] No camera found for GT frame {frame_name}; "
                    "skipping it"
                )
                continue

            gt_mask = _load_gt_mask(gt_path)
            _, alpha, _ = render_gsplat_camera(
                gaussians,
                camera,
                camera.image_width,
                camera.image_height,
                backgrounds=background,
            )
            pred_alpha = np.squeeze(alpha[0].detach().cpu().numpy())
            if pred_alpha.shape != gt_mask.shape:
                pred_alpha = cv2.resize(
                    pred_alpha,
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            pred_mask = (pred_alpha > alpha_threshold).astype(np.uint8)
            per_frame_results.append(
                {
                    "frame": frame_name,
                    "iou": calculate_iou(pred_mask, gt_mask),
                    "pixel_accuracy": calculate_pixel_accuracy(
                        pred_mask, gt_mask
                    ),
                    "pred_mask": pred_mask,
                    "gt_mask": gt_mask,
                }
            )
    finally:
        del gaussians
        torch.cuda.empty_cache()

    if not per_frame_results:
        raise ValueError(
            f"No evaluation camera matched object {object_name!r}"
        )

    miou = float(np.mean([item["iou"] for item in per_frame_results]))
    macc = float(
        np.mean([item["pixel_accuracy"] for item in per_frame_results])
    )
    return miou, macc, per_frame_results


def _mask_thumbnail(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    gray = ((1 - mask) * 255).astype(np.uint8)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)


def save_comparison_grid(
    per_frame_results: list[dict[str, Any]],
    save_path: str | Path,
    thumbnail_size: tuple[int, int] = (256, 256),
) -> None:
    """Save prediction/GT comparisons ordered from lowest to highest IoU."""
    if not per_frame_results:
        return

    results = sorted(per_frame_results, key=lambda item: item["iou"])
    thumb_width, thumb_height = thumbnail_size
    padding = 10
    label_height = 30
    row_label_width = 40
    gap = 6
    canvas_width = (
        padding * 2
        + row_label_width
        + len(results) * thumb_width
        + max(0, len(results) - 1) * gap
    )
    canvas_height = padding * 2 + label_height + thumb_height * 2 + gap
    canvas = np.full(
        (canvas_height, canvas_width, 3), 255, dtype=np.uint8
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    for index, result in enumerate(results):
        x = padding + row_label_width + index * (thumb_width + gap)
        pred_y = padding + label_height
        gt_y = pred_y + thumb_height + gap
        label = f"f{result['frame']}  {result['iou']:.3f}"
        cv2.putText(
            canvas,
            label,
            (x + 2, padding + 20),
            font,
            0.36,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
        canvas[pred_y:pred_y + thumb_height, x:x + thumb_width] = (
            _mask_thumbnail(result["pred_mask"], thumbnail_size)
        )
        canvas[gt_y:gt_y + thumb_height, x:x + thumb_width] = (
            _mask_thumbnail(result["gt_mask"], thumbnail_size)
        )

    cv2.putText(
        canvas,
        "Pred",
        (padding, padding + label_height + thumb_height // 2),
        font,
        0.42,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "GT",
        (
            padding,
            padding + label_height + thumb_height + gap
            + thumb_height // 2,
        ),
        font,
        0.42,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(save_path), canvas):
        raise OSError(f"Could not save evaluation comparison: {save_path}")
