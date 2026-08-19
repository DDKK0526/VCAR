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


def camera_rgb_image(camera: Any) -> np.ndarray:
    """Return a dataset camera's source image as an RGB uint8 array."""
    image = camera.original_image.detach().cpu().numpy()
    if image.ndim != 3:
        raise ValueError(
            f"Expected a 3D camera image, got shape {image.shape}"
        )
    if image.shape[0] in (3, 4):
        image = np.transpose(image[:3], (1, 2, 0))
    elif image.shape[2] in (3, 4):
        image = image[:, :, :3]
    else:
        raise ValueError(
            f"Expected an RGB camera image, got shape {image.shape}"
        )
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    return np.ascontiguousarray(image.astype(np.uint8))


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
            render_rgb, alpha, _ = render_gsplat_camera(
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
            target_size = (gt_mask.shape[1], gt_mask.shape[0])
            if render_rgb.shape[:2] != gt_mask.shape:
                render_rgb = cv2.resize(
                    render_rgb, target_size, interpolation=cv2.INTER_AREA
                )
            source_rgb = camera_rgb_image(camera)
            if source_rgb.shape[:2] != gt_mask.shape:
                source_rgb = cv2.resize(
                    source_rgb, target_size, interpolation=cv2.INTER_AREA
                )
            per_frame_results.append(
                {
                    "frame": frame_name,
                    "iou": calculate_iou(pred_mask, gt_mask),
                    "pixel_accuracy": calculate_pixel_accuracy(
                        pred_mask, gt_mask
                    ),
                    "pred_mask": pred_mask,
                    "gt_mask": gt_mask,
                    "source_rgb": source_rgb,
                    "render_rgb": render_rgb,
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
    gray = ((mask > 0).astype(np.uint8) * 255)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)


def _error_thumbnail(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    pred = pred_mask > 0
    gt = gt_mask > 0
    image = np.zeros((*pred.shape, 3), dtype=np.uint8)
    image[np.logical_and(pred, gt)] = (0, 180, 0)
    image[np.logical_and(pred, np.logical_not(gt))] = (0, 0, 255)
    image[np.logical_and(np.logical_not(pred), gt)] = (255, 0, 0)
    return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)


def _rgb_thumbnail(image_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)


def _save_result_grid(
    per_frame_results: list[dict[str, Any]],
    save_path: str | Path,
    row_labels: list[str],
    panel_builder: Any,
    thumbnail_size: tuple[int, int],
) -> None:
    if not per_frame_results:
        return

    results = sorted(per_frame_results, key=lambda item: item["iou"])
    thumb_width, thumb_height = thumbnail_size
    padding = 10
    label_height = 30
    row_label_width = 120
    gap = 6
    canvas_width = (
        padding * 2
        + row_label_width
        + len(results) * thumb_width
        + max(0, len(results) - 1) * gap
    )
    canvas_height = (
        padding * 2
        + label_height
        + len(row_labels) * thumb_height
        + max(0, len(row_labels) - 1) * gap
    )
    canvas = np.full(
        (canvas_height, canvas_width, 3), 255, dtype=np.uint8
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    for column, result in enumerate(results):
        x = padding + row_label_width + column * (thumb_width + gap)
        frame = str(result["frame"])
        frame_label = f"f{frame}" if frame.isdigit() else frame
        label = f"{frame_label}  IoU {result['iou']:.3f}"
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
        panels = panel_builder(result, thumbnail_size)
        if len(panels) != len(row_labels):
            raise ValueError("Panel count does not match comparison row labels")
        for row, panel in enumerate(panels):
            y = padding + label_height + row * (thumb_height + gap)
            canvas[y:y + thumb_height, x:x + thumb_width] = panel

    for row, label in enumerate(row_labels):
        y = (
            padding
            + label_height
            + row * (thumb_height + gap)
            + thumb_height // 2
        )
        cv2.putText(
            canvas,
            label,
            (padding, y),
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


def save_mask_comparison_grid(
    per_frame_results: list[dict[str, Any]],
    save_path: str | Path,
    thumbnail_size: tuple[int, int] = (256, 256),
) -> None:
    """Save Prediction/GT/error masks ordered from lowest to highest IoU."""
    def build_panels(result: dict[str, Any], size: tuple[int, int]):
        return [
            _mask_thumbnail(result["pred_mask"], size),
            _mask_thumbnail(result["gt_mask"], size),
            _error_thumbnail(result["pred_mask"], result["gt_mask"], size),
        ]

    _save_result_grid(
        per_frame_results,
        save_path,
        ["Prediction", "GT", "TP/FP/FN"],
        build_panels,
        thumbnail_size,
    )


def save_rgb_comparison_grid(
    per_frame_results: list[dict[str, Any]],
    save_path: str | Path,
    thumbnail_size: tuple[int, int] = (256, 256),
) -> None:
    """Save source/GT-masked/render RGB panels in mask-grid order."""
    def build_panels(result: dict[str, Any], size: tuple[int, int]):
        source_rgb = result["source_rgb"]
        render_rgb = result["render_rgb"]
        gt_mask = result["gt_mask"] > 0
        if gt_mask.shape != source_rgb.shape[:2]:
            gt_mask = cv2.resize(
                gt_mask.astype(np.uint8),
                (source_rgb.shape[1], source_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        white_background = np.full_like(source_rgb, 255)
        gt_masked_source = np.where(
            gt_mask[:, :, None], source_rgb, white_background
        ).astype(np.uint8)
        return [
            _rgb_thumbnail(source_rgb, size),
            _rgb_thumbnail(gt_masked_source, size),
            _rgb_thumbnail(render_rgb, size),
        ]

    _save_result_grid(
        per_frame_results,
        save_path,
        ["Source RGB", "GT-masked RGB", "3DGS render"],
        build_panels,
        thumbnail_size,
    )


def save_comparison_grid(
    per_frame_results: list[dict[str, Any]],
    save_path: str | Path,
    thumbnail_size: tuple[int, int] = (256, 256),
) -> None:
    """Backward-compatible alias for the Mask comparison grid."""
    save_mask_comparison_grid(per_frame_results, save_path, thumbnail_size)
