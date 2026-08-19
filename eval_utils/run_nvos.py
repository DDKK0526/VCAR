#!/usr/bin/env python3
"""
Reproduce and evaluate VCAR NVOS results from trained 3DGS models.

Validate the data and parameter table without loading CUDA:
    python eval_utils/run_nvos.py --dry-run

A full run requires the local SAM3 service. The NVOS paper protocol requires
the second fine-segmentation round for every target. This script validates
force_round2 and verifies that Round 2 was executed. The default data and
output directories are data/nvos and outputs/nvos.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import shutil
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "nvos" / "nvos.csv"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "nvos"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "nvos"
SUMMARY_FIELDS = [
    "scene_name",
    "object_name",
    "status",
    "error",
    "iou",
    "acc",
    "current_stage",
    "round2_executed",
    "segment_ply_path",
    "config_row",
    "random_seed",
]


@dataclass(frozen=True)
class Experiment:
    config_row: int
    scene_name: str
    object_name: str
    gt_mask_file: str
    camera_name: str
    prompt_text: str | None
    first_frame_idx: int
    fg_points: list[list[int]]
    bg_points: list[list[int]]
    box: list[int] | None
    threshold_r1: float
    threshold_r2: float
    min_visible_ratio: float
    angular_gap_threshold: float
    n_layers: int
    n_points_per_layer: int
    force_round2: bool
    boundary_refine: bool
    min_compression: float
    tolerance_ratio: float
    api_url: str
    random_seed: int


def _required(row: dict[str, str], name: str) -> str:
    value = (row.get(name) or "").strip()
    if not value:
        raise ValueError(f"Missing required field: {name}")
    return value


def _json_field(
    row: dict[str, str],
    name: str,
    default: Any,
) -> Any:
    value = (row.get(name) or "").strip()
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Field {name} is not valid JSON: {value}") from error


def _number_field(
    row: dict[str, str],
    name: str,
    default: int | float,
    number_type: type[int] | type[float],
) -> int | float:
    value = (row.get(name) or "").strip()
    return number_type(value) if value else default


def _bool_field(
    row: dict[str, str],
    name: str,
    default: bool,
) -> bool:
    value = (row.get(name) or "").strip().lower()
    if not value:
        return default
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"Field {name} is not a valid boolean: {value}")


def load_experiments(config_path: Path) -> list[Experiment]:
    experiments: list[Experiment] = []
    with config_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row_index, row in enumerate(reader):
            try:
                prompt_text = (row.get("prompt_text") or "").strip()
                experiment = Experiment(
                    config_row=row_index,
                    scene_name=_required(row, "scene_name"),
                    object_name=_required(row, "object_name"),
                    gt_mask_file=_required(row, "gt_mask_file"),
                    camera_name=_required(row, "camera_name"),
                    prompt_text=prompt_text or None,
                    first_frame_idx=int(
                        _number_field(row, "first_frame_idx", 0, int)
                    ),
                    fg_points=_json_field(row, "fg_points", []),
                    bg_points=_json_field(row, "bg_points", []),
                    box=_json_field(row, "box", None),
                    threshold_r1=float(
                        _number_field(row, "threshold_r1", 0.5, float)
                    ),
                    threshold_r2=float(
                        _number_field(row, "threshold_r2", 0.8, float)
                    ),
                    min_visible_ratio=float(
                        _number_field(
                            row,
                            "min_visible_ratio",
                            0.01,
                            float,
                        )
                    ),
                    angular_gap_threshold=float(
                        _number_field(
                            row,
                            "angular_gap_threshold",
                            90.0,
                            float,
                        )
                    ),
                    n_layers=int(
                        _number_field(row, "n_layers", 4, int)
                    ),
                    n_points_per_layer=int(
                        _number_field(
                            row,
                            "n_points_per_layer",
                            8,
                            int,
                        )
                    ),
                    force_round2=_bool_field(
                        row,
                        "force_round2",
                        True,
                    ),
                    boundary_refine=_bool_field(
                        row,
                        "boundary_refine",
                        True,
                    ),
                    min_compression=float(
                        _number_field(
                            row,
                            "min_compression",
                            0.1,
                            float,
                        )
                    ),
                    tolerance_ratio=float(
                        _number_field(
                            row,
                            "tolerance_ratio",
                            0.6,
                            float,
                        )
                    ),
                    api_url=(
                        row.get("api_url") or "http://localhost:8000"
                    ).strip(),
                    random_seed=int(
                        _number_field(row, "random_seed", 42, int)
                    ),
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Could not parse configuration "
                    f"{config_path}:{row_index + 2}: "
                    f"{error}"
                ) from error

            has_spatial_prompt = bool(
                experiment.fg_points
                or experiment.bg_points
                or experiment.box
            )
            if not experiment.prompt_text and not has_spatial_prompt:
                raise ValueError(
                    f"Configuration requires a text, point, or box prompt: "
                    f"{config_path}:"
                    f"{row_index + 2}"
                )
            if experiment.prompt_text and has_spatial_prompt:
                raise ValueError(
                    f"Text prompts cannot be combined with point or box "
                    f"prompts: {config_path}:"
                    f"{row_index + 2}"
                )
            if not experiment.force_round2:
                raise ValueError(
                    f"The NVOS paper protocol requires force_round2=true: "
                    f"{config_path}:{row_index + 2}"
                )
            if experiment.first_frame_idx < 0:
                raise ValueError("first_frame_idx cannot be negative")
            if not 0.0 <= experiment.min_visible_ratio <= 1.0:
                raise ValueError(
                    f"min_visible_ratio must be between 0 and 1: "
                    f"{config_path}:{row_index + 2}"
                )
            if experiment.random_seed < 0:
                raise ValueError("random_seed cannot be negative")
            experiments.append(experiment)
    return experiments


def experiment_paths(
    experiment: Experiment,
    data_root: Path,
) -> tuple[Path, Path, Path]:
    scene_root = (data_root / experiment.scene_name).resolve()
    return (
        scene_root / "model",
        scene_root,
        scene_root / "masks" / experiment.gt_mask_file,
    )


def validate_experiments(
    experiments: list[Experiment],
    data_root: Path,
) -> list[str]:
    errors: list[str] = []
    seen_objects: set[tuple[str, str]] = set()
    checked_scenes: set[str] = set()
    for experiment in experiments:
        object_key = (experiment.scene_name, experiment.object_name)
        if object_key in seen_objects:
            errors.append(
                f"Duplicate target in configuration: {experiment.scene_name}/"
                f"{experiment.object_name}"
            )
        seen_objects.add(object_key)

        model_path, source_path, gt_mask_path = experiment_paths(
            experiment,
            data_root,
        )
        if experiment.scene_name not in checked_scenes:
            checked_scenes.add(experiment.scene_name)
            if not (source_path / "images").is_dir():
                errors.append(f"Scene is missing images/: {source_path}")
            if not (source_path / "sparse" / "0").is_dir():
                errors.append(f"Scene is missing sparse/0: {source_path}")
            if not (model_path / "cfg_args").is_file():
                errors.append(f"Missing 3DGS cfg_args: {model_path}")
            point_cloud = (
                model_path
                / "point_cloud"
                / "iteration_30000"
                / "point_cloud.ply"
            )
            if not point_cloud.is_file():
                errors.append(f"Missing trained point cloud: {point_cloud}")
        if not gt_mask_path.is_file():
            errors.append(f"Missing GT mask: {gt_mask_path}")
    return errors


def set_random_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_SCENE_CACHE: dict[
    tuple[Path, Path],
    tuple[list[Any], Any],
] = {}


def load_cameras_and_context(
    model_path: Path,
    source_path: Path,
) -> tuple[list[Any], Any]:
    cache_key = (model_path, source_path)
    if cache_key in _SCENE_CACHE:
        return _SCENE_CACHE[cache_key]

    from gaussiansplatting.scene import Scene
    from gaussiansplatting.scene.gaussian_model import GaussianModel
    from seg_utils.pipeline import create_segmentation_context

    context = create_segmentation_context(
        str(model_path),
        str(source_path),
    )
    gaussians = GaussianModel(context.sh_degree)
    scene = Scene(
        context.dataset,
        gaussians,
        load_iteration=context.args.iteration,
        shuffle=False,
    )
    train_cameras = list(scene.getTrainCameras())
    _SCENE_CACHE[cache_key] = (train_cameras, context)
    return train_cameras, context


def find_camera(cameras: list[Any], camera_name: str) -> Any:
    for camera in cameras:
        if Path(str(camera.image_name)).stem == camera_name:
            return camera
    raise ValueError(f"Evaluation camera not found: {camera_name}")


def load_gt_mask(path: Path) -> Any:
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read GT mask: {path}")
    if image.ndim == 3 and image.shape[2] == 4:
        gray = image[:, :, 3]
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return (gray > 128).astype(np.uint8)


def render_prediction(
    segment_path: Path,
    camera: Any,
    context: Any,
    alpha_threshold: float,
) -> tuple[Any, Any]:
    import numpy as np
    import torch
    from gaussiansplatting.scene.gaussian_model import GaussianModel
    from seg_utils.render_utils import render_gsplat_camera

    gaussians = GaussianModel(context.sh_degree)
    gaussians.load_ply(str(segment_path))
    render_rgb, alphas, _ = render_gsplat_camera(
        gaussians,
        camera,
        camera.image_width,
        camera.image_height,
        backgrounds=context.background,
    )
    pred_alpha = np.squeeze(alphas[0].detach().cpu().numpy())
    prediction = (pred_alpha > alpha_threshold).astype(np.uint8)
    del gaussians
    torch.cuda.empty_cache()
    return prediction, render_rgb


def calculate_metrics(prediction: Any, ground_truth: Any) -> tuple[float, float]:
    import numpy as np

    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = float(intersection) / float(union) if union else 0.0
    accuracy = float(np.logical_not(np.logical_xor(pred, gt)).sum()) / float(
        gt.size
    )
    return iou, accuracy


def read_existing_stage(output_dir: Path) -> tuple[str, bool]:
    params_path = output_dir / "segmentation_params.json"
    if not params_path.is_file():
        return "existing", False
    params = json.loads(params_path.read_text(encoding="utf-8"))
    round2 = params.get("segmentation", {}).get("round2")
    return str(params.get("current_stage", "existing")), round2 is not None


def process_experiment(
    experiment: Experiment,
    data_root: Path,
    output_root: Path,
    api_url_override: str | None,
    alpha_threshold: float,
    evaluate_only: bool,
) -> dict[str, Any]:
    from eval_utils.evaluate_2d_masks import (
        camera_rgb_image,
        save_mask_comparison_grid,
        save_rgb_comparison_grid,
    )
    from seg_utils.pipeline import run_two_round_pipeline

    model_path, source_path, gt_mask_path = experiment_paths(
        experiment,
        data_root,
    )
    object_output = output_root / experiment.object_name
    segment_path = object_output / "segment.ply"
    result: dict[str, Any] = {
        "scene_name": experiment.scene_name,
        "object_name": experiment.object_name,
        "status": "success",
        "error": "",
        "iou": 0.0,
        "acc": 0.0,
        "current_stage": "",
        "round2_executed": False,
        "segment_ply_path": str(segment_path),
        "config_row": experiment.config_row,
        "random_seed": experiment.random_seed,
    }

    print(
        f"\n[RUN] {experiment.scene_name}/{experiment.object_name} "
        f"(seed: {experiment.random_seed})"
    )
    try:
        set_random_seed(experiment.random_seed)
        cameras, context = load_cameras_and_context(
            model_path,
            source_path,
        )
        if evaluate_only:
            if not segment_path.is_file():
                raise FileNotFoundError(
                    f"--evaluate-only requires an existing result: "
                    f"{segment_path}"
                )
            current_stage, round2_executed = read_existing_stage(
                object_output
            )
            result["current_stage"] = current_stage
            result["round2_executed"] = round2_executed
        else:
            pipeline_result = run_two_round_pipeline(
                ctx=context,
                cached_train_cameras=cameras,
                first_frame_idx=experiment.first_frame_idx,
                fg_points=experiment.fg_points,
                bg_points=experiment.bg_points,
                box=experiment.box,
                text=experiment.prompt_text,
                threshold_round1=experiment.threshold_r1,
                threshold_round2=experiment.threshold_r2,
                save_background=False,
                n_layers=experiment.n_layers,
                n_points_per_layer=experiment.n_points_per_layer,
                angular_gap_threshold=(
                    experiment.angular_gap_threshold
                ),
                min_visible_ratio=experiment.min_visible_ratio,
                skip_round2_if_covered=not experiment.force_round2,
                object_name=experiment.object_name,
                api_url=api_url_override or experiment.api_url,
                output_dir=str(object_output),
                boundary_refine=experiment.boundary_refine,
                min_compression=experiment.min_compression,
                tolerance_ratio=experiment.tolerance_ratio,
                skip_render=True,
                temp_frames_folder=str(object_output / "temp_frames"),
            )
            if pipeline_result["current_stage"] == "failed":
                raise ValueError("Round 1 produced no foreground Gaussians")
            if pipeline_result["round2_info"] is None:
                raise RuntimeError("NVOS run did not execute Round 2")
            segment_path = Path(pipeline_result["seg_path"])
            result["segment_ply_path"] = str(segment_path)
            result["current_stage"] = pipeline_result["current_stage"]
            result["round2_executed"] = True

        camera = find_camera(cameras, experiment.camera_name)
        ground_truth = load_gt_mask(gt_mask_path)
        prediction, render_rgb = render_prediction(
            segment_path,
            camera,
            context,
            alpha_threshold,
        )
        source_rgb = camera_rgb_image(camera)
        import cv2

        target_size = (ground_truth.shape[1], ground_truth.shape[0])
        if prediction.shape != ground_truth.shape:
            prediction = cv2.resize(
                prediction,
                target_size,
                interpolation=cv2.INTER_NEAREST,
            )
        if render_rgb.shape[:2] != ground_truth.shape:
            render_rgb = cv2.resize(
                render_rgb, target_size, interpolation=cv2.INTER_AREA
            )
        if source_rgb.shape[:2] != ground_truth.shape:
            source_rgb = cv2.resize(
                source_rgb, target_size, interpolation=cv2.INTER_AREA
            )
        iou, accuracy = calculate_metrics(prediction, ground_truth)
        result["iou"] = iou
        result["acc"] = accuracy
        frame_results = [{
            "frame": experiment.camera_name,
            "iou": iou,
            "pred_mask": prediction,
            "gt_mask": ground_truth,
            "source_rgb": source_rgb,
            "render_rgb": render_rgb,
        }]
        thumbnail_height = min(512, ground_truth.shape[0])
        thumbnail_size = (
            max(
                1,
                round(
                    ground_truth.shape[1]
                    * thumbnail_height
                    / ground_truth.shape[0]
                ),
            ),
            thumbnail_height,
        )
        mask_comparison_path = object_output / "eval_mask_comparison.png"
        save_mask_comparison_grid(
            frame_results, mask_comparison_path, thumbnail_size
        )
        save_rgb_comparison_grid(
            frame_results,
            object_output / "eval_rgb_comparison.png",
            thumbnail_size,
        )
        # Preserve the historical filename for downstream scripts.
        shutil.copy2(mask_comparison_path, object_output / "eval_comparison.png")
        print(f"[DONE] IoU={iou:.4f}, Acc={accuracy:.4f}")
    except Exception as error:
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = str(error)
    finally:
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
    return result


def _read_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return [
            row
            for row in csv.DictReader(file)
            if row.get("status") != "summary"
        ]


def write_summary(
    output_root: Path,
    new_results: list[dict[str, Any]],
) -> Path:
    summary_path = output_root / "nvos_summary.csv"
    result_by_object = {
        row["object_name"]: row
        for row in _read_existing_results(summary_path)
    }
    for row in new_results:
        result_by_object[row["object_name"]] = row

    data_rows = sorted(
        result_by_object.values(),
        key=lambda row: int(row.get("config_row", 0)),
    )
    success_rows = [row for row in data_rows if row["status"] == "success"]
    rows_to_write = list(data_rows)
    if success_rows:
        rows_to_write.append(
            {
                "scene_name": "MEAN",
                "object_name": "[ALL_AVG]",
                "status": "summary",
                "iou": sum(float(row["iou"]) for row in success_rows)
                / len(success_rows),
                "acc": sum(float(row["acc"]) for row in success_rows)
                / len(success_rows),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows_to_write)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce NVOS results from trained 3DGS models"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Reproduction parameter table; defaults to configs/nvos/nvos.csv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="NVOS data directory; defaults to data/nvos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Segmentation and evaluation output; defaults to outputs/nvos",
    )
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        help="Run selected zero-based configuration rows",
    )
    parser.add_argument(
        "--api-url",
        help="Override the SAM3 service URL from the CSV",
    )
    parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=0.1,
        help="Alpha threshold for evaluation rendering; defaults to 0.1",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip segmentation and evaluate existing segment.ply files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate parameters and data without loading CUDA",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        print(f"[ERROR] Configuration file does not exist: {config_path}")
        return 2
    try:
        experiments = load_experiments(config_path)
    except ValueError as error:
        print(f"[ERROR] {error}")
        return 2

    if args.rows is not None:
        invalid_rows = [
            row for row in args.rows if row < 0 or row >= len(experiments)
        ]
        if invalid_rows:
            print(f"[ERROR] Configuration row indices out of range: {invalid_rows}")
            return 2
        experiments = [experiments[row] for row in args.rows]

    data_root = args.data_root.resolve()
    validation_errors = validate_experiments(experiments, data_root)
    if validation_errors:
        print("[ERROR] NVOS reproduction input validation failed:")
        for error in validation_errors:
            print(f"  - {error}")
        return 2

    print(
        f"[OK] Configuration and input validation passed for "
        f"{len(experiments)} targets; Round 2 is enforced"
    )
    if args.dry_run:
        return 0

    output_root = args.output.resolve()
    results = [
        process_experiment(
            experiment=experiment,
            data_root=data_root,
            output_root=output_root,
            api_url_override=args.api_url,
            alpha_threshold=args.alpha_threshold,
            evaluate_only=args.evaluate_only,
        )
        for experiment in experiments
    ]
    summary_path = write_summary(output_root, results)
    failed_count = sum(row["status"] == "failed" for row in results)
    print(f"[DONE] Summary saved to: {summary_path}")
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
