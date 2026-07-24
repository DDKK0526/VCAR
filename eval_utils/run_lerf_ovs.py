#!/usr/bin/env python3
"""
Reproduce VCAR LERF-OVS segmentation and evaluation from trained 3DGS models.

Run with --dry-run first to validate models, source scenes, curated GT, and
configuration files. Remove --dry-run after validation. The default data root
is data/lerf_ovs; each scene contains images, sparse, model, and masks.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import re
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = sorted(
    (PROJECT_ROOT / "configs" / "lerf_ovs").glob("*.csv")
)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "lerf_ovs"
MASK_PATTERN = re.compile(r"^(?P<target>.+)_frame_\d+_mask\.jpg$")
SUMMARY_FIELDS = [
    "scene_name",
    "object_name",
    "gt_object_name",
    "status",
    "error",
    "miou",
    "macc",
    "n_views",
    "current_stage",
    "segment_ply_path",
    "config_file",
    "config_row",
    "random_seed",
]


@dataclass(frozen=True)
class Experiment:
    config_file: Path
    config_row: int
    scene_name: str
    object_name: str
    gt_object_name: str
    prompt_text: str | None
    first_frame_idx: int
    fg_points: list[list[int]]
    bg_points: list[list[int]]
    box: list[int] | None
    threshold_r1: float
    threshold_r2: float
    angular_gap_threshold: float
    n_layers: int
    n_points_per_layer: int
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


def load_experiments(config_paths: list[Path]) -> list[Experiment]:
    experiments: list[Experiment] = []
    for config_path in config_paths:
        with config_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            for row_index, row in enumerate(reader):
                try:
                    scene_name = _required(row, "scene_name")
                    object_name = _required(row, "object_name")
                    prompt_text = (row.get("prompt_text") or "").strip()
                    experiment = Experiment(
                        config_file=config_path,
                        config_row=row_index,
                        scene_name=scene_name,
                        object_name=object_name,
                        gt_object_name=(
                            row.get("gt_object_name") or object_name
                        ).strip(),
                        prompt_text=prompt_text or None,
                        first_frame_idx=int(
                            _number_field(row, "first_frame_idx", 0, int)
                        ),
                        fg_points=_json_field(row, "fg_points", []),
                        bg_points=_json_field(row, "bg_points", []),
                        box=_json_field(row, "box", None),
                        threshold_r1=float(
                            _number_field(row, "threshold_r1", 0.7, float)
                        ),
                        threshold_r2=float(
                            _number_field(row, "threshold_r2", 0.5, float)
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
                                row, "n_points_per_layer", 8, int
                            )
                        ),
                        boundary_refine=_bool_field(
                            row, "boundary_refine", True
                        ),
                        min_compression=float(
                            _number_field(
                                row, "min_compression", 0.2, float
                            )
                        ),
                        tolerance_ratio=float(
                            _number_field(
                                row, "tolerance_ratio", 0.2, float
                            )
                        ),
                        api_url=(
                            row.get("api_url")
                            or "http://localhost:8000"
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
                        f"{config_path}:{row_index + 2}"
                    )
                if experiment.prompt_text and has_spatial_prompt:
                    raise ValueError(
                        f"Text prompts cannot be combined with point or box "
                        f"prompts: "
                        f"{config_path}:{row_index + 2}"
                    )
                if experiment.first_frame_idx < 0:
                    raise ValueError(
                        f"first_frame_idx cannot be negative: "
                        f"{config_path}:{row_index + 2}"
                    )
                if experiment.random_seed < 0:
                    raise ValueError(
                        f"random_seed cannot be negative: "
                        f"{config_path}:{row_index + 2}"
                    )
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
        scene_root / "masks",
    )


def validate_experiments(
    experiments: list[Experiment],
    data_root: Path,
) -> list[str]:
    errors: list[str] = []
    checked_scenes: set[tuple[Path, Path]] = set()
    for experiment in experiments:
        model_path, source_path, gt_mask_dir = experiment_paths(
            experiment, data_root
        )
        scene_key = (model_path, source_path)
        if scene_key not in checked_scenes:
            checked_scenes.add(scene_key)
            if not (model_path / "cfg_args").is_file():
                errors.append(f"Missing 3DGS cfg_args: {model_path}")
            point_cloud = (
                model_path
                / "point_cloud"
                / "iteration_30000"
                / "point_cloud.ply"
            )
            if not point_cloud.is_file():
                errors.append(f"Missing trained point_cloud.ply: {point_cloud}")
            if not source_path.is_dir():
                errors.append(
                    f"Source scene directory does not exist: {source_path}"
                )
            elif not (source_path / "images").is_dir():
                errors.append(
                    f"Source scene is missing images/: {source_path}"
                )
            elif not (source_path / "sparse" / "0").is_dir():
                errors.append(
                    f"Source scene is missing sparse/0: {source_path}"
                )

        gt_pattern = f"{experiment.gt_object_name}_frame_*_mask.jpg"
        if not any(gt_mask_dir.glob(gt_pattern)):
            errors.append(
                f"Missing GT mask: {gt_mask_dir / gt_pattern}"
            )
    return errors


def validate_unique_experiments(
    experiments: list[Experiment],
) -> list[str]:
    keys = [
        (experiment.scene_name, experiment.gt_object_name)
        for experiment in experiments
    ]
    return [
        f"Duplicate GT target in configuration: {scene}/{target}"
        for (scene, target), count in Counter(keys).items()
        if count > 1
    ]


def validate_full_config_coverage(
    experiments: list[Experiment],
    data_root: Path,
) -> list[str]:
    configured = {
        (experiment.scene_name, experiment.gt_object_name)
        for experiment in experiments
    }
    expected: set[tuple[str, str]] = set()
    for mask_dir in sorted(data_root.glob("*/masks")):
        scene = mask_dir.parent.name
        for path in mask_dir.glob("*_mask.jpg"):
            match = MASK_PATTERN.match(path.name)
            if match is None:
                return [f"Could not parse GT mask filename: {path}"]
            expected.add((scene, match.group("target")))

    errors = [
        f"Final GT has no reproduction configuration: {scene}/{target}"
        for scene, target in sorted(expected - configured)
    ]
    errors.extend(
        f"Reproduction configuration has no final GT: {scene}/{target}"
        for scene, target in sorted(configured - expected)
    )
    return errors


def set_random_seed(seed: int) -> None:
    """Reset random state before each target to reduce order-dependent drift."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_SCENE_CACHE: dict[
    tuple[Path, Path],
    tuple[list[Any], list[Any], Any],
] = {}


def load_cameras_and_context(
    model_path: Path,
    source_path: Path,
) -> tuple[list[Any], list[Any], Any]:
    cache_key = (model_path, source_path)
    if cache_key in _SCENE_CACHE:
        return _SCENE_CACHE[cache_key]

    from gaussiansplatting.scene import Scene
    from gaussiansplatting.scene.gaussian_model import GaussianModel
    from seg_utils.pipeline import create_segmentation_context

    context = create_segmentation_context(
        str(model_path), str(source_path)
    )
    gaussians = GaussianModel(context.sh_degree)
    scene = Scene(
        context.dataset,
        gaussians,
        load_iteration=context.args.iteration,
        shuffle=False,
    )
    train_cameras = list(scene.getTrainCameras())
    test_cameras = list(scene.getTestCameras())
    all_cameras = train_cameras + test_cameras
    _SCENE_CACHE[cache_key] = (train_cameras, all_cameras, context)
    return train_cameras, all_cameras, context


def process_experiment(
    experiment: Experiment,
    data_root: Path,
    output_root: Path,
    api_url_override: str | None,
    alpha_threshold: float,
    evaluate_only: bool,
) -> dict[str, Any]:
    from eval_utils.evaluate_2d_masks import (
        evaluate_segmentation,
        save_comparison_grid,
    )
    from seg_utils.pipeline import run_two_round_pipeline

    model_path, source_path, gt_mask_dir = experiment_paths(
        experiment, data_root
    )
    object_output = (
        output_root / experiment.scene_name / experiment.object_name
    )
    segment_path = object_output / "segment.ply"
    result: dict[str, Any] = {
        "scene_name": experiment.scene_name,
        "object_name": experiment.object_name,
        "gt_object_name": experiment.gt_object_name,
        "status": "success",
        "error": "",
        "miou": 0.0,
        "macc": 0.0,
        "n_views": 0,
        "current_stage": "",
        "segment_ply_path": str(segment_path),
        "config_file": str(experiment.config_file),
        "config_row": experiment.config_row,
        "random_seed": experiment.random_seed,
    }

    print(
        f"\n[RUN] {experiment.scene_name}/{experiment.object_name} "
        f"(GT: {experiment.gt_object_name}, seed: {experiment.random_seed})"
    )
    try:
        set_random_seed(experiment.random_seed)
        train_cameras, cameras, context = load_cameras_and_context(
            model_path, source_path
        )
        train_camera_count = len(train_cameras)
        if evaluate_only:
            if not segment_path.is_file():
                raise FileNotFoundError(
                    f"--evaluate-only requires an existing result: "
                    f"{segment_path}"
                )
            result["current_stage"] = "existing"
        else:
            pipeline_result = run_two_round_pipeline(
                ctx=context,
                cached_train_cameras=train_cameras,
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
                skip_round2_if_covered=True,
                object_name=experiment.object_name,
                api_url=api_url_override or experiment.api_url,
                output_dir=str(object_output),
                boundary_refine=experiment.boundary_refine,
                min_compression=experiment.min_compression,
                tolerance_ratio=experiment.tolerance_ratio,
                skip_render=True,
            )
            segment_path = Path(pipeline_result["seg_path"])
            result["segment_ply_path"] = str(segment_path)
            result["current_stage"] = pipeline_result["current_stage"]
            if pipeline_result["current_stage"] == "failed":
                raise ValueError("Round 1 produced no foreground Gaussians")

        miou, macc, frame_results = evaluate_segmentation(
            segment_ply_path=segment_path,
            gt_mask_dir=gt_mask_dir,
            object_name=experiment.gt_object_name,
            cameras=cameras,
            background=context.background,
            sh_degree=context.sh_degree,
            alpha_threshold=alpha_threshold,
        )
        result["miou"] = miou
        result["macc"] = macc
        result["n_views"] = len(frame_results)
        save_comparison_grid(
            frame_results, object_output / "eval_comparison.png"
        )
        print(
            f"[DONE] mIoU={miou:.4f}, mAcc={macc:.4f}, "
            f"views={len(frame_results)}, train_cameras={train_camera_count}"
        )
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
    summary_path = output_root / "lerf_ovs_summary.csv"
    result_by_key = {
        (row["scene_name"], row["object_name"]): row
        for row in _read_existing_results(summary_path)
    }
    for row in new_results:
        result_by_key[(row["scene_name"], row["object_name"])] = row

    data_rows = sorted(
        result_by_key.values(),
        key=lambda row: (row["scene_name"], row["object_name"]),
    )
    success_rows = [row for row in data_rows if row["status"] == "success"]
    summary_rows: list[dict[str, Any]] = []
    for scene_name in sorted({row["scene_name"] for row in success_rows}):
        scene_rows = [
            row for row in success_rows if row["scene_name"] == scene_name
        ]
        summary_rows.append(
            {
                "scene_name": scene_name,
                "object_name": f"[{scene_name}_AVG]",
                "status": "summary",
                "miou": sum(float(row["miou"]) for row in scene_rows)
                / len(scene_rows),
                "macc": sum(float(row["macc"]) for row in scene_rows)
                / len(scene_rows),
            }
        )
    if success_rows:
        summary_rows.append(
            {
                "scene_name": "OVERALL",
                "object_name": "[ALL_AVG]",
                "status": "summary",
                "miou": sum(float(row["miou"]) for row in success_rows)
                / len(success_rows),
                "macc": sum(float(row["macc"]) for row in success_rows)
                / len(success_rows),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(data_rows + summary_rows)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce LERF-OVS results from trained 3DGS models"
    )
    parser.add_argument(
        "--config",
        type=Path,
        nargs="+",
        default=DEFAULT_CONFIGS,
        help=(
            "One or more scene CSV files; defaults to "
            "configs/lerf_ovs/*.csv"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Directory containing the four self-contained scenes; "
            "defaults to data/lerf_ovs"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "lerf_ovs",
        help="Output directory for segmentations and summary metrics",
    )
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        help="Run selected zero-based rows from one configuration file",
    )
    parser.add_argument(
        "--api-url",
        help="Override the SAM API URL from the CSV files",
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
        help="Validate configurations and data without loading CUDA models",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_paths = [path.resolve() for path in args.config]
    missing_configs = [path for path in config_paths if not path.is_file()]
    if missing_configs:
        print(f"[ERROR] Configuration files do not exist: {missing_configs}")
        return 2
    if args.rows is not None and len(config_paths) != 1:
        print("[ERROR] --rows requires exactly one --config file")
        return 2

    try:
        experiments = load_experiments(config_paths)
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
    validation_errors = validate_unique_experiments(experiments)
    validation_errors.extend(validate_experiments(experiments, data_root))
    default_configs = {path.resolve() for path in DEFAULT_CONFIGS}
    if args.rows is None and set(config_paths) == default_configs:
        validation_errors.extend(
            validate_full_config_coverage(experiments, data_root)
        )
    if validation_errors:
        print("[ERROR] Reproduction input validation failed:")
        for error in validation_errors:
            print(f"  - {error}")
        return 2

    print(
        f"[OK] Configuration and input validation passed: "
        f"{len(experiments)} objects across "
        f"{len({item.scene_name for item in experiments})} scenes"
    )
    if args.dry_run:
        return 0

    output_root = args.output.resolve()
    results = []
    for experiment in experiments:
        results.append(
            process_experiment(
                experiment=experiment,
                data_root=data_root,
                output_root=output_root,
                api_url_override=args.api_url,
                alpha_threshold=args.alpha_threshold,
                evaluate_only=args.evaluate_only,
            )
        )
    summary_path = write_summary(output_root, results)
    failed_count = sum(row["status"] == "failed" for row in results)
    print(f"[DONE] Summary saved to: {summary_path}")
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
