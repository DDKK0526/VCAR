"""Core segmentation pipeline for SAM-guided 3DGS segmentation."""

import gc
import os
import json
import pickle
import shutil
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from gaussiansplatting.arguments import ModelParams, PipelineParams
from gaussiansplatting.scene import Scene
from gaussiansplatting.scene.gaussian_model import GaussianModel

from seg_utils.multi_sphere_sampling import sample_multi_sphere_views
from seg_utils.ensemble_utils import mask_inverse, ensemble
from seg_utils.boundary_compression import compute_boundary_compression_axes
from seg_utils.mask_filter import filter_low_quality_masks
from seg_utils.post_filter import robust_center_outlier_filter_indices, compute_robust_sphere_center
from seg_utils.render_utils import (
    RenderConfig, build_virtual_camera, create_pinhole_intrinsics,
    create_look_at_view_matrix, render_gsplat_camera
)
from seg_utils.general_utils import (
    get_combined_args, call_sam_video_segment, load_masks_from_folder,
    render_segmented_results, save_gs, save_background_gs
)
from seg_utils.frame_mapping import (
    LATEST_FRAME_MAPPING_FILENAME,
    build_frame_camera_mapping,
    collect_mapped_train_masks,
    load_frame_camera_mapping,
    save_frame_camera_mapping,
)

SEGMENT_FILENAME = 'segment.ply'
BACKGROUND_FILENAME = 'background.ply'
COARSE_FILENAME = 'coarse.ply'
FINE_FILENAME = 'fine.ply'
COMPRESSION_FILENAME = 'compression.ply'
MASKS_FILENAME = 'multiview_masks.pkl'
PARAMS_FILENAME = 'segmentation_params.json'
VIEW_COVERAGE_MIN_VISIBLE_RATIO = 0.2


@dataclass
class SegmentationContext:
    """Model configuration and render settings for one segmentation scene."""
    model_path: str
    source_path: str
    dataset: Any
    pipeline: Any
    background: torch.Tensor
    sh_degree: int
    args: Any


def _patch_gaussian_splatting_dataset_defaults(dataset: Any) -> Any:
    """Fill defaults added by newer Gaussian Splatting versions.

    Older trained scenes may have cfg_args files without recently added
    ModelParams fields. Scene still expects these attributes at runtime.
    """
    defaults = {
        "depths": "",
        "train_test_exp": False,
    }
    for name, value in defaults.items():
        if not hasattr(dataset, name):
            setattr(dataset, name, value)
    return dataset


def _resolve_source_path(
    model_path: str,
    configured_source_path: str,
    source_path_override: Optional[str],
) -> str:
    """Resolve training data and allow overriding a non-portable cfg_args."""
    source_path = source_path_override or configured_source_path
    if not source_path:
        raise ValueError(
            "The 3DGS configuration has no source_path; provide the "
            "training-data path explicitly"
        )

    source_path = os.path.expanduser(source_path)
    if not os.path.isabs(source_path):
        model_relative_path = os.path.join(model_path, source_path)
        if os.path.isdir(model_relative_path):
            source_path = model_relative_path

    source_path = os.path.abspath(source_path)
    if not os.path.isdir(source_path):
        raise FileNotFoundError(
            "The 3DGS training-data path does not exist. cfg_args often "
            "contains an absolute path from the training machine; override it "
            f"with source_path. Current path: {source_path}"
        )
    return source_path


def create_segmentation_context(
    model_path: str,
    source_path: Optional[str] = None,
) -> SegmentationContext:
    """Create a context and optionally override cfg_args training data."""
    model_path = os.path.abspath(os.path.expanduser(model_path))
    parser = ArgumentParser(description="Segmentation Context")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser, model_path)
    args.source_path = _resolve_source_path(
        model_path=model_path,
        configured_source_path=getattr(args, "source_path", ""),
        source_path_override=source_path,
    )

    configs = RenderConfig(white_background=True, debug=False,
                           compute_cov3D_python=False, convert_SHs_python=True)
    bg_color = [1, 1, 1] if configs.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    dataset = model.extract(args)
    dataset = _patch_gaussian_splatting_dataset_defaults(dataset)
    dataset.model_path = args.model_path

    return SegmentationContext(
        model_path=model_path,
        source_path=dataset.source_path,
        dataset=dataset,
        pipeline=pipeline,
        background=background,
        sh_degree=dataset.sh_degree,
        args=args
    )


def _get_original_ply_path(ctx: SegmentationContext) -> str:
    return os.path.join(ctx.model_path, "point_cloud", "iteration_30000", "point_cloud.ply")


def _load_gaussians(ctx: SegmentationContext, ply_path: Optional[str] = None) -> GaussianModel:
    gaussians = GaussianModel(ctx.sh_degree)
    gaussians.load_ply(ply_path or _get_original_ply_path(ctx))
    return gaussians


def _clear_cuda(*_objs):
    gc.collect()
    torch.cuda.empty_cache()


def _select_render_cameras(render_cameras: Optional[List], fallback_cameras: List) -> List:
    return render_cameras if render_cameras is not None else fallback_cameras


def _render_current_results(
    ctx: SegmentationContext,
    seg_path: str,
    cameras: List,
    save_dir: str,
    bg_path: Optional[str] = None,
    save_background: bool = False,
    save_mask: bool = True,
    show_previews: bool = True,
    show_start_frame: int = 0,
    show_frames: int = 6,
):
    render_dir, preview_images, mask_dir = render_segmented_results(
        seg_path, cameras, save_dir, ctx.pipeline, ctx.background,
        ctx.sh_degree, save_mask=save_mask, show_previews=show_previews,
        show_start_frame=show_start_frame, show_frames=show_frames
    )

    if save_background and bg_path and os.path.exists(bg_path):
        render_segmented_results(
            bg_path, cameras, save_dir, ctx.pipeline, ctx.background,
            ctx.sh_degree, save_mask=False, show_previews=False
        )

    return render_dir, preview_images, mask_dir


def _build_sam_prompts(fg_points: List, bg_points: List, box: Optional[List]):
    points = [[pt[0], pt[1], 1] for pt in fg_points]
    points.extend([[pt[0], pt[1], 0] for pt in bg_points])
    boxes = [[box[0], box[1], box[2], box[3]]] if box else None
    return points, boxes


def _prepare_sam_frames(
    output_dir: str,
    temp_frames_folder: str,
    all_cameras: List,
    first_frame_idx: int,
    is_iterative: bool,
):
    frames_folder = os.path.join(output_dir, "frames")
    if os.path.exists(frames_folder):
        shutil.rmtree(frames_folder)
    os.makedirs(frames_folder)

    cameras_for_masks = all_cameras
    source_indices_for_masks = list(range(len(all_cameras)))
    sam_frame_idx = first_frame_idx

    if is_iterative:
        keep_indices = _filter_empty_frames(temp_frames_folder, len(all_cameras), first_frame_idx)
        n_filtered = len(all_cameras) - len(keep_indices)
        if n_filtered > 0:
            print(
                f"[INFO] Filtered {n_filtered} empty frames; "
                f"kept {len(keep_indices)}"
            )
        cameras_for_masks = []
        source_indices_for_masks = keep_indices
        for new_idx, orig_idx in enumerate(keep_indices):
            shutil.copy2(
                os.path.join(temp_frames_folder, f"{orig_idx:05d}.jpg"),
                os.path.join(frames_folder, f"{new_idx:05d}.jpg")
            )
            cameras_for_masks.append(all_cameras[orig_idx])
            if orig_idx == first_frame_idx:
                sam_frame_idx = new_idx
    else:
        for filename in sorted(os.listdir(temp_frames_folder)):
            if filename.endswith('.jpg'):
                shutil.copy2(
                    os.path.join(temp_frames_folder, filename),
                    os.path.join(frames_folder, filename)
                )

    return (
        frames_folder,
        cameras_for_masks,
        sam_frame_idx,
        source_indices_for_masks,
    )


def _get_mask_folder(
    frames_folder: str,
    sam_frame_idx: int,
    points: List,
    boxes: Optional[List],
    text: Optional[str],
    object_name: str,
    api_url: str,
    pre_mask_folder: Optional[str],
) -> str:
    if pre_mask_folder is not None and os.path.exists(pre_mask_folder):
        print(
            f"[INFO] Using precomputed masks and skipping SAM: "
            f"{pre_mask_folder}"
        )
        return pre_mask_folder

    result = call_sam_video_segment(
        frames_folder, sam_frame_idx, points, boxes,
        text=text, object_name=object_name, api_url=f"{api_url}/segment"
    )
    return result['result_paths'][object_name]['mask_folder']


def _collect_valid_masks(cameras: List, masks_dict: Dict[int, np.ndarray]):
    valid_masks, valid_cameras = [], []
    for frame_idx, cam in enumerate(cameras):
        if frame_idx in masks_dict:
            valid_masks.append(masks_dict[frame_idx])
            valid_cameras.append(cam)
    return valid_masks, valid_cameras


def _vote_masks(xyz: torch.Tensor, cameras: List, masks: List[np.ndarray], desc: str):
    multiview_masks = []
    for cam, mask in tqdm(zip(cameras, masks), desc=desc):
        mask_tensor = torch.from_numpy(mask).to("cuda")
        point_mask, _ = mask_inverse(xyz, cam, mask_tensor)
        multiview_masks.append(point_mask.unsqueeze(-1))
    return multiview_masks


def _save_multiview_masks(
    output_dir: str,
    multiview_masks: List,
    multiview_masks_original: List,
    foreground_model_path: str,
    save_background: bool,
):
    with open(os.path.join(output_dir, MASKS_FILENAME), 'wb') as f:
        pickle.dump({
            'mmasks': _stack_and_compress_masks(multiview_masks),
            'mmasks_original': _stack_and_compress_masks(multiview_masks_original) if save_background else None,
            'foreground_model_path': foreground_model_path,
        }, f)


def _summarize_segmentation_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'final_mask_count': result['final_mask_count'],
        'total_points': result['total_points'],
        'valid_masks_count': result['valid_masks_count'],
        'removed_masks_count': result.get('removed_masks_count', 0),
        'removed_mask_indices': result.get('removed_mask_indices', []),
        'filtered_empty_frames': result.get('filtered_empty_frames', 0),
        'filtered_source_frame_indices': result.get(
            'filtered_source_frame_indices', []
        ),
        'filtered_train_camera_indices': result.get(
            'filtered_train_camera_indices', []
        ),
        'filtered_sphere_camera_indices': result.get(
            'filtered_sphere_camera_indices', []
        ),
        'frame_mapping_file': (
            os.path.basename(result['frame_mapping_path'])
            if result.get('frame_mapping_path') else None
        ),
    }


def _build_segmentation_params(
    model_path: str,
    source_path: str,
    object_name: str,
    first_frame_idx: int,
    fg_points: List,
    bg_points: List,
    box: Optional[List],
    text: Optional[str],
    threshold_round1: float,
    threshold_round2: float,
    save_background: bool,
    cached_train_cameras: List,
    round1_info: Dict[str, Any],
    coverage: Dict[str, Any],
    need_sphere: bool,
    round2_info: Optional[Dict[str, Any]],
    n_layers: int,
    n_points_per_layer: int,
    angular_gap_threshold: float,
    boundary_refine: bool,
    current_stage: str,
    abr_result: Optional[Dict[str, Any]],
    min_compression: float,
    tolerance_ratio: float,
) -> Dict[str, Any]:
    params = {
        'model_path': model_path,
        'source_path': source_path,
        'object_name': object_name,
        'annotations': {
            'text': text,
            'foreground_points': fg_points,
            'background_points': bg_points,
            'box': box,
            'annotated_frame_idx': first_frame_idx,
        },
        'segmentation': {
            'threshold_round1': threshold_round1,
            'threshold_round2': threshold_round2,
            'save_background': save_background,
            'round1': {
                'views': 'train_only',
                'num_cameras': len(cached_train_cameras),
                'valid_masks': round1_info['valid_masks_count'],
                'removed_low_quality_masks': round1_info['removed_masks_count'],
                'removed_low_quality_mask_indices': round1_info['removed_mask_indices'],
                'filtered_empty_frames': round1_info['filtered_empty_frames'],
                'filtered_source_frame_indices': round1_info['filtered_source_frame_indices'],
                'filtered_train_camera_indices': round1_info['filtered_train_camera_indices'],
                'filtered_sphere_camera_indices': round1_info['filtered_sphere_camera_indices'],
                'frame_mapping_file': round1_info['frame_mapping_file'],
                'output': COARSE_FILENAME,
            },
            'coverage_analysis': {
                'total_cameras': coverage['total_cameras'],
                'valid_cameras': coverage['valid_cameras'],
                'invalid_cameras': coverage['invalid_cameras'],
                'min_visible_ratio': coverage['min_visible_ratio'],
                'max_angular_gap_deg': coverage['max_angular_gap_deg'],
                'mean_angular_gap_deg': coverage['mean_angular_gap_deg'],
                'angular_gap_threshold_deg': angular_gap_threshold,
                'need_sphere_sampling': need_sphere,
            },
            'round2': {
                'views': round2_info['views'],
                'num_train_cameras': len(cached_train_cameras),
                'num_sphere_cameras': round2_info['num_sphere_cameras'],
                'n_layers': n_layers,
                'n_points_per_layer': n_points_per_layer,
                'valid_masks': round2_info['valid_masks_count'],
                'removed_low_quality_masks': round2_info['removed_masks_count'],
                'removed_low_quality_mask_indices': round2_info['removed_mask_indices'],
                'filtered_empty_frames': round2_info['filtered_empty_frames'],
                'filtered_source_frame_indices': round2_info['filtered_source_frame_indices'],
                'filtered_train_camera_indices': round2_info['filtered_train_camera_indices'],
                'filtered_sphere_camera_indices': round2_info['filtered_sphere_camera_indices'],
                'frame_mapping_file': round2_info['frame_mapping_file'],
                'output': FINE_FILENAME,
            } if round2_info is not None else None,
        },
        'post_processing': {
            'outlier_filter': {'applied': False},
            'boundary_compression': {'applied': boundary_refine and current_stage == 'compression'},
        },
        'current_stage': current_stage,
    }
    if boundary_refine and current_stage == 'compression' and abr_result is not None:
        params['post_processing']['boundary_compression'].update({
            'min_compression': min_compression,
            'tolerance_ratio': tolerance_ratio,
            'original_points': abr_result['original_points'],
            'compressed_points': abr_result['compressed_points'],
            'aligned_masks': abr_result['aligned_masks'],
            'used_explicit_frame_mapping': abr_result[
                'used_explicit_frame_mapping'
            ],
            'frame_mapping_file': abr_result['frame_mapping_file'],
            'output': COMPRESSION_FILENAME,
        })
    return params


def load_views(
    ctx: SegmentationContext,
    use_train_views: bool,
    use_random_views: bool,
    n_layers: int,
    n_points_per_layer: int,
    prev_seg_path: Optional[str],
    cached_train_cameras: Optional[List],
    output_folder: str,
    first_frame_idx: int = 0,
    use_test_views: bool = False
) -> Tuple[List, List, List]:
    """Load views and render them to an output directory.
    
    Args:
        use_test_views: Include test views in segmentation. Disabled by
            default and recommended only for ablation studies.
    
    Returns:
        (all_cameras, cached_train_cameras, test_cameras)
        test_cameras is populated only when Scene is first loaded.
    """
    # Rebuild the rendered-frame directory.
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Merge and sort train/test views only on first load; later rounds reuse
    # the cached cameras without rebuilding Scene.
    test_cameras_out = []
    scene = None
    if cached_train_cameras is None:
        print("[INFO] Loading cameras for the first time")
        original_gaussians = GaussianModel(ctx.sh_degree)
        scene = Scene(ctx.dataset, original_gaussians, 
                       load_iteration=ctx.args.iteration, shuffle=False)
        train_cameras = list(scene.getTrainCameras())
        test_cameras_out = list(scene.getTestCameras())
        if use_test_views:
            # Preserve source frame order after merging train and test views.
            cameras = sorted(train_cameras + test_cameras_out, key=lambda c: c.image_name)
            print(
                f"[INFO] Merged {len(train_cameras)} training and "
                f"{len(test_cameras_out)} test cameras: {len(cameras)} total"
            )
        else:
            cameras = train_cameras
        cached_train_cameras = cameras
    else:
        print("[INFO] Using cached cameras")
        cameras = cached_train_cameras
        original_gaussians = _load_gaussians(ctx)

    # Select the Gaussian model to render.
    if prev_seg_path and os.path.exists(prev_seg_path):
        gaussians = _load_gaussians(ctx, prev_seg_path)
    else:
        gaussians = original_gaussians

    all_cameras = []
    frame_idx = 0

    # Render dataset views in their cached order.
    if use_train_views:
        for cam in tqdm(cameras, desc="Rendering dataset views"):
            rendering, _, _ = render_gsplat_camera(
                gaussians, cam, cam.image_width, cam.image_height, backgrounds=ctx.background
            )
            rendering = cv2.cvtColor(rendering, cv2.COLOR_RGB2BGR)
            filename = f"{frame_idx:05d}.jpg"
            cv2.imwrite(os.path.join(output_folder, filename), rendering)
            all_cameras.append(cam)
            frame_idx += 1

    # Render supplemental spherical views.
    if use_random_views and n_layers > 0 and n_points_per_layer > 0:
        xyz = gaussians.get_xyz
        ref_idx = min(first_frame_idx, len(cameras) - 1)
        v_ref = cameras[ref_idx]
        sphere_center = compute_robust_sphere_center(xyz)
        sphere_radius = torch.dist(v_ref.camera_center.reshape(1, 3),
                                   sphere_center.reshape(1, 3)).item() * 1.1
        width, height = v_ref.image_width, v_ref.image_height
        znear, zfar = 0.01, 100

        sampled_points, _ = sample_multi_sphere_views(
            sphere_center.cpu().detach().numpy(), sphere_radius, n_layers,
            n_points_per_layer, random_seed=1024
        )

        for cam_pos in tqdm(sampled_points, desc="Rendering spherical views"):
            camera_pos = torch.tensor(cam_pos, dtype=torch.float32)
            intrinsics = create_pinhole_intrinsics(width, height)[None, :, :].to("cuda")
            world_to_camera = create_look_at_view_matrix(camera_pos, sphere_center)[None, :, :].to("cuda")
            virtual_camera = build_virtual_camera(
                intrinsics, world_to_camera, width, height, znear, zfar,
                custom_id=frame_idx, device="cuda"
            )
            rendering, _, _ = render_gsplat_camera(
                gaussians, virtual_camera, virtual_camera.image_width,
                virtual_camera.image_height, backgrounds=ctx.background
            )
            rendering = cv2.cvtColor(rendering, cv2.COLOR_RGB2BGR)
            filename = f"{frame_idx:05d}.jpg"
            cv2.imwrite(os.path.join(output_folder, filename), rendering)
            all_cameras.append(virtual_camera)
            frame_idx += 1

    print(
        f"[INFO] Loaded {len(all_cameras)} views "
        f"(training: {use_train_views}, test: {use_test_views}, "
        f"spherical: {use_random_views})"
    )

    # Release GaussianModel and Scene GPU memory after rendering.
    if gaussians is not original_gaussians:
        del gaussians
    del original_gaussians
    if scene is not None:
        del scene
    _clear_cuda()

    return all_cameras, cached_train_cameras, test_cameras_out


def _filter_empty_frames(temp_frames_folder, num_frames, first_frame_idx, content_threshold=0.005):
    """Return nonempty frame indices while always retaining the prompt frame.

    content_threshold is the minimum ratio of non-white pixels.
    """
    keep_indices = []
    for idx in range(num_frames):
        if idx == first_frame_idx:
            keep_indices.append(idx)
            continue
        filepath = os.path.join(temp_frames_folder, f"{idx:05d}.jpg")
        if not os.path.exists(filepath):
            continue
        img = cv2.imread(filepath)
        content_ratio = np.any(img < 250, axis=2).mean()
        if content_ratio >= content_threshold:
            keep_indices.append(idx)
    return keep_indices


def _stack_and_compress_masks(multiview_masks_list: List) -> Optional[np.ndarray]:
    """Stack multi-view masks into a compact CPU int8 array."""
    if len(multiview_masks_list) == 0:
        return None
    stacked = torch.cat(multiview_masks_list, dim=1).to('cpu')
    stacked = stacked.clamp(min=-1, max=1).to(torch.int8)
    return stacked.numpy()


def run_segmentation(
    ctx: SegmentationContext,
    all_cameras: List,
    cached_train_cameras: List,
    first_frame_idx: int,
    fg_points: List,
    bg_points: List,
    box: Optional[List],
    text: Optional[str],
    threshold: float,
    save_background: bool,
    object_name: str,
    api_url: str,
    output_dir: str,
    prev_seg_path: Optional[str] = None,
    temp_frames_folder: Optional[str] = None,
    render_cameras: Optional[List] = None,
    skip_render: bool = False,
    pre_mask_folder: Optional[str] = None,
    stage_name: str = "segmentation",
) -> Dict[str, Any]:
    """Run the core segmentation workflow.
    
    Returns:
        Paths, rendered previews, and segmentation statistics.
    """
    points, boxes = _build_sam_prompts(fg_points, bg_points, box)
    os.makedirs(output_dir, exist_ok=True)

    # Load foreground and optional full-scene models.
    foreground_model_path = prev_seg_path if prev_seg_path and os.path.exists(prev_seg_path) else _get_original_ply_path(ctx)
    gaussians = _load_gaussians(ctx, foreground_model_path)
    original_gaussians = _load_gaussians(ctx) if save_background else None
    xyz = gaussians.get_xyz

    # Step 1: prepare frames and filter empty iterative views.
    if temp_frames_folder is None:
        temp_frames_folder = os.path.join(ctx.model_path, "temp_frames")

    is_iterative = prev_seg_path is not None and os.path.exists(str(prev_seg_path))
    (
        frames_folder,
        cameras_for_masks,
        sam_frame_idx,
        source_indices_for_masks,
    ) = _prepare_sam_frames(
        output_dir, temp_frames_folder, all_cameras, first_frame_idx, is_iterative
    )

    # Persist the explicit relationship before SAM runs.  Iterative filtering
    # renumbers frames, so mask filenames alone cannot recover camera identity.
    frame_camera_mapping = build_frame_camera_mapping(
        source_frame_indices=source_indices_for_masks,
        total_source_frames=len(all_cameras),
        num_train_cameras=len(cached_train_cameras),
        stage=stage_name,
    )
    frame_mapping_path = save_frame_camera_mapping(
        output_dir, frame_camera_mapping
    )

    # Step 2: call SAM or use precomputed masks.
    sam_mask_folder = _get_mask_folder(
        frames_folder, sam_frame_idx, points, boxes, text,
        object_name, api_url, pre_mask_folder
    )
    mask_folder = sam_mask_folder
    absolute_mask_folder = os.path.abspath(mask_folder)
    absolute_output_dir = os.path.abspath(output_dir)
    try:
        mask_is_within_output = os.path.commonpath([
            absolute_mask_folder, absolute_output_dir
        ]) == absolute_output_dir
    except ValueError:
        mask_is_within_output = False
    frame_camera_mapping['mask_folder'] = (
        os.path.relpath(absolute_mask_folder, absolute_output_dir)
        if mask_is_within_output else absolute_mask_folder
    )
    frame_mapping_path = save_frame_camera_mapping(
        output_dir, frame_camera_mapping
    )

    # Step 3: filter empty and low-quality masks.
    removed_indices = filter_low_quality_masks(mask_folder=mask_folder)
    frame_camera_mapping['removed_low_quality_mask_indices'] = [
        int(idx) for idx in removed_indices
    ]
    frame_mapping_path = save_frame_camera_mapping(
        output_dir, frame_camera_mapping
    )

    # Step 4: back-project valid masks and vote.
    masks_dict = load_masks_from_folder(mask_folder)
    valid_masks, valid_cameras = _collect_valid_masks(cameras_for_masks, masks_dict)
    multiview_masks = _vote_masks(
        xyz, valid_cameras, valid_masks, desc="Back-projecting and voting"
    )
    _, final_mask = ensemble(multiview_masks, threshold=threshold)

    # Avoid an uncatchable gsplat SIGFPE when voting selects no foreground.
    if len(final_mask) == 0:
        print("[WARN] Voting selected no foreground Gaussians; skipping render")
        seg_path = os.path.join(output_dir, SEGMENT_FILENAME)
        # Preserve the normal output structure with an empty PLY.
        save_gs(gaussians, final_mask, seg_path)

        del gaussians
        if original_gaussians is not None:
            del original_gaussians
        del multiview_masks
        _clear_cuda()

        return {
            'seg_path': seg_path,
            'bg_path': None,
            'render_dir': None,
            'mask_dir': None,
            'sam_mask_folder': sam_mask_folder,
            'preview_images': [],
            'final_mask_count': 0,
            'total_points': len(xyz),
            'valid_masks_count': len(valid_masks),
            'removed_masks_count': len(removed_indices),
            'removed_mask_indices': removed_indices,
            'filtered_empty_frames': len(
                frame_camera_mapping['filtered_source_frame_indices']
            ),
            'filtered_source_frame_indices': frame_camera_mapping[
                'filtered_source_frame_indices'
            ],
            'filtered_train_camera_indices': frame_camera_mapping[
                'filtered_train_camera_indices'
            ],
            'filtered_sphere_camera_indices': frame_camera_mapping[
                'filtered_sphere_camera_indices'
            ],
            'frame_camera_mapping': frame_camera_mapping,
            'frame_mapping_path': frame_mapping_path,
        }

    # Repeat voting on the original model when saving the background.
    multiview_masks_original = []
    final_mask_original = None
    if save_background and original_gaussians is not None:
        original_xyz = original_gaussians.get_xyz
        multiview_masks_original = _vote_masks(
            original_xyz,
            valid_cameras,
            valid_masks,
            desc="Back-projecting onto the original model",
        )
        _, final_mask_original = ensemble(multiview_masks_original, threshold=threshold)

    # Persist compressed multi-view votes for threshold updates.
    _save_multiview_masks(
        output_dir, multiview_masks, multiview_masks_original,
        foreground_model_path, save_background
    )

    # Step 5: save the segmented PLY.
    seg_path = os.path.join(output_dir, SEGMENT_FILENAME)
    save_gs(gaussians, final_mask, seg_path)

    bg_path = None
    if save_background and final_mask_original is not None:
        bg_path = os.path.join(output_dir, BACKGROUND_FILENAME)
        save_background_gs(original_gaussians, final_mask_original, bg_path)

    # Step 6: render with explicit cameras or cached training cameras.
    render_dir = None
    preview_images = []
    mask_dir = None
    if not skip_render:
        _render_cams = _select_render_cameras(render_cameras, cached_train_cameras)
        render_dir, preview_images, mask_dir = _render_current_results(
            ctx, seg_path, _render_cams, output_dir,
            bg_path=bg_path, save_background=save_background,
            save_mask=True, show_previews=True,
            show_start_frame=first_frame_idx, show_frames=6
        )

    # Release segmentation memory.
    del gaussians
    if original_gaussians is not None:
        del original_gaussians
    del multiview_masks
    if multiview_masks_original:
        del multiview_masks_original
    _clear_cuda()

    return {
        'seg_path': seg_path,
        'bg_path': bg_path,
        'render_dir': render_dir,
        'mask_dir': mask_dir,
        'sam_mask_folder': sam_mask_folder,
        'preview_images': preview_images,
        'final_mask_count': len(final_mask),
        'total_points': len(xyz),
        'valid_masks_count': len(valid_masks),
        'removed_masks_count': len(removed_indices),
        'removed_mask_indices': removed_indices,
        'filtered_empty_frames': len(
            frame_camera_mapping['filtered_source_frame_indices']
        ),
        'filtered_source_frame_indices': frame_camera_mapping[
            'filtered_source_frame_indices'
        ],
        'filtered_train_camera_indices': frame_camera_mapping[
            'filtered_train_camera_indices'
        ],
        'filtered_sphere_camera_indices': frame_camera_mapping[
            'filtered_sphere_camera_indices'
        ],
        'frame_camera_mapping': frame_camera_mapping,
        'frame_mapping_path': frame_mapping_path,
    }


def apply_outlier_filter(
    ctx: SegmentationContext,
    save_dir: str,
    outlier_std_mul: float,
    cached_train_cameras: List,
    save_background: bool = True,
    render_cameras: Optional[List] = None,
) -> Optional[Dict[str, Any]]:
    """Apply geometric outlier filtering.

    Returns:
        Filter statistics, or ``None`` when filtering cannot be applied.
    """
    if not save_dir or not os.path.exists(save_dir):
        return None

    seg_path = os.path.join(save_dir, SEGMENT_FILENAME)
    if not os.path.exists(seg_path):
        return None

    # Load the current foreground model.
    gaussians = _load_gaussians(ctx, seg_path)

    # Filter geometric outliers from the foreground.
    xyz = gaussians.get_xyz
    all_indices = torch.arange(len(xyz), device=xyz.device)
    final_mask_filtered, removed_outliers, _ = robust_center_outlier_filter_indices(
        xyz, all_indices, std_mul=float(outlier_std_mul), sample_size=50000
    )

    # Abort when filtering would leave too few Gaussians.
    if final_mask_filtered.numel() < max(50, int(0.05 * all_indices.numel())):
        del gaussians
        _clear_cuda()
        return None

    # Save the filtered foreground model.
    save_gs(gaussians, final_mask_filtered, seg_path)

    # Rebuild the background by removing the filtered foreground.
    bg_path = os.path.join(save_dir, BACKGROUND_FILENAME)
    if save_background:
        original_gaussians = _load_gaussians(ctx)
        save_background_gs(original_gaussians, final_mask_filtered, bg_path)
        del original_gaussians

    # Release the model before rendering; the renderer reloads the PLY.
    del gaussians
    _clear_cuda()

    # Prefer explicit render cameras, then fall back to cached training cameras.
    _render_cams = _select_render_cameras(render_cameras, cached_train_cameras)
    _, preview_images, _ = _render_current_results(
        ctx, seg_path, _render_cams, save_dir,
        bg_path=bg_path, save_background=save_background
    )

    return {
        'original_points': all_indices.numel(),
        'filtered_points': final_mask_filtered.numel(),
        'removed_points': removed_outliers.numel(),
        'seg_path': seg_path,
        'preview_images': preview_images,
    }


def apply_boundary_compression(
    ctx: SegmentationContext,
    save_dir: str,
    min_compression: float,
    tolerance_ratio: float,
    cached_train_cameras: List,
    render_cameras: Optional[List] = None,
    mask_folder: Optional[str] = None,
    frame_mapping_path: Optional[str] = None,
    allow_legacy_positional_alignment: bool = False,
) -> Optional[Dict[str, Any]]:
    """Apply standalone Axis-aware Boundary Refinement (ABR).

    The method uses a saved ``segment.ply`` and SAM masks. It recovers the
    endpoints of each Gaussian ellipse from gsplat conics, then uses
    multi-view voting to compress axes that extend beyond the 2D boundaries.

    Args:
        mask_folder: Directory containing SAM masks. When omitted, search
            ``save_dir`` for a ``*_mask`` directory to support Gradio runs.
        frame_mapping_path: JSON manifest mapping renumbered SAM frames to
            their original cameras. New runs persist this automatically.
        allow_legacy_positional_alignment: Opt in to positional alignment for
            an audited legacy output that has no mapping manifest.

    Returns:
        Compression statistics, or ``None`` when ABR cannot be applied.
    """
    if not save_dir or not os.path.exists(save_dir):
        return None

    seg_path = os.path.join(save_dir, SEGMENT_FILENAME)
    if not os.path.exists(seg_path):
        return None

    # Resolve the manifest before searching for masks.  New manifests record
    # the exact mask directory, which avoids selecting another round's masks
    # when a run directory contains more than one ``*_mask`` folder.
    resolved_mapping_path = frame_mapping_path
    mapping_was_explicit = resolved_mapping_path is not None
    if resolved_mapping_path and not os.path.isabs(resolved_mapping_path):
        resolved_mapping_path = os.path.join(save_dir, resolved_mapping_path)
    if resolved_mapping_path is None:
        latest_mapping_path = os.path.join(
            save_dir, LATEST_FRAME_MAPPING_FILENAME
        )
        if os.path.exists(latest_mapping_path):
            resolved_mapping_path = latest_mapping_path

    candidate_mapping = None
    recorded_mask_folder = None
    if resolved_mapping_path and os.path.exists(resolved_mapping_path):
        candidate_mapping = load_frame_camera_mapping(resolved_mapping_path)
        recorded_mask_folder = candidate_mapping.get('mask_folder')
        if recorded_mask_folder and not os.path.isabs(recorded_mask_folder):
            recorded_mask_folder = os.path.join(save_dir, recorded_mask_folder)
        if (
            (not mask_folder or not os.path.exists(mask_folder))
            and recorded_mask_folder
            and os.path.exists(recorded_mask_folder)
        ):
            mask_folder = recorded_mask_folder
    elif mapping_was_explicit:
        print(
            f"[ABR][ERROR] Frame-camera mapping not found: "
            f"{resolved_mapping_path}"
        )
        return None

    # Search for an interactive-run mask directory when none was provided.
    if not mask_folder or not os.path.exists(mask_folder):
        mask_folder = None
        for entry in os.listdir(save_dir):
            sub = os.path.join(save_dir, entry)
            if os.path.isdir(sub):
                for sub_entry in os.listdir(sub):
                    if sub_entry.endswith('_mask') and os.path.isdir(os.path.join(sub, sub_entry)):
                        mask_folder = os.path.join(sub, sub_entry)
                        break
            if mask_folder:
                break

    if not mask_folder or not os.path.exists(mask_folder):
        print("[ABR] No mask directory found; skipping ABR")
        return None

    print(f"[ABR] Using mask directory: {mask_folder}")

    frame_camera_mapping = None
    if candidate_mapping is not None:
        mapping_matches_masks = (
            not recorded_mask_folder
            or os.path.abspath(recorded_mask_folder) == os.path.abspath(mask_folder)
        )
        if not mapping_matches_masks:
            print(
                "[ABR][ERROR] Frame-camera mapping and mask directory do "
                "not belong to the same segmentation round; skipping ABR"
            )
            return None
        frame_camera_mapping = candidate_mapping

    if (
        frame_camera_mapping is None
        and not allow_legacy_positional_alignment
    ):
        print(
            "[ABR][ERROR] No frame-camera mapping is available. Refusing "
            "legacy positional alignment because filtered training frames "
            "cannot be ruled out."
        )
        return None

    # Load the current foreground model.
    gaussians = _load_gaussians(ctx, seg_path)
    num_points = len(gaussians.get_xyz)

    # Guard against empty PLY files, which can crash the gsplat CUDA kernel.
    if num_points == 0:
        print("[ABR] segment.ply contains no Gaussians; skipping ABR")
        del gaussians
        _clear_cuda()
        return None

    # Align SAM masks with cached training cameras using the explicit mapping
    # persisted by run_segmentation().  Positional alignment is available only
    # through an explicit opt-in for an independently audited legacy output.
    masks_dict = load_masks_from_folder(mask_folder)
    if frame_camera_mapping is not None:
        valid_masks_2d, valid_cams = collect_mapped_train_masks(
            cached_train_cameras, masks_dict, frame_camera_mapping
        )
        print(
            f"[ABR] Explicitly aligned {len(valid_masks_2d)} masks to "
            "training cameras"
        )
    else:
        print(
            "[ABR][WARN] Using explicitly enabled legacy positional alignment"
        )
        valid_masks_2d, valid_cams = _collect_valid_masks(
            cached_train_cameras, masks_dict
        )

    if len(valid_masks_2d) == 0:
        del gaussians
        _clear_cuda()
        return None

    # Render gsplat conics for each valid view.
    meta_list = []
    with torch.no_grad():
        for cam in valid_cams:
            _, _, meta = render_gsplat_camera(
                gaussians, cam, cam.image_width, cam.image_height,
                backgrounds=ctx.background
            )
            meta_list.append(meta)

    # Keep the 3D data required for anisotropic compression on the GPU.
    # Quaternion order is (w, x, y, z); scales are already activated.
    rotations = gaussians.get_rotation.detach()   # [N, 4] GPU tensor
    scales_3d = gaussians.get_scaling.detach()     # [N, 3] GPU tensor
    compute_device = rotations.device

    # Build world-to-camera matrices and intrinsics for valid views.
    view_matrices = []
    intrinsics_list = []
    for cam in valid_cams:
        # Construct the 4x4 view matrix.
        R_w2c = torch.tensor(cam.R, dtype=torch.float32, device=compute_device).T
        t_w2c = torch.tensor(cam.T, dtype=torch.float32, device=compute_device)
        W = torch.eye(4, dtype=torch.float32, device=compute_device)
        W[:3, :3] = R_w2c
        W[:3, 3] = t_w2c
        view_matrices.append(W)

        # Camera intrinsics.
        fx = cam.image_width / (2.0 * np.tan(cam.FoVx / 2.0))
        fy = cam.image_height / (2.0 * np.tan(cam.FoVy / 2.0))
        cx = cam.image_width / 2.0
        cy = cam.image_height / 2.0
        intrinsics_list.append((fx, fy, cx, cy))

    # Convert masks once instead of inside the compression loop.
    valid_masks_torch = [torch.from_numpy(m) for m in valid_masks_2d]

    # Compute per-axis compression factors on the GPU.
    per_axis_factors = compute_boundary_compression_axes(
        num_points, meta_list, valid_masks_torch,
        rotations=rotations,
        scales_3d=scales_3d,
        view_matrices=view_matrices,
        intrinsics_list=intrinsics_list,
        min_compression=min_compression,
        tolerance_ratio=tolerance_ratio,
        device=compute_device,
    )

    # Apply per-axis factors to log-space Gaussian scales.
    compress_mask = (per_axis_factors < 1.0).any(dim=1)  # [N]
    n_compressed = int(compress_mask.sum())

    if n_compressed > 0:
        print(
            f"[ABR] Compressed {n_compressed} boundary-overflowing Gaussians "
            f"(minimum factor: {min_compression}, tolerance: {tolerance_ratio})"
        )
        log_factors = torch.log(
            per_axis_factors[compress_mask].clamp(min=1e-6)
        ).to(gaussians._scaling.device)  # [K, 3]
        compress_indices = torch.where(compress_mask)[0].to(gaussians._scaling.device)
        gaussians._scaling.data[compress_indices] += log_factors

    # Release intermediate GPU data before rendering.
    del meta_list, rotations, scales_3d, view_matrices, valid_masks_torch, per_axis_factors
    _clear_cuda()

    # Save the refined model.
    all_indices_tensor = torch.arange(num_points, device=gaussians.get_xyz.device)
    save_gs(gaussians, all_indices_tensor, seg_path)

    # Release the model before rendering; the renderer reloads the PLY.
    del gaussians
    _clear_cuda()

    # Prefer explicit render cameras, then fall back to cached training cameras.
    _render_cams = _select_render_cameras(render_cameras, cached_train_cameras)
    _, preview_images, _ = _render_current_results(ctx, seg_path, _render_cams, save_dir)

    return {
        'original_points': num_points,
        'compressed_points': n_compressed,
        'aligned_masks': len(valid_masks_2d),
        'used_explicit_frame_mapping': frame_camera_mapping is not None,
        'frame_mapping_file': (
            os.path.basename(resolved_mapping_path)
            if frame_camera_mapping is not None else None
        ),
        'seg_path': seg_path,
        'preview_images': preview_images,
    }


def update_threshold(
    ctx: SegmentationContext,
    save_dir: str,
    new_threshold: float,
    cached_train_cameras: List,
    save_background: bool = True,
    render_cameras: Optional[List] = None,
) -> Optional[Dict[str, Any]]:
    """Recompute segmentation with a new voting threshold.

    Returns:
        Updated statistics, or ``None`` when cached votes are unavailable.
    """
    masks_data_path = os.path.join(save_dir, MASKS_FILENAME)
    params_json_path = os.path.join(save_dir, PARAMS_FILENAME)

    if not os.path.exists(masks_data_path):
        return None

    # Load compressed multi-view votes.
    with open(masks_data_path, 'rb') as f:
        data = pickle.load(f)

    mm = torch.from_numpy(data['mmasks'].astype(np.int8))
    multiview_masks_list = list(mm.split(1, dim=1))
    num_points = mm.shape[0]

    # Recompute foreground votes.
    _, final_mask = ensemble(multiview_masks_list, threshold=new_threshold)

    # Recompute votes against the original model for background export.
    final_mask_original = None
    if save_background and 'mmasks_original' in data and data['mmasks_original'] is not None:
        mm_orig = torch.from_numpy(data['mmasks_original'].astype(np.int8))
        multiview_masks_original_list = list(mm_orig.split(1, dim=1))
        _, final_mask_original = ensemble(multiview_masks_original_list, threshold=new_threshold)

    # Load parameters to identify the source model.
    with open(params_json_path, 'r', encoding='utf-8') as f:
        params = json.load(f)

    # Multi-view votes from Round 2 index the subset model. Reload exactly
    # that model to keep vote indices aligned.
    foreground_model_path = data.get('foreground_model_path')
    if foreground_model_path and os.path.exists(foreground_model_path):
        print(f"[update_threshold] Loading foreground vote source: {foreground_model_path}")
    else:
        foreground_model_path = _get_original_ply_path(ctx)
        print(f"[update_threshold] Falling back to the original model: {foreground_model_path}")
    gaussians = _load_gaussians(ctx, foreground_model_path)

    original_gaussians = None
    if save_background:
        original_gaussians = _load_gaussians(ctx)

    # Save the updated segmentation.
    seg_path = os.path.join(save_dir, SEGMENT_FILENAME)
    save_gs(gaussians, final_mask, seg_path)

    bg_path = None
    if save_background and final_mask_original is not None and original_gaussians is not None:
        bg_path = os.path.join(save_dir, BACKGROUND_FILENAME)
        save_background_gs(original_gaussians, final_mask_original, bg_path)
        del original_gaussians
        original_gaussians = None

    # Release models before rendering; the renderer reloads the PLY files.
    if original_gaussians is not None:
        del original_gaussians
    del gaussians
    _clear_cuda()

    # Prefer explicit render cameras, then fall back to cached training cameras.
    _render_cams = _select_render_cameras(render_cameras, cached_train_cameras)
    _, preview_images, _ = _render_current_results(
        ctx, seg_path, _render_cams, save_dir,
        bg_path=bg_path, save_background=save_background
    )

    # Persist the new threshold.
    params['threshold'] = new_threshold
    params['save_background'] = save_background
    with open(params_json_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    return {
        'seg_path': seg_path,
        'bg_path': bg_path,
        'preview_images': preview_images,
        'final_mask_count': len(final_mask),
        'num_points': num_points,
    }


# ============================================================
# Automatic view-coverage analysis
# ============================================================

def _fibonacci_sphere(n: int) -> np.ndarray:
    """Sample ``n`` unit vectors with a Fibonacci lattice.

    Returns:
        An ``[n, 3]`` float64 array of unit direction vectors.
    """
    indices = np.arange(n, dtype=np.float64)
    phi = np.arccos(1 - 2 * (indices + 0.5) / n)
    theta = np.pi * (1 + np.sqrt(5)) * indices
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=-1)


def compute_view_coverage(
    segment_ply_path: str,
    train_cameras: List,
    sh_degree: int,
    min_visible_ratio: float = VIEW_COVERAGE_MIN_VISIBLE_RATIO,
    angular_gap_threshold: float = 90.0,
    n_probe_points: int = 2000,
) -> Dict[str, Any]:
    """Measure how evenly the training cameras surround the foreground.

    The check has two stages:
      1. Project the coarse foreground into every training view. A camera is
         valid when its visible foreground ratio reaches ``min_visible_ratio``.
      2. Sample directions around the foreground center and measure the angle
         to the nearest valid camera. The largest value is the angular gap.

    Args:
        segment_ply_path: Path to the coarse ``segment.ply``.
        train_cameras: Training cameras.
        sh_degree: Spherical-harmonic degree.
        min_visible_ratio: Minimum visible foreground ratio for a valid view.
        angular_gap_threshold: Maximum acceptable angular gap in degrees.
        n_probe_points: Number of spherical probe directions.

    Returns:
        Coverage statistics, including camera counts, angular gaps, whether
        spherical sampling is needed, and valid camera indices.
    """
    from seg_utils.ensemble_utils import project_to_2d
    from seg_utils.post_filter import compute_robust_sphere_center

    # Load the coarse foreground.
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(segment_ply_path)
    xyz = gaussians.get_xyz  # [N, 3] GPU tensor
    n_fg = len(xyz)

    # Homogeneous coordinates, [N, 4].
    xyz_pad = torch.nn.functional.pad(xyz, (0, 1), mode='constant', value=1)

    # ----------------------------------------------------------------
    # Step 1: reject cameras with insufficient foreground visibility.
    # ----------------------------------------------------------------
    per_camera_visible_ratio = []
    valid_camera_indices = []

    for cam_idx, cam in enumerate(train_cameras):
        w2c_matrix = cam.world_view_transform  # [4, 4]
        h = cam.image_height
        w = cam.image_width

        # Positive depth indicates a point in front of the camera.
        p_cam = (xyz_pad @ w2c_matrix).transpose(0, 1)  # [4, N]
        valid_depth = p_cam[2, :] > 0

        # Project into pixel coordinates.
        point_image = project_to_2d(cam, xyz_pad)
        point_image = point_image.long()
        valid_x = (point_image[:, 0] >= 0) & (point_image[:, 0] < w)
        valid_y = (point_image[:, 1] >= 0) & (point_image[:, 1] < h)
        visible = valid_x & valid_y & valid_depth

        ratio = float(visible.sum()) / max(1, n_fg)
        per_camera_visible_ratio.append(ratio)

        if ratio >= min_visible_ratio:
            valid_camera_indices.append(cam_idx)

    n_total = len(train_cameras)
    n_valid = len(valid_camera_indices)
    n_invalid = n_total - n_valid

    print(
        f"[ViewCoverage] Cameras: {n_total} total, {n_valid} valid, "
        f"{n_invalid} invalid (visibility threshold: {min_visible_ratio * 100:.0f}%)"
    )

    # ----------------------------------------------------------------
    # Step 2: compute the maximum angular gap.
    # ----------------------------------------------------------------
    # Estimate a robust foreground center.
    obj_center = compute_robust_sphere_center(xyz)  # [3] GPU tensor
    obj_center_np = obj_center.detach().cpu().numpy()

    # Release model memory.
    del gaussians, xyz, xyz_pad
    _clear_cuda()

    if n_valid == 0:
        print("[ViewCoverage] No valid cameras; spherical sampling is required")
        return {
            'total_cameras': n_total,
            'valid_cameras': 0,
            'invalid_cameras': n_invalid,
            'min_visible_ratio': float(min_visible_ratio),
            'max_angular_gap_deg': 360.0,
            'mean_angular_gap_deg': 360.0,
            'need_sphere_sampling': True,
            'per_camera_visible_ratio': per_camera_visible_ratio,
            'valid_camera_indices': valid_camera_indices,
            'object_center': obj_center_np.tolist(),
        }

    # Unit directions from the object center to valid cameras.
    valid_directions = []
    for cam_idx in valid_camera_indices:
        cam = train_cameras[cam_idx]
        cam_pos = cam.camera_center.detach().cpu().numpy().reshape(3)
        direction = cam_pos - obj_center_np
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            valid_directions.append(direction / norm)
    valid_directions = np.array(valid_directions)  # [V, 3]

    if len(valid_directions) == 0:
        return {
            'total_cameras': n_total,
            'valid_cameras': 0,
            'invalid_cameras': n_invalid,
            'min_visible_ratio': float(min_visible_ratio),
            'max_angular_gap_deg': 360.0,
            'mean_angular_gap_deg': 360.0,
            'need_sphere_sampling': True,
            'per_camera_visible_ratio': per_camera_visible_ratio,
            'valid_camera_indices': valid_camera_indices,
            'object_center': obj_center_np.tolist(),
        }

    # Uniformly sample spherical probe directions.
    probe_points = _fibonacci_sphere(n_probe_points)  # [P, 3]

    # Angular distance from each probe to its nearest valid camera.
    # Dot product: [P, V] = probe_points @ valid_directions.T.
    dots = probe_points @ valid_directions.T  # [P, V]
    dots = np.clip(dots, -1.0, 1.0)
    min_angles = np.arccos(dots.max(axis=1))  # [P], radians
    min_angles_deg = np.degrees(min_angles)

    max_angular_gap = float(min_angles_deg.max())
    mean_angular_gap = float(min_angles_deg.mean())
    need_sphere = max_angular_gap > angular_gap_threshold

    status = "required" if need_sphere else "not required"
    print(
        f"[ViewCoverage] Maximum angular gap: {max_angular_gap:.1f} deg, "
        f"mean: {mean_angular_gap:.1f} deg, "
        f"threshold: {angular_gap_threshold:.0f} deg; "
        f"spherical sampling is {status}"
    )

    return {
        'total_cameras': n_total,
        'valid_cameras': n_valid,
        'invalid_cameras': n_invalid,
        'min_visible_ratio': float(min_visible_ratio),
        'max_angular_gap_deg': max_angular_gap,
        'mean_angular_gap_deg': mean_angular_gap,
        'need_sphere_sampling': need_sphere,
        'per_camera_visible_ratio': per_camera_visible_ratio,
        'valid_camera_indices': valid_camera_indices,
        'object_center': obj_center_np.tolist(),
    }


# ============================================================
# Headless two-round segmentation entry point for batch use.
# ============================================================

def run_two_round_pipeline(
    ctx: SegmentationContext,
    cached_train_cameras: List,
    first_frame_idx: int,
    fg_points: List,
    bg_points: List,
    box: Optional[List],
    text: Optional[str],
    threshold_round1: float,
    threshold_round2: float,
    save_background: bool,
    n_layers: int,
    n_points_per_layer: int,
    angular_gap_threshold: float,
    skip_round2_if_covered: bool,
    object_name: str,
    api_url: str,
    output_dir: str,
    boundary_refine: bool = False,
    min_compression: float = 0.2,
    tolerance_ratio: float = 0.2,
    render_cameras: Optional[List] = None,
    skip_render: bool = False,
    pre_mask_folder_round1: Optional[str] = None,
    temp_frames_folder: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the complete two-round segmentation pipeline without a UI.

    This entry point shares segmentation logic with the Gradio application but
    omits UI state and progress yields, making it suitable for batch runs.

    Workflow:
      Round 1: training views -> SAM -> voting -> ``coarse.ply``
      Coverage: ``compute_view_coverage(angular_gap_threshold)``
      Round 2: training and optional spherical views -> ``fine.ply``
      Optional ABR boundary refinement -> ``compression.ply``
      ``segment.ply`` always contains the latest result.

    Args:
        ctx: Segmentation context.
        cached_train_cameras: Training cameras.
        first_frame_idx: Annotated frame index.
        fg_points: Foreground point prompts.
        bg_points: Background point prompts.
        box: Optional box prompt.
        text: Optional text prompt.
        threshold_round1: Round 1 voting threshold.
        threshold_round2: Round 2 voting threshold.
        save_background: Whether to save the complementary background.
        n_layers: Number of spherical sampling layers.
        n_points_per_layer: Points per spherical sampling layer.
        angular_gap_threshold: Coverage threshold in degrees.
        skip_round2_if_covered: Skip Round 2 when training views suffice.
        object_name: Object name used for outputs and SAM requests.
        api_url: SAM API base URL, for example ``http://localhost:8000``.
        output_dir: Full output directory without a timestamp.
        boundary_refine: Whether to apply ABR.
        min_compression: Minimum ABR compression factor.
        tolerance_ratio: ABR boundary tolerance.
        render_cameras: Cameras for final rendering; defaults to training views.
        skip_render: Skip final rendering for faster batch execution.
        pre_mask_folder_round1: Optional precomputed Round 1 masks.
        temp_frames_folder: Shared render-frame cache for both rounds.

    Returns:
        Paths, coverage statistics, per-round summaries, serialized
        parameters, and the current stage name.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = ctx.model_path
    if temp_frames_folder is None:
        temp_frames_folder = os.path.join(model_path, "temp_frames")
    round2_result = None
    abr_result = None

    # Prepare Round 1 frames. A previous object's Round 2 may have added
    # spherical frames, so rebuild unless the cache exactly matches training.
    n_expected = len(cached_train_cameras)
    existing_count = len([f for f in os.listdir(temp_frames_folder)
                          if f.endswith('.jpg')]) if os.path.exists(temp_frames_folder) else 0
    if existing_count != n_expected:
        print(
            f"[Pipeline] Frame cache mismatch ({existing_count} found, "
            f"{n_expected} expected); rendering training views"
        )
        load_views(
            ctx=ctx,
            use_train_views=True,
            use_random_views=False,
            n_layers=0, n_points_per_layer=0,
            prev_seg_path=None,
            cached_train_cameras=cached_train_cameras,
            output_folder=temp_frames_folder,
            first_frame_idx=first_frame_idx,
        )
    else:
        print(f"[Pipeline] Reusing {n_expected} cached training frames")

    # Round 1: training views only.
    print("\n[Pipeline] ==================== Round 1: coarse segmentation ====================")

    round1_result = run_segmentation(
        ctx=ctx,
        all_cameras=cached_train_cameras,
        cached_train_cameras=cached_train_cameras,
        first_frame_idx=first_frame_idx,
        fg_points=fg_points, bg_points=bg_points, box=box,
        text=text, threshold=threshold_round1,
        save_background=save_background,
        object_name=object_name,
        api_url=api_url,
        output_dir=output_dir,
        prev_seg_path=None,
        temp_frames_folder=temp_frames_folder,
        render_cameras=render_cameras,
        skip_render=True,  # Render only after the final stage is known.
        pre_mask_folder=pre_mask_folder_round1,
        stage_name="round1",
    )

    round1_seg_path = round1_result['seg_path']
    round1_info = _summarize_segmentation_result(round1_result)
    print(
        f"[Pipeline] Round 1: foreground "
        f"{round1_info['final_mask_count']}/{round1_info['total_points']} "
        f"({round1_info['final_mask_count'] / max(1, round1_info['total_points']) * 100:.1f}%), "
        f"{round1_info['valid_masks_count']} valid masks"
    )

    # Stop early when Round 1 returns an empty foreground.
    if round1_info['final_mask_count'] == 0:
        print("[Pipeline][WARN] Round 1 produced no foreground; skipping later stages")
        segment_path = os.path.join(output_dir, SEGMENT_FILENAME)
        # run_segmentation() has already created an empty segment.ply.
        params = {
            'model_path': model_path,
            'source_path': ctx.source_path,
            'object_name': object_name,
            'error': 'Round 1 produced 0 foreground points',
            'current_stage': 'failed',
        }
        with open(os.path.join(output_dir, PARAMS_FILENAME), 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
        return {
            'seg_path': segment_path,
            'coarse_path': None,
            'fine_path': None,
            'coverage': None,
            'round1_info': round1_info,
            'round2_info': None,
            'params': params,
            'current_stage': 'failed',
        }

    # Analyze training-view coverage.
    print("\n[Pipeline] ==================== Coverage analysis ====================")

    coverage = compute_view_coverage(
        segment_ply_path=round1_seg_path,
        train_cameras=list(cached_train_cameras),
        sh_degree=ctx.sh_degree,
        min_visible_ratio=VIEW_COVERAGE_MIN_VISIBLE_RATIO,
        angular_gap_threshold=angular_gap_threshold,
    )

    need_sphere = coverage['need_sphere_sampling']

    # Decide whether to run Round 2.
    coarse_path = os.path.join(output_dir, COARSE_FILENAME)
    segment_path = os.path.join(output_dir, SEGMENT_FILENAME)
    fine_path = None
    round2_info = None
    current_stage = 'coarse'

    if not need_sphere and skip_round2_if_covered:
        print("[Pipeline] Training-view coverage is sufficient; skipping Round 2")
        shutil.copy2(round1_seg_path, coarse_path)
        if round1_seg_path != segment_path:
            shutil.copy2(round1_seg_path, segment_path)
    else:
        r2_mode = "training + spherical views" if need_sphere else "training views only"
        print(
            f"\n[Pipeline] ==================== Round 2: fine segmentation "
            f"({r2_mode}) ===================="
        )

        # Preserve the Round 1 result.
        os.rename(round1_seg_path, coarse_path)

        # Load Round 2 views.
        all_cameras_r2, _, _ = load_views(
            ctx=ctx,
            use_train_views=True,
            use_random_views=need_sphere,
            n_layers=n_layers,
            n_points_per_layer=n_points_per_layer,
            prev_seg_path=coarse_path,
            cached_train_cameras=cached_train_cameras,
            output_folder=temp_frames_folder,
            first_frame_idx=first_frame_idx,
        )

        round2_result = run_segmentation(
            ctx=ctx,
            all_cameras=all_cameras_r2,
            cached_train_cameras=cached_train_cameras,
            first_frame_idx=first_frame_idx,
            fg_points=fg_points, bg_points=bg_points, box=box,
            text=text, threshold=threshold_round2,
            save_background=save_background,
            object_name=object_name,
            api_url=api_url,
            output_dir=output_dir,
            prev_seg_path=coarse_path,
            temp_frames_folder=temp_frames_folder,
            render_cameras=render_cameras,
            skip_render=True,  # Render after optional ABR.
            stage_name="round2",
        )

        round2_seg_path = round2_result.get('seg_path')
        fine_path = os.path.join(output_dir, FINE_FILENAME)
        shutil.copy2(round2_seg_path, fine_path)
        if round2_seg_path != segment_path:
            shutil.copy2(round2_seg_path, segment_path)

        round2_info = _summarize_segmentation_result(round2_result)
        round2_info.update({
            'views': 'train+sphere' if need_sphere else 'train_only',
            'num_sphere_cameras': len(all_cameras_r2) - len(cached_train_cameras) if need_sphere else 0,
        })
        current_stage = 'fine'
        print(
            f"[Pipeline] Round 2: foreground "
            f"{round2_info['final_mask_count']}/{round2_info['total_points']} "
            f"({round2_info['final_mask_count'] / max(1, round2_info['total_points']) * 100:.1f}%), "
            f"{round2_info['valid_masks_count']} valid masks"
        )

    # Apply optional ABR boundary refinement.
    if boundary_refine:
        print("\n[Pipeline] ==================== ABR boundary refinement ====================")
        # Release residual GPU allocations before gsplat rendering.
        _clear_cuda()
        # Use masks from the latest completed segmentation round.
        abr_mask_folder = (round2_result.get('sam_mask_folder') if round2_result is not None
                           else round1_result['sam_mask_folder'])
        try:
            abr_result = apply_boundary_compression(
                ctx=ctx,
                save_dir=output_dir,
                min_compression=min_compression,
                tolerance_ratio=tolerance_ratio,
                cached_train_cameras=cached_train_cameras,
                render_cameras=render_cameras,
                mask_folder=abr_mask_folder,
                frame_mapping_path=(
                    round2_result['frame_mapping_path']
                    if round2_result is not None
                    else round1_result['frame_mapping_path']
                ),
            )
        except RuntimeError as e:
            print(f"[Pipeline][WARN] ABR rendering failed: {e}")
            print("[Pipeline] Skipping ABR and retaining the current segment.ply")
            abr_result = None
        if abr_result is not None:
            compression_path = os.path.join(output_dir, COMPRESSION_FILENAME)
            shutil.copy2(abr_result['seg_path'], compression_path)
            current_stage = 'compression'
            print(
                f"[Pipeline] ABR compressed {abr_result['compressed_points']}/"
                f"{abr_result['original_points']} Gaussians"
            )
        else:
            print("[Pipeline] ABR skipped because masks were unavailable or rendering failed")

    # Render the final result.
    if not skip_render:
        print("\n[Pipeline] Rendering final result")
        _render_cams = _select_render_cameras(render_cameras, cached_train_cameras)
        _render_current_results(
            ctx, segment_path, _render_cams, output_dir,
            save_mask=True, show_previews=False
        )

    # Save the complete parameter record.
    params = _build_segmentation_params(
        model_path=model_path,
        source_path=ctx.source_path,
        object_name=object_name,
        first_frame_idx=first_frame_idx,
        fg_points=fg_points,
        bg_points=bg_points,
        box=box,
        text=text,
        threshold_round1=threshold_round1,
        threshold_round2=threshold_round2,
        save_background=save_background,
        cached_train_cameras=cached_train_cameras,
        round1_info=round1_info,
        coverage=coverage,
        need_sphere=need_sphere,
        round2_info=round2_info,
        n_layers=n_layers,
        n_points_per_layer=n_points_per_layer,
        angular_gap_threshold=angular_gap_threshold,
        boundary_refine=boundary_refine,
        current_stage=current_stage,
        abr_result=abr_result,
        min_compression=min_compression,
        tolerance_ratio=tolerance_ratio,
    )

    with open(os.path.join(output_dir, PARAMS_FILENAME), 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    print(f"\n[Pipeline] Completed: {output_dir} (stage: {current_stage})")

    return {
        'seg_path': segment_path,
        'coarse_path': coarse_path,
        'fine_path': fine_path,
        'coverage': coverage,
        'round1_info': round1_info,
        'round2_info': round2_info,
        'params': params,
        'current_stage': current_stage,
    }
