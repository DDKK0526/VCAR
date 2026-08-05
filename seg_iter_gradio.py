"""
Gradio interface for interactive SAM-guided 3DGS segmentation.

Tab [1]: view loading and annotation
Tab [2]: segmentation results and post-processing
"""

import json
import os
import shutil
import cv2
import gradio as gr
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw

from seg_utils.pipeline import (
    create_segmentation_context, load_views, run_segmentation,
    apply_outlier_filter, apply_boundary_compression, update_threshold,
    compute_view_coverage, VIEW_COVERAGE_MIN_VISIBLE_RATIO
)
from seg_utils.general_utils import render_segmented_results


def _get_default_model_path():
    """Return the optional default model path configured for the UI."""
    return os.environ.get("VCAR_DEFAULT_MODEL_PATH", "")


def _get_default_source_path():
    """Return the optional 3DGS training-data path configured for the UI."""
    return os.environ.get("VCAR_DEFAULT_SOURCE_PATH", "")


def _get_allowed_paths():
    """Return Gradio allowed paths from VCAR_ALLOWED_PATHS or the project root."""
    configured = os.environ.get("VCAR_ALLOWED_PATHS")
    if configured:
        return [p for p in configured.split(os.pathsep) if p]
    return [os.getcwd()]


# ==================== View loading ====================

def load_views_for_annotation(
    model_path,
    source_path,
    cached_train_cameras,
    frame_choice=None,
):
    """Load training views for annotation-frame selection."""
    first_frame_idx = 0
    if frame_choice is not None:
        try:
            first_frame_idx = int(frame_choice.split()[1])
        except (IndexError, ValueError, AttributeError):
            first_frame_idx = 0

    ctx = create_segmentation_context(model_path, source_path or None)
    temp_frames_folder = os.path.join(model_path, "temp_frames")

    all_cameras, cached_train_cameras, test_cameras = load_views(
        ctx=ctx,
        use_train_views=True,
        use_random_views=False,
        n_layers=0,
        n_points_per_layer=0,
        prev_seg_path=None,
        cached_train_cameras=cached_train_cameras,
        output_folder=temp_frames_folder,
        first_frame_idx=first_frame_idx
    )

    num_frames = len(all_cameras)
    frame_choices = [f"Frame {i}" for i in range(num_frames)]
    status = (
        f"Loaded {num_frames} training views "
        f"(test views: {len(test_cameras)})"
    )

    return gr.update(choices=frame_choices, value=frame_choices[0] if frame_choices else None), \
           status, all_cameras, cached_train_cameras, test_cameras


# ==================== Frame loading ====================

def get_frame_by_idx(model_path, frame_idx):
    """Load a rendered frame by index."""
    temp_frames_folder = os.path.join(model_path, "temp_frames")
    if not os.path.exists(temp_frames_folder):
        return None, "Load the views first"

    filename = f"{frame_idx:05d}.jpg"
    filepath = os.path.join(temp_frames_folder, filename)
    if not os.path.exists(filepath):
        return None, f"Invalid frame index: {frame_idx}"

    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, f"Current frame: {frame_idx}"


# ==================== Annotation handling ====================

def draw_annotations_on_image(image, fg_points, bg_points, box):
    """Draw point and box annotations on an image."""
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)

    for pt in fg_points:
        x, y = pt
        r = 8
        draw.ellipse([x - r, y - r, x + r, y + r], fill='green', outline='white', width=2)

    for pt in bg_points:
        x, y = pt
        r = 8
        draw.ellipse([x - r, y - r, x + r, y + r], fill='red', outline='white', width=2)

    if box is not None:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline='blue', width=3)

    return np.array(img_pil)


def handle_image_click(image, fg_points, bg_points, box, point_mode, evt: gr.SelectData):
    """Handle an annotation click on the image."""
    x, y = evt.index[0], evt.index[1]

    if point_mode == "Foreground points":
        fg_points.append([x, y])
    elif point_mode == "Background points":
        bg_points.append([x, y])
    elif point_mode == "Box annotation - start":
        box = [x, y, x, y]
    elif point_mode == "Box annotation - end" and box is not None:
        box[2], box[3] = x, y

    annotated_img = draw_annotations_on_image(image, fg_points, bg_points, box)

    info = f"Foreground points ({len(fg_points)}): {fg_points}\n"
    info += f"Background points ({len(bg_points)}): {bg_points}\n"
    if box:
        info += f"Bounding box: {box}"

    return annotated_img, fg_points, bg_points, box, info


def clear_annotations():
    """Clear all annotations."""
    return [], [], None, "Annotations cleared"


# ==================== Segmentation workflow ====================

def _merge_render_cameras(cached_train_cameras, cached_test_cameras):
    """Combine training and test cameras for result rendering."""
    if cached_test_cameras:
        return list(cached_train_cameras) + list(cached_test_cameras)
    return cached_train_cameras


def _frame_alignment_metadata(segmentation_result):
    """Return the persisted frame-alignment fields for one SAM round."""
    mapping_path = segmentation_result.get('frame_mapping_path')
    return {
        'removed_low_quality_masks': segmentation_result.get(
            'removed_masks_count', 0
        ),
        'removed_low_quality_mask_indices': segmentation_result.get(
            'removed_mask_indices', []
        ),
        'filtered_empty_frames': segmentation_result.get(
            'filtered_empty_frames', 0
        ),
        'filtered_source_frame_indices': segmentation_result.get(
            'filtered_source_frame_indices', []
        ),
        'filtered_train_camera_indices': segmentation_result.get(
            'filtered_train_camera_indices', []
        ),
        'filtered_sphere_camera_indices': segmentation_result.get(
            'filtered_sphere_camera_indices', []
        ),
        'frame_mapping_file': (
            os.path.basename(mapping_path) if mapping_path else None
        ),
    }


def segment_3dgs_two_round_pipeline(
    model_path, source_path, cached_train_cameras, cached_test_cameras,
    first_frame_idx,
    fg_points, bg_points, box, text,
    threshold_round1, threshold_round2, save_background,
    n_layers, n_points_per_layer,
    skip_round2_if_covered, angular_gap_threshold,
    object_name="object", api_url_base="http://localhost:8000"
):
    """Run the two-round automatic segmentation workflow with progress yields.

    Round 1: training views -> SAM -> voting (threshold_round1)
    Coverage test: compute_view_coverage(angular_gap_threshold)
    Round 2: conditional training+spherical views -> SAM -> voting
    """
    if text is None and cached_train_cameras is None:
        yield "Error: training cameras are not loaded", None, [], None, None
        return

    ctx = create_segmentation_context(model_path, source_path or None)
    render_cameras = _merge_render_cameras(cached_train_cameras, cached_test_cameras)

    # Create a timestamped output directory.
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    save_dir = os.path.join(model_path, f"{object_name}-{timestamp}")

    # ==================== Round 1: training views ====================
    yield "[Round 1] Running coarse segmentation on training views...", None, [], None, None

    round1_result = run_segmentation(
        ctx=ctx,
        all_cameras=cached_train_cameras,
        cached_train_cameras=cached_train_cameras,
        first_frame_idx=first_frame_idx,
        fg_points=fg_points, bg_points=bg_points, box=box,
        text=text, threshold=threshold_round1,
        save_background=save_background,
        object_name=object_name,
        api_url=api_url_base,
        output_dir=save_dir,
        prev_seg_path=None,
        render_cameras=render_cameras,
        skip_render=True,
        stage_name="round1",
    )

    round1_seg_path = round1_result['seg_path']
    round1_info = (
        f"Foreground: {round1_result['final_mask_count']}/{round1_result['total_points']} "
        f"({round1_result['final_mask_count']/round1_result['total_points']*100:.1f}%), "
        f"valid masks: {round1_result['valid_masks_count']}"
    )

    # ==================== Coverage analysis ====================
    yield (
        f"[Coverage] {round1_info}; analyzing angular view coverage...",
        None,
        [],
        None,
        None,
    )

    coverage = compute_view_coverage(
        segment_ply_path=round1_seg_path,
        train_cameras=list(cached_train_cameras),
        sh_degree=ctx.sh_degree,
        min_visible_ratio=VIEW_COVERAGE_MIN_VISIBLE_RATIO,
        angular_gap_threshold=angular_gap_threshold,
    )

    coverage_info = (
        f"valid cameras: {coverage['valid_cameras']}/{coverage['total_cameras']}, "
        f"maximum angular gap: {coverage['max_angular_gap_deg']:.1f}°"
    )
    need_sphere = coverage['need_sphere_sampling']

    # ==================== Round 2 decision ====================
    if not need_sphere and skip_round2_if_covered:
        # Skip Round 2 and preserve the stage outputs.
        coarse_path = os.path.join(save_dir, 'coarse.ply')
        segment_path = os.path.join(save_dir, 'segment.ply')
        shutil.copy2(round1_seg_path, coarse_path)
        # segment.ply always points to the latest result.
        if round1_seg_path != segment_path:
            shutil.copy2(round1_seg_path, segment_path)

        # Save run parameters.
        params = {
            'model_path': model_path,
            'source_path': ctx.source_path,
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
                    'valid_masks': round1_result['valid_masks_count'],
                    **_frame_alignment_metadata(round1_result),
                    'output': 'coarse.ply',
                },
                'coverage_analysis': {
                    'total_cameras': coverage['total_cameras'],
                    'valid_cameras': coverage['valid_cameras'],
                    'invalid_cameras': coverage['invalid_cameras'],
                    'min_visible_ratio': coverage['min_visible_ratio'],
                    'max_angular_gap_deg': coverage['max_angular_gap_deg'],
                    'mean_angular_gap_deg': coverage['mean_angular_gap_deg'],
                    'angular_gap_threshold_deg': angular_gap_threshold,
                    'need_sphere_sampling': False,
                },
                'round2': None,
            },
            'post_processing': {
                'outlier_filter': {'applied': False},
                'boundary_compression': {'applied': False},
            },
            'current_stage': 'coarse',
        }
        with open(os.path.join(save_dir, 'segmentation_params.json'), 'w', encoding='utf-8') as f:
            json.dump(params, f, indent=2, ensure_ascii=False)

        # Round 1 skipped rendering; render now after confirming Round 2
        # is unnecessary.
        _, preview_images, _ = render_segmented_results(
            segment_path, render_cameras, save_dir, ctx.pipeline, ctx.background,
            ctx.sh_degree, save_mask=True, show_previews=True,
            show_start_frame=first_frame_idx, show_frames=6
        )
        if save_background:
            bg_path = os.path.join(save_dir, 'background.ply')
            if os.path.exists(bg_path):
                render_segmented_results(
                    bg_path, render_cameras, save_dir, ctx.pipeline, ctx.background,
                    ctx.sh_degree, save_mask=False, show_previews=False
                )

        summary = f"""
Segmentation complete (single round; Round 2 skipped).

Round 1 (training views): {round1_info}
Coverage: {coverage_info}
Training views provide sufficient coverage; spherical sampling is unnecessary.

Output: coarse.ply -> segment.ply
"""
        yield summary, segment_path, preview_images, save_dir, segment_path
        return

    # ==================== Round 2 ====================
    r2_mode = "training+spherical views" if need_sphere else "training views only"
    yield (
        f"[Round 2] {coverage_info}; running fine segmentation with "
        f"{r2_mode}...",
        None,
        [],
        None,
        None,
    )

    # Preserve Round 1 as coarse.ply before Round 2 replaces segment.ply.
    # multiview_masks.pkl records coarse.ply as its foreground model so
    # update_threshold can align voting indices correctly.
    coarse_path = os.path.join(save_dir, "coarse.ply")
    os.rename(round1_seg_path, coarse_path)

    # Load Round 2 views.
    temp_frames_folder = os.path.join(model_path, "temp_frames")
    all_cameras_r2, _, _ = load_views(
        ctx=ctx,
        use_train_views=True,
        use_random_views=need_sphere,
        n_layers=n_layers,
        n_points_per_layer=n_points_per_layer,
        prev_seg_path=coarse_path,
        cached_train_cameras=cached_train_cameras,
        output_folder=temp_frames_folder,
        first_frame_idx=first_frame_idx
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
        api_url=api_url_base,
        output_dir=save_dir,
        prev_seg_path=coarse_path,
        render_cameras=render_cameras,
        stage_name="round2",
    )

    fine_path = os.path.join(save_dir, 'fine.ply')
    segment_path = os.path.join(save_dir, 'segment.ply')
    shutil.copy2(round2_result['seg_path'], fine_path)
    # segment.ply always points to the latest result.
    if round2_result['seg_path'] != segment_path:
        shutil.copy2(round2_result['seg_path'], segment_path)

    # Save run parameters.
    n_sphere = len(all_cameras_r2) - len(cached_train_cameras) if need_sphere else 0
    params = {
        'model_path': model_path,
        'source_path': ctx.source_path,
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
                'valid_masks': round1_result['valid_masks_count'],
                **_frame_alignment_metadata(round1_result),
                'output': 'coarse.ply',
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
                'views': 'train+sphere' if need_sphere else 'train_only',
                'num_train_cameras': len(cached_train_cameras),
                'num_sphere_cameras': n_sphere,
                'n_layers': n_layers,
                'n_points_per_layer': n_points_per_layer,
                'valid_masks': round2_result['valid_masks_count'],
                **_frame_alignment_metadata(round2_result),
                'output': 'fine.ply',
            },
        },
        'post_processing': {
            'outlier_filter': {'applied': False},
            'boundary_compression': {'applied': False},
        },
        'current_stage': 'fine',
    }
    with open(os.path.join(save_dir, 'segmentation_params.json'), 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    round2_info = (
        f"Foreground: {round2_result['final_mask_count']}/{round2_result['total_points']} "
        f"({round2_result['final_mask_count']/round2_result['total_points']*100:.1f}%), "
        f"valid masks: {round2_result['valid_masks_count']}"
    )

    summary = f"""
Two-round segmentation complete.

Round 1 (training views): {round1_info}
Coverage: {coverage_info} -> {'spherical sampling required' if need_sphere else 'coverage sufficient; Round 2 was requested'}

Round 2 ({r2_mode}): {round2_info}

Output: coarse.ply + fine.ply -> segment.ply
"""

    yield summary, segment_path, round2_result['preview_images'], save_dir, segment_path


# ==================== Post-processing wrappers ====================

def apply_outlier_filter_wrapper(save_dir, outlier_std_mul, cached_train_cameras, cached_test_cameras,
                                 save_background=True):
    """Apply geometric outlier filtering and update run metadata."""
    if not save_dir or not os.path.exists(save_dir):
        return "Complete the initial segmentation first", None, [], None

    if cached_train_cameras is None:
        return "Error: training cameras are not loaded", None, [], None

    params_json_path = os.path.join(save_dir, 'segmentation_params.json')
    if not os.path.exists(params_json_path):
        return "Error: parameter file not found", None, [], None

    with open(params_json_path, 'r', encoding='utf-8') as f:
        params = json.load(f)

    ctx = create_segmentation_context(
        params['model_path'], params.get('source_path')
    )
    render_cameras = _merge_render_cameras(cached_train_cameras, cached_test_cameras)

    result = apply_outlier_filter(
        ctx=ctx,
        save_dir=save_dir,
        outlier_std_mul=outlier_std_mul,
        cached_train_cameras=cached_train_cameras,
        save_background=save_background,
        render_cameras=render_cameras
    )

    if result is None:
        return "Filtering failed or left too few points", None, [], None

    # Preserve a named copy for this processing stage.
    outlier_path = os.path.join(save_dir, 'outlier.ply')
    shutil.copy2(result['seg_path'], outlier_path)

    summary = f"""
Outlier filtering complete.

Statistics:
- Original points: {result['original_points']}
- Filtered points: {result['filtered_points']}
- Removed points: {result['removed_points']}

Output: outlier.ply -> segment.ply
"""

    # Update run metadata.
    params['post_processing']['outlier_filter'] = {
        'applied': True,
        'std_mul': outlier_std_mul,
        'original_points': result['original_points'],
        'filtered_points': result['filtered_points'],
        'removed_points': result['removed_points'],
        'output': 'outlier.ply',
    }
    params['current_stage'] = 'outlier'
    with open(params_json_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    return summary, result['seg_path'], result['preview_images'], result['seg_path']


def apply_boundary_compression_wrapper(save_dir, min_compression, tolerance_ratio,
                                       cached_train_cameras, cached_test_cameras):
    """Apply ABR boundary refinement and update run metadata."""
    if not save_dir or not os.path.exists(save_dir):
        return "Complete the initial segmentation first", None, [], None

    if cached_train_cameras is None:
        return "Error: training cameras are not loaded", None, [], None

    params_json_path = os.path.join(save_dir, 'segmentation_params.json')
    if not os.path.exists(params_json_path):
        return "Error: parameter file not found", None, [], None

    with open(params_json_path, 'r', encoding='utf-8') as f:
        params = json.load(f)

    ctx = create_segmentation_context(
        params['model_path'], params.get('source_path')
    )
    render_cameras = _merge_render_cameras(cached_train_cameras, cached_test_cameras)

    segmentation_params = params.get('segmentation', {})
    latest_round_params = (
        segmentation_params.get('round2')
        or segmentation_params.get('round1')
        or {}
    )
    frame_mapping_path = latest_round_params.get('frame_mapping_file')

    result = apply_boundary_compression(
        ctx=ctx,
        save_dir=save_dir,
        min_compression=min_compression,
        tolerance_ratio=tolerance_ratio,
        cached_train_cameras=cached_train_cameras,
        render_cameras=render_cameras,
        frame_mapping_path=frame_mapping_path,
    )

    if result is None:
        return (
            "ABR boundary refinement failed; masks or the frame-camera "
            "mapping may be missing or mismatched",
            None,
            [],
            None,
        )

    # Preserve a named copy for this processing stage.
    compression_path = os.path.join(save_dir, 'compression.ply')
    shutil.copy2(result['seg_path'], compression_path)

    summary = f"""
ABR boundary refinement complete.

Statistics:
- Total points: {result['original_points']}
- Compressed points: {result['compressed_points']}

Output: compression.ply -> segment.ply
"""

    # Update run metadata.
    params['post_processing']['boundary_compression'] = {
        'applied': True,
        'min_compression': min_compression,
        'tolerance_ratio': tolerance_ratio,
        'original_points': result['original_points'],
        'compressed_points': result['compressed_points'],
        'aligned_masks': result['aligned_masks'],
        'used_explicit_frame_mapping': result[
            'used_explicit_frame_mapping'
        ],
        'frame_mapping_file': result['frame_mapping_file'],
        'output': 'compression.ply',
    }
    params['current_stage'] = 'compression'
    with open(params_json_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    return summary, result['seg_path'], result['preview_images'], result['seg_path']


def update_threshold_and_resegment(save_dir, new_threshold, cached_train_cameras, cached_test_cameras,
                                   save_background=True):
    """Re-segment with an updated voting threshold."""
    if not save_dir or not os.path.exists(save_dir):
        return "Complete the initial segmentation first", None, [], None

    if cached_train_cameras is None:
        return "Error: training cameras are not loaded", None, [], None

    params_json_path = os.path.join(save_dir, 'segmentation_params.json')
    if not os.path.exists(params_json_path):
        return "Error: parameter file not found", None, [], None

    with open(params_json_path, 'r', encoding='utf-8') as f:
        params = json.load(f)

    ctx = create_segmentation_context(
        params['model_path'], params.get('source_path')
    )
    render_cameras = _merge_render_cameras(cached_train_cameras, cached_test_cameras)

    result = update_threshold(
        ctx=ctx,
        save_dir=save_dir,
        new_threshold=new_threshold,
        cached_train_cameras=cached_train_cameras,
        save_background=save_background,
        render_cameras=render_cameras
    )

    if result is None:
        return "Threshold update failed", None, [], None

    summary = f"""
Voting threshold updated to {new_threshold}.

Statistics:
- Foreground points: {result['final_mask_count']} / {result['num_points']}
- Segmentation ratio: {result['final_mask_count'] / max(1, result['num_points']) * 100:.2f}%
"""

    # The updated threshold applies to the final segmentation round.
    if 'segmentation' in params:
        params['segmentation']['threshold_round2'] = new_threshold
    else:
        params['threshold'] = new_threshold
    with open(params_json_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    return summary, result['seg_path'], result['preview_images'], result['seg_path']


# ==================== Cache reset ====================

def clear_all_cache():
    """Clear all UI state and cached cameras."""
    return (
        None, None, None, None, [], [], None, None, None, None,
        gr.update(choices=[], value=None), "", "Cache cleared; load a new dataset"
    )


# ==================== Gradio interface ====================

def create_gradio_interface():
    """Create the interactive Gradio interface."""

    with gr.Blocks(title="3DGS Segmentation") as demo:
        gr.Markdown("# 3DGS Segmentation")

        # UI state
        all_cameras_state = gr.State(None)
        cached_train_cameras_state = gr.State(None)
        cached_test_cameras_state = gr.State(None)
        original_image = gr.State(None)
        fg_points_state = gr.State([])
        bg_points_state = gr.State([])
        box_state = gr.State(None)
        current_save_dir = gr.State(None)
        current_seg_path_state = gr.State(None)

        with gr.Tab("[1] Views and annotations"):
            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.Markdown("### 3DGS paths")
                    model_path = gr.Textbox(
                        label="Trained 3DGS model path",
                        value=_get_default_model_path(),
                    )
                    source_path = gr.Textbox(
                        label="3DGS training-data path",
                        value=_get_default_source_path(),
                        info=(
                            "Overrides a machine-specific or stale absolute "
                            "source_path in cfg_args"
                        ),
                    )

                    gr.Markdown("Clear the cache before switching datasets")
                    clear_cache_btn = gr.Button("Clear cache", variant="secondary")

                    gr.Markdown("### View configuration")
                    n_layers = gr.Slider(
                        minimum=0,
                        maximum=20,
                        step=1,
                        value=4,
                        label="Spherical spiral layers",
                    )
                    n_points_per_layer = gr.Slider(
                        minimum=0,
                        maximum=20,
                        step=1,
                        value=8,
                        label="Spherical spiral points per layer",
                    )
                    skip_round2_if_covered = gr.Checkbox(
                        label="Skip Round 2 when spherical sampling is unnecessary",
                        value=True
                    )

                    load_views_btn = gr.Button("Load views", variant="primary")
                    load_status = gr.Textbox(label="Loading status", lines=2)

                    frame_idx_dropdown = gr.Dropdown(
                        label="Annotation frame",
                        choices=[],
                        value=None,
                        interactive=True,
                    )

                with gr.Column():
                    gr.Markdown("### Segmentation prompt")
                    text = gr.Textbox(label="Open-vocabulary text", value=None)

                    point_mode = gr.Radio(
                        choices=["Foreground points", "Background points", "Box annotation - start", "Box annotation - end"],
                        value="Foreground points",
                        label="Annotation mode"
                    )

                    clear_btn = gr.Button("Clear annotations", variant="secondary")
                    annotation_info = gr.Textbox(label="Current annotations", lines=3)

                    gr.Markdown("### Segmentation parameters")
                    threshold_round1 = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, value=0.7,
                        label="Round 1 voting threshold (coarse)",
                    )
                    threshold_round2 = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, value=0.5,
                        label="Round 2 voting threshold (fine)",
                    )
                    angular_gap_threshold = gr.Slider(
                        minimum=0.0, maximum=360.0, step=1.0, value=90.0,
                        label="Angular-gap threshold (0=always Round 2, 360=never)",
                    )
                    save_background = gr.Checkbox(
                        label="Save background model",
                        value=False,
                    )
                    gr.Markdown("### Basic configuration")
                    object_name = gr.Textbox(label="Object name", value="object")
                    api_url = gr.Textbox(
                        label="SAM API URL",
                        value="http://localhost:8000",
                    )
                    run_btn = gr.Button(
                        "Start segmentation",
                        variant="primary",
                        size="lg",
                    )


            with gr.Row():
                annotation_image = gr.Image(
                    label="Click the image to annotate",
                    type="numpy",
                )

        with gr.Tab("[2] Results"):
            with gr.Row(equal_height=True):
                output_text = gr.Textbox(label="Run status", lines=8)

            with gr.Row(equal_height=True):
                with gr.Column():
                    output_seg = gr.File(label="Segmentation result (.ply)")
                    seg_gallery = gr.Gallery(
                        label="Rendered segmentation results",
                        columns=2,
                        height=600,
                    )
                with gr.Column():
                    gr.Markdown("### Adjust voting threshold")
                    new_threshold = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, value=0.7,
                        label="Voting threshold",
                    )
                    update_threshold_btn = gr.Button(
                        "Update voting threshold",
                        variant="secondary",
                    )

                    gr.Markdown(
                        "### ABR: Axis-aware Boundary Refinement"
                    )
                    boundary_min_compression = gr.Slider(
                        minimum=0.01, maximum=1.0, step=0.01, value=0.2,
                        label=(
                            "Minimum compression (default 0.2; limits "
                            "over-compression and holes)"
                        ),
                    )
                    boundary_tolerance_ratio = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01, value=0.2,
                        label=(
                            "Directional overflow tolerance (default 0.2)"
                        ),
                    )
                    apply_boundary_btn = gr.Button(
                        "Apply ABR refinement",
                        variant="secondary",
                    )

                    gr.Markdown("### Geometric outlier filtering")
                    outlier_std_mul = gr.Slider(
                        minimum=0.5, maximum=10.0, step=0.01, value=5.0,
                        label="Threshold multiplier (std_mul)",
                    )
                    apply_outlier_filter_save_bg = gr.Checkbox(
                        label="Also process the background model",
                        value=False,
                    )
                    apply_outlier_filter_btn = gr.Button(
                        "Apply outlier filtering",
                        variant="secondary",
                    )

                    gr.Markdown("---")

        # ==================== Event bindings ====================

        # Clear cache.
        clear_cache_btn.click(
            fn=clear_all_cache,
            inputs=[],
            outputs=[
                all_cameras_state, cached_train_cameras_state, cached_test_cameras_state,
                original_image,
                fg_points_state, bg_points_state, box_state,
                current_save_dir, current_seg_path_state,
                annotation_image, frame_idx_dropdown, annotation_info, load_status
            ]
        )

        # Load views.
        load_views_btn.click(
            fn=load_views_for_annotation,
            inputs=[
                model_path,
                source_path,
                cached_train_cameras_state,
                frame_idx_dropdown,
            ],
            outputs=[frame_idx_dropdown, load_status, all_cameras_state,
                     cached_train_cameras_state, cached_test_cameras_state]
        )

        # Select the annotation frame.
        def on_frame_select(model_path_val, frame_choice):
            if frame_choice is None or not model_path_val:
                return None, [], [], None, "", "Load the views first"
            frame_idx = int(frame_choice.split()[1])
            img, status = get_frame_by_idx(model_path_val, frame_idx)
            return img, [], [], None, "", status

        frame_idx_dropdown.change(
            fn=on_frame_select,
            inputs=[model_path, frame_idx_dropdown],
            outputs=[original_image, fg_points_state, bg_points_state, box_state, annotation_info, load_status]
        ).then(fn=lambda img: img if img is not None else None, inputs=[original_image], outputs=[annotation_image])

        # Annotate the image.
        annotation_image.select(
            fn=handle_image_click,
            inputs=[original_image, fg_points_state, bg_points_state, box_state, point_mode],
            outputs=[annotation_image, fg_points_state, bg_points_state, box_state, annotation_info]
        )

        clear_btn.click(
            fn=clear_annotations,
            outputs=[fg_points_state, bg_points_state, box_state, annotation_info]
        ).then(fn=lambda img: img, inputs=[original_image], outputs=[annotation_image])

        # Run the automatic two-round segmentation workflow.
        def run_segmentation_wrapper(model_path_val, source_path_val,
                                     cached_train_cameras, cached_test_cameras,
                                     frame_choice,
                                     fg_points, bg_points, box, text,
                                     threshold_r1, threshold_r2,
                                     save_background,
                                     n_layers_val, n_points_per_layer_val,
                                     skip_round2_val, angular_gap_val,
                                     object_name_val, api_url_val):
            if cached_train_cameras is None or frame_choice is None:
                yield "Load views and select an annotation frame first", None, [], None, None
                return
            frame_idx = int(frame_choice.split()[1])

            for result in segment_3dgs_two_round_pipeline(
                model_path_val, source_path_val,
                cached_train_cameras, cached_test_cameras,
                frame_idx,
                fg_points, bg_points, box, text,
                threshold_r1, threshold_r2, save_background,
                int(n_layers_val), int(n_points_per_layer_val),
                skip_round2_val, angular_gap_val,
                object_name=object_name_val,
                api_url_base=api_url_val
            ):
                yield result

        run_btn.click(
            fn=run_segmentation_wrapper,
            inputs=[
                model_path, source_path,
                cached_train_cameras_state, cached_test_cameras_state,
                frame_idx_dropdown,
                fg_points_state, bg_points_state, box_state, text,
                threshold_round1, threshold_round2,
                save_background,
                n_layers, n_points_per_layer,
                skip_round2_if_covered, angular_gap_threshold,
                object_name, api_url
            ],
            outputs=[output_text, output_seg, seg_gallery, current_save_dir, current_seg_path_state]
        ).then(fn=lambda t: t, inputs=[threshold_round2], outputs=[new_threshold])

        # Post-processing actions.
        update_threshold_btn.click(
            fn=update_threshold_and_resegment,
            inputs=[current_save_dir, new_threshold, cached_train_cameras_state,
                    cached_test_cameras_state, save_background],
            outputs=[output_text, output_seg, seg_gallery, current_seg_path_state]
        )

        apply_outlier_filter_btn.click(
            fn=apply_outlier_filter_wrapper,
            inputs=[current_save_dir, outlier_std_mul, cached_train_cameras_state,
                    cached_test_cameras_state, apply_outlier_filter_save_bg],
            outputs=[output_text, output_seg, seg_gallery, current_seg_path_state]
        )

        apply_boundary_btn.click(
            fn=apply_boundary_compression_wrapper,
            inputs=[current_save_dir, boundary_min_compression, boundary_tolerance_ratio,
                    cached_train_cameras_state, cached_test_cameras_state],
            outputs=[output_text, output_seg, seg_gallery, current_seg_path_state]
        )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                allowed_paths=_get_allowed_paths())
