import os
import cv2
import torch
import requests
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from plyfile import PlyData, PlyElement
from gaussiansplatting.scene.gaussian_model import GaussianModel
from seg_utils.render_utils import render_gsplat_camera


def get_combined_args(parser : ArgumentParser, model_path):
    # cmdlne_string = sys.argv[1:]
    # cfgfile_string = "Namespace()"
    cmdlne_string = ['--model_path', model_path]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)


    try:
        cfgfilepath = os.path.join(model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)


def save_gs(pc, indices_mask, save_path):
    indices_mask = indices_mask.detach().cpu()
    xyz = pc._xyz.detach().cpu()[indices_mask].numpy()
    normals = np.zeros_like(xyz)
    f_dc = pc._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu()[indices_mask].numpy()
    f_rest = pc._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu()[indices_mask].numpy()
    opacities = pc._opacity.detach().cpu()[indices_mask].numpy()
    scale = pc._scaling.detach().cpu()[indices_mask].numpy()
    rotation = pc._rotation.detach().cpu()[indices_mask].numpy()

    dtype_full = [(attribute, 'f4') for attribute in pc.construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(save_path)


def save_background_gs(pc, foreground_indices_mask, save_path):
    """Save background GS (remove foreground points)"""
    total_points = pc._xyz.shape[0]
    all_indices = torch.arange(total_points)

    # Background contains every point outside the foreground mask.
    background_mask = torch.ones(total_points, dtype=torch.bool)
    if isinstance(foreground_indices_mask, torch.Tensor):
        if foreground_indices_mask.device != torch.device('cpu'):
            foreground_indices_mask = foreground_indices_mask.cpu()
    else:
        foreground_indices_mask = torch.tensor(foreground_indices_mask)

    background_mask[foreground_indices_mask] = False
    background_indices = all_indices[background_mask]

    save_gs(pc, background_indices, save_path)


def load_masks_from_folder(mask_folder):
    """Load masks from folder, return mask dictionary, key is frame index"""
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.jpg')]
    masks_dict = {}

    for mask_file in mask_files:
        # Extract frame index from file name, e.g. 00005.jpg -> 5
        frame_idx = int(os.path.splitext(mask_file)[0])
        mask_path = os.path.join(mask_folder, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 128).astype(np.uint8)
        masks_dict[frame_idx] = mask

    return masks_dict


def call_sam_video_segment(frame_folder, first_frame_idx, points, boxes, text=None, object_name=None,
                           background_color=[255, 255, 255], api_url="http://localhost:8000/segment"):
    """Call SAM video segmentation API"""
    if object_name is None:
        object_name = "object"

    annotations = []
    annotation = {
        "object_name": object_name,
        "frame_idx": first_frame_idx,
        "text": text
    }

    # Multiple prompt points
    if points is not None and len(points) > 0:
        annotation["points"] = [{"x": int(p[0]), "y": int(p[1]), "label": int(p[2])}
                                for p in points]
    # Only support single box
    if boxes is not None and len(boxes) > 0:
        box = boxes[0]
        annotation["box"] = {"x1": int(box[0]), "y1": int(box[1]),
                             "x2": int(box[2]), "y2": int(box[3])}

    annotations.append(annotation)

    data = {
        "frame_folder": frame_folder,
        "annotations": annotations,
        "background_color": background_color,
    }

    response = requests.post(api_url, json=data)
    return response.json()


def render_segmented_results(ply_path, cameras, save_dir, pipeline, background, sh_degree, 
                                save_mask=True, show_previews=True, show_start_frame=0, show_frames=6):
    """Render segmented 3DGS results, optional save mask"""
    seg_gaussians = GaussianModel(sh_degree)
    seg_gaussians.load_ply(ply_path)

    # An empty PLY can trigger an uncatchable SIGFPE in the gsplat CUDA kernel.
    if len(seg_gaussians.get_xyz) == 0:
        print(f"[WARN] PLY contains no Gaussians; skipping render: {ply_path}")
        del seg_gaussians
        torch.cuda.empty_cache()
        return None, [], None

    base_name = os.path.basename(ply_path).split('.')[0]
    render_seg_dir = os.path.join(save_dir, 'render_' + base_name)
    os.makedirs(render_seg_dir, exist_ok=True)

    # If save mask, create mask directory
    render_mask_dir = None
    if save_mask:
        render_mask_dir = os.path.join(save_dir, 'render_mask_' + base_name)
        os.makedirs(render_mask_dir, exist_ok=True)

    rendered_images = []
    for idx, view in enumerate(
        tqdm(cameras, desc=f"Rendering {os.path.basename(ply_path)}")
    ):
        image_name = f"{view.image_name}"
        render_image_gsplat, alphas, meta = render_gsplat_camera(
            seg_gaussians, view, view.image_width, view.image_height, backgrounds=background
        )
        render_image_bgr = cv2.cvtColor(render_image_gsplat, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(render_seg_dir, f'{image_name}.jpg')
        cv2.imwrite(save_path, render_image_bgr)

        # If save mask, create mask directory
        mask = None
        if save_mask:
            # Generate mask: foreground (colored region) is white, background (black region) is black
            # Judgement criterion: RGB three channels are all close to 0 are background
            mask_path = os.path.join(render_mask_dir, f'{image_name}_mask.jpg')
            mask = (alphas[0].detach().cpu().numpy() > 0.1).astype(np.uint8) * 255
            cv2.imwrite(mask_path, mask)

        # Save 6 images for preview
        if show_previews and idx >= show_start_frame and idx < show_start_frame + show_frames:
            rendered_images.append(render_image_gsplat)
            if mask is not None:
                rendered_images.append(mask.squeeze())

    # Release GaussianModel GPU memory after rendering.
    del seg_gaussians
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return render_seg_dir, rendered_images, render_mask_dir
