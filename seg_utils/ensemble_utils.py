import torch
import torch.nn.functional as F
import numpy as np
import cv2

def project_to_2d(viewpoint_camera, points3D):
    """Project 3D points to 2D image plane"""
    full_matrix = viewpoint_camera.full_proj_transform
    if points3D.shape[-1] != 4:
        points3D = F.pad(input=points3D, pad=(0, 1), mode='constant', value=1)
    p_hom = (points3D @ full_matrix).transpose(0, 1)
    p_w = 1.0 / (p_hom[-1, :] + 0.0000001)
    p_proj = p_hom[:3, :] * p_w

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width

    point_image = 0.5 * ((p_proj[:2] + 1) * torch.tensor([w, h]).unsqueeze(-1).to(p_proj.device) - 1)
    point_image = point_image.detach().clone()
    point_image = torch.round(point_image.transpose(0, 1))
    return point_image


def mask_inverse(xyz, viewpoint_camera, sam_mask):
    """Inverse project 2D mask to 3D space
       use full 4x4 world_view_transform to calculate camera coordinates Z>0
    """
    w2c_matrix = viewpoint_camera.world_view_transform  # [4,4]

    # Expand to homogeneous coordinates [N,3] -> [N,4]
    xyz_pad = F.pad(input=xyz, pad=(0, 1), mode='constant', value=1)

    # Use full 4x4 view matrix to get camera coordinates, Z>0 is considered valid
    p_cam = (xyz_pad @ w2c_matrix).transpose(0, 1)  # [4,N]
    depth = p_cam[2, :].detach().clone()            # Z in camera coordinate system
    valid_depth = depth > 0

    h = viewpoint_camera.image_height
    w = viewpoint_camera.image_width

    if isinstance(sam_mask, np.ndarray):
        sam_mask = torch.from_numpy(sam_mask).to("cuda")

    if sam_mask.shape[0] != h or sam_mask.shape[1] != w:
        sam_mask = F.interpolate(sam_mask.unsqueeze(0).unsqueeze(0).float(),
                                 size=(h, w), mode='nearest').squeeze().long()
    else:
        sam_mask = sam_mask.long()

    point_image = project_to_2d(viewpoint_camera, xyz_pad)
    point_image = point_image.long()

    valid_x = (point_image[:, 0] >= 0) & (point_image[:, 0] < w)
    valid_y = (point_image[:, 1] >= 0) & (point_image[:, 1] < h)
    valid_mask = valid_x & valid_y & valid_depth

    point_mask = torch.full((point_image.shape[0],), -1, device=sam_mask.device if isinstance(sam_mask, torch.Tensor) else "cuda")
    point_mask[valid_mask] = sam_mask[point_image[valid_mask, 1], point_image[valid_mask, 0]]

    indices_mask = torch.where(point_mask == 1)[0]
    return point_mask, indices_mask


def ensemble(multiview_masks, threshold=0.7):
    """Multi-view label voting (supports threshold=0 for union semantic, and ignores -1 for invisible labels)
    Input: list of tensors, each shape is [N,1], values in {-1,0,1}
    Rules:
      - First count the foreground ratio in visible views (!= -1): pos/vis.
      - keep condition: pos/vis >= threshold.
      - When threshold == 0, equivalent to union semantic: any visible view is 1 is retained.
    Return:
      - vote_labels: [N], 1 for retained, 0 for discarded (compatible with old interface, caller only uses indices)
      - indices_mask: indices of foreground points to keep (CPU)
    """
    if len(multiview_masks) == 0:
        return torch.empty(0, dtype=torch.int8), torch.empty(0, dtype=torch.long)

    X = torch.cat(multiview_masks, dim=1)  # [N, V], values in {-1,0,1}
    device = X.device
    N, V = X.shape

    # Visible views (ignore -1)
    visible = (X != -1)
    vis = visible.sum(dim=1)  # [N]
    pos = (X == 1).sum(dim=1)  # [N]

    # Avoid NaN when denominator is 0, and ensure ratio is 0
    vis_safe = torch.clamp(vis, min=1)
    ratios = pos.float() / vis_safe.float()

    if threshold <= 0:
        # Union semantic: any visible view is 1 is retained (and must have at least one visible view is 1)
        keep = pos > 0
    else:
        keep = ratios >= float(threshold)

    vote_labels = torch.where(keep, torch.ones(N, dtype=torch.int8, device=device),
                              torch.zeros(N, dtype=torch.int8, device=device))
    indices_mask = torch.where(keep)[0].detach().cpu()
    return vote_labels, indices_mask

