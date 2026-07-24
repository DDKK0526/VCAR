import torch
import collections
import numpy as np
from gsplat import rasterization
from gaussiansplatting.utils.graphics_utils import getWorld2View, getProjectionMatrix

RenderConfig = collections.namedtuple(
    "RenderConfig", ["white_background", "debug", "compute_cov3D_python", "convert_SHs_python"])

class VirtualCamera:
    def __init__(self, custom_id, R, T, FoVx, FoVy, image_width, image_height,
                 znear=0.01, zfar=100.0, trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        self.uid = custom_id
        self.colmap_id = custom_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = image_width
        self.image_height = image_height

        self.znear = znear
        self.zfar = zfar

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View(R, T)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def create_pinhole_intrinsics(width, height, fov_degrees=45):
    """
    Create a pinhole camera intrinsic matrix from image size and field of view.
    """
    fov_rad = np.radians(fov_degrees)

    # Use the smaller dimension so the effective field of view is stable across aspect ratios.
    min_dim = min(width, height)
    focal_length = min_dim / (2 * np.tan(fov_rad / 2))

    # Set principal point to image center
    cx = width / 2.0
    cy = height / 2.0

    intrinsics = torch.tensor([
        [focal_length, 0., cx],
        [0., focal_length, cy],
        [0., 0., 1.]
    ], dtype=torch.float32)
    return intrinsics

def create_look_at_view_matrix(position, target, up=None, device=None):
    """
    Create camera transformation matrix (extrinsic matrix/view matrix) based on observation point and target point
    :param position: Camera position [x, y, z]
    :param target: Target point [x, y, z]
    :param up: Up direction vector, default is [0, 1, 0]
    :param device: Specify device (optional)
    :return:
    """
    position = torch.as_tensor(position, dtype=torch.float32, device=device)
    target = torch.as_tensor(target, dtype=torch.float32, device=position.device)

    # Default to y-axis as vertical direction
    if up is None:
        up = torch.tensor([0., 1., 0.], dtype=torch.float32, device=position.device)
    else:
        up = torch.as_tensor(up, dtype=torch.float32, device=position.device)

    # Calculate forward vector and normalize
    forward = target - position
    forward = forward / torch.linalg.norm(forward)

    # Calculate right vector
    right = torch.cross(forward, up, dim=0)
    right_norm = torch.linalg.norm(right)

    # Handle special case: forward and up direction are parallel
    if right_norm < 1e-6:
        if abs(forward[0]) > 1e-6 or abs(forward[2]) > 1e-6:
            alt_up = torch.tensor([0., 1., 0.], device=position.device)
        else:
            alt_up = torch.tensor([0., 0., 1.], device=position.device)
        right = torch.cross(forward, alt_up)
        right_norm = torch.linalg.norm(right)
    right = right / right_norm

    # Recalculate up direction to ensure orthogonality
    up_new = torch.cross(right, forward, dim=0)

    # Build rotation matrix and translation vector
    R = torch.stack([right, up_new, forward], dim=0)
    t = -torch.matmul(R, position)

    viewmat = torch.eye(4, dtype=torch.float32, device=position.device)
    viewmat[:3, :3] = R
    viewmat[:3, 3] = t
    return viewmat

def _ensure_float32(*tensors):
    """
    Ensure all tensors are float32 type
    """
    return tuple(t.float() if t.dtype != torch.float32 else t for t in tensors)

def _render_gsplat_from_matrices(
    gaussians,
    intrinsics,
    world_to_camera,
    width,
    height,
    znear=0.01,
    zfar=100,
    device="cuda",
    backgrounds=None,
):
    """
    Render image from explicit camera intrinsic and extrinsic matrices.
    
    Parameters:
        backgrounds: Background color tensor, shape is (3,), RGB format. If None, use transparent background
    """
    # Get Gaussian attributes for rendering, and ensure data type is correct
    means = gaussians.get_xyz.to(device)
    quats = gaussians.get_rotation.to(device)
    scales = gaussians.get_scaling.to(device)
    opacities = gaussians.get_opacity.to(device).reshape(-1)
    colors = gaussians.get_features

    means, quats, scales, opacities, colors = _ensure_float32(means, quats, scales, opacities, colors)
    
    # If background color is provided, ensure format is correct
    if backgrounds is not None:
        backgrounds = backgrounds.to(device).float()
    
    render_image, alphas, meta = rasterization(means, quats, scales, opacities, colors,
                                               world_to_camera, intrinsics, width, height,
                                               sh_degree=gaussians.max_sh_degree, near_plane=znear, far_plane=zfar,
                                               backgrounds=backgrounds)
    render_image = render_image[0].detach().cpu().numpy()
    render_image = (255 * np.clip(render_image, 0, 1)).astype(np.uint8)
    return render_image, alphas, meta

def render_gsplat_camera(gaussians, camera, width, height,
                         znear=0.01, zfar=100, device="cuda", backgrounds=None):
    """
    Render using gsplat with a dataset camera or VirtualCamera.
    
    Parameters:
        gaussians: Gaussian model
        camera: Camera-like object with R, T, FoVx, and FoVy
        width: Rendered image width
        height: Rendered image height
        znear: Near plane
        zfar: Far plane
        device: Calculation device
        backgrounds: Background color tensor, shape is (3,)
    
    Returns:
        render_image: Rendered image (H, W, 3) numpy array
        alphas: alpha channel (used to generate mask)
        meta: Meta information dictionary
    """
    # Camera.R is camera-to-world rotation; transpose it to get world-to-camera.
    R_w2c = torch.tensor(camera.R, dtype=torch.float32, device=device).T  # [3, 3]
    t_w2c = torch.tensor(camera.T, dtype=torch.float32, device=device)  # [3]

    world_to_camera = torch.eye(4, dtype=torch.float32, device=device)
    world_to_camera[:3, :3] = R_w2c
    world_to_camera[:3, 3] = t_w2c
    world_to_camera = world_to_camera.unsqueeze(0)  # [1, 4, 4]
    
    fx = width / (2.0 * np.tan(camera.FoVx / 2.0))
    fy = height / (2.0 * np.tan(camera.FoVy / 2.0))
    cx = width / 2.0
    cy = height / 2.0
    
    intrinsics = torch.tensor([
        [fx, 0., cx],
        [0., fy, cy],
        [0., 0., 1.]
    ], dtype=torch.float32, device=device).unsqueeze(0)  # [1, 3, 3]

    render_image, alphas, meta = _render_gsplat_from_matrices(
        gaussians, intrinsics, world_to_camera, width, height,
        znear=znear, zfar=zfar, device=device, backgrounds=backgrounds
    )
    
    return render_image, alphas, meta

def build_virtual_camera(intrinsics, world_to_camera, image_width, image_height, znear, zfar, custom_id=None, device="cuda"):
    """
    Build a VirtualCamera from camera intrinsic and world-to-camera matrices.
    """
    # Ensure tensor format and correct device
    if isinstance(world_to_camera, np.ndarray):
        world_to_camera = torch.from_numpy(world_to_camera).float().to(device)
    if isinstance(intrinsics, np.ndarray):
        intrinsics = torch.from_numpy(intrinsics).float().to(device)

    world_to_camera = world_to_camera.to(device)
    intrinsics = intrinsics.to(device)

    # Handle batch dimension
    if world_to_camera.dim() == 3 and world_to_camera.shape[0] == 1:
        world_to_camera = world_to_camera.squeeze(0)
    if intrinsics.dim() == 3 and intrinsics.shape[0] == 1:
        intrinsics = intrinsics.squeeze(0)

    # Extract rotation matrix R and translation vector T from extrinsic matrix
    # In the original Camera class, the R passed in through getWorld2View is the rotation matrix from camera to world, so it needs to be transposed
    R = world_to_camera[:3, :3].detach().cpu().numpy().T
    T = world_to_camera[:3, 3].detach().cpu().numpy()

    # Extract focal length from intrinsic matrix
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]

    # Calculate field of view angle based on focal length and image size, FoV = 2 * arctan(image_size / (2 * focal_length))
    FoVx = float(2.0 * torch.atan(torch.tensor(float(image_width)) / (2.0 * fx)).detach().cpu())
    FoVy = float(2.0 * torch.atan(torch.tensor(float(image_height)) / (2.0 * fy)).detach().cpu())
    return VirtualCamera(
        custom_id=custom_id,
        R=R, T=T, FoVx=FoVx, FoVy=FoVy,
        image_width=image_width, image_height=image_height,
        znear=znear, zfar=zfar,data_device=device
    )
