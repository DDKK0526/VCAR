"""Axis-aware Boundary Refinement (ABR) with multi-view voting.

The implementation uses PyTorch tensor operations and selects CUDA when
available. It restores each 2D covariance ellipse from gsplat conics, samples
the four major/minor-axis endpoints, and checks them against the mask. For an
overflowing Gaussian, the depth-free form of T = J * W * R retains first-order
perspective coupling and attributes the 2D overflow direction to its
contributing 3D axes so only the dominant axis is compressed.
"""

import numpy as np
import cv2
import torch

from seg_utils.axis_attribution import depth_free_perspective_axis_projection


# Chunk size used to limit peak GPU memory during batched sampling.
_BOUNDARY_CHUNK_SIZE = 50000


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _get_device(device=None):
    """Prefer CUDA when available and otherwise use the CPU."""
    if device is not None:
        return torch.device(device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------------------------------------------------------------
# Vectorized quaternion-to-rotation conversion
# ---------------------------------------------------------------------------

def _quaternion_to_rotation_matrix(quats: torch.Tensor) -> torch.Tensor:
    """Convert a batch of quaternions to 3x3 rotation matrices.

    This follows gaussiansplatting.utils.general_utils.build_rotation.

    Args:
        quats: [N, 4] float32 quaternions in (w, x, y, z) order.

    Returns:
        [N, 3, 3] float32 rotation matrices.
    """
    # Normalize input quaternions.
    norm = torch.norm(quats, dim=1, keepdim=True).clamp(min=1e-12)
    q = quats / norm

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros(len(q), 3, 3, dtype=quats.dtype, device=quats.device)
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - w*x)
    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)

    return R


# ---------------------------------------------------------------------------
# Vectorized 2D inverse covariance to ellipse axes
# ---------------------------------------------------------------------------

def _conics_to_axes(conics: torch.Tensor, sigma_cutoff: float = 3.0):
    """Recover ellipse axes and radii from gsplat conics.

    Args:
        conics: [N, 3] upper triangle (A, B, C) of the inverse covariance.
        sigma_cutoff: Ellipse cutoff in standard deviations; gsplat uses 3.

    Returns:
        Unit directions and pixel radii for the long and short axes, followed
        by the corresponding covariance eigenvalues.
    """
    A = conics[:, 0]
    B = conics[:, 1]
    C = conics[:, 2]

    # Determinant of the inverse covariance.
    det_inv = (A * C - B * B).clamp(min=1e-12)

    # Covariance = (1 / det) * [[C, -B], [-B, A]].
    inv_det = 1.0 / det_inv
    s_a = C * inv_det   # Σ[0,0]
    s_b = -B * inv_det  # Σ[0,1] = Σ[1,0]
    s_c = A * inv_det   # Σ[1,1]

    # Eigenvalues of a symmetric 2x2 matrix.
    trace = s_a + s_c
    diff = s_a - s_c
    disc = torch.sqrt((diff * diff + 4.0 * s_b * s_b).clamp(min=0.0))

    lam1 = ((trace + disc) * 0.5).clamp(min=1e-12)
    lam2 = ((trace - disc) * 0.5).clamp(min=1e-12)

    # Semi-axis length = sigma cutoff * sqrt(eigenvalue).
    len_long = sigma_cutoff * torch.sqrt(lam1)
    len_short = sigma_cutoff * torch.sqrt(lam2)

    # Normalize the eigenvector (b, lambda1 - a) for the larger eigenvalue.
    ex = s_b.clone()
    ey = lam1 - s_a

    norm = torch.sqrt(ex * ex + ey * ey)
    # A degenerate vector indicates an axis-aligned ellipse.
    degenerate = norm < 1e-8
    safe_norm = norm.clamp(min=1e-12)
    ex = torch.where(degenerate, torch.ones_like(ex), ex / safe_norm)
    ey = torch.where(degenerate, torch.zeros_like(ey), ey / safe_norm)

    axes_long = torch.stack([ex, ey], dim=-1)        # [N, 2]
    axes_short = torch.stack([-ey, ex], dim=-1)

    return axes_long, axes_short, len_long, len_short, lam1, lam2


# ---------------------------------------------------------------------------
# Batched directional mask-boundary distance
# ---------------------------------------------------------------------------

def _batched_directional_boundary_distance(
    cx: torch.Tensor, cy: torch.Tensor,
    disp_x: torch.Tensor, disp_y: torch.Tensor,
    mask: torch.Tensor,
    n_steps: int = 128,
) -> torch.Tensor:
    """Measure mask-boundary distance along a batch of displacement vectors.

    Uniform samples are taken between each center and endpoint. The returned
    distance ends immediately before the first sample outside the mask.

    Args:
        cx, cy: [B] center pixel coordinates.
        disp_x, disp_y: [B] displacements to semi-axis endpoints in pixels.
        mask: [H, W] boolean mask.
        n_steps: Fixed sample count used for vectorized evaluation.

    Returns:
        [B] directional distances from centers to the mask boundary.
    """
    B = cx.shape[0]
    if B == 0:
        return torch.zeros(0, device=cx.device, dtype=cx.dtype)

    H, W = mask.shape
    device = cx.device

    lengths = torch.sqrt(disp_x * disp_x + disp_y * disp_y)  # [B]
    valid = lengths > 1e-6

    # Sample t in [0, 1], including center and endpoint.
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)  # [S+1]

    # Evaluate all sample coordinates as [B, S+1].
    sx = cx.unsqueeze(1) + ts.unsqueeze(0) * disp_x.unsqueeze(1)
    sy = cy.unsqueeze(1) + ts.unsqueeze(0) * disp_y.unsqueeze(1)

    su = torch.round(sx).to(torch.int32)  # [B, S+1]
    sv = torch.round(sy).to(torch.int32)  # [B, S+1]

    # Check image bounds.
    in_bounds = (su >= 0) & (su < W) & (sv >= 0) & (sv < H)  # [B, S+1]

    # Clamp coordinates before indexing the flattened mask.
    su_clip = su.clamp(0, W - 1)
    sv_clip = sv.clamp(0, H - 1)
    flat_idx = sv_clip.long() * W + su_clip.long()  # [B, S+1]
    mask_flat = mask.reshape(-1)                      # [H*W]
    in_mask_vals = mask_flat[flat_idx]                 # [B, S+1] bool

    # Out-of-image samples are outside the mask.
    in_mask = in_mask_vals & in_bounds  # [B, S+1]

    # Find the first outside sample on each line.
    outside = ~in_mask  # [B, S+1]
    has_outside = outside.any(dim=1)  # [B]

    # Add a sentinel for fully inside rows before locating the first True.
    outside_sentinel = outside.clone()
    outside_sentinel[~has_outside, -1] = True
    first_out = outside_sentinel.to(torch.uint8).argmax(dim=1)  # [B]

    # Estimate the boundary at the midpoint of adjacent samples.
    first_out_long = first_out.long()
    t_prev = ts[torch.clamp(first_out_long - 1, min=0)]  # [B]
    t_curr = ts[first_out_long]                            # [B]
    t_boundary = (t_prev + t_curr) * 0.5
    t_boundary[first_out == 0] = 0.0

    d_dir = t_boundary * lengths  # [B]
    d_dir[~has_outside] = lengths[~has_outside]
    d_dir[~valid] = 0.0

    return d_dir


def _chunked_directional_boundary_distance(
    cx, cy, disp_x, disp_y, mask, n_steps=128, chunk_size=_BOUNDARY_CHUNK_SIZE
):
    """Run directional boundary sampling in memory-bounded chunks."""
    B = cx.shape[0]
    if B <= chunk_size:
        return _batched_directional_boundary_distance(
            cx, cy, disp_x, disp_y, mask, n_steps
        )

    results = []
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk = _batched_directional_boundary_distance(
            cx[start:end], cy[start:end],
            disp_x[start:end], disp_y[start:end],
            mask, n_steps
        )
        results.append(chunk)
    return torch.cat(results)


# ---------------------------------------------------------------------------
# Multi-view endpoint voting and axis-attributed compression
# ---------------------------------------------------------------------------

def compute_boundary_compression_axes(
    xyz_len: int,
    meta_list: list,
    masks: list,
    rotations: torch.Tensor = None,
    scales_3d: torch.Tensor = None,
    view_matrices: list = None,
    intrinsics_list: list = None,
    min_compression: float = 0.2,
    tolerance_ratio: float = 0.2,
    sigma_cutoff: float = 3.0,
    device=None,
):
    """Compute per-axis compression using endpoint sampling and view voting.

    Args:
        xyz_len: Number of foreground Gaussians.
        meta_list: gsplat metadata dictionaries for each view.
        masks: Binary 2D masks corresponding to meta_list.
        rotations: [N, 4] quaternions in (w, x, y, z) order.
        scales_3d: [N, 3] activated 3D scales.
        view_matrices: Per-view 4x4 world-to-camera matrices.
        intrinsics_list: Per-view (fx, fy, cx, cy) tuples.
        min_compression: Lower bound that prevents surface holes.
        tolerance_ratio: Minimum overflowing-view ratio for compression.
        sigma_cutoff: Gaussian cutoff; defaults to gsplat's 3 sigma.
        device: Compute device, or automatic CUDA/CPU selection when None.

    Returns:
        [xyz_len, 3] per-axis factors; 1.0 means no compression.
    """
    dev = _get_device(device)

    # Anisotropic attribution requires the complete 3D camera geometry.
    anisotropic = (rotations is not None and scales_3d is not None
                   and view_matrices is not None and intrinsics_list is not None)

    if anisotropic:
        rotations = rotations.to(dev).float()
        scales_3d = scales_3d.to(dev).float()
        # Precompute all Gaussian rotation matrices.
        all_rot_matrices = _quaternion_to_rotation_matrix(rotations)

    visible_counts = torch.zeros(xyz_len, dtype=torch.int32, device=dev)
    overflow_counts = torch.zeros(xyz_len, dtype=torch.int32, device=dev)

    # Keep the strictest factor observed for each Gaussian axis.
    min_factors_per_axis = torch.ones(xyz_len, 3, dtype=torch.float32, device=dev)
    # Track which axes have contributed to an overflow.
    axis_has_overflow = torch.zeros(xyz_len, 3, dtype=torch.bool, device=dev)

    for view_idx, (meta, mask_raw) in enumerate(zip(meta_list, masks)):
        # Move the mask to the selected compute device.
        if isinstance(mask_raw, np.ndarray):
            mask_t = torch.from_numpy(mask_raw).to(dev)
        elif isinstance(mask_raw, torch.Tensor):
            mask_t = mask_raw.to(dev)
        else:
            mask_t = torch.as_tensor(mask_raw, device=dev)
        mask_bool = mask_t > 0  # [H, W] bool
        h, w = mask_bool.shape

        # Keep gsplat metadata on the compute device.
        gaussian_ids = meta["gaussian_ids"].to(dev).flatten().long()
        conics = meta["conics"].to(dev).reshape(-1, 3).float()
        means2d = meta["means2d"].to(dev).reshape(-1, 2).float()

        radii_raw = meta["radii"].to(dev)
        if radii_raw.ndim > 1 and radii_raw.shape[-1] == 2:
            radii_max = radii_raw.max(dim=-1).values.float()
        else:
            radii_max = radii_raw.flatten().float()

        n = len(gaussian_ids)
        if n == 0:
            continue

        # Recover ellipse axes and eigenvalues.
        axes_long, axes_short, len_long, len_short, lam1_arr, lam2_arr = \
            _conics_to_axes(conics, sigma_cutoff)

        # Ellipse centers.
        cx = means2d[:, 0]
        cy = means2d[:, 1]

        # Four endpoints: +/- long axis and +/- short axis.
        endpoints = torch.zeros(n, 4, 2, dtype=torch.float32, device=dev)
        endpoints[:, 0, 0] = cx + axes_long[:, 0] * len_long
        endpoints[:, 0, 1] = cy + axes_long[:, 1] * len_long
        endpoints[:, 1, 0] = cx - axes_long[:, 0] * len_long
        endpoints[:, 1, 1] = cy - axes_long[:, 1] * len_long
        endpoints[:, 2, 0] = cx + axes_short[:, 0] * len_short
        endpoints[:, 2, 1] = cy + axes_short[:, 1] * len_short
        endpoints[:, 3, 0] = cx - axes_short[:, 0] * len_short
        endpoints[:, 3, 1] = cy - axes_short[:, 1] * len_short

        # Unit overflow direction for each endpoint.
        ep_dirs = torch.zeros(n, 4, 2, dtype=torch.float32, device=dev)
        ep_dirs[:, 0] = axes_long
        ep_dirs[:, 1] = -axes_long
        ep_dirs[:, 2] = axes_short
        ep_dirs[:, 3] = -axes_short

        # Center pixel coordinates.
        cu = torch.round(cx).to(torch.int32)
        cv = torch.round(cy).to(torch.int32)
        cu_clip = cu.clamp(0, w - 1)
        cv_clip = cv.clamp(0, h - 1)

        # A view is valid only when the Gaussian center lies inside the mask.
        center_in_mask = mask_bool[cv_clip.long(), cu_clip.long()]
        if not center_in_mask.any():
            continue

        valid_idx = torch.where(center_in_mask)[0]  # [V]
        valid_gids = gaussian_ids[valid_idx]
        visible_counts.scatter_add_(
            0, valid_gids,
            torch.ones_like(valid_gids, dtype=torch.int32)
        )

        # Test endpoint overflow.
        ep = endpoints[valid_idx]                              # [V, 4, 2]
        ep_u = torch.round(ep[:, :, 0]).to(torch.int32)        # [V, 4]
        ep_v = torch.round(ep[:, :, 1]).to(torch.int32)        # [V, 4]

        # Image-boundary overflow counts as mask overflow.
        out_of_bounds = (ep_u < 0) | (ep_u >= w) | (ep_v < 0) | (ep_v >= h)

        # Clamp endpoints before querying the mask.
        ep_u_clip = ep_u.clamp(0, w - 1)
        ep_v_clip = ep_v.clamp(0, h - 1)
        ep_mask_val = mask_bool[ep_v_clip.long(), ep_u_clip.long()]  # [V, 4]

        # An endpoint is outside when the mask is false or it exceeds the image.
        ep_outside = (~ep_mask_val) | out_of_bounds  # [V, 4]

        # Any outside endpoint marks the Gaussian as overflowing in this view.
        any_overflow = ep_outside.any(dim=1)  # [V]

        if not any_overflow.any():
            continue

        overflow_local_idx = torch.where(any_overflow)[0]  # [K]
        overflow_idx = valid_idx[overflow_local_idx]
        overflow_gids = gaussian_ids[overflow_idx]

        overflow_counts.scatter_add_(
            0, overflow_gids,
            torch.ones_like(overflow_gids, dtype=torch.int32)
        )

        if anisotropic:
            # ============================================================
            # Anisotropic 3D-axis attribution and directional distance
            # ============================================================
            W_3x3 = view_matrices[view_idx][:3, :3].to(dev).float()  # [3, 3]
            fx, fy, principal_x, principal_y = (
                float(value) for value in intrinsics_list[view_idx]
            )

            # Flatten all overflowing (Gaussian, endpoint) pairs.
            overflow_ep_flags = ep_outside[overflow_local_idx]  # [K, 4]
            ov_gs_local, ov_ep_idx = torch.where(overflow_ep_flags)  # [B], [B]

            if len(ov_gs_local) == 0:
                continue

            # Map the flattened pairs back to view and global indices.
            ov_local_i = overflow_local_idx[ov_gs_local]
            ov_vi = valid_idx[ov_local_i]
            ov_gid = gaussian_ids[ov_vi]

            # Endpoint directions and ellipse geometry.
            ov_dirs = ep_dirs[ov_vi, ov_ep_idx]   # [B, 2]
            is_long = ov_ep_idx < 2               # [B]
            ov_semi_len = torch.where(is_long, len_long[ov_vi], len_short[ov_vi])   # [B]
            ov_sigma_sq = torch.where(is_long, lam1_arr[ov_vi], lam2_arr[ov_vi])    # [B]

            # Centers and endpoint displacements.
            ov_cx = cx[ov_vi]
            ov_cy = cy[ov_vi]
            ov_disp_x = ov_dirs[:, 0] * ov_semi_len
            ov_disp_y = ov_dirs[:, 1] * ov_semi_len

            # Batched directional boundary distance.
            d_dir = _chunked_directional_boundary_distance(
                ov_cx, ov_cy, ov_disp_x, ov_disp_y, mask_bool
            )  # [B]

            # Target variance where the compressed 3-sigma extent equals d_dir.
            target_var = (d_dir / sigma_cutoff) ** 2  # [B]

            # Keep endpoints whose target variance requires compression.
            need_compress_ep = target_var < ov_sigma_sq  # [B]
            if not need_compress_ep.any():
                continue

            # Extract the compression subset.
            nc_idx = torch.where(need_compress_ep)[0]  # [C]
            nc_gid = ov_gid[nc_idx]
            nc_dirs = ov_dirs[nc_idx]              # [C, 2]
            nc_sigma_sq = ov_sigma_sq[nc_idx]      # [C]
            nc_target_var = target_var[nc_idx]      # [C]
            nc_cx = ov_cx[nc_idx]                   # [C]
            nc_cy = ov_cy[nc_idx]                   # [C]
            C_len = len(nc_idx)

            # Batched 3D-axis contribution analysis.
            # W·R: [C, 3, 3]
            R_gs_batch = all_rot_matrices[nc_gid]              # [C, 3, 3]
            WR_batch = W_3x3.unsqueeze(0) @ R_gs_batch         # [C, 3, 3]

            # Retain the complete first-order pinhole perspective coupling.
            # The exact Jacobian also has a shared 1/z factor; it cancels in
            # axis ranking and normalized contribution, so only q_bar is used.
            projected_x, projected_y = depth_free_perspective_axis_projection(
                axis_x=WR_batch[:, 0, :],
                axis_y=WR_batch[:, 1, :],
                axis_z=WR_batch[:, 2, :],
                mean_x=nc_cx.unsqueeze(1),
                mean_y=nc_cy.unsqueeze(1),
                fx=fx,
                fy=fy,
                principal_x=principal_x,
                principal_y=principal_y,
            )

            # q_bar contains the depth-free projected direction of every axis.
            q_batch = torch.zeros(C_len, 3, 2, dtype=torch.float32, device=dev)
            q_batch[:, :, 0] = projected_x  # [C, 3]
            q_batch[:, :, 1] = projected_y  # [C, 3]

            # Dot product u * q[d].
            dot = (nc_dirs.unsqueeze(1) * q_batch).sum(dim=2)  # [C, 3]

            # w[d] = s_3d[d]² × (u·q[d])²
            s_3d_batch = scales_3d[nc_gid]  # [C, 3]
            w = s_3d_batch ** 2 * dot ** 2   # [C, 3]

            d_star = w.argmax(dim=1)                                       # [C]
            w_total = w.sum(dim=1)                                         # [C]
            arange_C = torch.arange(C_len, device=dev)
            w_dominant = w[arange_C, d_star]                               # [C]

            # Discard numerically invalid contributions.
            valid_w = (w_dominant > 1e-12) & (w_total > 1e-12)
            if not valid_w.any():
                continue

            vw_idx = torch.where(valid_w)[0]
            vw_gid = nc_gid[vw_idx]
            vw_d_star = d_star[vw_idx]
            vw_sigma_sq = nc_sigma_sq[vw_idx]
            vw_target_var = nc_target_var[vw_idx]
            vw_w_dominant = w_dominant[vw_idx]
            vw_w_total = w_total[vw_idx]

            # Relative contribution, independent of depth.
            alpha_d_star = vw_w_dominant / vw_w_total
            # Dominant-axis contribution in squared pixels.
            w_dominant_true = alpha_d_star * vw_sigma_sq

            # Compression factors.
            # σ'²_u = (σ²_u - w_{d*}) + f²·w_{d*} = target_var
            residual = vw_sigma_sq - w_dominant_true
            numerator = vw_target_var - residual

            f_d = torch.where(
                numerator <= 0,
                torch.full_like(numerator, min_compression),
                torch.sqrt(
                    (numerator / w_dominant_true.clamp(min=1e-12)).clamp(min=1e-12)
                ).clamp(min_compression, 1.0)
            )

            # Apply the strictest value for each (Gaussian, axis) pair.
            flat_idx_out = vw_gid * 3 + vw_d_star  # [M]
            min_factors_per_axis.view(-1).scatter_reduce_(
                0, flat_idx_out, f_d, reduce='amin', include_self=True
            )

            # Mark axes that contributed to an overflow.
            axis_has_overflow.view(-1)[flat_idx_out] = True

        else:
            # ============================================================
            # Legacy isotropic CPU fallback using a distance transform
            # ============================================================
            mask_np = mask_t.cpu().numpy()
            mask_uint8 = (mask_np * 255).astype(np.uint8) if mask_np.max() <= 1 else mask_np.astype(np.uint8)
            dt = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5).astype(np.float32)

            oi_cpu = overflow_idx.cpu()
            safe_r_np = dt[cv_clip[oi_cpu].cpu().numpy(), cu_clip[oi_cpu].cpu().numpy()]
            safe_r = torch.from_numpy(safe_r_np).to(dev)
            overflow_radii = radii_max[overflow_idx]

            factors = torch.where(
                overflow_radii > 0,
                safe_r / overflow_radii.clamp(min=1e-6),
                torch.ones_like(overflow_radii)
            )

            # Use the same strictest factor for all three axes.
            for k in range(len(overflow_gids)):
                gid = overflow_gids[k]
                for d in range(3):
                    axis_has_overflow[gid, d] = True
                    if factors[k] < min_factors_per_axis[gid, d]:
                        min_factors_per_axis[gid, d] = factors[k]

    # Vectorized voting across views.
    per_axis_factors = torch.ones(xyz_len, 3, dtype=torch.float32, device=dev)
    safe_visible = visible_counts.clamp(min=1).float()
    overflow_ratios = overflow_counts.float() / safe_visible

    need_compress = (overflow_ratios > tolerance_ratio) & (overflow_counts > 0)

    if need_compress.any():
        # Compress only axes with recorded overflow evidence.
        apply_mask = need_compress.unsqueeze(1) & axis_has_overflow  # [N, 3]
        per_axis_factors[apply_mask] = \
            min_factors_per_axis[apply_mask].clamp(min_compression, 1.0)

    # Release intermediate GPU memory.
    del visible_counts, overflow_counts, min_factors_per_axis, axis_has_overflow
    if anisotropic:
        del all_rot_matrices
    if dev.type == 'cuda':
        torch.cuda.empty_cache()

    return per_axis_factors
