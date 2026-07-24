import torch

def compute_robust_sphere_center(xyz, sample_size=50000, std_mul=3.0):
    """
    Lightweight outlier removal: mean center + standard deviation threshold, avoid full sorting/quantiles.
    - Subsampling to estimate distance distribution, reduce computational cost for large point clouds
    - Threshold: mean_dist + std_mul * std_dist
    """
    n = xyz.shape[0]
    if n < 10:
        return xyz.mean(dim=0)

    center0 = xyz.mean(dim=0)
    distances = torch.norm(xyz - center0, dim=1)

    # Subsampling to estimate distance distribution
    if n > sample_size:
        idx = torch.randperm(n, device=xyz.device)[:sample_size]
        dist_sample = distances[idx]
    else:
        dist_sample = distances

    mean_dist = dist_sample.mean()
    std_dist = dist_sample.std(unbiased=False)
    threshold = mean_dist + std_mul * std_dist

    mask = distances <= threshold
    # Prevent over-removal, keep at least 10% or 10 points
    if mask.sum() < max(10, int(0.1 * n)):
        k = max(10, int(0.1 * n))
        # Only take top k smallest distances, avoid full sorting
        topk_thresh = torch.topk(distances, k, largest=False).values.max()
        mask = distances <= topk_thresh

    filtered_xyz = xyz[mask]
    if filtered_xyz.shape[0] == 0:
        return center0
    return filtered_xyz.mean(dim=0)


def robust_center_outlier_filter_indices(
    xyz: torch.Tensor,
    indices: torch.Tensor,
    std_mul: float = 3.0,
    sample_size: int = 50000,
):
    """Outlier removal based on distance to robust center。

    Returns:
      kept_indices: torch.Tensor[int64]  Global point indices to keep (on original xyz)
      removed_indices: torch.Tensor[int64] Global point indices to remove
      stats: dict
    """
    if not isinstance(xyz, torch.Tensor):
        xyz = torch.as_tensor(xyz)
    if not isinstance(indices, torch.Tensor):
        indices = torch.as_tensor(indices)

    device = xyz.device
    indices = indices.to(device=device, dtype=torch.long)

    n = int(indices.numel())
    if n == 0:
        return indices, indices[:0], {"n_in": 0, "n_kept": 0, "n_removed": 0}

    if n < 3:
        return indices, indices[:0], {"n_in": n, "n_kept": n, "n_removed": 0}

    # Get subset of xyz
    subset_xyz = xyz[indices]

    # Compute robust center
    robust_center = compute_robust_sphere_center(subset_xyz, sample_size=sample_size, std_mul=std_mul)

    # Compute distance to center for each point
    distances = torch.norm(subset_xyz - robust_center.unsqueeze(0), dim=1)

    # Compute threshold: mean + std_mul * std
    mean_dist = distances.mean()
    std_dist = distances.std(unbiased=False)
    threshold = mean_dist + float(std_mul) * std_dist

    # Mark outliers
    is_outlier = distances > threshold
    kept = indices[~is_outlier]
    removed = indices[is_outlier]

    stats = {
        "n_in": n,
        "robust_center": robust_center.detach().cpu().tolist(),
        "mean_dist": float(mean_dist.detach().cpu()),
        "std_dist": float(std_dist.detach().cpu()),
        "threshold": float(threshold.detach().cpu()),
        "n_removed": int(removed.numel()),
        "n_kept": int(kept.numel()),
        "std_mul": float(std_mul),
        "sample_size": int(sample_size),
    }

    return kept, removed, stats
