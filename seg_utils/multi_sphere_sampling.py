"""Spherical spiral sampling for smooth supplemental 3DGS views.

The sampler generates a single continuous camera trajectory on a sphere.
Azimuth rotates uniformly for ``n_layers`` revolutions while elevation moves
smoothly from ``phi_max_deg`` to ``phi_min_deg``.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


def sample_multi_sphere_views(center, radius, n_layers, n_points_per_layer,
                               random_seed=None, phi_max_deg=60.0, phi_min_deg=-10.0):
    """Sample camera positions along a spherical spiral.

    Parameter semantics remain compatible with the legacy sampler:
      - n_layers: spiral revolutions;
      - n_points_per_layer: samples per revolution;
      - total_points = n_layers * n_points_per_layer.

    Parameters
    ----------
    center : array-like, shape (3,)
        Sphere center, usually the object center.
    radius : float
        Sphere radius, or camera distance from the center.
    n_layers : int
        Number of spiral revolutions.
    n_points_per_layer : int
        Samples per revolution.
    random_seed : int, optional
        Retained for API compatibility; sampling itself is deterministic.
    phi_max_deg : float
        Initial elevation in degrees; defaults to 60.
    phi_min_deg : float
        Final elevation in degrees; defaults to -10.

    Returns
    -------
    all_points : ndarray, shape (N, 3)
        Sample coordinates in spiral order.
    spiral_info : dict
        Sampling parameters and generated points.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    center = np.array(center, dtype=np.float64)

    total_points = n_layers * n_points_per_layer

    # Elevation transitions linearly from phi_max to phi_min.
    phi_max = np.deg2rad(phi_max_deg)
    phi_min = np.deg2rad(phi_min_deg)
    phi = np.linspace(phi_max, phi_min, total_points)

    # Azimuth rotates uniformly through n_layers revolutions.
    theta = np.linspace(0, 2 * np.pi * n_layers, total_points, endpoint=False)

    # Convert spherical to Cartesian coordinates. Phi is elevation:
    # zero lies on the equator and positive values lie above it.
    x = center[0] + radius * np.cos(phi) * np.cos(theta)
    y = center[1] + radius * np.cos(phi) * np.sin(theta)
    z = center[2] + radius * np.sin(phi)

    all_points = np.stack([x, y, z], axis=-1).astype(np.float64)

    spiral_info = {
        'type': 'spiral',
        'radius': radius,
        'n_layers': n_layers,
        'n_points_per_layer': n_points_per_layer,
        'total_points': total_points,
        'phi_max_deg': phi_max_deg,
        'phi_min_deg': phi_min_deg,
        'points': all_points,
    }

    return all_points, spiral_info


def visualize_multi_sphere_sampling(center, radius, spiral_info, save_path=None):
    """Visualize spherical spiral sampling.

    A color-gradient path shows sampling order, with a green start and red end.

    Parameters
    ----------
    center : array-like
        Sphere center.
    radius : float
        Sphere radius.
    spiral_info : dict
        Metadata returned by sample_multi_sphere_views.
    save_path : str, optional
        Output image path; show interactively when omitted.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    if isinstance(center, torch.Tensor):
        center = center.cpu().detach().numpy()
    else:
        center = np.array(center)

    # Draw the sphere wireframe.
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.15, color='lightgray')

    # Load sampled points.
    points = spiral_info['points']
    n = len(points)

    # Color the spiral path using a viridis gradient.
    colors = plt.cm.viridis(np.linspace(0, 1, n))

    # Draw each segment separately to preserve the gradient.
    for i in range(n - 1):
        ax.plot(points[i:i+2, 0], points[i:i+2, 1], points[i:i+2, 2],
                color=colors[i], linewidth=1.5, alpha=0.8)

    # Draw sample points.
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=np.arange(n), cmap='viridis', s=30, alpha=0.9,
               label=f'Spiral ({n} points, {spiral_info["n_layers"]} revolutions)')

    # Mark the first and last samples.
    ax.scatter(*points[0], c='lime', s=120, marker='^', label='Start', zorder=10, edgecolors='black')
    ax.scatter(*points[-1], c='red', s=120, marker='v', label='End', zorder=10, edgecolors='black')

    # Mark the sphere center.
    ax.scatter(center[0], center[1], center[2],
               c='gold', s=100, marker='*', label='Center', zorder=10)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper left', fontsize=9)

    title = (f'Spherical Spiral Sampling\n'
             f'revolutions={spiral_info["n_layers"]}, '
             f'points/rev={spiral_info["n_points_per_layer"]}, '
             f'φ: {spiral_info["phi_max_deg"]}° → {spiral_info["phi_min_deg"]}°')
    ax.set_title(title, fontsize=11)

    # Use equal coordinate ranges.
    max_range = radius * 1.3
    ax.set_xlim([center[0] - max_range, center[0] + max_range])
    ax.set_ylim([center[1] - max_range, center[1] + max_range])
    ax.set_zlim([center[2] - max_range, center[2] + max_range])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Spherical spiral sampling visualization saved to: {save_path}")
    else:
        plt.show()

    plt.close()


# Example usage
if __name__ == "__main__":
    sphere_center = np.array([0.0, 0.0, 0.0])
    radius = 30
    n_layers = 4        # 4 revolutions
    n_points_per_layer = 8  # 8 points per revolution → 32 total

    all_points, spiral_info = sample_multi_sphere_views(
        sphere_center, radius, n_layers, n_points_per_layer, random_seed=42
    )

    print(f"Total sampled points: {len(all_points)}")
    print(f"Consecutive distance stats:")
    dists = np.linalg.norm(np.diff(all_points, axis=0), axis=1)
    print(f"  mean: {dists.mean():.2f}, std: {dists.std():.2f}, "
          f"max: {dists.max():.2f}, min: {dists.min():.2f}")

    visualize_multi_sphere_sampling(
        sphere_center, radius, spiral_info,
        save_path="./sphere_spiral_sampling_visualization.png"
    )
