"""Projection formulas used by ABR 3D-axis attribution."""


def depth_free_perspective_axis_projection(
    axis_x,
    axis_y,
    axis_z,
    mean_x,
    mean_y,
    fx,
    fy,
    principal_x,
    principal_y,
):
    """Project camera-space axes while retaining perspective coupling.

    The exact pinhole Jacobian contains a common outer factor ``1 / z``.
    That factor cancels when ABR ranks the three axes or normalizes their
    contributions, so this function returns the remaining depth-free terms::

        q_x_bar = fx * (a_x - xi * a_z)
        q_y_bar = fy * (a_y - eta * a_z)

    where ``xi = (mean_x - principal_x) / fx`` and
    ``eta = (mean_y - principal_y) / fy``. Inputs may be Python numbers or
    broadcast-compatible tensor objects.
    """
    xi = (mean_x - principal_x) / fx
    eta = (mean_y - principal_y) / fy
    projected_x = fx * (axis_x - xi * axis_z)
    projected_y = fy * (axis_y - eta * axis_z)
    return projected_x, projected_y
