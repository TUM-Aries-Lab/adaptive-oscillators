"""Utility functions for testing."""

import numpy as np
from numpy.typing import NDArray


def sample_walking_data(
    period: float, t_start: float = 0.0, t_end: float = 100.0, dt: float = 0.01
) -> tuple[NDArray, NDArray, NDArray]:
    """Sample walking trajectory."""
    two_rad = 2 * np.pi
    t_vals = np.arange(t_start, t_end, dt)
    theta_il = np.sin(period * two_rad * t_vals)
    theta_il_dot = period * two_rad * np.cos(period * two_rad * t_vals)
    return t_vals, theta_il, theta_il_dot
