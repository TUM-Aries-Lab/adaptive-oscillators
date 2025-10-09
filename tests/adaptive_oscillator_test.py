"""Test controller.py."""

import numpy as np
import pytest

from adaptive_oscillator.oscillator import (
    AOParameters,
    GaitPhaseEstimator,
    LowLevelController,
)
from tests.utils_for_test import sample_walking_data


@pytest.mark.parametrize(
    "gait_freq",
    [
        0.4,
        0.5,
        0.6,
    ],
)
def test_adaptive_oscillator(gait_freq: float) -> None:
    """Test the controller."""
    # Arrange
    params = AOParameters()
    estimator = GaitPhaseEstimator(params)
    controller = LowLevelController()

    # Act
    dt = 0.01
    t_vals, theta_il, theta_il_dot = sample_walking_data(period=gait_freq)

    theta_m = 0.0
    for t, th, dth in zip(t_vals, theta_il, theta_il_dot):
        phi = estimator.update(t, th, dth)
        omega_cmd = controller.compute(phi, theta_m, dt)
        theta_m += omega_cmd * dt

    # Assert
    np.testing.assert_almost_equal(estimator.ao.omega, gait_freq * 2 * np.pi, decimal=1)
