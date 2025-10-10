"""Integration test for the controller.py module."""

import os

import numpy as np

from adaptive_oscillator.controller import AOController
from adaptive_oscillator.utils.parser_utils import LogFiles, LogParser


def test_ao_controller():
    """Test the AOController class."""
    # Arrange
    log_dir = os.path.join("data", "walk_5")
    log_files = LogFiles(log_dir)
    log_data = LogParser(log_files)

    # Act
    controller = AOController(show_plots=False)
    for _ii, ang_deg in enumerate(log_data.data.left.hip.angles.x_deg):
        th = np.deg2rad(ang_deg)
        dth = np.deg2rad(ang_deg)
        t = log_data.data.left.hip.time[_ii] - log_data.data.left.hip.time[0]
        controller.step(t=t, x=th, x_dot=dth)
