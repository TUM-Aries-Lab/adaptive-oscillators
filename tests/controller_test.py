"""Integration test for the controller.py module."""

import os

import numpy as np

from adaptive_oscillator.controller import AOController
from adaptive_oscillator.log_files import LogFiles, LogParser

TEST_DATA_DIR = os.path.join("data", "2025_10_old_data", "walk_5")


def test_ao_controller():
    """Test the AOController class."""
    # Arrange
    log_dir = TEST_DATA_DIR
    log_files = LogFiles(log_dir)
    log_data = LogParser(log_files)

    # Act
    controller = AOController(show_plots=False)
    for _ii, ang_deg in enumerate(log_data.data.left.hip.angles.x):
        th = np.deg2rad(ang_deg)
        dth = np.deg2rad(ang_deg)
        t = log_data.data.left.hip.time[_ii] - log_data.data.left.hip.time[0]
        controller.step(t=t, x=th, x_dot=dth)
