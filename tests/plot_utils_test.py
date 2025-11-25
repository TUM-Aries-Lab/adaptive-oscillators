"""Integration test for the plot_utils.py module."""

import os

from adaptive_oscillator.log_files import LogFiles

TEST_DATA_DIR = os.path.join("data", "2025_10_old_data", "walk_5")


def test_plot_log_data():
    """Test the plot_log_data class."""
    # Arrange
    log_dir = TEST_DATA_DIR
    log_files = LogFiles(log_dir)

    # Act
    log_files.plot()
