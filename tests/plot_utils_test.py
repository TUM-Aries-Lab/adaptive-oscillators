"""Integration test for the plot_utils.py module."""

from adaptive_oscillator.utils.parser_utils import LogFiles


def test_plot_log_data():
    """Test the plot_log_data class."""
    # Arrange
    log_dir = "data/walk_5"
    log_files = LogFiles(log_dir)

    # Act
    log_files.plot()
