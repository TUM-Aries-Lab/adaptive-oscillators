"""Integration test for the controller.py module."""

from adaptive_oscillator.controller import AOController


def test_ao_controller():
    """Test the AOController class."""
    # Arrange
    log_dir = "data/walk_5"

    # Act
    controller = AOController(real_time=False, plot=False)
    controller.replay(log_dir=log_dir)
