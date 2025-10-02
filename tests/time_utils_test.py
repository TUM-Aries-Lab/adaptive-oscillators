"""Test the time utils."""

import pytest

from iit_rehab.time_utils import time_str_to_seconds


@pytest.mark.parametrize(
    "time_str, expected_seconds",
    [
        ("12:34:56.789", 45296.789),
        ("12:34:56.0", 45296),
        ("12:34:00.0", 45240),
    ],
)
def test_time_str_to_seconds(time_str: str, expected_seconds: float):
    """Test the time string to seconds conversion."""
    # Act
    time_float = time_str_to_seconds(time_str)

    # Assert
    assert time_float == expected_seconds
