"""Test the parser modules."""

from pathlib import Path

import numpy as np
import pytest

from adaptive_oscillator.definitions import LogFileKeys
from adaptive_oscillator.parser_utils import (
    AngleParser,
    IMUParser,
    LogFiles,
    LogParser,
    QuaternionParser,
)

TEST_DIR = Path(__file__).parent.parent / "data" / "walk_4"
DECIMAL_ACCURACY = 2


def test_log_files() -> None:
    """Test the log files dataclass."""
    # Arrange
    test_dir = TEST_DIR

    # Act
    log_files = LogFiles(test_dir)

    # Assert
    assert log_files.accel.right == test_dir / f"{LogFileKeys.ACCEL}_right.txt"
    assert log_files.accel.left == test_dir / f"{LogFileKeys.ACCEL}_left.txt"

    assert log_files.angle.right == test_dir / f"{LogFileKeys.ANGLE}_right.txt"
    assert log_files.angle.left == test_dir / f"{LogFileKeys.ANGLE}_left.txt"

    assert log_files.gravity.right == test_dir / f"{LogFileKeys.GRAVITY}_right.txt"
    assert log_files.gravity.left == test_dir / f"{LogFileKeys.GRAVITY}_left.txt"

    assert log_files.gyro.right == test_dir / f"{LogFileKeys.GYRO}_right.txt"
    assert log_files.gyro.left == test_dir / f"{LogFileKeys.GYRO}_left.txt"

    assert log_files.quat.right == test_dir / f"{LogFileKeys.QUAT}_right.txt"
    assert log_files.quat.left == test_dir / f"{LogFileKeys.QUAT}_left.txt"


@pytest.mark.parametrize(
    "filename, expected_data",
    [
        (
            LogFileKeys.ACCEL + "_left.txt",
            np.array(
                [
                    60720.15,
                    -0.01,
                    -8.70,
                    0.77,
                    -1.66,
                    -10.18,
                    0.04,
                    2.28,
                    -10.16,
                    1.57,
                    -0.31,
                    10.61,
                    1.04,
                ]
            ),
        ),
    ],
)
def test_imu_parser(filename: str, expected_data: np.ndarray) -> None:
    """Test the file parser."""
    # Arrange
    test_file = TEST_DIR / filename

    # Act
    data = IMUParser(test_file)
    data.parse()

    # Assert
    np.testing.assert_almost_equal(
        data.time[0], expected_data[0], decimal=DECIMAL_ACCURACY
    )
    np.testing.assert_almost_equal(
        data.pelvis[0], expected_data[1:4], decimal=DECIMAL_ACCURACY
    )

    np.testing.assert_almost_equal(
        data.upper_leg[0], expected_data[4:7], decimal=DECIMAL_ACCURACY
    )

    np.testing.assert_almost_equal(
        data.lower_leg[0], expected_data[7:10], decimal=DECIMAL_ACCURACY
    )

    np.testing.assert_almost_equal(
        data.foot[0], expected_data[10:13], decimal=DECIMAL_ACCURACY
    )


@pytest.mark.parametrize(
    "filename, expected_data",
    [
        (
            LogFileKeys.QUAT + "_left.txt",
            np.array(
                [
                    60720.15,
                    0.45,
                    -0.40,
                    0.57,
                    -0.54,
                    0.49,
                    -0.51,
                    0.54,
                    -0.44,
                    -0.57,
                    0.56,
                    -0.36,
                    0.46,
                    0.71,
                    0.62,
                    0.19,
                    0.25,
                ]
            ),
        ),
    ],
)
def test_quaternion_parser(filename: str, expected_data: np.ndarray) -> None:
    """Test the file parser."""
    # Arrange
    test_file = TEST_DIR / filename

    # Act
    data = QuaternionParser(test_file)
    data.parse()

    # Assert
    np.testing.assert_almost_equal(
        data.time[0], expected_data[0], decimal=DECIMAL_ACCURACY
    )
    np.testing.assert_almost_equal(
        data.pelvis[0], expected_data[1:5], decimal=DECIMAL_ACCURACY
    )

    np.testing.assert_almost_equal(
        data.upper_leg[0], expected_data[5:9], decimal=DECIMAL_ACCURACY
    )

    np.testing.assert_almost_equal(
        data.lower_leg[0], expected_data[9:13], decimal=DECIMAL_ACCURACY
    )

    np.testing.assert_almost_equal(
        data.foot[0], expected_data[13:17], decimal=DECIMAL_ACCURACY
    )


@pytest.mark.parametrize(
    "filename, expected_data",
    [
        (
            LogFileKeys.ANGLE + "_left.txt",
            np.array(
                [
                    60720.154283,
                    -8.83,
                    -15.56,
                    -0.04,
                    13.92,
                    -15.30,
                    -12.88,
                    -0.26,
                    -71.06,
                    178.82,
                ]
            ),
        ),
    ],
)
def test_angle_parser(filename: str, expected_data: np.ndarray) -> None:
    """Test the file parser."""
    # Arrange
    test_file = TEST_DIR / filename

    # Act
    data = AngleParser(test_file)
    data.parse()

    # Assert
    np.testing.assert_almost_equal(
        data.time[0], expected_data[0], decimal=DECIMAL_ACCURACY
    )
    np.testing.assert_almost_equal(
        data.hip[0], expected_data[1:4], decimal=DECIMAL_ACCURACY
    )

    np.testing.assert_almost_equal(
        data.knee[0], expected_data[4:7], decimal=DECIMAL_ACCURACY
    )

    np.testing.assert_almost_equal(
        data.ankle[0], expected_data[7:10], decimal=DECIMAL_ACCURACY
    )


def test_log_parser() -> None:
    """Test the log parser."""
    # Arrange
    test_dir = TEST_DIR
    log_files = LogFiles(test_dir)

    # Act
    log_data = LogParser(log_files)

    # Assert
    np.testing.assert_array_equal(
        log_data.data.right.foot.time, log_data.data.left.foot.time
    )
