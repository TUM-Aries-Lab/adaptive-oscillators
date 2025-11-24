"""Common definitions for my module."""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from loguru import logger

np.set_printoptions(precision=3, floatmode="fixed", suppress=True)

# plot definitions
FIG_SIZE = (12, 6)  # width, height in inches
ALPHA = 0.8
LEGEND_LOC = "upper right"

# --- Directories ---
ROOT_DIR: Path = Path("src").parent
DATA_DIR: Path = ROOT_DIR / "data"
RESULTS_DIR: Path = DATA_DIR / "results"
RECORDINGS_DIR: Path = DATA_DIR / "recordings"
LOG_DIR: Path = DATA_DIR / "logs"

# Default encoding
ENCODING: str = "utf-8"

DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"
TIME_FORMAT = "%H:%M:%S.%f"


@dataclass
class LogLevel:
    """Log level."""

    trace: str = "TRACE"
    debug: str = "DEBUG"
    info: str = "INFO"
    success: str = "SUCCESS"
    warning: str = "WARNING"
    error: str = "ERROR"
    critical: str = "CRITICAL"

    def __iter__(self):
        """Iterate over log levels."""
        return iter(asdict(self).values())


DEFAULT_LOG_LEVEL = LogLevel.info
DEFAULT_LOG_FILENAME = "log_file"

LOG_FILE_EXT = ".txt"
logger.configure(handlers=[{"sink": sys.stderr, "level": LogLevel.info}])

NUMPY_PRINT_PRECISION = 3
np.set_printoptions(precision=NUMPY_PRINT_PRECISION)


@dataclass
class AOParameters:
    """Adaptive Oscillator parameters."""

    eta: float = 1
    nu_phi: float = 10
    nu_omega: float = 10
    n_harmonics: int = 3
    omega_init: float = 1e-3


@dataclass
class PIDGains:
    """PID gains."""

    kp: float = 5.0
    ki: float = 0.0
    kd: float = 0.1


DEFAULT_DELTA_TIME = 0.01


class LogFileKeys:
    """Enum for the log file categories."""

    ACCEL = "Accelerometers"
    ANGLE = "Angles"
    GRAVITY = "Gravity"
    GYRO = "Gyroscopes"
    QUAT = "Quaternions"


class Segments:
    """Body segments where each IMU is attached."""

    PELVIS = "Pelvis"
    UPPER_LEG = "Upper Leg"
    LOWER_LEG = "Lower Leg"
    FOOT = "Foot"


class Joints:
    """Body joints where each IMU is attached."""

    HIP = "Hip"
    KNEE = "Knee"
    ANKLE = "Ankle"


class IMUHeader:
    """Headers for the IMU sensor data."""

    TIME = "Time"
    PELVIS_X = "Pelvis_x"
    PELVIS_Y = "Pelvis_y"
    PELVIS_Z = "Pelvis_z"
    UPPLEG_X = "UppLeg_x"
    UPPLEG_Y = "UppLeg_y"
    UPPLEG_Z = "UppLeg_z"
    LOWLEG_X = "LowLeg_x"
    LOWLEG_Y = "LowLeg_y"
    LOWLEG_Z = "LowLeg_z"
    FOOT_X = "Foot_x"
    FOOT_Y = "Foot_y"
    FOOT_Z = "Foot_z"


class QuaternionHeader:
    """Headers for the quaternion data."""

    TIME = "Time"
    PELVIS_W = "Pelvis_w"
    PELVIS_X = "Pelvis_x"
    PELVIS_Y = "Pelvis_y"
    PELVIS_Z = "Pelvis_z"
    UPPLEG_W = "UppLeg_w"
    UPPLEG_X = "UppLeg_x"
    UPPLEG_Y = "UppLeg_y"
    UPPLEG_Z = "UppLeg_z"
    LOWLEG_W = "LowLeg_w"
    LOWLEG_X = "LowLeg_x"
    LOWLEG_Y = "LowLeg_y"
    LOWLEG_Z = "LowLeg_z"
    FOOT_W = "Foot_w"
    FOOT_X = "Foot_x"
    FOOT_Y = "Foot_y"
    FOOT_Z = "Foot_z"


class AnglesHeader:
    """Headers for the angle data."""

    TIME = "Time"
    HIP_X = "Hip_x"
    HIP_Y = "Hip_y"
    HIP_Z = "Hip_z"
    KNEE_X = "Knee_x"
    KNEE_Y = "Knee_y"
    KNEE_Z = "Knee_z"
    ANKLE_X = "Ankle_x"
    ANKLE_Y = "Ankle_y"
    ANKLE_Z = "Ankle_z"


IMU_SEGMENT_FIELDS = {
    "pelvis": [
        IMUHeader.PELVIS_X,
        IMUHeader.PELVIS_Y,
        IMUHeader.PELVIS_Z,
    ],
    "upper_leg": [
        IMUHeader.UPPLEG_X,
        IMUHeader.UPPLEG_Y,
        IMUHeader.UPPLEG_Z,
    ],
    "lower_leg": [
        IMUHeader.LOWLEG_X,
        IMUHeader.LOWLEG_Y,
        IMUHeader.LOWLEG_Z,
    ],
    "foot": [
        IMUHeader.FOOT_X,
        IMUHeader.FOOT_Y,
        IMUHeader.FOOT_Z,
    ],
}

QUATERNION_SEGMENT_FIELDS = {
    "pelvis": [
        QuaternionHeader.PELVIS_W,
        QuaternionHeader.PELVIS_X,
        QuaternionHeader.PELVIS_Y,
        QuaternionHeader.PELVIS_Z,
    ],
    "upper_leg": [
        QuaternionHeader.UPPLEG_W,
        QuaternionHeader.UPPLEG_X,
        QuaternionHeader.UPPLEG_Y,
        QuaternionHeader.UPPLEG_Z,
    ],
    "lower_leg": [
        QuaternionHeader.LOWLEG_W,
        QuaternionHeader.LOWLEG_X,
        QuaternionHeader.LOWLEG_Y,
        QuaternionHeader.LOWLEG_Z,
    ],
    "foot": [
        QuaternionHeader.FOOT_W,
        QuaternionHeader.FOOT_X,
        QuaternionHeader.FOOT_Y,
        QuaternionHeader.FOOT_Z,
    ],
}

ANGLES_SEGMENT_FIELDS = {
    "hip": [
        AnglesHeader.HIP_X,
        AnglesHeader.HIP_Y,
        AnglesHeader.HIP_Z,
    ],
    "knee": [
        AnglesHeader.KNEE_X,
        AnglesHeader.KNEE_Y,
        AnglesHeader.KNEE_Z,
    ],
    "ankle": [
        AnglesHeader.ANKLE_X,
        AnglesHeader.ANKLE_Y,
        AnglesHeader.ANKLE_Z,
    ],
}
