"""Common definitions for my module."""

import sys

import numpy as np
from loguru import logger

# plot definitions
FIG_SIZE = (14, 10)  # width=8 inches, height=6 inches
ALPHA = 0.8

TIME_FORMAT = "%H:%M:%S.%f"

LOG_FILE_EXT = ".txt"
logger.configure(handlers=[{"sink": sys.stderr, "level": "INFO"}])

NUMPY_PRINT_PRECISION = 3
np.set_printoptions(precision=NUMPY_PRINT_PRECISION)

# AOParameters
ETA = 0.05
N_HARMONICS = 3
NU_PHI = 0.5
NU_OMEGA = 0.5

DEFAULT_DELTA_TIME = 0.01


class LogFileKeys:
    """Enum for the log file categories."""

    ACCEL = "Accelerometers"
    ANGLE = "Angles"
    GRAVITY = "Gravity"
    GYRO = "Gyroscopes"
    QUAT = "Quaternions"


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
