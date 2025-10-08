"""Parser utils for log file data."""

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt

from adaptive_oscillator.base_classes import (
    AngleXYZ,
    Body,
    Joint,
    LeftRight,
    Limb,
    Quaternion,
    SensorFile,
    VectorXYZ,
)
from adaptive_oscillator.definitions import (
    ALPHA,
    ANGLES_SEGMENT_FIELDS,
    FIG_SIZE,
    IMU_SEGMENT_FIELDS,
    QUATERNION_SEGMENT_FIELDS,
    AnglesHeader,
    LogFileKeys,
    QuaternionHeader,
)
from adaptive_oscillator.utils.time_utils import time_str_to_seconds


class LogFiles:
    """Main entry point for accessing all sensor log files."""

    def __init__(self, base_path: str | Path) -> None:
        self._path = Path(base_path)
        self.accel = SensorFile(LogFileKeys.ACCEL, self._path)
        self.angle = SensorFile(LogFileKeys.ANGLE, self._path)
        self.gravity = SensorFile(LogFileKeys.GRAVITY, self._path)
        self.gyro = SensorFile(LogFileKeys.GYRO, self._path)
        self.quat = SensorFile(LogFileKeys.QUAT, self._path)

    def __repr__(self) -> str:  # pragma: no cover
        """Return a string representation of the LogFiles object."""
        return (
            f"Log files for dir: '{self._path}'"
            f"\n\t{self.accel.left}"
            f"\n\t{self.accel.right}"
            f"\n\t{self.angle.left}"
            f"\n\t{self.angle.right}"
            f"\n\t{self.gravity.left}"
            f"\n\t{self.gravity.right}"
            f"\n\t{self.gyro.left}"
            f"\n\t{self.gyro.right}"
            f"\n\t{self.quat.left}"
            f"\n\t{self.quat.right})"
        )

    def plot(self):
        """Plot log files."""
        logger.info("Plotting data.")
        accel_data = IMUParser(self.accel.right)
        accel_data.parse()
        accel_data.plot()

        gyro_data = IMUParser(self.gyro.right)
        gyro_data.parse()
        gyro_data.plot()

        quat_data = QuaternionParser(self.quat.right)
        quat_data.parse()
        quat_data.plot()

        accel_data = IMUParser(self.accel.left)
        accel_data.parse()
        accel_data.plot()

        gyro_data = IMUParser(self.gyro.left)
        gyro_data.parse()
        gyro_data.plot()

        quat_data = QuaternionParser(self.quat.left)
        quat_data.parse()
        quat_data.plot()


class IMUParser:
    """Parser for log files with limb information."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.time = np.array([])
        self.pelvis = VectorXYZ()
        self.upper_leg = VectorXYZ()
        self.lower_leg = VectorXYZ()
        self.foot = VectorXYZ()

    def parse(self):
        """Parse the log file and return a DataFrame."""
        raw_data = pd.read_csv(self.filepath, sep="\t+", engine="python")
        logger.debug(f"Parsing {self.filepath}")
        logger.debug(f"Columns: {raw_data.shape}")

        time_str = raw_data[AnglesHeader.TIME]
        self.time = np.array([time_str_to_seconds(t) for t in time_str])

        for segment_name, fields in IMU_SEGMENT_FIELDS.items():
            x = raw_data[fields[0]].to_numpy()
            y = raw_data[fields[1]].to_numpy()
            z = raw_data[fields[2]].to_numpy()
            setattr(self, segment_name, VectorXYZ(x, y, z))

    def plot(self):  # pragma: no cover
        """Plot the x, y, z data."""
        _, ax = plt.subplots(figsize=FIG_SIZE, sharex=True, nrows=4, ncols=1)

        for ii, (name, segment) in enumerate(
            zip(
                ["Pelvis", "Upper Leg", "Lower Leg", "Foot"],
                [self.pelvis, self.upper_leg, self.lower_leg, self.foot],
            )
        ):
            time = self.time - self.time[0]
            for axis in ["x", "y", "z"]:
                imu_signal = getattr(segment, axis)
                ax[ii].plot(time, imu_signal, label=f"{name}-{axis}", alpha=ALPHA)
            ax[ii].set_title(f"{name} - {self.filepath.stem}")
            ax[ii].set_xlabel("Time (s)")
            ax[ii].set_ylabel("Quaternion")
            ax[ii].legend(loc="upper right")
            ax[ii].grid(True)
            plt.tight_layout()


class AngleParser:
    """Parser for log files with angle."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.time = np.array([])
        self.hip = AngleXYZ()
        self.knee = AngleXYZ()
        self.ankle = AngleXYZ()

    def parse(self):
        """Parse the log file and return a DataFrame."""
        raw_data = pd.read_csv(self.filepath, sep="\t+", engine="python")
        logger.debug(f"Parsing {self.filepath}")
        logger.debug(f"Columns: {raw_data.shape}")

        time_str = raw_data[AnglesHeader.TIME]
        self.time = np.array([time_str_to_seconds(t) for t in time_str])

        for segment_name, fields in ANGLES_SEGMENT_FIELDS.items():
            x_deg = raw_data[fields[0]].to_numpy()
            y_deg = raw_data[fields[1]].to_numpy()
            z_deg = raw_data[fields[2]].to_numpy()
            setattr(self, segment_name, AngleXYZ(x_deg, y_deg, z_deg))


class QuaternionParser:
    """Parser for log files with quaternion information."""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.time = np.array([])
        self.pelvis = Quaternion()
        self.upper_leg = Quaternion()
        self.lower_leg = Quaternion()
        self.foot = Quaternion()

    def parse(self):
        """Parse the log file and return a DataFrame."""
        raw_data = pd.read_csv(self.filepath, sep="\t")
        logger.debug(f"Parsing {self.filepath}")
        logger.debug(f"Columns: {raw_data.shape}")

        time_str = raw_data[QuaternionHeader.TIME]
        self.time = np.array([time_str_to_seconds(t) for t in time_str])

        for segment_name, fields in QUATERNION_SEGMENT_FIELDS.items():
            w = raw_data[fields[0]].to_numpy()
            x = raw_data[fields[1]].to_numpy()
            y = raw_data[fields[2]].to_numpy()
            z = raw_data[fields[3]].to_numpy()
            setattr(self, segment_name, Quaternion(w, x, y, z))

    def plot(self):  # pragma: no cover
        """Plot the Quaternion data."""
        _, ax = plt.subplots(figsize=FIG_SIZE, sharex=True, nrows=4, ncols=1)

        for ii, (name, segment) in enumerate(
            zip(
                ["Pelvis", "Upper Leg", "Lower Leg", "Foot"],
                [self.pelvis, self.upper_leg, self.lower_leg, self.foot],
            )
        ):
            time = self.time - self.time[0]
            for axis in ["w", "x", "y", "z"]:
                quat_component = getattr(segment, axis)
                ax[ii].plot(time, quat_component, label=f"{name}-{axis}", alpha=ALPHA)
            ax[ii].set_title(f"{name} Orientation")
            ax[ii].set_xlabel("Time (s)")
            ax[ii].set_ylabel("Quaternion")
            ax[ii].legend(loc="upper right")
            ax[ii].grid(True)
            plt.tight_layout()


class LogParser:
    """Parser for log files with limb information."""

    def __init__(self, log_files: LogFiles):
        logger.info(f"Parsing {log_files}")
        accel_data_right = IMUParser(log_files.accel.right)
        accel_data_right.parse()
        accel_data_left = IMUParser(log_files.accel.left)
        accel_data_left.parse()

        gyro_data_right = IMUParser(log_files.gyro.right)
        gyro_data_right.parse()
        gyro_data_left = IMUParser(log_files.gyro.left)
        gyro_data_left.parse()

        quat_data_right = QuaternionParser(log_files.quat.right)
        quat_data_right.parse()
        quat_data_left = QuaternionParser(log_files.quat.left)
        quat_data_left.parse()

        angles_right = AngleParser(log_files.angle.right)
        angles_right.parse()
        angles_left = AngleParser(log_files.angle.left)
        angles_left.parse()

        time = accel_data_right.time

        pelvis_right = Limb(
            time=time,
            accel=accel_data_right.pelvis,
            gyro=gyro_data_right.pelvis,
            quat=quat_data_right.pelvis,
        )
        upper_leg_right = Limb(
            time=time,
            accel=accel_data_right.upper_leg,
            gyro=gyro_data_right.upper_leg,
            quat=quat_data_right.upper_leg,
        )
        lower_leg_right = Limb(
            time=time,
            accel=accel_data_right.lower_leg,
            gyro=gyro_data_right.lower_leg,
            quat=quat_data_right.lower_leg,
        )
        foot_right = Limb(
            time=time,
            accel=accel_data_right.foot,
            gyro=gyro_data_right.foot,
            quat=quat_data_right.foot,
        )
        hip_right = Joint(time=time, angles=angles_right.hip)
        knee_right = Joint(time=time, angles=angles_right.knee)
        ankle_right = Joint(time=time, angles=angles_right.ankle)

        pelvis_left = Limb(
            time=time,
            accel=accel_data_left.pelvis,
            gyro=gyro_data_left.pelvis,
            quat=quat_data_left.pelvis,
        )
        upper_leg_left = Limb(
            time=time,
            accel=accel_data_left.upper_leg,
            gyro=gyro_data_left.upper_leg,
            quat=quat_data_left.upper_leg,
        )
        lower_leg_left = Limb(
            time=time,
            accel=accel_data_left.lower_leg,
            gyro=gyro_data_left.lower_leg,
            quat=quat_data_left.lower_leg,
        )
        foot_left = Limb(
            time=time,
            accel=accel_data_left.foot,
            gyro=gyro_data_left.foot,
            quat=quat_data_left.foot,
        )
        hip_left = Joint(time=time, angles=angles_left.hip)
        knee_left = Joint(time=time, angles=angles_left.knee)
        ankle_left = Joint(time=time, angles=angles_left.ankle)

        self.log_files = log_files
        self.time = accel_data_right.time
        self.data = LeftRight(
            left=Body(
                pelvis=pelvis_left,
                upper_leg=upper_leg_left,
                lower_leg=lower_leg_left,
                foot=foot_left,
                hip=hip_left,
                knee=knee_left,
                ankle=ankle_left,
            ),
            right=Body(
                pelvis=pelvis_right,
                upper_leg=upper_leg_right,
                lower_leg=lower_leg_right,
                foot=foot_right,
                hip=hip_right,
                knee=knee_right,
                ankle=ankle_right,
            ),
        )
