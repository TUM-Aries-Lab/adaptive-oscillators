"""Parser utils for log file data."""

from pathlib import Path

from loguru import logger

from adaptive_oscillator.base_classes import SensorFile
from adaptive_oscillator.definitions import LogFileKeys
from adaptive_oscillator.log_files.parsers import (
    AngleParser,
    IMUParser,
    QuaternionParser,
)


class LogFiles:
    """Main entry point for accessing all sensor log files."""

    def __init__(self, base_path: str | Path) -> None:
        self._path = Path(base_path)
        if not self._path.is_dir():
            msg = f"Path '{self._path}' does not exist."
            logger.error(msg)
            raise FileNotFoundError(msg)
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

    def plot(self, euler_only: bool = False, add_offset: bool = False) -> None:
        """Plot log files.

        :param euler_only: Plot only euler angles.
        :param add_offset: Add offset to angles.
        :return: None
        """
        logger.info("Plotting log file data.")

        for side in ["left", "right"]:
            if not euler_only:
                accel_data = IMUParser(getattr(self.accel, side))
                accel_data.parse()
                accel_data.plot(y_label="Acceleration (m/s2)")

                gyro_data = IMUParser(getattr(self.gyro, side))
                gyro_data.parse()
                gyro_data.plot(y_label="Angular Velocity (deg/s)")

                quat_data = QuaternionParser(getattr(self.quat, side))
                quat_data.parse()
                quat_data.plot()

            angle_data = AngleParser(getattr(self.angle, side))
            angle_data.parse(add_offset=add_offset)
            angle_data.plot(y_label="Euler Angle (deg)")
