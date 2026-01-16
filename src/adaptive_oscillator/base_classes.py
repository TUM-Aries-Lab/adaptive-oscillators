"""Common base classes."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as Rot

from adaptive_oscillator.definitions import LOG_FILE_EXT

TWO_PI = 2 * np.pi
PI = np.pi


@dataclass
class VectorXYZ:
    """XYZ axes vectors."""

    x: NDArray = field(default_factory=lambda: np.array([]))
    y: NDArray = field(default_factory=lambda: np.array([]))
    z: NDArray = field(default_factory=lambda: np.array([]))

    def __getitem__(self, index: int | slice) -> NDArray:
        """Return stacked XYZ components as a 2D array, or select one component.

        :param index: Index or slice for accessing stacked vector components.
        :return: Stacked NumPy array of shape (3, N) or (K, N).
        """
        stacked = np.stack(arrays=[self.x, self.y, self.z], axis=1)
        return stacked[index].T

    def __len__(self) -> int:
        """Return the number of elements in the vector."""
        return len(self.x)


@dataclass
class Quaternion:
    """Quaternion vectors."""

    w: NDArray | float = field(default_factory=lambda: np.array([]))
    x: NDArray | float = field(default_factory=lambda: np.array([]))
    y: NDArray | float = field(default_factory=lambda: np.array([]))
    z: NDArray | float = field(default_factory=lambda: np.array([]))

    def __getitem__(self, index: int | slice) -> "Quaternion":
        """Return stacked XYZ components as a 2D array, or select one component.

        :param index: Index or slice for accessing stacked vector components.
        :return: Stacked NumPy array of shape (3, N) or (K, N).
        """
        stacked = np.stack([self.w, self.x, self.y, self.z], axis=1)
        w, x, y, z = stacked[index].T
        return Quaternion(w, x, y, z)

    def __mul__(self, quat_b: "Quaternion") -> "Quaternion":
        """Multiply two quaternions.

        :param quat_b: Quaternion to multiply with.
        :return: New Quaternion representing the product.
        """
        q = quat_b
        w = self.w * q.w - self.x * q.x - self.y * q.y - self.z * q.z
        x = self.w * q.x + self.x * q.w + self.y * q.z - self.z * q.y
        y = self.w * q.y - self.x * q.z + self.y * q.w + self.z * q.x
        z = self.w * q.z + self.x * q.y - self.y * q.x + self.z * q.w
        return Quaternion(w, x, y, z)

    def remap(self, rotation_matrix: NDArray) -> "Quaternion":
        """Remap the quaternion to a new coordinate frame."""
        pose_se3 = Rot.from_quat(self.as_list()).as_matrix()
        pose_se3 = pose_se3 @ rotation_matrix
        quat = Rot.from_matrix(pose_se3).as_quat()
        x, y, z, w = quat
        return Quaternion(float(w), float(x), float(y), float(z))

    def as_list(self, scalar_first: bool = False) -> list:
        """Return a list of quaternions."""
        if scalar_first:
            quat = [self.w, self.x, self.y, self.z]
        else:
            quat = [self.x, self.y, self.z, self.w]
        return quat


@dataclass
class AngleXYZ:
    """XYZ Angle Vector."""

    x: NDArray = field(default_factory=lambda: np.array([]))
    y: NDArray = field(default_factory=lambda: np.array([]))
    z: NDArray = field(default_factory=lambda: np.array([]))

    def __getitem__(self, index: int | slice) -> NDArray:
        """Return stacked XYZ components as a 2D array, or select one component.

        :param index: Index or slice for accessing stacked vector components.
        :return: Stacked NumPy array of shape (3, N) or (K, N).
        """
        stacked = np.stack([self.x, self.y, self.z], axis=1)
        return stacked[index].T

    def __len__(self) -> int:
        """Return the number of elements in the vector."""
        return len(self.x)

    def add_offset(self) -> None:
        """Offset angles so the signals don't wrap from max to min.

        :return: None
        """
        axes = ["x", "y", "z"]

        for axis in axes:
            old = False
            angles_deg = getattr(self, axis)
            angles_rad = np.deg2rad(angles_deg)
            if old:
                diff = max(angles_deg) - min(angles_deg)
                if diff > 355:
                    offset = 180
                else:
                    offset = 0
                logger.info(f"Adding {offset:.2f} deg offset for {axis} axis.")

                angles_deg += offset
                mask_upper = angles_deg > 180
                mask_lower = angles_deg < -180
                angles_deg[mask_upper] -= 360
                angles_deg[mask_lower] += 360
                setattr(self, axis, angles_deg)
            else:
                offset = -angles_rad[0]
                ang_old = None
                for ii, ang in enumerate(angles_rad):
                    if ang_old is not None:
                        diff = ang - ang_old
                        if abs(diff) > TWO_PI:
                            logger.debug(diff)
                        if diff > PI:
                            offset -= TWO_PI
                        if diff < -PI:
                            offset += TWO_PI

                    angles_rad[ii] += offset
                    ang_old = ang

                setattr(self, axis, np.rad2deg(angles_rad))


class SensorFile:
    """Represent a sensor category with left and right side access."""

    def __init__(self, category: str, base_path: Path) -> None:
        """Initialize a sensor category with left and right side access.

        :param category: Sensor category name.
        :param base_path: Path to the sensor directory.
        :return: None
        """
        self.left = base_path / f"{category}_left{LOG_FILE_EXT}"
        self.right = base_path / f"{category}_right{LOG_FILE_EXT}"


@dataclass
class Limb:
    """Represent a side of the body."""

    time: NDArray
    accel: VectorXYZ
    gyro: VectorXYZ
    quat: Quaternion


@dataclass
class Joint:
    """Represent a side of the body."""

    time: NDArray
    angles: AngleXYZ


@dataclass
class Body:
    """Represent a body."""

    pelvis: Limb
    upper_leg: Limb
    lower_leg: Limb
    foot: Limb
    hip: Joint
    knee: Joint
    ankle: Joint


@dataclass
class LeftRight:
    """Represent a body."""

    left: Body
    right: Body

    def __iter__(self):
        """Iterate over the left and right sides."""
        yield "left", self.left
        yield "right", self.right


@dataclass
class AdaptiveOscillatorStepResult:
    """Represent a result of the adaptive oscillator step."""

    timestamp: float
    theta: float
    theta_hat: float
    omega: float
    gait_phase: float
    offset: float

    def __repr__(self):
        """Represent the result as a string."""
        msg = (
            f"time: {self.timestamp:.3f}, "
            f"theta: {self.theta:.3f}, "
            f"theta_hat: {self.theta_hat:.3f}, "
            f"omega: {self.omega:.3f}, "
            f"gait phase: {self.gait_phase:.3f}"
            f"offset: {self.offset:.3f}"
        )
        return msg

    def __iter__(self):
        """Iterate over the results."""
        yield self.timestamp
        yield self.theta
        yield self.theta_hat
        yield self.omega
        yield self.gait_phase
        yield self.offset
