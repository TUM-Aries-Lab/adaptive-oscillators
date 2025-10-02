"""Common base classes."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from adaptive_oscillator.definitions import LOG_FILE_EXT


@dataclass
class VectorXYZ:
    """XYZ Vector."""

    x: NDArray = field(default_factory=lambda: np.array([]))
    y: NDArray = field(default_factory=lambda: np.array([]))
    z: NDArray = field(default_factory=lambda: np.array([]))

    def __getitem__(self, index: int | slice) -> NDArray:
        """Return stacked XYZ components as a 2D array, or select one component.

        :param index: Index or slice for accessing stacked vector components.
                      0 = x, 1 = y, 2 = z; or a slice like 0:2
        :return: Stacked NumPy array of shape (3, N) or (K, N).
        :raises IndexError: If index is out of bounds.
        """
        stacked = np.stack(arrays=[self.x, self.y, self.z], axis=1)
        return stacked[index].T

    def __len__(self) -> int:
        """Return the number of elements in the vector."""
        return len(self.x)


@dataclass
class Quaternion:
    """Quaternion."""

    w: NDArray = field(default_factory=lambda: np.array([]))
    x: NDArray = field(default_factory=lambda: np.array([]))
    y: NDArray = field(default_factory=lambda: np.array([]))
    z: NDArray = field(default_factory=lambda: np.array([]))

    def __getitem__(self, index: int | slice) -> NDArray:
        """Return stacked XYZ components as a 2D array, or select one component.

        :param index: Index or slice for accessing stacked vector components.
                      0 = x, 1 = y, 2 = z; or a slice like 0:2
        :return: Stacked NumPy array of shape (3, N) or (K, N).
        :raises IndexError: If index is out of bounds.
        """
        stacked = np.stack([self.w, self.x, self.y, self.z], axis=1)
        return stacked[index].T

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


@dataclass
class AngleXYZ:
    """XYZ Angle Vector."""

    x_deg: NDArray = field(default_factory=lambda: np.array([]))
    y_deg: NDArray = field(default_factory=lambda: np.array([]))
    z_deg: NDArray = field(default_factory=lambda: np.array([]))

    def __getitem__(self, index: int | slice) -> NDArray:
        """Return stacked XYZ components as a 2D array, or select one component.

        :param index: Index or slice for accessing stacked vector components.
                      0 = x, 1 = y, 2 = z; or a slice like 0:2
        :return: Stacked NumPy array of shape (3, N) or (K, N).
        :raises IndexError: If index is out of bounds.
        """
        stacked = np.stack([self.x_deg, self.y_deg, self.z_deg], axis=1)
        return stacked[index].T

    def __len__(self) -> int:
        """Return the number of elements in the vector."""
        return len(self.x_deg)


class SensorFile:
    """Represent a sensor category with left and right side access."""

    def __init__(self, category: str, base_path: Path) -> None:
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
