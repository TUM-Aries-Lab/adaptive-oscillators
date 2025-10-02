"""Tests for the base classes."""

import numpy as np

from iit_rehab.base_classes import AngleXYZ, Quaternion, VectorXYZ


def test_angle_xyz():
    """Test the AngleXYZ class."""
    # Arrange
    array = np.array([1, 2, 3, 4])
    num_el = len(array)

    # Act
    angle_xyz = AngleXYZ(array, array, array)

    # Assert
    assert angle_xyz.x_deg[0] == 1
    assert angle_xyz.y_deg[1] == 2
    assert angle_xyz.z_deg[2] == 3
    assert len(angle_xyz) == num_el
    assert np.shape(angle_xyz[:]) == (3, num_el)


def test_vector_xyz():
    """Test the VectorXYZ class."""
    # Arrange
    array = np.array([1, 2, 3, 4])
    num_el = len(array)

    # Act
    vec_xyz = VectorXYZ(array, array, array)

    # Assert
    assert vec_xyz.x[0] == 1
    assert vec_xyz.y[1] == 2
    assert vec_xyz.z[2] == 3
    assert len(vec_xyz) == num_el
    assert np.shape(vec_xyz[:]) == (3, num_el)


def test_quaternion():
    """Test the Quaternion class."""
    # Arrange
    exp_quat = Quaternion(
        w=np.array([1]),
        x=np.array([0]),
        y=np.array([0]),
        z=np.array([0]),
    )

    quat_a = Quaternion(
        w=np.array([1, 1]),
        x=np.array([0, 0]),
        y=np.array([0, 0]),
        z=np.array([0, 0]),
    )

    quat_b = Quaternion(
        w=np.array([1]),
        x=np.array([0]),
        y=np.array([0]),
        z=np.array([0]),
    )

    # Act
    quat_new = quat_a * quat_b

    # Assert
    for q in quat_new:
        quat = Quaternion(q[0], q[1], q[2], q[3])
        assert quat.x == exp_quat.x
        assert quat.y == exp_quat.y
        assert quat.z == exp_quat.z
        assert quat.w == exp_quat.w
