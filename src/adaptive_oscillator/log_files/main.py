"""Run the log file plotter."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as Rot

from adaptive_oscillator.log_files import LogFiles
from adaptive_oscillator.log_files.parser import QuaternionParser


def remap_rotation_matrix(rot_mat) -> NDArray:
    """Remap the rotation matrix so that it matches the original rotation matrix."""
    remap_mat = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    return remap_mat.T @ rot_mat


def quat_to_euler_remapped(quat):
    """Convert quaternion to euler angles and remap."""
    quat_as_rot_mat = Rot.from_quat(quat).as_matrix()
    rot_mat_remapped = remap_rotation_matrix(rot_mat=quat_as_rot_mat)
    euler_remapped = Rot.from_matrix(rot_mat_remapped).as_euler("xyz", degrees=True)
    return euler_remapped


def plot_joint_angles(time, hip: list, knee: list, ankle: list) -> None:
    """Plot the joint angles."""
    fig, ax = plt.subplots(figsize=(20, 9), sharex=True, nrows=3, ncols=1)
    ax[0].plot(time, hip)
    ax[0].set_ylabel("hip")
    ax[0].grid(True)

    ax[1].plot(time, knee)
    ax[1].set_ylabel("knee")
    ax[1].grid(True)

    ax[2].plot(time, ankle)
    ax[2].set_ylabel("ankle")
    ax[2].grid(True)

    plt.legend()
    plt.show()


def calculate_joint_angles(quat_parser: QuaternionParser) -> tuple[list, list, list]:
    """Calculate the joint angles."""
    pelvis = quat_parser.pelvis
    upper_leg = quat_parser.upper_leg
    lower_leg = quat_parser.lower_leg
    foot = quat_parser.foot

    angles = []
    for q_pel, q_upleg, q_lowleg, q_foot in zip(pelvis, upper_leg, lower_leg, foot):
        euler_pelvis = quat_to_euler_remapped(q_pel)
        euler_upper = quat_to_euler_remapped(q_upleg)
        euler_lower = quat_to_euler_remapped(q_lowleg)
        euler_foot = quat_to_euler_remapped(q_foot)

        hip = euler_pelvis - euler_upper
        knee = euler_upper - euler_lower
        ankle = euler_lower - euler_foot
        pitch = (hip[1], knee[1], ankle[1])
        angles.append(pitch)

    hip_list = [ang[0] for ang in angles]
    knee_list = [ang[1] for ang in angles]
    ankle_list = [ang[2] for ang in angles]
    return hip_list, knee_list, ankle_list


if __name__ == "__main__":  # pragma: no cover
    """Plot data from log files."""
    parser = argparse.ArgumentParser(description="Plot the data from a log dir.")
    parser.add_argument(
        "-l",
        "--log-dir",
        required=True,
        help="Path to the log directory.",
    )
    parser.add_argument(
        "-e",
        "--euler-only",
        action="store_true",
        help="Plot only the euler angles.",
    )
    parser.add_argument(
        "--remap",
        required=False,
        default=False,
        action="store_true",
        help="Remap the Euler angles.",
    )
    args = parser.parse_args()

    log_files = LogFiles(args.log_dir)
    quat_parser = QuaternionParser(log_files.quat.right)
    quat_parser.parse()
    hip, knee, ankle = calculate_joint_angles(quat_parser)
    plot_joint_angles(quat_parser.time, hip, knee, ankle)
