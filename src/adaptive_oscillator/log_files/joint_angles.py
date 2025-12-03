"""Run the log file plotter."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as Rot

from adaptive_oscillator.definitions import FIG_SIZE
from adaptive_oscillator.log_files.parser import QuaternionParser


@dataclass
class JointAngles:
    """Store the joint angles for a side of the body."""

    hip: list
    knee: list
    ankle: list
    time: list

    def plot(self) -> None:
        """Plot the joint angles."""
        self.time -= self.time[0]
        fig, ax = plt.subplots(figsize=FIG_SIZE, sharex=True, nrows=3, ncols=1)
        ax[0].plot(self.time, self.hip)
        ax[0].set_ylabel("Hip Angle (deg)")

        ax[1].plot(self.time, self.knee)
        ax[1].set_ylabel("Knee Angle (deg)")

        ax[2].plot(self.time, self.ankle)
        ax[2].set_ylabel("Ankle Angle (deg)")

        for i in range(3):
            ax[i].grid(True)
        ax[-1].set_xlabel("Time (s)")

        plt.show()


def get_joint_angles(quat_data: QuaternionParser) -> JointAngles:
    """Calculate the joint angles."""
    pelvis = quat_data.pelvis
    upper_leg = quat_data.upper_leg
    lower_leg = quat_data.lower_leg
    foot = quat_data.foot

    rot_pelvis = np.array(
        [
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0],
        ]
    )
    rot_upper = np.array(
        [
            [-1, 0, 0],
            [0, 0, -1],
            [0, -1, 0],
        ]
    )
    rot_lower = np.array(
        [
            [-1, 0, 0],
            [0, 0, -1],
            [0, -1, 0],
        ]
    )
    rot_foot = np.array(
        [
            [-1, 0, 0],
            [0, 0, -1],
            [0, -1, 0],
        ]
    )

    angles = []
    for q_pel, q_upleg, q_lowleg, q_foot in zip(pelvis, upper_leg, lower_leg, foot):
        q_pel_new = q_pel.remap(rotation_matrix=rot_pelvis.T).as_list()
        q_upleg_new = q_upleg.remap(rotation_matrix=rot_upper.T).as_list()
        q_lowleg_new = q_lowleg.remap(rotation_matrix=rot_lower.T).as_list()
        q_foot_new = q_foot.remap(rotation_matrix=rot_foot.T).as_list()

        euler_pelvis = Rot.from_quat(q_pel_new).as_euler("xyz", degrees=True)
        euler_upper = Rot.from_quat(q_upleg_new).as_euler("xyz", degrees=True)
        euler_lower = Rot.from_quat(q_lowleg_new).as_euler("xyz", degrees=True)
        euler_foot = Rot.from_quat(q_foot_new).as_euler("xyz", degrees=True)

        hip = euler_pelvis - euler_upper
        knee = euler_lower - euler_upper
        ankle = euler_lower + euler_foot
        pitch = (hip, knee, ankle)

        angles.append(pitch)


    hip_list = [ang[0] for ang in angles]
    knee_list = [ang[1] for ang in angles]
    ankle_list = [ang[2] for ang in angles]

    time = quat_data.time - quat_data.time[0]
    joint_angles = JointAngles(
        hip=hip_list, knee=knee_list, ankle=ankle_list, time=time
    )
    logger.info("Joint angles processed.")

    return joint_angles
