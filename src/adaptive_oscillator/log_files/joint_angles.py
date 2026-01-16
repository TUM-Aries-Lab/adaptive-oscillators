"""Run the log file plotter."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as Rot

from adaptive_oscillator.definitions import FIG_SIZE
from adaptive_oscillator.log_files.parser import QuaternionParser

SEQ = "zyx"


@dataclass
class JointAngles:
    """Store the joint angles for a side of the body."""

    hip: list
    knee: list
    ankle: list
    time: list

    def plot(self) -> None:
        """Plot the joint angles."""
        logger.info("Plotting joint angles.")
        self.time -= self.time[0]
        fig, ax = plt.subplots(figsize=FIG_SIZE, sharex=True, nrows=3, ncols=1)
        self._plot_joint(joint=self.hip, ax=ax[0])
        ax[0].set_ylabel("Hip Angle (deg)")

        self._plot_joint(joint=self.knee, ax=ax[1])
        ax[1].set_ylabel("Knee Angle (deg)")

        self._plot_joint(joint=self.ankle, ax=ax[2])
        ax[2].set_ylabel("Ankle Angle (deg)")

        for i in range(3):
            ax[i].grid(True)
        ax[-1].set_xlabel("Time (s)")

        try:
            plt.show()
        except KeyboardInterrupt:
            plt.close()

    def _plot_joint(self, joint, ax: plt.Axes) -> None:
        if isinstance(joint[0], np.ndarray):
            ax.plot(self.time, [j[0] for j in joint], label="x")
            ax.plot(self.time, [j[1] for j in joint], label="y")
            ax.plot(self.time, [j[2] for j in joint], label="z")
            ax.legend(loc="upper right")
        else:
            ax.plot(self.time, [j.x for j in joint])
            ax.plot(self.time, [j.y for j in joint])
            ax.plot(self.time, [j.z for j in joint])
            ax.plot(self.time, [j.w for j in joint])


def get_joint_angles(quat_data: QuaternionParser) -> JointAngles:
    """Calculate the joint angles."""
    pelvis = quat_data.pelvis
    upper_leg = quat_data.upper_leg
    lower_leg = quat_data.lower_leg
    foot = quat_data.foot

    angles = []
    eye3 = np.eye(3)
    for q_pel, q_upleg, q_lowleg, q_foot in zip(pelvis, upper_leg, lower_leg, foot):
        q_pel_new = q_pel.remap(rotation_matrix=eye3).as_list()
        q_upleg_new = q_upleg.remap(rotation_matrix=eye3).as_list()
        q_lowleg_new = q_lowleg.remap(rotation_matrix=eye3).as_list()
        q_foot_new = q_foot.remap(rotation_matrix=eye3).as_list()
        logger.trace(
            f"pelvis: {q_pel_new}, upleg: {q_upleg_new}, lowleg: {q_lowleg_new}, foot: {q_foot_new}"
        )

        euler_pelvis = Rot.from_quat(q_pel.as_list()).as_euler(seq="zyx", degrees=True)
        euler_upper = Rot.from_quat(q_upleg.as_list()).as_euler(seq="zyx", degrees=True)
        euler_lower = Rot.from_quat(q_lowleg.as_list()).as_euler(
            seq="zyx", degrees=True
        )
        euler_foot = Rot.from_quat(q_foot.as_list()).as_euler(seq="zyx", degrees=True)

        hip = euler_pelvis[0] - euler_upper[2]
        knee = -euler_upper[0] - euler_lower[0]
        ankle = euler_foot[0] - euler_lower[0]
        logger.trace(f"hip: {hip}, knee: {knee}, ankle: {ankle}")

        pitch = (euler_pelvis, euler_upper, euler_lower)
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
