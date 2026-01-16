"""Run the log file plotter."""

import argparse

import numpy as np
from matplotlib import pyplot as plt
from py_imu.fusion.madgwick import Madgwick, Vector3D
from scipy.spatial.transform import Rotation as Rot

from adaptive_oscillator.definitions import FIG_SIZE
from adaptive_oscillator.log_files import LogFiles
from adaptive_oscillator.log_files.joint_angles import get_joint_angles
from adaptive_oscillator.log_files.parser import IMUParser, QuaternionParser

if __name__ == "__main__":  # pragma: no cover
    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Plot the data from a log dir.")
        parser.add_argument(
            "-l",
            "--log-dir",
            required=True,
            help="Path to the log directory.",
        )
        args = parser.parse_args()

        log_files = LogFiles(args.log_dir)

        accel = IMUParser(log_files.accel.left)
        accel.parse()

        gyro = IMUParser(log_files.gyro.left)
        gyro.parse()

        poses = []
        limb = "upper_leg"
        time, accel, gyro = accel.time, getattr(accel, limb), getattr(gyro, limb)
        madgwick = Madgwick(frequency=100.0, gain=0.033)
        last = None
        angles = []
        for t, acc, gyr in zip(time, accel, gyro):
            if last is None:
                dt = 0.01
            else:
                dt = t - last
                last = t
            gyr_rad = np.deg2rad(gyr)
            pose = madgwick.update(gyr=Vector3D(gyr_rad), acc=Vector3D(acc), dt=dt)
            poses.append(pose)

            euler = Rot.from_quat([pose.x, pose.y, pose.z, pose.w]).as_euler(
                seq="xyz", degrees=True
            )
            angles.append(euler)

        plt.figure(figsize=FIG_SIZE)
        for axis in ["x", "y", "z", "w"]:
            plt.plot(time - time[0], [getattr(q, axis) for q in poses], label=axis)
        plt.grid(True)
        plt.show()

        plt.figure(figsize=FIG_SIZE)
        for i in range(3):
            plt.plot(time - time[0], [a[i] for a in angles])
        plt.grid(True)
        plt.show()

        quat_parser = QuaternionParser(log_files.quat.right)
        quat_parser.parse()
        quat_parser.plot()

        joint_angles = get_joint_angles(quat_data=quat_parser)
        joint_angles.plot()
