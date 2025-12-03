"""Run the log file plotter."""

import argparse

from adaptive_oscillator.log_files import LogFiles
from adaptive_oscillator.log_files.joint_angles import get_joint_angles
from adaptive_oscillator.log_files.parser import QuaternionParser

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

        quat_parser = QuaternionParser(log_files.quat.right)
        quat_parser.parse()

        joint_angles = get_joint_angles(quat_data=quat_parser)
        joint_angles.plot()
