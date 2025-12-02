"""Run the log file plotter."""

import argparse

from adaptive_oscillator.log_files import LogFiles
from adaptive_oscillator.log_files.joint_angles import (
    calculate_joint_angles,
    plot_joint_angles,
)
from adaptive_oscillator.log_files.parser import QuaternionParser

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
