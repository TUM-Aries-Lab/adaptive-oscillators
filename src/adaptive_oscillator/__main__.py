"""Run the Adaptive Oscillator controller."""

import argparse

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from adaptive_oscillator.controller import AOController
from adaptive_oscillator.definitions import RESULTS_DIR, LogLevel
from adaptive_oscillator.log_files import LogFiles, QuaternionParser
from adaptive_oscillator.log_files.joint_angles import (
    calculate_joint_angles,
    plot_joint_angles,
)
from adaptive_oscillator.utils import setup_logger


def process_joint_data(joint_data: list | NDArray, time_stamps: list | NDArray) -> None:
    """Run the adaptive oscillator on a joint.

    :param joint_data: recorded joint data.
    :param time_stamps: time stamps.
    :return: None
    """
    controller = AOController()
    logger.info("Processing data.")
    for _ii, (t, ang_deg) in enumerate(zip(time_stamps, joint_data)):
        th = np.deg2rad(ang_deg)
        dth = np.deg2rad(ang_deg)  # TODO: replace with actual derivative if available
        controller.step(t=t, x=th, x_dot=dth)

    controller.plot_results(joint="joint", side="side", save_plot=False)
    controller.write_results(filepath=RESULTS_DIR / f"results_{'joint'}_{'side'}.txt")


def main(log_dir: str, show_plots: bool) -> None:
    """Run the AO controller with optional plotting.

    :param log_dir: Path to the log directory.
    :param show_plots: Show plots.
    """
    log_files = LogFiles(log_dir)

    quat_parser = QuaternionParser(log_files.quat.right)
    quat_parser.parse()
    hip, knee, ankle = calculate_joint_angles(quat_parser)
    plot_joint_angles(time=quat_parser.time, hip=hip, knee=knee, ankle=ankle)

    for joint in [hip, knee, ankle]:
        process_joint_data(joint_data=joint, time_stamps=quat_parser.time)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run adaptive oscillator controller.")
    parser.add_argument("--debug", action="store_true", help="Output debug statements.")
    parser.add_argument("--log-dir", required=True, help="Path to the log directory.")
    parser.add_argument("--plot", action="store_true", help="Plot simulation results.")
    args = parser.parse_args()

    if args.debug:
        setup_logger(log_level=LogLevel.debug, stderr_level=LogLevel.debug)
    else:
        setup_logger()

    main(log_dir=args.log_dir, show_plots=args.plot)
