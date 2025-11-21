"""Run the AO controller with optional plotting."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from adaptive_oscillator.base_classes import Joint
from adaptive_oscillator.controller import AOController
from adaptive_oscillator.definitions import ANGLES_SEGMENT_FIELDS, LogLevel
from adaptive_oscillator.log_files import LogFiles, LogParser
from adaptive_oscillator.utils import setup_logger


def process_joint_data(joint_data: Joint) -> None:
    """Run the adaptive oscillator on a joint.

    :param joint_data: recorded joint data.
    :return: None
    """
    joint_data.angles.add_offset(offsets=[180, 0, 180])
    signal = -joint_data.angles.x
    time_stamps = joint_data.time - joint_data.time[0]

    controller = AOController()
    for _ii, (t, ang_deg) in enumerate(zip(time_stamps, signal)):
        th = np.deg2rad(ang_deg)
        dth = np.deg2rad(ang_deg)  # TODO: replace with actual derivative if available
        controller.step(t=t, x=th, x_dot=dth)

    controller.plot_results()


def main(log_dir: str, show_plots: bool) -> None:
    """Run the AO controller with optional plotting.

    :param log_dir: Path to the log directory.
    :param show_plots: Show plots.
    """
    log_files = LogFiles(log_dir)
    log_data = LogParser(log_files)

    for side in log_data.data:
        for key in ANGLES_SEGMENT_FIELDS:
            process_joint_data(joint_data=getattr(side, key))

    if show_plots:
        log_files.plot(euler_only=True)
        plt.show()


if __name__ == "__main__":
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
