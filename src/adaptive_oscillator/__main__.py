"""Run the AO controller with optional plotting."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from adaptive_oscillator.controller import AOController
from adaptive_oscillator.definitions import DEFAULT_LOG_LEVEL, LogLevel
from adaptive_oscillator.log_files import LogFiles, LogParser
from adaptive_oscillator.utils import setup_logger


def main(log_dir: str, show_plots: bool, ssh: bool) -> None:
    """Run the AO controller with optional plotting."""
    log_files = LogFiles(log_dir)
    log_data = LogParser(log_files)

    log_data.data.left.hip.angles.add_offset(offsets=[180, 0, 180])
    signal = log_data.data.left.hip.angles.x

    controller = AOController(show_plots=show_plots, ssh=ssh)
    for _ii, ang_deg in enumerate(signal):
        th = np.deg2rad(ang_deg)
        dth = np.deg2rad(ang_deg)  # TODO: replace with actual derivative if available
        t = log_data.data.left.hip.time[_ii] - log_data.data.left.hip.time[0]
        controller.step(t=t, x=th, x_dot=dth)

    if controller.plotter is not None:  # pragma: no cover
        log_files.plot(euler_only=True)
        plt.show()

    logger.success(f"Finished controller with log data from {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AO controller with optional plotting."
    )
    parser.add_argument(
        "--log-level",
        default=DEFAULT_LOG_LEVEL,
        choices=list(LogLevel()),
        help="Set the log level.",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--stderr-level",
        default=DEFAULT_LOG_LEVEL,
        choices=list(LogLevel()),
        help="Set the std err level.",
        required=False,
        type=str,
    )

    parser.add_argument(
        "-l", "--log-dir", required=True, help="Path to the log directory."
    )
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot simulation results."
    )
    parser.add_argument(
        "-s", "--ssh", action="store_true", help="Connect to an SSH server."
    )
    args = parser.parse_args()

    setup_logger(log_level=args.log_level, stderr_level=args.stderr_level)

    main(log_dir=args.log_dir, show_plots=args.plot, ssh=args.ssh)
