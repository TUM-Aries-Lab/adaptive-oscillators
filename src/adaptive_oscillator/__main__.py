"""Run the AO controller with optional plotting."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from adaptive_oscillator.controller import AOController
from adaptive_oscillator.definitions import LogLevel
from adaptive_oscillator.log_files import LogFiles, LogParser
from adaptive_oscillator.utils import setup_logger


def main(log_dir: str, show_plots: bool, ssh: bool) -> None:
    """Run the AO controller with optional plotting.

    :param log_dir: Path to the log directory.
    :param show_plots: Show plots.
    :param ssh: Use SSH.
    """
    log_files = LogFiles(log_dir)
    log_data = LogParser(log_files)

    log_data.data.right.hip.angles.add_offset(offsets=[180, 0, 180])
    log_data.data.left.hip.angles.add_offset(offsets=[180, 0, 180])

    signal = -log_data.data.right.hip.angles.x

    controller = AOController(show_plots=show_plots, ssh=ssh)
    for _ii, ang_deg in enumerate(signal):
        th = np.deg2rad(ang_deg)
        dth = np.deg2rad(ang_deg)  # TODO: replace with actual derivative if available
        t = log_data.data.left.hip.time[_ii] - log_data.data.left.hip.time[0]
        controller.step(t=t, x=th, x_dot=dth)

    if controller.plotter is not None:  # pragma: no cover
        log_files.plot(euler_only=True)
        plt.show()

    controller.plot_results()

    logger.success(f"Finished controller with log data from {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AO controller with optional plotting."
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Output debug statements.",
    )

    parser.add_argument("--log-dir", required=True, help="Path to the log directory.")
    parser.add_argument("--plot", action="store_true", help="Plot simulation results.")
    parser.add_argument("--ssh", action="store_true", help="Connect to an SSH server.")
    args = parser.parse_args()

    if args.debug:
        setup_logger(log_level=LogLevel.debug, stderr_level=LogLevel.debug)
    else:
        setup_logger()

    main(log_dir=args.log_dir, show_plots=args.plot, ssh=args.ssh)
