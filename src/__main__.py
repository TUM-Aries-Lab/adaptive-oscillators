"""Run the AO controller with optional plotting."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from adaptive_oscillator.controller import AOController
from adaptive_oscillator.utils.parser_utils import LogFiles, LogParser


def main() -> None:
    """Run the AO controller with optional plotting."""
    parser = argparse.ArgumentParser(
        description="Run AO controller with optional plotting."
    )
    parser.add_argument(
        "-l", "--log-dir", required=True, help="Path to the log directory."
    )
    parser.add_argument(
        "-p", "--plot-results", action="store_true", help="Plot simulation results."
    )
    parser.add_argument(
        "-s", "--ssh", action="store_true", help="Connect to an SSH server."
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    log_files = LogFiles(log_dir)
    log_data = LogParser(log_files)

    controller = AOController(show_plots=args.plot_results, ssh=args.ssh)
    for _ii, ang_deg in enumerate(log_data.data.left.hip.angles.x_deg):
        th = np.deg2rad(ang_deg)
        dth = np.deg2rad(ang_deg)  # TODO: replace with actual derivative if available
        t = log_data.data.left.hip.time[_ii] - log_data.data.left.hip.time[0]
        controller.step(t=t, th=th, dth=dth)

    if controller.plotter is not None:  # pragma: no cover
        log_files.plot()
        plt.show()

    logger.success(f"Finished controller with log data from {log_dir}")


if __name__ == "__main__":
    main()
