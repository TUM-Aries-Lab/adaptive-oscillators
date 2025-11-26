"""Run the log file plotter."""

import argparse

import matplotlib.pyplot as plt

from adaptive_oscillator.log_files import LogFiles

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
    log_files.plot(euler_only=args.euler_only, add_offset=args.remap)
    plt.show()
