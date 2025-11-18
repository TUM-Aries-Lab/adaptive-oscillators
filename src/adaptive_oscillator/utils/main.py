"""Run the AO controller with optional plotting."""

import argparse

import matplotlib.pyplot as plt

from adaptive_oscillator.utils.parser_utils import LogFiles


def main() -> None:
    """Run the AO controller with optional plotting."""
    parser = argparse.ArgumentParser(
        description="Run AO controller with optional plotting."
    )
    parser.add_argument(
        "-l", "--log-dir", required=True, help="Path to the log directory."
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    log_files = LogFiles(log_dir)
    log_files.plot()
    plt.show()


if __name__ == "__main__":
    main()
