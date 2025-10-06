"""Run the AO controller with optional plotting."""

import argparse

from adaptive_oscillator.controller import AOController


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run AO controller with optional plotting."
    )
    parser.add_argument(
        "-l", "--log-dir", required=True, help="Path to the log directory."
    )
    parser.add_argument(
        "-p", "--plot-results", action="store_true", help="Plot simulation results."
    )
    return parser.parse_args()


def main() -> None:
    """Run the AO controller with optional plotting."""
    args = parse_args()
    controller = AOController(show_plots=args.show_plots)
    controller.replay(log_dir=args.log_dir)


if __name__ == "__main__":
    main()
