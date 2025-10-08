"""Run the AO controller with optional plotting."""

import argparse

from adaptive_oscillator.controller import AOController


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
    controller = AOController(show_plots=args.plot_results, ssh=args.ssh)
    controller.replay(log_dir=args.log_dir)


if __name__ == "__main__":
    main()
