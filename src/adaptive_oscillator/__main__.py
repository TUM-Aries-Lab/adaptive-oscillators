"""Run the Adaptive Oscillator controller."""

import argparse

import numpy as np
from loguru import logger

from adaptive_oscillator.controller import AOController
from adaptive_oscillator.definitions import RESULTS_DIR, AOParameters, LogLevel
from adaptive_oscillator.log_files import LogFiles, QuaternionParser
from adaptive_oscillator.log_files.joint_angles import JointAngles, get_joint_angles
from adaptive_oscillator.utils import setup_logger


def process_joint_data(joint_data: JointAngles, side: str) -> None:
    """Run the adaptive oscillator on a joint.

    :param joint_data: recorded joint data.
    :param side: side of the body.
    :return: None
    """
    for joint in ["hip", "knee", "ankle"]:
        logger.info(f"Processing '{joint}' joint.")

        time = joint_data.time
        angles = getattr(joint_data, joint)
        velocities = np.diff(angles)

        if joint in ["hip", "knee"]:
            ao_config = AOParameters(n_harmonics=3, omega_init=1)
        else:
            ao_config = AOParameters()
        controller = AOController(config=ao_config)
        for t, ang_deg, vel_deg in zip(time[1:], angles[1:], velocities):
            th = np.deg2rad(ang_deg[1])
            dth = np.deg2rad(vel_deg[1])
            controller.step(t=t, x=th, x_dot=dth)

        controller.plot_results(joint=joint, side=side, save_plot=False)
        controller.write_results(filepath=RESULTS_DIR / f"results_{joint}_{side}_5.txt")


def main(log_dir: str, show_plots: bool) -> None:
    """Run the AO controller with optional plotting.

    :param log_dir: Path to the log directory.
    :param show_plots: Show plots.
    """
    log_files = LogFiles(log_dir)

    side = "right"
    logger.info(f"Processing '{side}' log file.")
    quat_parser = QuaternionParser(getattr(log_files.quat, side))
    quat_parser.parse()
    joint_angles = get_joint_angles(quat_data=quat_parser)
    if show_plots:
        joint_angles.plot()

    process_joint_data(joint_data=joint_angles, side=side)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run adaptive oscillator controller.")
    parser.add_argument("--debug", action="store_true", help="Output debug statements.")
    parser.add_argument(
        "--log-dir", "-l", required=True, help="Path to the log directory."
    )
    parser.add_argument("--plot", action="store_true", help="Plot simulation results.")
    args = parser.parse_args()

    if args.debug:
        setup_logger(log_level=LogLevel.debug, stderr_level=LogLevel.debug)
    else:
        setup_logger()

    main(log_dir=args.log_dir, show_plots=args.plot)
