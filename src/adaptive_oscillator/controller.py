"""Controller module for the Adaptive Oscillator."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from adaptive_oscillator.definitions import DEFAULT_DELTA_TIME
from adaptive_oscillator.oscillator import (
    AOParameters,
    GaitPhaseEstimator,
    LowLevelController,
)
from adaptive_oscillator.utils.parser_utils import LogFiles, LogParser
from adaptive_oscillator.utils.plot_utils import RealtimeAOPlotter


class AOController:
    """Encapsulate the AO control loop and optional real-time plotting."""

    def __init__(self, show_plots: bool, ssh: bool = False):
        """Initialize controller.

        :param show_plots: Plot IMU logs before running the control loop.
        """
        self.params = AOParameters()
        self.estimator = GaitPhaseEstimator(self.params)
        self.controller = LowLevelController()
        self.theta_m = 0.0
        self.last_time: float | None = None

        self.ang_idx = 0

        self.motor_output: list[float] = []
        self.theta_hat_output: list[float] = []
        self.phi_gp_output: list[float] = []
        self.omegas: list[float] = []

        self.plotter: RealtimeAOPlotter | None = None
        if show_plots:  # pragma: no cover
            self.plotter = RealtimeAOPlotter(ssh=ssh)
            self.plotter.run()

    def replay(self, log_dir: str | Path):
        """Run the AO simulation loop."""
        logger.info(f"Running controller with log data from {log_dir}")
        log_files = LogFiles(log_dir)
        log_data = LogParser(log_files)

        time_vec = log_data.data.left.hip.time
        angle_vec = log_data.data.left.hip.angles

        try:
            for i in range(len(angle_vec) - 1):
                th = np.deg2rad(angle_vec[i][self.ang_idx])
                dth = np.deg2rad(
                    angle_vec[i][self.ang_idx]
                )  # TODO: replace with actual derivative if available
                t = time_vec[i] - time_vec[0]
                self.step(t=t, th=th, dth=dth)

                logger.info(f"t={t:.2f}, th={th:.2f}, dth={dth:.2f}")
        except KeyboardInterrupt:  # pragma: no cover
            logger.warning("Controller interrupted.")

        if self.plotter is not None:  # pragma: no cover
            log_files.plot()
            plt.show()

        logger.success(f"Finished controller with log data from {log_dir}")

    def step(self, t: float, th: float, dth: float) -> None:
        """Step the AO ahead with one frame of data from the IMU."""
        if self.last_time is None:
            dt = DEFAULT_DELTA_TIME
        else:
            dt = t - self.last_time
        self.last_time = t

        phi = self.estimator.update(t=t, theta_il=th, theta_il_dot=dth)
        omega_cmd = self.controller.compute(phi=phi, theta_m=self.theta_m, dt=dt)
        self.theta_m += omega_cmd * dt

        # Store outputs
        self.motor_output.append(self.theta_m)
        self.theta_hat_output.append(self.estimator.ao.theta_hat)
        self.phi_gp_output.append(self.estimator.phi_gp)
        self.omegas.append(self.estimator.ao.omega)

        # Update live plot if enabled
        if self.plotter is not None:  # pragma: no cover
            self.plotter.update_data(
                t=t,
                theta_il=th,
                theta_hat=self.estimator.ao.theta_hat,
                omega=self.estimator.ao.omega,
                phi_gp=self.estimator.phi_gp,
            )
            time.sleep(dt)
