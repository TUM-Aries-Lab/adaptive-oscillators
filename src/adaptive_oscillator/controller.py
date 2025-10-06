"""Controller module for the Adaptive Oscillator."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from adaptive_oscillator.oscillator import (
    AOParameters,
    GaitPhaseEstimator,
    LowLevelController,
)
from adaptive_oscillator.utils.parser_utils import LogFiles, LogParser
from adaptive_oscillator.utils.plot_utils import RealtimeAOPlotter


class AOController:
    """Encapsulate the AO control loop and optional real-time plotting."""

    def __init__(self, show_plots: bool = False):
        """Initialize controller.

        :param plot: Plot IMU logs before running the control loop.
        """
        self.params = AOParameters()
        self.estimator = GaitPhaseEstimator(self.params)
        self.controller = LowLevelController()
        self.theta_m = 0.0

        self.ang_idx = 0

        self.motor_output: list[float] = []
        self.theta_hat_output: list[float] = []
        self.phi_gp_output: list[float] = []
        self.omegas: list[float] = []

        self.plotter: RealtimeAOPlotter | None = None
        if show_plots:  # pragma: no cover
            self.plotter = RealtimeAOPlotter()
            self.plotter.run()

    def replay(self, log_dir: str | Path):
        """Run the AO simulation loop."""
        logger.info(f"Running controller with log data from {log_dir}")
        log_files = LogFiles(log_dir)
        log_data = LogParser(log_files)

        if self.plotter is not None:  # pragma: no cover
            log_files.plot()
            plt.show()

        time_vec = log_data.data.left.hip.time
        angle_vec = log_data.data.left.hip.angles
        for i in range(len(angle_vec) - 1):
            t = time_vec[i] - time_vec[0]
            dt = time_vec[i + 1] - time_vec[i]

            th_deg = angle_vec[i][self.ang_idx]
            dth_deg = angle_vec[i][
                self.ang_idx
            ]  # TODO: replace with actual derivative if available
            th = np.deg2rad(th_deg)
            dth = np.deg2rad(dth_deg)

            self.step(t=t, dt=dt, th=th, dth=dth)

    def run(self) -> None:
        """Run the AO simulation loop."""
        # TODO: implement the controller that doesn't replay data
        try:
            while True:
                t, th, dth = 0.0, 0.0, 0.0
                dt = 0.01

                self.step(t=t, dt=dt, th=th, dth=dth)

        except KeyboardInterrupt:
            logger.info("Stopping AO simulation.")
            return

    def step(self, t: float, dt: float, th: float, dth: float) -> None:
        """Step the AO ahead with one frame of data from the IMU."""
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
                t,
                th,
                self.estimator.ao.theta_hat,
                self.estimator.ao.omega,
                self.estimator.phi_gp,
            )
