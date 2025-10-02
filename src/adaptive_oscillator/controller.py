"""Controller module for the Adaptive Oscillator."""

import time
from pathlib import Path

import numpy as np
from loguru import logger

from adaptive_oscillator.oscillator import (
    AOParameters,
    GaitPhaseEstimator,
    LowLevelController,
)
from adaptive_oscillator.utils.parser_utils import LogFiles, LogParser
from adaptive_oscillator.utils.plot_utils import (
    RealtimeAOPlotter,
    plot_log_data,
    plot_sim_results,
)


class AOController:
    """Encapsulate the AO control loop and optional real-time plotting."""

    def __init__(self, real_time: bool = False, plot: bool = False):
        """Initialize controller.

        :param real_time: Enable real-time plotting with Dash.
        :param plot: Plot IMU logs before running the control loop.
        """
        self.plot_logs = plot
        self.plot_results = real_time

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
        if self.plot_results:
            self.plotter = RealtimeAOPlotter()
            self.plotter.run()

    def replay(self, log_dir: str | Path):
        """Run the AO simulation loop."""
        logger.info(f"Running controller with log data from {log_dir}")
        log_files = LogFiles(log_dir)
        log_data = LogParser(log_files)

        if self.plot_logs:
            plot_log_data(log_files)

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

            # Update live plot if enabled
            if self.plot_results and self.plotter:  # pragma: no cover
                self.plotter.update_data(
                    t,
                    th,
                    self.estimator.ao.theta_hat,
                    self.estimator.ao.omega,
                    self.estimator.phi_gp,
                )
                time.sleep(dt)

        if self.plot_results:  # pragma: no cover
            t0 = time_vec[0]
            time_data = time_vec[:-1] - t0
            theta_il = np.array(
                [
                    log_data.data.left.hip.angles[t][self.ang_idx]
                    for t in range(len(time_data))
                ]
            )
            theta_il = np.deg2rad(theta_il)

            plot_sim_results(
                time_data,
                theta_il,
                self.theta_hat_output,
                self.omegas,
                self.phi_gp_output,
            )

    def run(self) -> None:
        """Run the AO simulation loop."""
        # TODO: implement the controller that doesn't replay data
        try:
            while True:
                t, th, dth = 0.0, 0.0, 0.0
                dt = 0.01

                self.step(t=t, dt=dt, th=th, dth=dth)

                # Update live plot if enabled
                if self.plot_results and self.plotter:  # pragma: no cover
                    self.plotter.update_data(
                        t,
                        th,
                        self.estimator.ao.theta_hat,
                        self.estimator.ao.omega,
                        self.estimator.phi_gp,
                    )
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
