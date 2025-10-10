"""Controller module for the Adaptive Oscillator."""

import time

from loguru import logger

from adaptive_oscillator.definitions import DEFAULT_DELTA_TIME, AOParameters, PIDGains
from adaptive_oscillator.oscillator import (
    GaitPhaseEstimator,
    LowLevelController,
)
from adaptive_oscillator.utils.plot_utils import RealtimeAOPlotter


class AOController:
    """Encapsulate the AO control loop and optional real-time plotting."""

    def __init__(
        self,
        config: AOParameters | None = None,
        pid_gains: PIDGains | None = None,
        show_plots: bool = False,
        ssh: bool = False,
    ):
        """Initialize controller.

        :param show_plots: Plot IMU logs before running the control loop.
        """
        self.estimator = GaitPhaseEstimator(config)
        self.controller = LowLevelController(pid_gains)
        self.theta_m = 0.0
        self.last_time: float | None = None

        self.plotter: RealtimeAOPlotter | None = None
        if show_plots:  # pragma: no cover
            self.plotter = RealtimeAOPlotter(ssh=ssh)
            self.plotter.run()

    def step(self, t: float, x: float, x_dot: float) -> tuple[float, float, float]:
        """Step the AO ahead with one frame of data from the IMU."""
        if self.last_time is None:
            dt = DEFAULT_DELTA_TIME
        else:
            dt = t - self.last_time
        self.last_time = t

        phi = self.estimator.update(t=t, theta_il=x, theta_il_dot=x_dot)
        omega_cmd = self.controller.compute(phi=phi, theta_m=self.theta_m, dt=dt)
        self.theta_m += omega_cmd * dt

        # Store outputs
        theta_hat = self.estimator.ao.theta_hat
        omega = self.estimator.ao.omega
        phi_gp = self.estimator.phi_gp
        logger.info(
            f"theta_hat: {theta_hat:.2f}, omega: {omega:.2f}, phi_gp: {phi_gp:.2f}"
        )

        # Update live plot if enabled
        if self.plotter is not None:  # pragma: no cover
            self.plotter.update_data(
                t=t,
                theta_il=x,
                theta_hat=self.estimator.ao.theta_hat,
                omega=self.estimator.ao.omega,
                phi_gp=self.estimator.phi_gp,
            )
            time.sleep(dt)

        return theta_hat, omega, phi_gp
