"""Controller module for the Adaptive Oscillator."""

import time

from loguru import logger

from adaptive_oscillator.definitions import DEFAULT_DELTA_TIME
from adaptive_oscillator.oscillator import (
    AOParameters,
    GaitPhaseEstimator,
    LowLevelController,
)
from adaptive_oscillator.utils.plot_utils import RealtimeAOPlotter


class AOController:
    """Encapsulate the AO control loop and optional real-time plotting."""

    def __init__(
        self,
        config: AOParameters | None = None,
        show_plots: bool = False,
        ssh: bool = False,
    ):
        """Initialize controller.

        :param show_plots: Plot IMU logs before running the control loop.
        """
        self.params = AOParameters() if config is None else config
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

    def step(self, t: float, th: float, dth: float) -> tuple[float, float, float]:
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
                theta_il=th,
                theta_hat=self.estimator.ao.theta_hat,
                omega=self.estimator.ao.omega,
                phi_gp=self.estimator.phi_gp,
            )
            time.sleep(dt)

        return theta_hat, omega, phi_gp
