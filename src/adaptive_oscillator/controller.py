"""Controller module for the Adaptive Oscillator."""

import time

import matplotlib.pyplot as plt
from loguru import logger

from adaptive_oscillator.base_classes import AdaptiveOscillatorStepResult
from adaptive_oscillator.definitions import (
    DEFAULT_DELTA_TIME,
    FIG_SIZE,
    AOParameters,
    PIDGains,
)
from adaptive_oscillator.oscillator import GaitPhaseEstimator, LowLevelController
from adaptive_oscillator.utils import PlotData, RealtimeAOPlotter


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

        :param config: AOParameters object or None
        :param pid_gains: PIDGains object or None
        :param show_plots: Plot IMU logs before running the control loop.
        :param ssh: Use SSH tunneling.
        """
        logger.info("Initializing controller.")
        self.results: list[AdaptiveOscillatorStepResult] = []
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
        logger.debug(f"Step: t={t:.2f}, x={x:.2f}, x_dot={x_dot:.2f}")
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
        logger.debug(
            f"theta_hat: {theta_hat:.2f}, omega: {omega:.2f}, phi_gp: {phi_gp:.2f}"
        )

        # Update live plot if enabled
        if self.plotter is not None:  # pragma: no cover
            data = PlotData(
                t=t,
                theta_il=x,
                theta_hat=self.estimator.ao.theta_hat,
                omega=self.estimator.ao.omega,
                phi_gp=self.estimator.phi_gp,
                aux_1=self.estimator.ao.alpha_0,
                aux_2=0.0,
            )
            self.plotter.update_data(data=data)
            time.sleep(0.01)

        step_result = AdaptiveOscillatorStepResult(
            timestamp=t,
            theta_hat=theta_hat,
            omega=omega,
            theta=x,
            phi_gp=phi_gp,
            offset=self.estimator.ao.alpha_0,
        )
        self.results.append(step_result)
        return theta_hat, omega, phi_gp

    def plot_results(self) -> None:
        """Plot controller results.

        :return: None
        """
        logger.info("Plotting results")
        (timestamps, thetas, theta_hats, omegas, phi_gps, offsets) = zip(
            *[
                (r.timestamp, r.theta, r.theta_hat, r.omega, r.phi_gp, r.offset)
                for r in self.results
            ]
        )

        _, axs = plt.subplots(3, 1, figsize=FIG_SIZE, sharex=True)

        axs[0].plot(timestamps, thetas, label="θ_IL (input)")
        axs[0].plot(timestamps, theta_hats, label="θ̂_IL (estimated)")
        axs[0].set_ylabel("Angle (rad)")
        axs[0].set_title("Input vs Estimated Hip Angle")

        axs[1].plot(timestamps, omegas, label="Motor θ", color="green")
        axs[1].set_ylabel("Angle (rad)")
        axs[1].set_title("Omega Estimate")

        axs[2].plot(timestamps, phi_gps, label="φ_GP (Gait Phase)", color="purple")
        axs[2].set_ylabel("Phase (rad)")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_title("Estimated Gait Phase")

        legend_loc = "upper right"

        for i in range(3):
            axs[i].legend(loc=legend_loc)
            axs[i].grid(True)

        plt.tight_layout()
        plt.show()
