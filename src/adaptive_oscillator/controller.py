"""Controller module for the Adaptive Oscillator."""

import time

import matplotlib.pyplot as plt
from loguru import logger

from adaptive_oscillator.base_classes import AdaptiveOscillatorStepResult
from adaptive_oscillator.definitions import (
    DEFAULT_DELTA_TIME,
    FIG_SIZE,
    LEGEND_LOC,
    AOParameters,
    PIDGains,
)
from adaptive_oscillator.oscillator import GaitPhaseEstimator, LowLevelController
from adaptive_oscillator.utils import RealtimeAOPlotter


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

    def step(self, t: float, x: float, x_dot: float) -> AdaptiveOscillatorStepResult:
        """Step the AO ahead with one frame of data from the IMU."""
        logger.trace(f"Step: t={t:.2f}, x={x:.2f}, x_dot={x_dot:.2f}")
        dt = self._calculate_dt(t=t)
        phi = self.estimator.update(t=t, theta_il=x, theta_il_dot=x_dot)
        omega_cmd = self.controller.compute(phi=phi, theta_m=self.theta_m, dt=dt)
        self.theta_m += omega_cmd * dt

        # Store outputs
        step_result = AdaptiveOscillatorStepResult(
            timestamp=t,
            theta=x,
            theta_hat=self.estimator.ao.theta_hat,
            omega=self.estimator.ao.omega,
            gait_phase=self.estimator.phi_gp,
            offset=self.estimator.ao.alpha_0,
        )
        self.results.append(step_result)
        logger.debug(f"Step result: {step_result}")

        # Update live plot if enabled
        if self.plotter is not None:  # pragma: no cover
            self.plotter.update_data(data=step_result)
            time.sleep(dt)

        return step_result

    def _calculate_dt(self, t: float) -> float:
        """Calculate the change in time since the last step.

        :param t: time in seconds.
        :return: delta time in seconds.
        """
        if self.last_time is None:
            dt = DEFAULT_DELTA_TIME
        else:
            dt = t - self.last_time
        self.last_time = t
        return dt

    def _unpack_results(self) -> tuple:
        """Unpack results list from the controller."""
        (timestamps, thetas, theta_hats, omegas, phi_gps, offsets) = zip(
            *[
                (r.timestamp, r.theta, r.theta_hat, r.omega, r.gait_phase, r.offset)
                for r in self.results
            ]
        )
        return timestamps, thetas, theta_hats, omegas, phi_gps, offsets

    def plot_results(self) -> None:
        """Plot controller results.

        :return: None
        """
        logger.info("Plotting results...")
        t, thetas, theta_hats, omegas, gait_phase, offsets = self._unpack_results()

        _, axs = plt.subplots(4, 1, figsize=FIG_SIZE, sharex=True)

        axs[0].plot(t, thetas, label="input")
        axs[0].plot(t, theta_hats, label="estimated")
        axs[0].set_ylabel("Hip Angle (rad)")
        axs[0].set_title("Input vs Estimated Hip Angle")
        axs[0].legend(loc=LEGEND_LOC)

        axs[1].plot(t, omegas, color="green")
        axs[1].set_ylabel("Frequency (rad/s)")
        axs[1].set_title("Omega Estimate")

        axs[2].plot(t, gait_phase, color="purple")
        axs[2].set_ylabel("Gait Phase (rad)")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_title("Estimated Gait Phase")

        axs[3].plot(t, offsets, color="red")
        axs[3].set_ylabel("Offset (rad)")
        axs[3].set_xlabel("Time (s)")
        axs[3].set_title("Adaptive Oscillator Offset")

        for i in range(4):
            axs[i].grid(True)
        plt.tight_layout()

        try:
            plt.show()
        except KeyboardInterrupt:
            logger.debug("Closing the controller results plot.")
            plt.close()
