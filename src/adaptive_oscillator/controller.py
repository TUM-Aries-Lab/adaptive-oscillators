"""Controller module for the Adaptive Oscillator."""

import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from loguru import logger

from adaptive_oscillator.base_classes import AdaptiveOscillatorStepResult
from adaptive_oscillator.definitions import (
    DATE_FORMAT,
    DEFAULT_DELTA_TIME,
    FIG_SIZE,
    LEGEND_LOC,
    RESULTS_DIR,
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
        if show_plots:
            self.plotter = RealtimeAOPlotter(ssh=ssh)
            self.plotter.run()

    def step(self, t: float, x: float, x_dot: float) -> AdaptiveOscillatorStepResult:
        """Step the AO ahead with one frame of data from the IMU."""
        logger.trace(f"Step: t={t:.2f}, x={x:.2f}, x_dot={x_dot:.2f}")
        dt = self._calculate_dt(t=t)
        phi = self.estimator.update(t=t, theta_il=x, theta_il_dot=x_dot)
        omega_cmd = self.controller.get_command(phi=phi, theta_m=self.theta_m, dt=dt)
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
        if self.plotter is not None:
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

    def plot_results(
        self,
        joint: str,
        side: str,
        save_plot: bool = False,
        add_timestamp: bool = False,
    ) -> None:
        """Plot controller results.

        :param joint: joint string
        :param side: side string
        :param save_plot: If True, save the plot.
        :param add_timestamp: If True, add timestamp to plot.
        :return: None
        """
        logger.info("Plotting results...")
        t, thetas, theta_hats, omegas, gait_phase, offsets = self._unpack_results()

        _, axs = plt.subplots(4, 1, figsize=FIG_SIZE, sharex=True)

        axs[0].plot(t, thetas, label="input")
        axs[0].plot(t, theta_hats, label="estimated")
        axs[0].set_ylabel(f"{joint.capitalize()} Angle (rad)")
        axs[0].set_title(f"Input vs Estimated {joint.capitalize()} Angle")
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

        if save_plot:
            if add_timestamp:
                timestamp = datetime.now().strftime(DATE_FORMAT)
                filename = f"results_{side}_{joint}_{timestamp}.png"
            else:
                filename = f"results_{side}_{joint}.png"
            plt.savefig(RESULTS_DIR / filename)
        else:
            try:
                plt.show()
            except KeyboardInterrupt:
                logger.debug("Closing the controller results plot.")
                plt.close()

    def write_results(self, filepath: Path) -> None:
        """Write results to file.

        :param filepath: Path to the file to write.
        :return: None
        """
        logger.info(f"Writing results to '{filepath}'.")
        headers = ["t", "thetas", "theta_hats", "omegas", "gait_phase", "offsets"]

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w") as f:
            f.write(", ".join(headers) + "\n")
            for row in self.results:
                f.write(", ".join([f"{i:.3f}" for i in row]) + "\n")

        logger.success(f"Results written to '{filepath}'.")
