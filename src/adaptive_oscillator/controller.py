"""Controller module for the Adaptive Oscillator.

Supports multi-joint input (hip, knee, ankle) with per-joint harmonics
and weighted aggregation of outputs.
"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from scipy.signal import butter, sosfilt_zi

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

JOINT_NAMES = ("hip", "knee", "ankle")


class OnlineLowPassFilter:
    """Causal 2nd-order Butterworth low-pass filter applied sample-by-sample.

    Suitable for real-time / streaming use: each call to :meth:`__call__`
    processes exactly one sample and returns the filtered value.
    """

    def __init__(self, cutoff_hz: float, fs_hz: float, order: int = 2):
        """Design the filter.

        :param cutoff_hz: Cut-off frequency in Hz.
        :param fs_hz: Sampling frequency in Hz.
        :param order: Butterworth filter order (default 2).
        """
        nyq = fs_hz / 2.0
        if cutoff_hz >= nyq:
            raise ValueError(
                f"cutoff_hz ({cutoff_hz}) must be < Nyquist ({nyq}) "
                f"for fs_hz={fs_hz}"
            )
        self.sos = butter(N=order, Wn=cutoff_hz / nyq, btype="low", output="sos")
        # Initial filter state (steady-state for the first sample)
        self._zi = sosfilt_zi(self.sos)  # shape (n_sections, 2)
        self._initialised = False

    def __call__(self, sample: float) -> float:
        """Filter one sample and return the filtered value."""
        x = np.array([[sample]])
        if not self._initialised:
            # Scale initial conditions so the filter starts at the first value
            self._zi = self._zi * sample
            self._initialised = True
        from scipy.signal import sosfilt

        y, self._zi = sosfilt(self.sos, x.ravel(), zi=self._zi)
        return float(y[0])


class AOController:
    """Encapsulate the multi-joint AO control loop and optional real-time plotting.

    Each joint (hip, knee, ankle) runs its own Adaptive Oscillator with
    independent ``n_harmonics``.  The per-joint outputs are aggregated into
    a single result using configurable weights.
    """

    def __init__(
        self,
        joint_configs: dict[str, AOParameters] | None = None,
        joint_weights: dict[str, float] | None = None,
        pid_gains: PIDGains | None = None,
        filter_cutoff_hz: float | None = 6.0,
        sample_rate_hz: float = 100.0,
        show_plots: bool = False,
        ssh: bool = False,
    ):
        """Initialize multi-joint controller.

        :param joint_configs: Mapping of joint name -> AOParameters.  Each
            AOParameters may specify a different ``n_harmonics``.  If *None*,
            default AOParameters are used for every joint in ``JOINT_NAMES``.
        :param joint_weights: Mapping of joint name -> aggregation weight.
            Weights are normalised internally so they sum to 1.  If *None*,
            equal weights are used.
        :param pid_gains: PIDGains object or None.
        :param filter_cutoff_hz: Low-pass Butterworth cut-off frequency (Hz).
            Set to *None* to disable filtering.
        :param sample_rate_hz: Sampling rate of the input data (Hz).
        :param show_plots: Plot IMU logs before running the control loop.
        :param ssh: Use SSH tunneling.
        """
        logger.info("Initializing multi-joint controller.")

        # --- per-joint configs ------------------------------------------------
        if joint_configs is None:
            joint_configs = {name: AOParameters() for name in JOINT_NAMES}
        self.joint_names: tuple[str, ...] = tuple(joint_configs.keys())

        # --- per-joint weights (normalised) -----------------------------------
        if joint_weights is None:
            joint_weights = {name: 1.0 for name in self.joint_names}
        total_w = sum(joint_weights.values())
        self.joint_weights: dict[str, float] = {
            k: v / total_w for k, v in joint_weights.items()
        }

        # --- one GaitPhaseEstimator per joint ---------------------------------
        self.estimators: dict[str, GaitPhaseEstimator] = {
            name: GaitPhaseEstimator(config)
            for name, config in joint_configs.items()
        }

        # --- optional low-pass filter per joint per signal -----------------
        self.filters: dict[str, dict[str, OnlineLowPassFilter]] | None = None
        if filter_cutoff_hz is not None:
            self.filters = {
                name: {
                    "x": OnlineLowPassFilter(filter_cutoff_hz, sample_rate_hz),
                    "x_dot": OnlineLowPassFilter(filter_cutoff_hz, sample_rate_hz),
                }
                for name in self.joint_names
            }
            logger.info(
                f"Low-pass filter enabled: cutoff={filter_cutoff_hz} Hz, "
                f"fs={sample_rate_hz} Hz."
            )

        self.results: list[AdaptiveOscillatorStepResult] = []
        self.controller = LowLevelController(pid_gains)
        self.theta_m = 0.0
        self.last_time: float | None = None

        self.plotter: RealtimeAOPlotter | None = None
        if show_plots:
            self.plotter = RealtimeAOPlotter(ssh=ssh)
            self.plotter.run()

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(
        self,
        t: float,
        joint_data: dict[str, tuple[float, float]],
    ) -> AdaptiveOscillatorStepResult:
        """Step all joint AOs and return a weighted, aggregated result.

        :param t: Current timestamp (shared across joints).
        :param joint_data: Mapping of joint name -> ``(x, x_dot)`` where *x*
            is the joint angle and *x_dot* its derivative.
        :return: Aggregated :class:`AdaptiveOscillatorStepResult`.
        """
        logger.trace(f"Step: t={t:.2f}, joints={list(joint_data.keys())}")
        dt = self._calculate_dt(t=t)

        # --- run each joint's AO independently --------------------------------
        per_joint: dict[str, dict[str, float]] = {}
        for name, (x, x_dot) in joint_data.items():
            # Apply low-pass filter if enabled
            if self.filters is not None:
                x = self.filters[name]["x"](x)
                x_dot = self.filters[name]["x_dot"](x_dot)

            estimator = self.estimators[name]
            phi = estimator.update(t=t, theta_il=x, theta_il_dot=x_dot)
            per_joint[name] = {
                "theta": x,
                "theta_hat": estimator.ao.theta_hat,
                "omega": estimator.ao.omega,
                "gait_phase": estimator.phi_gp,
                "offset": estimator.ao.alpha_0,
                "phi": phi,
            }

        # --- weighted aggregation ---------------------------------------------
        w = self.joint_weights
        agg_theta = sum(w[n] * per_joint[n]["theta"] for n in per_joint)
        agg_theta_hat = sum(w[n] * per_joint[n]["theta_hat"] for n in per_joint)
        agg_omega = sum(w[n] * per_joint[n]["omega"] for n in per_joint)
        agg_gait_phase = sum(w[n] * per_joint[n]["gait_phase"] for n in per_joint)
        agg_offset = sum(w[n] * per_joint[n]["offset"] for n in per_joint)
        agg_phi = sum(w[n] * per_joint[n]["phi"] for n in per_joint)

        # --- motor command from aggregated phase ------------------------------
        omega_cmd = self.controller.get_command(
            phi=agg_phi, theta_m=self.theta_m, dt=dt
        )
        self.theta_m += omega_cmd * dt

        # Store aggregated outputs
        step_result = AdaptiveOscillatorStepResult(
            timestamp=t,
            theta=agg_theta,
            theta_hat=agg_theta_hat,
            omega=agg_omega,
            gait_phase=agg_gait_phase,
            offset=agg_offset,
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
        side: str = "",
        save_plot: bool = False,
        add_timestamp: bool = False,
    ) -> None:
        """Plot aggregated controller results.

        :param side: side string (e.g. 'left', 'right').
        :param save_plot: If True, save the plot.
        :param add_timestamp: If True, add timestamp to plot.
        :return: None
        """
        logger.info("Plotting aggregated results...")
        t, thetas, theta_hats, omegas, gait_phase, offsets = self._unpack_results()

        joints_label = "+".join(self.joint_names)

        _, axs = plt.subplots(4, 1, figsize=FIG_SIZE, sharex=True)

        axs[0].plot(t, thetas, label="input (weighted)")
        axs[0].plot(t, theta_hats, label="estimated (weighted)")
        axs[0].set_ylabel("Angle (rad)")
        axs[0].set_title(f"Aggregated Input vs Estimated Angle [{joints_label}]")
        axs[0].legend(loc=LEGEND_LOC)

        axs[1].plot(t, omegas, color="green")
        axs[1].set_ylabel("Frequency (rad/s)")
        axs[1].set_title("Aggregated Omega Estimate")

        axs[2].plot(t, gait_phase, color="purple")
        axs[2].set_ylabel("Gait Phase (rad)")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_title("Aggregated Estimated Gait Phase")

        axs[3].plot(t, offsets, color="red")
        axs[3].set_ylabel("Offset (rad)")
        axs[3].set_xlabel("Time (s)")
        axs[3].set_title("Aggregated Adaptive Oscillator Offset")

        for i in range(4):
            axs[i].grid(True)
        plt.tight_layout()

        if save_plot:
            if add_timestamp:
                timestamp = datetime.now().strftime(DATE_FORMAT)
                filename = f"results_{side}_{joints_label}_{timestamp}.png"
            else:
                filename = f"results_{side}_{joints_label}.png"
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
