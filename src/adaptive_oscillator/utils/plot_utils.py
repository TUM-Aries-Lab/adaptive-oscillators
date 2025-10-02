"""Real-time Plotly Dash interface for Adaptive Oscillator simulation."""

import threading
from collections import deque
from dataclasses import dataclass

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from loguru import logger
from numpy.typing import NDArray

from adaptive_oscillator.utils.parser_utils import IMUParser, QuaternionParser


@dataclass
class PlotMetrics:
    """Dictionary keys for the dash app."""

    time_data = "time_data"
    theta_hat = "theta_hat"
    theta_il = "theta_il"
    omega = "omega"
    phi_gp = "phi_gp"


class RealtimeAOPlotter:
    """Dash app for visualizing Adaptive Oscillator outputs in real time."""

    def __init__(self, window_sec: float = 5.0, frequency_hz: int = 100) -> None:
        """Initialize the real-time plotter.

        :param window_sec: Duration of the window to show in seconds.
        :param frequency_hz: Assumed data frequency in Hz.
        """
        self.window_sec = window_sec
        self.data_points = int(window_sec * frequency_hz)

        self.app = Dash(__name__)
        self._setup_layout()
        self._register_callbacks()

        # Data buffers
        self.data: dict = {
            PlotMetrics.time_data: deque(maxlen=self.data_points),
            PlotMetrics.theta_il: deque(maxlen=self.data_points),
            PlotMetrics.theta_hat: deque(maxlen=self.data_points),
            PlotMetrics.omega: deque(maxlen=self.data_points),
            PlotMetrics.phi_gp: deque(maxlen=self.data_points),
        }

    def _setup_layout(self) -> None:
        """Set up the layout for the Dash app with minimal spacing between plots."""
        self.app.layout = html.Div(
            [
                html.H1(
                    "Real-time Adaptive Oscillator Control Simulation",
                    style={"margin-bottom": "10px"},
                ),
                dcc.Graph(
                    id="hip-angle-graph", style={"height": "29vh", "margin": "0"}
                ),
                dcc.Graph(
                    id="omega-estimate-graph", style={"height": "29vh", "margin": "0"}
                ),
                dcc.Graph(
                    id="gait-phase-graph", style={"height": "29vh", "margin": "0"}
                ),
                dcc.Interval(id="interval-component", interval=100, n_intervals=0),
            ],
            style={"padding": "10px", "gap": "0px"},
        )

    def _register_callbacks(self) -> None:
        """Register Dash callbacks for updating plots."""

        @self.app.callback(
            [
                Output("hip-angle-graph", "figure"),
                Output("omega-estimate-graph", "figure"),
                Output("gait-phase-graph", "figure"),
            ],
            Input("interval-component", "n_intervals"),
        )
        def update_graphs(_):
            return self._generate_figures()

    def _generate_figures(self) -> tuple[go.Figure, go.Figure, go.Figure]:
        """Generate figures from current buffer contents."""
        time_data = list(self.data[PlotMetrics.time_data])
        theta_il = list(self.data[PlotMetrics.theta_il])
        theta_hat = list(self.data[PlotMetrics.theta_hat])
        omega = list(self.data[PlotMetrics.omega])
        phi_gp = list(self.data[PlotMetrics.phi_gp])

        if not time_data:
            empty_fig = go.Figure()
            return empty_fig, empty_fig, empty_fig

        # Crop data to latest 5 seconds
        if len(time_data) > 1:
            latest_time = time_data[-1]
            window_start = latest_time - self.window_sec
            start_idx = next(
                (i for i, t in enumerate(time_data) if t >= window_start), 0
            )

            time_data = time_data[start_idx:]
            theta_il = theta_il[start_idx:]
            theta_hat = theta_hat[start_idx:]
            omega = omega[start_idx:]
            phi_gp = phi_gp[start_idx:]

        margin = dict(l=30, r=10, t=30, b=30)

        # Hip angle plot
        hip_fig = go.Figure()
        hip_fig.add_trace(
            go.Scatter(x=time_data, y=theta_il, mode="lines", name="θ_IL (input)")
        )
        hip_fig.add_trace(
            go.Scatter(x=time_data, y=theta_hat, mode="lines", name="θ̂ (estimated)")
        )
        hip_fig.update_layout(
            title="Input vs Estimated Hip Angle",
            xaxis_title="Time (s)",
            yaxis_title="Angle (rad)",
            margin=margin,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        # Omega plot
        omega_fig = go.Figure()
        omega_fig.add_trace(
            go.Scatter(
                x=time_data, y=omega, mode="lines", name="ω", line=dict(color="green")
            )
        )
        omega_fig.update_layout(
            title="Omega Estimate",
            xaxis_title="Time (s)",
            yaxis_title="Angle (rad)",
            margin=margin,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        # Gait phase plot
        phase_fig = go.Figure()
        phase_fig.add_trace(
            go.Scatter(
                x=time_data,
                y=phi_gp,
                mode="lines",
                name="ϕ_GP",
                line=dict(color="purple"),
            )
        )
        phase_fig.update_layout(
            title="Estimated Gait Phase",
            xaxis_title="Time (s)",
            yaxis_title="Phase (rad)",
            margin=margin,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        return hip_fig, omega_fig, phase_fig

    def update_data(
        self, t: float, theta_il: float, theta_hat: float, omega: float, phi_gp: float
    ) -> None:
        """Append a new data point to the buffers.

        :param t: Timestamp in seconds.
        :param theta_il: Input joint angle.
        :param theta_hat: Estimated joint angle.
        :param omega: Estimated angular velocity.
        :param phi_gp: Gait phase estimate.
        """
        self.data[PlotMetrics.time_data].append(t)
        self.data[PlotMetrics.theta_il].append(theta_il)
        self.data[PlotMetrics.theta_hat].append(theta_hat)
        self.data[PlotMetrics.omega].append(omega)
        self.data[PlotMetrics.phi_gp].append(phi_gp)

    def run(self, threaded: bool = True) -> None:
        """Run the Dash server.

        :param threaded: If True, run in a background thread.
        """
        if threaded:
            threading.Thread(
                target=self.app.run,
                kwargs={"debug": False, "use_reloader": False},
                daemon=True,
            ).start()
        else:
            self.app.run(debug=False, use_reloader=False)
        logger.info("Dash app started. Open http://127.0.0.1:8050 in a browser.")


def plot_sim_results(
    t_vals: NDArray,
    theta_il: NDArray,
    theta_hat_output: list[float],
    omega_estimate: list[float],
    phi_gp_output: list[float],
) -> None:  # pragma: no cover
    """Plot results from the simulation."""
    logger.info("Plotting results")

    fig, axs = plt.subplots(3, 1, figsize=(15, 8), sharex=True)

    axs[0].plot(t_vals, theta_il, label="θ_IL (input)")
    axs[0].plot(t_vals, theta_hat_output, label="θ̂_IL (estimated)")
    axs[0].set_ylabel("Angle (rad)")
    axs[0].set_title("Input vs Estimated Hip Angle")

    axs[1].plot(t_vals, omega_estimate, label="Motor θ", color="green")
    axs[1].set_ylabel("Angle (rad)")
    axs[1].set_title("Omega Estimate")

    axs[2].plot(t_vals, phi_gp_output, label="φ_GP (Gait Phase)", color="purple")
    axs[2].set_ylabel("Phase (rad)")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_title("Estimated Gait Phase")

    legend_loc = "upper right"
    for i in range(3):
        axs[i].legend(loc=legend_loc)
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()


def plot_log_data(log_files) -> None:  # pragma: no cover
    """Plot log files."""
    logger.info("Plotting data.")
    accel_data = IMUParser(log_files.accel.right)
    accel_data.parse()
    accel_data.plot()

    gyro_data = IMUParser(log_files.gyro.right)
    gyro_data.parse()
    gyro_data.plot()

    quat_data = QuaternionParser(log_files.quat.right)
    quat_data.parse()
    quat_data.plot()

    accel_data = IMUParser(log_files.accel.left)
    accel_data.parse()
    accel_data.plot()

    gyro_data = IMUParser(log_files.gyro.left)
    gyro_data.parse()
    gyro_data.plot()

    quat_data = QuaternionParser(log_files.quat.left)
    quat_data.parse()
    quat_data.plot()

    plt.show()
