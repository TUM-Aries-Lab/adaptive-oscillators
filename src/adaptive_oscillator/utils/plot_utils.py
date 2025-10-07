"""Plot utilities module."""

import logging
import threading
from collections import deque
from dataclasses import dataclass

import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)


@dataclass
class PlotMetrics:
    """AO plot metrics."""

    time_data = "time_data"
    theta_hat = "theta_hat"
    theta_il = "theta_il"
    omega = "omega"
    phi_gp = "phi_gp"


class RealtimeAOPlotter:  # pragma: no cover
    """Real-time Data plotter for the Adaptive Oscillator."""

    def __init__(
        self, ssh: bool = False, window_sec: float = 5.0, frequency_hz: int = 100
    ) -> None:
        self.window_sec = window_sec
        self.data_points = int(window_sec * frequency_hz)

        self.app = Dash(__name__)
        self._setup_layout()
        self._register_callbacks()

        # Thread-safe buffers
        self._lock = threading.Lock()
        self.data: dict = {
            PlotMetrics.time_data: deque(maxlen=self.data_points),
            PlotMetrics.theta_il: deque(maxlen=self.data_points),
            PlotMetrics.theta_hat: deque(maxlen=self.data_points),
            PlotMetrics.omega: deque(maxlen=self.data_points),
            PlotMetrics.phi_gp: deque(maxlen=self.data_points),
        }

        self.host = "0.0.0.0" if ssh else "127.0.0.1"  # ruff: ignore B104,S104
        self.port = 8050

    def _setup_layout(self) -> None:
        self.app.layout = html.Div(
            [
                html.H1(
                    "Adaptive Oscillator Logfile Replay",
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
                dcc.Interval(
                    id="interval-component", interval=200, n_intervals=0
                ),  # 5 Hz is plenty over SSH
            ],
            style={"padding": "10px", "gap": "0px"},
        )

    def _register_callbacks(self) -> None:
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

    def _snapshot(
        self,
    ) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        """Copy current data under a lock to avoid torn reads."""
        with self._lock:
            time_data = list(self.data[PlotMetrics.time_data])
            theta_il = list(self.data[PlotMetrics.theta_il])
            theta_hat = list(self.data[PlotMetrics.theta_hat])
            omega = list(self.data[PlotMetrics.omega])
            phi_gp = list(self.data[PlotMetrics.phi_gp])
        return time_data, theta_il, theta_hat, omega, phi_gp

    def _generate_figures(self) -> tuple[go.Figure, go.Figure, go.Figure]:
        time_data, theta_il, theta_hat, omega, phi_gp = self._snapshot()
        if not time_data:
            empty = go.Figure()
            return empty, empty, empty

        # Keep last window
        latest = time_data[-1]
        window_start = latest - self.window_sec
        start_idx = next((i for i, t in enumerate(time_data) if t >= window_start), 0)
        time_data, theta_il, theta_hat, omega, phi_gp = (
            time_data[start_idx:],
            theta_il[start_idx:],
            theta_hat[start_idx:],
            omega[start_idx:],
            phi_gp[start_idx:],
        )

        margin = dict(l=30, r=10, t=30, b=30)

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
        """Update the data and plot it."""
        with self._lock:
            self.data[PlotMetrics.time_data].append(t)
            self.data[PlotMetrics.theta_il].append(theta_il)
            self.data[PlotMetrics.theta_hat].append(theta_hat)
            self.data[PlotMetrics.omega].append(omega)
            self.data[PlotMetrics.phi_gp].append(phi_gp)

    def run(self, threaded: bool = True) -> None:
        """Run the AO control loop."""
        if threaded:
            th = threading.Thread(
                target=self.app.run,
                kwargs={
                    "debug": False,
                    "use_reloader": False,
                    "host": self.host,
                    "port": self.port,
                },
                daemon=False,  # keep process alive
            )
            th.start()
        else:
            self.app.run_server(
                debug=False, use_reloader=False, host=self.host, port=self.port
            )
