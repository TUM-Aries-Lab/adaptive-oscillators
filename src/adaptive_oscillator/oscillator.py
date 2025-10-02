"""Adaptive Oscillator gait tracking."""

from dataclasses import dataclass

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from adaptive_oscillator.definitions import ETA, N_HARMONICS, NU_OMEGA, NU_PHI


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
@dataclass
class AOParameters:
    """Adaptive Oscillator parameters."""

    eta: float = ETA
    nu_phi: float = NU_PHI
    nu_omega: float = NU_OMEGA
    n_harmonics: int = N_HARMONICS


# -----------------------------------------------------------------------------
# Adaptive Oscillator
# -----------------------------------------------------------------------------
class AdaptiveOscillator:
    """Adaptive Oscillator tracking rhythmic signals like inter-limb hip angle."""

    def __init__(self, params: AOParameters, omega_init: float = 1.0):
        self.params = params
        self.n = params.n_harmonics
        self.omega = omega_init
        self.alpha_0 = 0.0
        self.alpha = np.zeros(self.n)
        self.phi = np.zeros(self.n)
        self.last_t = 0.0
        self.theta_hat = 0.0

    def _dynamics(self, t: float, y: NDArray, theta_il: float) -> NDArray:
        omega = y[0]
        alpha_0 = y[1]
        alpha = y[2 : 2 + self.n]
        phi = y[2 + self.n :]

        theta_hat = alpha_0 + np.sum(alpha * np.sin(phi))
        F = theta_il - theta_hat
        alpha_sum = np.sum(alpha) + 1e-6

        phi_dot = (
            omega * np.arange(1, self.n + 1)
            + self.params.nu_phi * F * np.cos(phi) / alpha_sum
        )
        omega_dot = self.params.nu_omega * F * np.cos(phi[0]) / alpha_sum
        dalpha = self.params.eta * F * np.sin(phi)
        dalpha_0 = self.params.eta * F

        return np.concatenate([[omega_dot, dalpha_0], dalpha, phi_dot])

    def update(self, t: float, theta_il: float, solver: str = "RK45") -> float:
        """Integrate the oscillator from self.last_t to t, return gait phase φ_GP(t)."""
        y0 = np.concatenate([[self.omega, self.alpha_0], self.alpha, self.phi])
        sol = solve_ivp(
            fun=lambda t_, y_: self._dynamics(t_, y_, theta_il),
            t_span=(self.last_t, t),
            y0=y0,
            method=solver,
            max_step=0.01,
        )

        y = sol.y[:, -1]
        self.omega = y[0]
        self.alpha_0 = y[1]
        self.alpha = y[2 : 2 + self.n]
        self.phi = y[2 + self.n :]
        self.last_t = t

        self.theta_hat = self.alpha_0 + np.sum(self.alpha * np.sin(self.phi))
        return np.mod(self.phi[0], 2 * np.pi)


# -----------------------------------------------------------------------------
# Gait Phase Estimation
# -----------------------------------------------------------------------------
class GaitPhaseEstimator:
    """Estimates corrected gait phase using AOs, event detection, and correction."""

    def __init__(self, params: AOParameters):
        self.ao = AdaptiveOscillator(params)
        self.last_t_start = -np.inf
        self.phi_error = 0.0
        self.ke = 1.0
        self.phi_gp = 0.0

    def detect_gait_event(
        self, t: float, theta_il: float, theta_il_dot: float, period: float
    ) -> bool:
        """Detect gait event based on theta_il and theta_il_dot."""
        return theta_il_dot < 0.01 and t - self.last_t_start > 0.7 * period

    def correct_phase(
        self, phi_gp: float, t: float, t_start: float, omega: float
    ) -> float:
        """Correct gait phase φ(t) using error correction."""
        Pe = -phi_gp if 0 <= phi_gp < np.pi else 2 * np.pi - phi_gp
        Ce = self.ke * (Pe - self.phi_error)
        self.phi_error += Ce * np.exp(-omega * (t - t_start))
        return np.mod(phi_gp + self.phi_error, 2 * np.pi)

    def update(self, t: float, theta_il: float, theta_il_dot: float) -> float:
        """Update gait phase and return corrected gait phase φ(t)."""
        self.phi_gp = self.ao.update(t, theta_il)
        omega = self.ao.omega
        period = 2 * np.pi / omega

        if self.detect_gait_event(t, theta_il, theta_il_dot, period):
            self.last_t_start = t

        phi = self.correct_phase(self.phi_gp, t, self.last_t_start, omega)

        logger.debug(
            f"t={t:.2f}, φ_GP={self.phi_gp:.2f}, "
            f"φ={phi:.2f}, ω={omega:.2f}, θ_hat={self.ao.theta_hat:.2f}"
        )
        return phi


# -----------------------------------------------------------------------------
# PID Controller
# -----------------------------------------------------------------------------
class PIDController:
    """PID controller."""

    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.last_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        """Compute PID output."""
        self.integral += error * dt
        derivative = (error - self.last_error) / dt
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


# -----------------------------------------------------------------------------
# Low-Level Motor Controller
# -----------------------------------------------------------------------------
class LowLevelController:
    """Low-level motor controller."""

    def __init__(
        self,
        kp: float = 5.0,
        ki: float = 0.0,
        kd: float = 0.1,
        gait_shape: NDArray | None = None,
    ):
        self.pid = PIDController(kp, ki, kd)
        x = np.linspace(0, 2 * np.pi, 100)
        y = gait_shape if gait_shape is not None else np.sin(x)
        self.spline = CubicSpline(x, y)

    def compute(self, phi: float, theta_m: float, dt: float) -> float:
        """Compute motor output."""
        theta_r = self.spline(phi - np.pi)
        error = theta_r - theta_m
        return self.pid.compute(error, dt)  # type: ignore[arg-type]


def sample_walking_data(
    period: float, t_start: float = 0.0, t_end: float = 100.0, dt: float = 0.01
) -> tuple[NDArray, NDArray, NDArray]:
    """Sample walking trajectory."""
    two_rad = 2 * np.pi
    t_vals = np.arange(t_start, t_end, dt)
    theta_il = np.sin(period * two_rad * t_vals)
    theta_il_dot = period * two_rad * np.cos(period * two_rad * t_vals)
    return t_vals, theta_il, theta_il_dot
