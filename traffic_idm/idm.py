"""The Intelligent Driver Model (Treiber, Hennecke & Helbing, 2000).

Acceleration of a vehicle with speed ``v``, gap ``s`` to its leader and
approach rate ``dv = v - v_leader``::

    a_idm = a * [ 1 - (v / v0)^delta - (s* / s)^2 ]
    s*    = s0 + max(0, v*T + v*dv / (2*sqrt(a*b)))

All functions are vectorised over numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IDMParams:
    """IDM driver/vehicle parameters (SI units)."""

    v0: float = 33.3  # desired speed [m/s] (~120 km/h)
    T: float = 1.5  # safe time headway [s]
    a: float = 0.73  # maximum acceleration [m/s^2]
    b: float = 1.67  # comfortable deceleration [m/s^2]
    s0: float = 2.0  # minimum jam gap [m]
    delta: float = 4.0  # acceleration exponent
    length: float = 5.0  # vehicle length [m]
    b_max: float = 9.0  # physical deceleration limit [m/s^2]


def idm_acceleration(v: np.ndarray, dv: np.ndarray, s: np.ndarray, p: IDMParams) -> np.ndarray:
    """Vectorised IDM acceleration.

    Parameters
    ----------
    v : own speeds [m/s]
    dv : approach rates ``v - v_leader`` [m/s]
    s : bumper-to-bumper gaps to the leader [m]; ``inf`` for a free road
    p : IDM parameters
    """
    v = np.asarray(v, dtype=float)
    dv = np.asarray(dv, dtype=float)
    s = np.asarray(s, dtype=float)
    s_star = p.s0 + np.maximum(0.0, v * p.T + v * dv / (2.0 * np.sqrt(p.a * p.b)))
    s_safe = np.maximum(s, 0.1)  # never divide by a non-positive gap
    acc = p.a * (1.0 - (v / p.v0) ** p.delta - (s_star / s_safe) ** 2)
    return np.clip(acc, -p.b_max, p.a)


def equilibrium_spacing(v: np.ndarray | float, p: IDMParams) -> np.ndarray:
    """Equilibrium gap s_e(v) for steady-state homogeneous traffic."""
    v = np.asarray(v, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (p.s0 + v * p.T) / np.sqrt(np.maximum(1.0 - (v / p.v0) ** p.delta, 1e-12))
    return s


def fundamental_diagram(p: IDMParams, n: int = 2000) -> dict[str, np.ndarray | float]:
    """Steady-state density-flow relation implied by the IDM.

    Returns density [veh/km], flow [veh/h] and speed [m/s] arrays plus the
    capacity ``q_max`` [veh/h] and the critical density ``rho_crit`` [veh/km].
    """
    v = np.linspace(0.0, p.v0 * (1 - 1e-6), n)
    s = equilibrium_spacing(v, p)
    rho = 1.0 / (s + p.length)  # veh/m
    q = rho * v  # veh/s
    i = int(np.argmax(q))
    return {
        "density": rho * 1000.0,
        "flow": q * 3600.0,
        "speed": v,
        "q_max": float(q[i] * 3600.0),
        "rho_crit": float(rho[i] * 1000.0),
        "v_crit": float(v[i]),
    }
