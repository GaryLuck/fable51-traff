"""Ramp-metering (adaptive entrance signal) controllers.

Every controller exposes ``update(t, density)`` returning the metering rate in
vehicles/hour for the next control interval.  ``density`` is the measured
mainline density [veh/km] just downstream of the merge, the quantity real
ALINEA installations estimate from loop-detector occupancy.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class MeteringController:
    interval: float = 30.0  # seconds between control decisions

    def update(self, t: float, density: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__


@dataclass
class NoMetering(MeteringController):
    """Entrance signal switched off: ramp vehicles enter as fast as they can."""

    interval: float = 30.0

    def update(self, t: float, density: float) -> float:
        return float("inf")

    @property
    def name(self) -> str:
        return "No metering"


@dataclass
class FixedRateMetering(MeteringController):
    """Time-of-day fixed release rate, the classic non-adaptive signal."""

    rate: float = 600.0  # veh/h
    interval: float = 30.0

    def update(self, t: float, density: float) -> float:
        return self.rate

    @property
    def name(self) -> str:
        return f"Fixed {self.rate:.0f} veh/h"


@dataclass
class ALINEAMetering(MeteringController):
    """ALINEA feedback controller (Papageorgiou et al., 1991).

    ``r(k+1) = r(k) + K_R * (rho_target - rho_measured)``

    The release rate is nudged up while the downstream mainline is below the
    target density and down when it exceeds it, keeping the merge section near
    capacity without letting it break down.
    """

    target_density: float = 25.0  # veh/km, slightly below critical density
    gain: float = 50.0  # (veh/h) per (veh/km) of density error
    r_min: float = 200.0  # never starve the ramp completely
    r_max: float = 1800.0
    interval: float = 30.0
    rate: float = field(default=900.0)

    def update(self, t: float, density: float) -> float:
        self.rate = float(min(self.r_max, max(self.r_min, self.rate + self.gain * (self.target_density - density))))
        return self.rate

    @property
    def name(self) -> str:
        return "ALINEA adaptive"
