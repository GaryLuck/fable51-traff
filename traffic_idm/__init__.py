"""Highway traffic simulation with the Intelligent Driver Model (IDM).

The package models a single-lane highway with an on-ramp and compares
ramp-metering ("adaptive entrance signal") strategies for mitigating jams.
"""

from .idm import IDMParams, idm_acceleration, equilibrium_spacing, fundamental_diagram
from .metering import NoMetering, FixedRateMetering, ALINEAMetering
from .highway import HighwayConfig, DemandProfile, HighwaySimulation, SimulationResult
from .metrics import summarize

__all__ = [
    "IDMParams",
    "idm_acceleration",
    "equilibrium_spacing",
    "fundamental_diagram",
    "NoMetering",
    "FixedRateMetering",
    "ALINEAMetering",
    "HighwayConfig",
    "DemandProfile",
    "HighwaySimulation",
    "SimulationResult",
    "summarize",
]
