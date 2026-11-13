"""
Settings for named scenarios that are set up in more than one place.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

FLIGHT_SCHOOL_SETTINGS: Mapping[str, Any] = MappingProxyType(
    {
        "scenario_name": "Xplus-Sector",
        "fl_limits": (300, 350),
        "num_starter_aircraft": 4,
        "initial_spawn_rate": 0.012,
        "spawn_rate_increment": 0.0015,
        "spawn_rate_increase_interval": 300,
        "max_spawn_rate": 0.03,
        "same_level_spawn_distance": 40,
    }
)
"""
Traffic and airspace settings of the Flight School challenge, as keyword arguments of ``Infinite.setup``.

Used by ``Simulator.from_category("Flight School")`` and as the default scenario of bluebird_gymnasium's
``CustomInfiniteEnv``.
"""
