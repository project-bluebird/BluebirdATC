from __future__ import annotations

import typing

from bluebird_gymnasium.actions.simple.outcomm import outcomm

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs.base import BaseEnv


def test_outcomm(gym_env: BaseEnv):
    """Test `outcomm` action function."""

    _sector = gym_env.get_active_airspace_sector()
    tracked_aircraft = gym_env.get_tracked_aircraft_data()

    callsign = next(iter(tracked_aircraft.keys()))
    exit_coord = tracked_aircraft[callsign].exit_coords[_sector]
    next_sector = exit_coord.to_sector

    action = outcomm(callsign, gym_env)
    assert action.kind == "outcomm"
    assert action.value == next_sector
