from __future__ import annotations

import typing

from bluebird_dt.core import Action

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs.base import BaseEnv


def outcomm(
    callsign: str,
    gym_env: BaseEnv,
    value: str | None = None, # noqa: ARG001
    agent: str = "Agent",
) -> Action:
    """Generate a simulator action to outcomm an aircraft.

    Args:
        callsign: defines the identifier of the aircraft.
        gym_env: defines the gymnasium environment.
        value: defines the value to pass to the simulator action, which is
            the next sector to which the aircraft is navigating.
            Defaults to None, which initiates the retrieval of the next sector
            (information of the aircraft) within the function.
        agent: defines the name of agent. Defaults to 'Agent'.

    Return:
        the generated simulator action.
    """

    # outcomm the aircraft
    # get next sector
    active_airspace_sector = gym_env.get_active_airspace_sector()
    tracked_data = gym_env.get_tracked_aircraft_data(callsign)
    exit_coord = tracked_data.exit_coords[active_airspace_sector]

    next_sector = exit_coord.to_sector
    return Action(callsign, "outcomm", next_sector, agent=agent)
