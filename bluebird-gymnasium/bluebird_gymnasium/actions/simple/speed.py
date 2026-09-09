from __future__ import annotations

import typing

from bluebird_dt.core import Action

from bluebird_gymnasium.actions import DEFAULT_RELATIVE_SPEED
from bluebird_gymnasium.utils.simulator_utils import get_aircraft_selected_cas

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs.base import BaseEnv
    from bluebird_gymnasium.utils.types import Number


def speed_increase(
    callsign: str,
    gym_env: BaseEnv,
    value: Number = DEFAULT_RELATIVE_SPEED,
    agent: str = "Agent",
) -> Action:
    """Generate a simulator action to increase an aircraft speed.

    A relative speed action.

    Args:
        callsign: defines the identifier of the aircraft.
        gym_env: defines the gymnasium environment.
        value: defines the value to pass to the simulator action.
            Defaults to DEFAULT_RELATIVE_SPEED.
        agent: defines the name of agent. Defaults to 'Agent'.

    Return:
        the generated simulator action.
    """

    assert value > 0

    # get the aircraft
    aircraft = gym_env.get_simulator_env().aircraft[callsign]

    # get the selected calibrated airspeed (cas) of the aircraft.
    selected_cas = round(get_aircraft_selected_cas(aircraft))

    # now compute the new cas from the relative measure
    new_cas = int(selected_cas + value)  # (increase speed is positive)

    return Action(callsign, "change_cas_to", new_cas, agent=agent)


def speed_decrease(
    callsign: str,
    gym_env: BaseEnv,
    value: Number = DEFAULT_RELATIVE_SPEED,
    agent: str = "Agent",
) -> Action:
    """Generate a simulator action to decrease an aircraft speed.

    A relative speed action.

    Args:
        callsign: defines the identifier of the aircraft.
        gym_env: defines the gymnasium environment.
        value: defines the value to pass to the simulator action.
            Defaults to DEFAULT_RELATIVE_SPEED.
        agent: defines the name of agent. Defaults to 'Agent'.

    Return:
        the generated simulator action.
    """

    assert value > 0

    # get the aircraft
    aircraft = gym_env.get_simulator_env().aircraft[callsign]

    # get the selected calibrated airspeed (cas) of the aircraft.
    selected_cas = round(get_aircraft_selected_cas(aircraft))

    # now compute the new cas from the relative measure
    new_cas = int(selected_cas - value)  # (decrease speed is negative)

    return Action(callsign, "change_cas_to", new_cas, agent=agent)


def speed_maintain_current(
    callsign: str,
    gym_env: BaseEnv,
    value: Number | None = None,  # noqa: ARG001
    agent: str = "Agent",
) -> Action:
    """Generate a simulator action to maintain an aircraft current speed.

    Args:
        callsign: defines the identifier of the aircraft.
        gym_env: defines the gymnasium environment.
        value: defines the value to pass to the simulator action.
            This is argument is ignored. Defaults to None.
        agent: defines the name of agent. Defaults to 'Agent'.

    Return:
        the generated simulator action.

    Note, the value is not used for this clearance. it's just set for
    consistency purpose with other simulator actions.
    """

    # get the aircraft
    aircraft = gym_env.get_simulator_env().aircraft[callsign]

    # get the selected calibrated airspeed (cas) of the aircraft.
    selected_cas = round(get_aircraft_selected_cas(aircraft))

    return Action(callsign, "change_cas_to", selected_cas, agent=agent)


def speed_choose_own(
    callsign: str,
    gym_env: BaseEnv,  # noqa: ARG001
    value: Number | None = None,  # noqa: ARG001
    agent: str = "Agent",
) -> Action:
    """Generate a simulator action to instruct an aircraft to choose own speed.

    Args:
        callsign: defines the identifier of the aircraft.
        gym_env: defines the gymnasium environment.
        value: defines the value to pass to the simulator action.
            This is argument is ignored. Defaults to None.
        agent: defines the name of agent. Defaults to 'Agent'.

    Return:
        the generated simulator action.

    Note, the value is not used for this clearance. it's just set for
    consistency purpose with other simulator actions.
    """

    return Action(callsign, "change_cas_to", None, agent=agent)
