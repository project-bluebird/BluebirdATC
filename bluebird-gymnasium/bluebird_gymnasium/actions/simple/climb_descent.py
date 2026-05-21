from __future__ import annotations

import typing

import numpy as np

from bluebird_dt.core import Action
from bluebird_gymnasium.actions import (
    DEFAULT_INTERVAL_FL,
    DEFAULT_RELATIVE_CLIMB_DESCENT,
)

from bluebird_gymnasium.utils.interaction_utils import (
    get_optimal_unblocked_flight_level,
)

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs.base import BaseEnv
    from bluebird_gymnasium.utils.types import Number

SELECTED_FL_MIN = 0
SELECTED_FL_MAX = 500


def fl_climb(
    callsign: str,
    gym_env: BaseEnv,
    value: Number | None = None,
    agent: str = "Agent",
) -> Action:
    """Generate a simulator action to relatively climb an aircraft.

    Args:
        callsign: defines the identifier of the aircraft.
        gym_env: defines the gymnasium environment.
        value: defines the relative value to pass to the simulator action.
            Defaults to None, which uses the default value in
            DEFAULT_RELATIVE_CLIMB_DESCENT.
        agent: defines the name of agent. Defaults to 'Agent'.

    Return:
        the generated simulator action.
    """

    value = DEFAULT_RELATIVE_CLIMB_DESCENT if value is None else value

    if value <= 0:
        raise ValueError("`value` should be a positive integer")
    elif value % DEFAULT_INTERVAL_FL != 0:
        raise ValueError(
            f"`value` should be in intervals of {DEFAULT_INTERVAL_FL}"
        )

    value = int(value)
    simulator_env = gym_env.get_simulator_env()
    value = simulator_env.aircraft[callsign].selected_fl + value
    value = np.clip(value, SELECTED_FL_MIN, SELECTED_FL_MAX).item()

    return Action(callsign, "change_flight_level_to", value, agent=agent)


def fl_descent(
    callsign: str,
    gym_env: BaseEnv,
    value: Number | None = None,
    agent: str = "Agent",
) -> Action:
    """Generate a simulator action to relatively descend an aircraft.

    Args:
        callsign: defines the identifier of the aircraft.
        gym_env: defines the gymnasium environment.
        value: defines the relative value to pass to the simulator action.
            Defaults to None, which uses the default value in
            DEFAULT_RELATIVE_CLIMB_DESCENT.
        agent: defines the name of agent. Defaults to 'Agent'.

    Return:
        the generated simulator action.
    """

    value = DEFAULT_RELATIVE_CLIMB_DESCENT if value is None else value

    if value <= 0:
        raise ValueError("`value` should be a positive integer")
    elif value % DEFAULT_INTERVAL_FL != 0:
        raise ValueError(
            f"`value` should be in intervals of {DEFAULT_INTERVAL_FL}"
        )

    value = int(value)
    simulator_env = gym_env.get_simulator_env()
    value = simulator_env.aircraft[callsign].selected_fl - value
    value = np.clip(value, SELECTED_FL_MIN, SELECTED_FL_MAX).item()

    return Action(callsign, "change_flight_level_to", value, agent=agent)


def fl_intermediate(
    callsign: str,
    gym_env: BaseEnv,
    value: Number | None = None,
    agent: str = "Agent",
) -> Action:
    """Generate a simulator action to climb/descend an aircraft.

    Climb or descend action to an intermediate flight level, the highest
    (climb) or lowest (descend) intermediate level available, which does
    not lead to a potential loss of separation with another aircraft.

    Note: an absolute climb or descend action.

    Args:
        callsign: defines the identifier of the aircraft.
        gym_env: defines the gymnasium environment.
        value: defines the value to pass to the simulator action.
            Defaults to None, which initiates the calculation of the
            intermediate exit flight level within the function.
        agent: defines the name of agent. Defaults to 'Agent'.

    Return:
        the generated simulator action.
    """

    simulator_env = gym_env.get_simulator_env()
    rollout_predictor = gym_env.get_rollout_predictor()
    _sector = gym_env.get_active_airspace_sector()

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    ac_exit_fl = tracked_aircraft[callsign].exit_coords[_sector].fl
    ac_selected_fl = simulator_env.aircraft[callsign].selected_fl

    if value is not None:
        chosen_fl = value

    else:
        if ac_selected_fl == ac_exit_fl:
            # if aircraft is already at its exit flight level, then the
            # intermediate level is the same as the exit flight level
            chosen_fl = ac_exit_fl

        else:
            # calculate the intermediate level here
            chosen_fl = get_optimal_unblocked_flight_level(
                callsign,
                _sector,
                tracked_aircraft,
                simulator_env,
                rollout_predictor,
                gym_env.get_traffic_monitor().get_relevant_traffic(callsign),
            )

    if ac_selected_fl <= chosen_fl:
        # climb or same level
        kind = "change_flight_level_to"
        _value = int(chosen_fl)
    else:
        # descend
        # despite being an intermediate flight level, assume that the
        # fix to level by is the exit fix.
        # NOTE: this could be re-implemented to select an intermdiate
        # fix. e.g., instead assume that the chosen fix is the one the
        # before the exit fix.
        # kind = "descend_now,level_by_fix"
        # ac_exit_fix = tracked_aircraft[callsign].exit_coords[_sector].fix
        # _value = (int(chosen_fl), ac_exit_fix)

        # option 2: stick to regular change_flight_level_to for intermediate
        # flight level descent
        kind = "change_flight_level_to"
        _value = int(chosen_fl)

    return Action(callsign, kind, _value, agent=agent)


def fl_exit(
    callsign: str,
    gym_env: BaseEnv,
    value: Number | None = None,
    agent: str = "Agent",
) -> Action:
    """Generate a simulator action to climb/descend an aircraft.

    Climb or descend action to its exit flight level.

    Note: an absolute climb or descend action.

    Args:
        callsign: defines the identifier of the aircraft.
        gym_env: defines the gymnasium environment.
        value: defines the value to pass to the simulator action.
            Defaults to None, which initiates the calculation of the exit
            flight level within the function.
        agent: defines the name of agent. Defaults to 'Agent'.

    Return:
        the generated simulator action.
    """

    active_airspace_sector = gym_env.get_active_airspace_sector()
    tracked_data = gym_env.get_tracked_aircraft_data(callsign)
    exit_coord = tracked_data.exit_coords[active_airspace_sector]

    if value is None:
        # set it here
        exit_fl = exit_coord.fl
    else:
        if value != exit_coord.fl:
            raise ValueError(
                "`value` should be set to the exit flight level for "
                f"{callsign}. It should be {exit_coord.fl} instead of {value}."
            )

    # current flight level
    ac_fl = gym_env.get_simulator_env().aircraft[callsign].fl

    if ac_fl <= exit_fl:
        # climb or same level
        kind = "change_flight_level_to"
        _value = int(exit_fl)
    else:
        # descend
        kind = "descend_now,level_by_fix"
        _value = (int(exit_fl), exit_coord.fix)

    return Action(callsign, kind, _value, agent=agent)
