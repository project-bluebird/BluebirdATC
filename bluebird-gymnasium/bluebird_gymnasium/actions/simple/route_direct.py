from __future__ import annotations

import typing

from bluebird_dt.core import Action

from bluebird_gymnasium.utils.simulator_utils import get_n_forward_fixes

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs.base import BaseEnv


def route_direct(
    callsign: str,
    gym_env: BaseEnv,
    value: int = 1,
    agent: str = "Agent",
) -> Action:
    """Generate a simulator action to route direct an aircraft.

    Instruct aircraft to follow the route in its flight plan.

    Route direct to its next closest forward fix or some fix after the next
    (skipping one or more fixes), and continue route following
    behaviour onwards.

    Note:
    - the list of fixes to which the aircraft can be route directed is
      truncated to start from the next forward fix and stop at the exit fix.
    - if the fix idx (`value`) is higher than the number of available fixes
      in the processed route list (next to exit fix), then the last fix in
      the list is chosen.
    - for route direct to any fix after the next forward fix, the operation
      reverts back to route direct to the next forward fix if any of the two
      conditions below holds true.
        - if the next fix is the last fix in the flight plan, or
        - if the fix after next is outside the sector. That is, the next
          fix is the exit fix.

    Args:
        callsign: defines the identifier of the aircraft.
        gym_env: defines the gymnasium environment.
        value: defines the value to pass to the simulator action.
            the fix to which to route direct. 1 for the next forward fix,
            2 for the fix after the next forward fix, and so on.
            Defaults to 1.
        agent: defines the name of agent. Defaults to 'Agent'.

    Return:
        the generated simulator action.

    Recommendation:
    `value` needs to be passed a positive integer. Even though it can be
    passed an arbitrarily large integer value, however, route direct actions
    in practice tend to be no more than the third forward fix. Therefore, it
    makes sense to pass a value no higher than 3 (i.e., so 1, 2 or 3).
    """

    forward_fixes_info = gym_env.get_forward_fixes_info()
    if value not in range(1, forward_fixes_info.num_fixes + 1):
        raise ValueError(
            f"`value` {value} should be set no higher than {forward_fixes_info.max_num_fixes} and no less than 1"
        )

    simulator_env = gym_env.get_simulator_env()
    if forward_fixes_info.use_filed_route:
        route = simulator_env.aircraft[callsign].flight_plan.route.filed
        next_fix = gym_env.get_tracked_aircraft_data(callsign).next_fix_fr
    else:
        route = simulator_env.aircraft[callsign].flight_plan.route.current
        next_fix = gym_env.get_tracked_aircraft_data(callsign).next_fix_cr

    if value == 1:
        # route direct to the next forward fix
        action = Action(callsign, "route_direct_to", next_fix, agent=agent)

    else:
        # route direct to the `value` forward fix.
        exit_coords = gym_env.get_tracked_aircraft_data(callsign).exit_coords
        sector = gym_env.get_active_airspace_sector()
        exit_fix = exit_coords[sector].fix

        # process the route to truncate it to *start from the next fix* and
        # *end at the exit fix*.
        if exit_fix in route:
            exit_fix_idx = route.index(exit_fix)
            route_trunc = route[: (exit_fix_idx + 1)]
        else:
            # exit fix is not in the route list. could happen if the current
            # route is being used and a route direct in a previous step was
            # issued to go to the fix just after the exit fix.
            # we should not get here as actions cannot be issued after sector
            # exit. however, add a failsafe, by starting it from the next fix.
            route_trunc = [
                exit_fix,
            ]

        # route_ff: route, forward fixes (ff)
        if next_fix in route_trunc:
            route_ff = get_n_forward_fixes(route_trunc, start_from=next_fix, n=value)
            route_ff = [fix for fix in route_ff if fix is not None]

        else:
            # likely that the next fix appears after the exit fix, which
            # indicates that the aircraft is about to exit or already
            # exited the sector.
            route_ff = route_trunc

        _max_idx = len(route_ff) - 1
        if (value - 1) > _max_idx:
            # the fix to skip to in the list is past the available fixes.
            # revert to the last fix in the list.
            _fix = route_ff[-1]

        else:
            # get the position of the fix after the next.
            forward_n_fix_idx = value - 1
            _fix = route_ff[forward_n_fix_idx]

        action = Action(callsign, "route_direct_to", _fix, agent=agent)

    return action
