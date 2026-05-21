from __future__ import annotations

import typing

from bluebird_dt.core import Action
from bluebird_gymnasium.actions import (
    DEFAULT_INTERVAL_HEADING,
    DEFAULT_RELATIVE_HEADING,
)

if typing.TYPE_CHECKING:
    from bluebird_dt.core import Aircraft
    from bluebird_dt.core import Airspace
    from bluebird_gymnasium.envs.base import BaseEnv
    from bluebird_gymnasium.utils.types import ACStateTracker, Number

DEGREES_MIN = 0
DEGREES_MAX = 360


def get_forward_segment_angle(
    aircraft: Aircraft,
    airspace: Airspace,
    aircraft_tracked_data: ACStateTracker,
    segment_idx: int,
    use_filed_route: bool,
    clip: bool = True,
) -> int | None:
    """Computes the angle of a track segment of an aircraft route.

    The segment is described as the line between two fixes in
    the aircraft's route. Also, forward segments refer to
    the aircraft's current segment in the route and other segments
    after the current.

    Args:
        aircraft: defines the aircraft for which to compute the forward
            segment angle.
        airspace: defines the airspace which contains the sector through
            which the aircraft is flying.
        aircraft_tracked_data: defines tracked information about the aircraft.
        segment_idx: the forward segment idx for which to compute the angle.
            The value should be >= 1.
            The idx implies the following,
            1 => for the current segment (thus to fly parallel to the current
                 segment the heading is set to the bearing from the previous
                 to next fix of the aircraft)
            2 => for the next segment (thus to fly parallel to this segment,
                 the heading is set to the bearing from the next fix to the
                 fix after the next).
            ...
        use_filed_route: defines whether to use the aircraft's filed or
            current route.
        clip: defines whether to clip the `segment_idx` to the last
            available forward segment if the originally passed `segment_idx`
            is greater than the number of available forward segment.
            Defaults to True.

    Returns:
        the angle computed based on the selected forward segment.

        Edge case returns:
        - if the `segment_idx` is greater than the number of available forward
          segment and `clip` is True, then the last segment is used to
          compute the returned angle.
        - if the `segment_idx` is greater than the number of available forward
          segment and `clip` is False, then `None` is returned.

        Note, this scenario could ocurr if an aircraft is approaching its last
        fix in the route, and an action action to fly parallel to segment after
        the current segment (i.e., segment_idx set to 2) is issued, this clip
        ensures that the current segment is used instead, as there is no next
        segment in such route.

    Raises:
        ValueError, if `segment_idx` is not an integer or if its set value is
            <= 1.
    """

    if not isinstance(segment_idx, int):
        raise ValueError("`segment_idx` should be an integer")

    if segment_idx < 1:
        raise ValueError(
            "`segment_idx should be set to value greater or equal to 1. "
            "See the docstring for more information."
        )

    if use_filed_route:
        route = aircraft.flight_plan.route.filed
        next_fix = aircraft_tracked_data.next_fix_fr
    else:
        route = aircraft.flight_plan.route.current
        next_fix = aircraft_tracked_data.next_fix_cr
    next_fix_idx = route.index(next_fix)

    places = airspace.fixes.places

    ####### get segment end position
    segment_end_fix_idx = next_fix_idx + (segment_idx - 1)

    if segment_end_fix_idx > (len(route) - 1) and not clip:
        return None

    # clip the segment end fix idx to the length of the route.
    segment_end_fix_idx = min(segment_end_fix_idx, len(route) - 1)
    segment_end_pos = places[route[segment_end_fix_idx]]

    ####### get the segment start position
    if segment_end_fix_idx == 0:
        # the segment end fix is the first in the route.
        # so use the position at the last route direct as
        # the start position of segment.
        segment_start_pos = aircraft_tracked_data.pos_at_last_route_direct

    else:
        # > 0
        segment_start_fix_idx = segment_end_fix_idx - 1
        segment_start_pos = places[route[segment_start_fix_idx]]

    ####### calculate the heading to fly
    return int(round(segment_start_pos.bearing_to(segment_end_pos), 0))


def heading_left(
    callsign: str,
    gym_env: BaseEnv,
    value: Number = DEFAULT_RELATIVE_HEADING,
    agent: str = "Agent",
) -> Action:
    """Generate a simulator action to turn an aircraft left.

    A relative heading action.

    Args:
        callsign: defines the identifier of the aircraft.
        gym_env: defines the gymnasium environment.
        value: defines the value to pass to the simulator action.
            Defaults to DEFAULT_RELATIVE_HEADING.
        agent: defines the name of agent. Defaults to 'Agent'.

    Return:
        the generated simulator action.
    """

    if value <= 0:
        raise ValueError("`value` should be a positive integer")
    elif value % DEFAULT_INTERVAL_HEADING != 0:
        raise ValueError(
            f"`value` should be in intervals of {DEFAULT_INTERVAL_HEADING}"
        )

    aircraft = gym_env.get_simulator_env().aircraft[callsign]
    if aircraft.selected_instructions.heading is not None:
        prev_selected_heading = aircraft.selected_instructions.heading
    else:
        prev_selected_heading = aircraft.heading

    # update the currently selected heading to a new value
    value = int(round(prev_selected_heading - value, 0))
    if value < DEGREES_MIN:
        value = DEGREES_MAX + value  # note, value is already a negative number
    return Action(callsign, "change_heading_to", value, agent=agent)


def heading_right(
    callsign: str,
    gym_env: BaseEnv,
    value: Number = DEFAULT_RELATIVE_HEADING,
    agent: str = "Agent",
) -> Action:
    """Generate a simulator action to turn an aircraft right.

    A relative heading action.

    Args:
        callsign: defines the identifier of the aircraft.
        gym_env: defines the gymnasium environment.
        value: defines the value to pass to the simulator action.
            Defaults to DEFAULT_RELATIVE_HEADING.
        agent: defines the name of agent. Defaults to 'Agent'.

    Return:
        the generated simulator action.
    """

    if value <= 0:
        raise ValueError("`value` should be a positive integer")
    elif value % DEFAULT_INTERVAL_HEADING != 0:
        raise ValueError(
            f"`value` should be in intervals of {DEFAULT_INTERVAL_HEADING}"
        )

    aircraft = gym_env.get_simulator_env().aircraft[callsign]
    if aircraft.selected_instructions.heading is not None:
        prev_selected_heading = aircraft.selected_instructions.heading
    else:
        prev_selected_heading = aircraft.heading

    # update the currently selected heading to a new value
    value = int(round(prev_selected_heading + value, 0))
    if value > DEGREES_MAX:
        value = value - DEGREES_MAX
    return Action(callsign, "change_heading_to", value, agent=agent)


def heading_route_parallel(
    callsign: str,
    gym_env: BaseEnv,
    value: Number = 1,
    agent: str = "Agent",
) -> Action:
    """Generate a simulator action to turn an aircraft parallel to route.

    Depending on the `value` passed, the computed angle could be based on
    the segment that the aircraft is currently in (i.e., bearing of its
    previous to next fix) or a future segment of its route (i.e., its next
    segment which is based on the bearing from its next fix to the fix
    after the next).

    An absolute heading action.

    Args:
        callsign: defines the identifier of the aircraft.
        gym_env: defines the gymnasium environment.
        value: defines the index of the forward segment in the aircraft's route,
            to use in computing the angle in which to fly parallel. For example,
            1 => for the current segment (thus to fly parallel to the current
                 segment the heading is set to the bearing from the previous
                 to next fix of the aircraft)
            2 => for the next segment (thus to fly parallel to this segment,
                 the heading is set to the bearing from the next fix to the
                 fix after the next).
            ...
            Defaults to 1, which uses the aircraft's current segment.
            The number of forward segments is capped based on the the
            maximum number of forward fixes in the environment configuration
        agent: defines the name of agent. Defaults to 'Agent'.

    Return:
        the generated simulator action.

    Note, if the `value` argument is greater than the number of available,
    forward segment, then the last segment is used. For example, this could
    occur if an aircraft is approaching its last fix in the route, and an
    action to fly parallel to segment after the current segment (i.e., value
    set to 2) is issued, this clip ensures that the current segment is used
    instead, as there is no next segment in such route.
    """

    forward_fixes_info = gym_env.get_forward_fixes_info()
    possible_options = list(range(1, forward_fixes_info.num_fixes + 1))

    if value not in possible_options:
        raise ValueError(
            f"`value` {value} should be set to one of the following values"
            f"{possible_options}."
        )

    simulator_env = gym_env.get_simulator_env()
    value = get_forward_segment_angle(
        simulator_env.aircraft[callsign],
        simulator_env.airspace,
        gym_env.get_tracked_aircraft_data(callsign),
        value,
        forward_fixes_info.use_filed_route,
    )
    value = int(round(value, 0))

    return Action(callsign, "change_heading_to", value, agent=agent)


def heading_maintain_current(
    callsign: str,
    gym_env: BaseEnv,
    value: Number | None = None,
    agent: str = "Agent",
) -> Action:
    """Generate a simulator action to maintain an aircraft's current heading.

    Args:
        callsign: defines the identifier of the aircraft.
        gym_env: defines the gymnasium environment.
        value: defines the value to pass to the simulator action.
        value: defines the value to pass to the simulator action.
            This is argument is ignored. Defaults to None.
        agent: defines the name of agent. Defaults to 'Agent'.

    Return:
        the generated simulator action.

    Note, the value is not used for this clearance. it's just set for
    consistency purpose with other simulator actions. the clearance is
    used to set the aircraft to stay on its current heading if it was
    previously following the route. if the aircraft was previously on a
    heading (off route), then the clearance does not change anything.
    """

    return Action(callsign, "maintain_current_heading", 0, agent=agent)
