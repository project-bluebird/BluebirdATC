from __future__ import annotations

import copy
import math
import typing

from bluebird_dt.core.pos2d import Pos2D
from bluebird_dt.predictor.route_follow_predictor import RouteFollowPredictor

from bluebird_gymnasium.utils.constants import (
    CUSTOM_FIX_AT_X,
    CUSTOM_FIX_BEFORE_X,
    CUSTOM_FIX_BETWEEN_X_AND_Y,
    CUSTOM_FIX_CURRENT_POS,
    CUSTOM_FIX_FUTURE_POS,
    CUSTOM_FIX_PREFIX,
)
from bluebird_gymnasium.utils.geo_utils import (
    angle_diff,
    angle_in_range,
    filter_route_fixes_in_sector_by_coordination,
    get_centreline_distance,
    get_route_segments,
    get_route_segments_in_sector,
    left_right_check,
    passed_location as _passed_location,
    path_intersection,
    path_overlap,
)
from bluebird_gymnasium.utils.simulator_utils import (
    aircraft_prev_next_fixes,
    distance_to_target_pos_along_route,
    get_aircraft_selected_heading,
    predict_trajectory,
)
from bluebird_gymnasium.utils.types import (
    InteractionCategory,
    InteractionDistance,
    InteractionInfo,
    InteractionRelevance,
    IntersectionInfo,
    MinAircraftSeparation,
    NamedLine,
    PositionStatus,
)

if typing.TYPE_CHECKING:
    from bluebird_dt.core.aircraft import Aircraft
    from bluebird_dt.core.airspace import Airspace
    from bluebird_dt.core.environment import Environment as SimulatorEnv
    from bluebird_dt.predictor import Predictor
    from bluebird_dt.core.pos4d import Pos4D
    from bluebird_gymnasium.envs.base import BaseEnv
    from bluebird_gymnasium.utils.types import ACStateTracker, Number

    try:
        from typing import Self
    except ImportError:
        from typing_extensions import Self


############ constants below
# used to determine if the relevance of a cross-track interaction when an
# aircraft pair haven't passed each other. if their time to the intersecting
# point is greater than the defined value, then it is irrelevant.
TIME_DIFF = 5  # in minutes


############ helper functions below
def f_traj(
    aircraft_tracked_data: ACStateTracker | None,
    aircraft: Aircraft,
    predictor: Predictor,
    duration: int,
    simulator_env: SimulatorEnv,
    trajectory_alias: str | None = None,
) -> list[Pos4D]:
    """get the future trajectory of a given aircraft"""

    # if aircraft_tracked_data is not None:
    #    return aircraft_tracked_data.future_trajectory
    # else:
    #    return predict_trajectory(
    #        aircraft,
    #        predictor,
    #        duration=duration,
    #        curr_time=simulator_env.time,
    #    )

    if trajectory_alias is None:
        if aircraft_tracked_data is not None:
            return aircraft_tracked_data.future_trajectory
        else:
            return predict_trajectory(
                aircraft,
                predictor,
                duration=duration,
                curr_time=simulator_env.time,
            )
    else:
        extra_tp = aircraft_tracked_data.extra_future_trajectory

        if trajectory_alias in extra_tp.keys():
            return extra_tp[trajectory_alias]
        else:
            trajectory = predict_trajectory(
                aircraft,
                predictor,
                duration=duration,
                curr_time=simulator_env.time,
            )
            # cache the trajectory in case it's need still within the current
            # traffic monitor update. note: before the end of `update(...)` in
            # traffic monitor (at the end of this file), the extra trajectory
            # buffer is cleared.
            aircraft_tracked_data.extra_future_trajectory[trajectory_alias] = (
                trajectory
            )

            return trajectory


def passed_location(
    aircraft: Aircraft,
    location: Pos2D,
    aircraft_tracked_data: ACStateTracker | None,
    simulator_env: SimulatorEnv,
    rollout_predictor: Predictor,
) -> bool:
    """Check whether a location has been passed."""

    ac_f_traj = f_traj(
        aircraft_tracked_data, aircraft, rollout_predictor, 300, simulator_env
    )

    return _passed_location(aircraft, location, ac_f_traj)


def route_distance_and_time_to_location(
    aircraft: Aircraft,
    stop_at: str | tuple[Pos2D, str],
    aircraft_tracked_data: ACStateTracker,
    simulator_env: SimulatorEnv,
    sector: str,
    use_filed_route: bool = True,
) -> tuple[Number, Number]:
    """Get an aircraft track/route distance and time to a location."""

    if use_filed_route:
        next_fix = aircraft_tracked_data.next_fix_fr
        route = aircraft.flight_plan.route.filed
    else:
        next_fix = aircraft_tracked_data.next_fix_cr
        route = aircraft.flight_plan.route.current

    start_from = (
        aircraft.pos2d(),
        next_fix,
    )

    # check: that start from fix appears before the stop at fix in the route.
    # if not, `get_route_segments_in_sector` will raise an exception.
    # therefore, catch this issue here and resolve it.
    start_from_pos, start_from_fix = start_from[0], start_from[1]
    if isinstance(stop_at, str):
        stop_at_pos = simulator_env.airspace.fixes.places[stop_at]
        stop_at_fix = stop_at
    else:
        stop_at_pos = stop_at[0]
        stop_at_fix = stop_at[1]

    # determine which route to use for the check: important when the current
    # route is being used. if that's the case and at least one fix is not
    # present in the route, we fall back to filed route - only for this check
    # note: this is only used for the check in the `if block`. the else block
    # reverts back to the chosen route (determined by `use_filed_route`).
    _flag = start_from_fix in route and stop_at_fix in route
    tmp_route = route if _flag else aircraft.flight_plan.route.filed

    if tmp_route.index(start_from_fix) >= tmp_route.index(stop_at_fix):
        aircraft_segments = [
            NamedLine((start_from_pos, stop_at_pos), (None, None)),
        ]
    else:
        aircraft_segments = get_route_segments_in_sector(
            route,
            simulator_env.airspace,
            sector,
            start_from=start_from,
            stop_at=stop_at,
        )

    ac_dist_ip = 0.0
    for idx in range(len(aircraft_segments)):
        start_pos, end_pos = aircraft_segments[idx].get_line()
        ac_dist_ip += start_pos.distance(end_pos)

    ac_time_ip = (ac_dist_ip / aircraft.speed_tas) * 60
    return ac_dist_ip, ac_time_ip


def flight_level_range_overlap(
    aircraft_1_fl_range: tuple[float, float],
    aircraft_2_fl_range: tuple[float, float],
) -> bool:
    """Check whether an overlap exists between two flight level ranges"""

    # get the min and max flight level of each range
    range_1_min, range_1_max = sorted(aircraft_1_fl_range)
    range_2_min, range_2_max = sorted(aircraft_2_fl_range)

    no_overlap = range_1_min >= (
        range_2_max + MinAircraftSeparation.VERTICAL
    ) or range_2_min >= (range_1_max + MinAircraftSeparation.VERTICAL)

    return not no_overlap


def get_previous_next_fixes_from_position(
    aircraft: Aircraft,
    custom_fix: tuple[Pos2D, str] | Pos2D,
    aircraft_tracked_data: ACStateTracker,
    simulator_env: SimulatorEnv,
    use_filed_route: bool,
    deepcopy_aircraft: bool = True,
) -> tuple[str, str]:
    """Get the previous and next standard fixes from a custom fix."""

    if isinstance(custom_fix, Pos2D):
        # get previous and next fix by using the only the position
        # relative to the aircraft route.

        if deepcopy_aircraft:
            dummy_aircraft = copy.deepcopy(aircraft)
        else:
            dummy_aircraft = aircraft
        dummy_aircraft.lat = custom_fix.lat
        dummy_aircraft.lon = custom_fix.lon
        previous_fix, next_fix = aircraft_prev_next_fixes(
            dummy_aircraft, simulator_env, use_filed_route
        )
        del dummy_aircraft

    else:
        # get previous and next fixes by parsing the custom fix name

        if use_filed_route:
            route = aircraft.flight_plan.route.filed
        else:
            route = aircraft.flight_plan.route.current

        location, location_name = custom_fix
        ret = location_name.split("///")

        if len(ret) == 1:  # at custom fix AT command
            location_command = "AT"
            delimiter = "__"
        else:
            location_command = ret[1]
            delimiter = "///"

        if location_command == "BEFORE":  # CUSTOM_FIX_BEFORE_X
            next_fix = location_name.split(delimiter)[-1]

            idx = route.index(next_fix)
            previous_fix = route[0] if idx == 0 else route[idx - 1]

        elif location_command == "AFTER":  # CUSTOM_FIX_AFTER_X
            previous_fix = location_name.split(delimiter)[-1]

            idx = route.index(previous_fix)
            next_fix = route[idx + 1] if idx < (len(route) - 1) else route[-1]

        elif location_command == "BETWEEN":  # CUSTOM_FIX_BETWEEN_X_AND_Y
            previous_fix = location_name.split(delimiter)[-2]
            next_fix = location_name.split(delimiter)[-1]

        elif location_command == "AT":  # CUSTOM_FIX_AT_X
            _name = location_name.split(delimiter)[-1]

            if _name == CUSTOM_FIX_CURRENT_POS:
                if use_filed_route:
                    previous_fix = aircraft_tracked_data.previous_fix_fr
                    next_fix = aircraft_tracked_data.next_fix_fr
                else:
                    previous_fix = aircraft_tracked_data.previous_fix_cr
                    next_fix = aircraft_tracked_data.next_fix_cr
            elif _name == CUSTOM_FIX_FUTURE_POS:
                if deepcopy_aircraft:
                    dummy_aircraft = copy.deepcopy(aircraft)
                else:
                    dummy_aircraft = aircraft

                dummy_aircraft.lat = location.lat
                dummy_aircraft.lon = location.lon
                previous_fix, next_fix = aircraft_prev_next_fixes(
                    dummy_aircraft, simulator_env, use_filed_route
                )
            else:
                raise ValueError(f"Unknown custom fix name: {location_name}")

        else:
            raise ValueError(f"Unknown custom fix name: {location_name}")

    return previous_fix, next_fix


def passed_fixes_intersections(
    callsign: str,
    interaction: InteractionInfo,
    intersect_fixes: list[IntersectionInfo],
    tracked_data: dict[str, ACStateTracker],
    simulator_env: SimulatorEnv,
    sector: str,
    rollout_predictor: Predictor,
    use_filed_route: bool,
    time_diff: Number = 5,
) -> tuple[bool, bool]:
    """Check whether aircraft pair (subject and other) has passed intersections

    The intersecting fixes is derived based on the aircraft's interaction
    with another aircraft.

    Args:
        callsign: defines the identifier of the aircraft for which its
            interactions with other aircraft is defined in the `interaction`
            list.
        interaction: captures information about the interaction with the other
            aircraft that the subject aircraft is in conflict with.
        intersect_fixes: defines a list of fixes or positions where
            the intersections occur.
        tracked_data: defines a data store that tracks information about all
            active aircraft in a sector.
        simulator_env: defines the underlying simulator.
        sector: defines the name of the sector that the aircraft is located.
        rollout_predictor: defines a trajectory predictor used to simulate a
            forward rollout to get future trajectories of aircraft.
        use_filed_route: defines whether to use the aircraft's filed or
            current route. Defaults to `True`.

    Returns:
        two-element tuple of bool (passed, relevant):
        - the first element is a flag that indicates whether the aircraft
          has passed the intersecting fixes.
        - the second element is a flag that indicates whether the
          interaction is relevant.

    Depending on the interaction type (same, cross, opposite track), either
    one or both aircraft would need to have passed the intersecting positions.
    If one or both aircraft have passed the intersections, then `passed` is
    True, and `relevant` is correspondingly set to False.

    Note for special condition when there's only *one* intersecting position:
    if the none of the aircraft have passed the position, then a further check
    is carried out to see if the subject aircraft or the other aircraft will
    arrive (and pass) the intersecting position *significantly* in time before
    the other arrives. If this is True, then the `passed` is set to False, but
    relevant is set to `False`.

    """

    # note: intersecting fixes are ordered based on the target/current
    # aircraft's route (i.e., not the other aircraft's route)

    aircraft = simulator_env.aircraft[callsign]
    other_callsign = interaction.other_callsign
    other_aircraft = simulator_env.aircraft[other_callsign]

    ac_tracked_data = tracked_data.get(callsign, None)
    other_ac_tracked_data = tracked_data.get(other_callsign, None)

    # initialization
    passed = False
    # when both aircraft have passed each other, it indicates that the
    # interactions is irrelevant by default. however, when both aircraft
    # haven't passed each other, the interaction is deemed relevant for
    # most use cases, except for a single case where the relevance is
    # determined independently, taking into account the time to intersecting
    # point (in cross track scenario).
    relevant = True

    args = [
        simulator_env,
        rollout_predictor,
    ]
    _cross_track_types = [
        InteractionCategory.CROSS_TRACK_LEFT,
        InteractionCategory.CROSS_TRACK_RIGHT,
    ]

    if interaction.track_category_ac_other == InteractionCategory.SAME_TRACK:
        location_name = intersect_fixes[-1].location_name
        location = intersect_fixes[-1].location

        s1 = passed_location(aircraft, location, ac_tracked_data, *args)
        s2 = passed_location(
            other_aircraft, location, other_ac_tracked_data, *args
        )
        passed = s1 or s2
        relevant = not passed

        if CUSTOM_FIX_PREFIX in location_name:
            # aircraft
            previous_fix, next_fix = get_previous_next_fixes_from_position(
                aircraft,
                (location, location_name),
                tracked_data[callsign],
                simulator_env,
                use_filed_route,
            )
            ac_stop_at: tuple[Pos2D, str] = (location, previous_fix)

            # other aircraft
            if use_filed_route:
                other_ac_route = other_aircraft.flight_plan.route.filed
            else:
                other_ac_route = other_aircraft.flight_plan.route.current

            if previous_fix in other_ac_route:
                other_ac_stop_at: tuple[Pos2D, str] = ac_stop_at
            elif next_fix in other_ac_route:
                idx = other_ac_route.index(next_fix)
                # previous fix for other aircraft
                _fix = other_ac_route[idx - 1] if idx > 0 else other_ac_route[0]
                other_ac_stop_at: tuple[Pos2D, str] = (location, _fix)
            else:
                # manually check
                other_previous_fix, _ = get_previous_next_fixes_from_position(
                    other_aircraft,
                    location,
                    tracked_data[other_callsign],
                    simulator_env,
                    use_filed_route,
                )
                other_ac_stop_at: tuple[Pos2D, str] = (
                    location,
                    other_previous_fix,
                )

        else:
            ac_stop_at: str = location_name  # a standard fix name
            other_ac_stop_at: str = location_name  # a standard fix name

        dist_ac_ip_1, _ = route_distance_and_time_to_location(
            aircraft,
            ac_stop_at,
            tracked_data[callsign],
            simulator_env,
            sector,
            use_filed_route,
        )
        dist_ac_ip_2, _ = route_distance_and_time_to_location(
            other_aircraft,
            other_ac_stop_at,
            tracked_data[other_callsign],
            simulator_env,
            sector,
            use_filed_route,
        )
        s1 = dist_ac_ip_1 == 0.0
        s2 = dist_ac_ip_2 == 0.0
        passed = s1 or s2
        relevant = not passed

    elif (
        interaction.track_category_ac_other
        == InteractionCategory.OPPOSITE_TRACK
    ):
        passed = False
        for _intersect_fix in intersect_fixes:
            location = _intersect_fix.location

            s1 = passed_location(aircraft, location, ac_tracked_data, *args)
            s2 = passed_location(
                other_aircraft, location, other_ac_tracked_data, *args
            )

            passed = s1 and s2
            if passed:
                # if there's an intersecting fix that both aircraft
                # have passed in opposite track interaction, then
                # the interaction is no longer relevant.
                break
        relevant = not passed

    elif (
        interaction.track_category_ac_other in _cross_track_types
        and len(intersect_fixes) > 1
    ):
        # cross track: start as cross track but later
        # transforms into same or opposite track.
        location = intersect_fixes[-1].location
        passed = passed_location(aircraft, location, ac_tracked_data, *args)
        relevant = not passed

    elif (
        interaction.track_category_ac_other in _cross_track_types
        and len(intersect_fixes) == 1
    ):
        # cross track with one intersecting fix.
        location_name = intersect_fixes[-1].location_name
        location = intersect_fixes[-1].location
        s1 = passed_location(aircraft, location, ac_tracked_data, *args)
        s2 = passed_location(
            other_aircraft, location, other_ac_tracked_data, *args
        )
        passed = s1 or s2

        if passed:
            relevant = not passed

        else:
            # final check: since one or both aircraft haven't passed the
            # intersecting location, check the time to reach it, take speed
            # into account. this determines the relevance, the one scenario
            # where relevance is determined independent of `passed` status.

            if CUSTOM_FIX_PREFIX in location_name:
                previous_fix, _ = get_previous_next_fixes_from_position(
                    aircraft,
                    (location, location_name),
                    tracked_data[callsign],
                    simulator_env,
                    use_filed_route,
                )
                ac_stop_at: tuple[Pos2D, str] = (location, previous_fix)

                previous_fix, _ = get_previous_next_fixes_from_position(
                    other_aircraft,
                    location,
                    tracked_data[other_callsign],
                    simulator_env,
                    use_filed_route,
                )
                other_ac_stop_at: tuple[Pos2D, str] = (location, previous_fix)
            else:
                ac_stop_at: str = location_name  # a standard fix name
                other_ac_stop_at: str = location_name  # a standard fix name

            ret = route_distance_and_time_to_location(
                aircraft,
                ac_stop_at,
                tracked_data[callsign],
                simulator_env,
                sector,
                use_filed_route=use_filed_route,
            )
            _, ac_time_ip = ret

            ret = route_distance_and_time_to_location(
                other_aircraft,
                other_ac_stop_at,
                tracked_data[other_callsign],
                simulator_env,
                sector,
                use_filed_route=use_filed_route,
            )
            _, other_ac_time_ip = ret

            relevant = abs(ac_time_ip - other_ac_time_ip) < time_diff

    else:
        passed = True
        relevant = False

    return passed, relevant


def get_discrete_track_category(
    aircraft_1: Aircraft,
    aircraft_2: Aircraft,
    aircraft_1_cps: list[Pos4D],
    aircraft_2_cps: list[Pos4D],
) -> InteractionCategory:
    """Compute the interaction type between two aircraft.

    Compute the interaction type between two aircraft discretized
    into a number of categories.

    `aircraft_1` serves as the reference point to `aircraft_2`.
    The computed interaction type depends `aircraft_2` position
    and heading relative to `aircraft_1` position and heading.

    Args:
        aircraft_1: the first aircraft
        aircraft_2: the second aircraft
        aircraft_1_cps: the predicted future trajectory of aircraft 1,
            defined using a list of control points.
        aircraft_2_cps: the predicted future trajectory of aircraft 2,
            defined using a list of control points.

    Returns:
        int (0 to 4), denoting one of four interaction type:
        cross track (left or right), opposite track, same track or
        no interaction.


    Examples
        SAME TRACK
                      .
                     . heading + 45 deg
                    .
                 AC1------>     AC2------>
                    .
                     . heading - 45 deg
                      .


        OPPOSITE TRACK
                      .
                     . (heading + 180) - 45 deg
                    .
                 AC1------>     <------AC2
                    .
                     .
                      . (heading + 180) + 45 deg



        CROSS TRACK LEFT
                                         AC2
                                          .
                                          .
                                          .
                                          v
                 heading+270-45     heading+270+45
                            .        .
                             .      .
                              .    .
                             AC1------>


        CROSS TRACK RIGHT

                         AC1------>
                          .    .
                         .      .
                        .        . heading+90+45
                 heading+90-45
                                      ^
                                      .
                                      .
                                      .
                                     AC2
    """

    FOV = 45.0  # field of view in degrees

    # get the field of view range for each interaction type
    track_range = {}

    track_range[InteractionCategory.SAME_TRACK] = (
        round((aircraft_1.heading - FOV) % 360, 4),
        round((aircraft_1.heading + FOV) % 360, 4),
    )

    ac1_facing_cross_track = aircraft_1.heading + 90.0
    ac1_from_cross_track_dir = ac1_facing_cross_track + 180.0
    track_range[InteractionCategory.CROSS_TRACK_RIGHT] = (
        round((ac1_from_cross_track_dir - FOV) % 360, 4),
        round((ac1_from_cross_track_dir + FOV) % 360, 4),
    )

    track_range[InteractionCategory.OPPOSITE_TRACK] = (
        round(((aircraft_1.heading + 180.0) - FOV) % 360, 4),
        round(((aircraft_1.heading + 180.0) + FOV) % 360, 4),
    )

    ac1_facing_cross_track = aircraft_1.heading + 270.0
    ac1_from_cross_track_dir = ac1_facing_cross_track + 180.0
    track_range[InteractionCategory.CROSS_TRACK_LEFT] = (
        round((ac1_from_cross_track_dir - FOV) % 360, 4),
        round((ac1_from_cross_track_dir + FOV) % 360, 4),
    )

    # determine interaction type based on distance between
    # aircraft control points and FOV range
    interaction_type = None
    if aircraft_1_cps[1].distance(aircraft_2_cps[1]) < aircraft_1_cps[
        0
    ].distance(aircraft_2_cps[0]):
        # both aircraft are drawing closer to each other. it could
        # be any of the interaction types. so, check for all types.
        ac2_heading = aircraft_2.heading

        if angle_in_range(
            ac2_heading, track_range[InteractionCategory.SAME_TRACK]
        ):
            interaction_type = InteractionCategory.SAME_TRACK

        elif angle_in_range(
            ac2_heading, track_range[InteractionCategory.CROSS_TRACK_LEFT]
        ):
            interaction_type = InteractionCategory.CROSS_TRACK_LEFT

        elif angle_in_range(
            ac2_heading, track_range[InteractionCategory.CROSS_TRACK_RIGHT]
        ):
            interaction_type = InteractionCategory.CROSS_TRACK_RIGHT

        elif angle_in_range(
            ac2_heading, track_range[InteractionCategory.OPPOSITE_TRACK]
        ):
            interaction_type = InteractionCategory.OPPOSITE_TRACK

        else:
            raise ValueError("We should never get here.")
    else:
        # even though the distance between aircraft in their current position
        # is greater or equal to the distance in their future position, it
        # could still be a same track interaction. therefore, check for that.
        ac2_heading = aircraft_2.heading

        if angle_in_range(
            ac2_heading, track_range[InteractionCategory.SAME_TRACK]
        ):
            interaction_type = InteractionCategory.SAME_TRACK
        else:
            interaction_type = InteractionCategory.NONE

    return interaction_type


def get_continuous_track_info(
    aircraft_1: Aircraft,
    aircraft_2: Aircraft,
    aircraft_1_cps: list[Pos4D],
    aircraft_2_cps: list[Pos4D],
) -> tuple[float, int, float, int, InteractionDistance]:
    """Compute the interaction type between two aircraft.

    Compute the interaction type between two aircraft in a
    continuous form using relative angle between the heading
    of both aircraft.

    `aircraft_1` serves as the reference point to `aircraft_2`.
    The computed interaction type depends on `aircraft_2` position
    and heading relative to `aircraft_1` position and heading.

    Args:
        aircraft_1: the first aircraft
        aircraft_2: the second aircraft
        aircraft_1_cps: the predicted future trajectory of aircraft 1,
            defined using a list of control points.
        aircraft_2_cps: the predicted future trajectory of aircraft 2,
            defined using a list of control points.

    Returns:
        tuple of three elements:
        - the relative angle between the heading of both aircraft,
        - the direction to the turn aircraft 1 to point to aircraft 2.
        - ternary value that indicates whether both aircraft are drawing
          closer to each other, maintaining same distance or growing apart.
    """

    angle_diff_ac1_ac2 = angle_diff(aircraft_1.heading, aircraft_2.heading)
    # turn_dir_ac1_ac2 = left_right_check(aircraft_1.heading, aircraft_2.heading)
    turn_dir_ac1_ac2 = left_right_check(
        aircraft_1.heading, aircraft_1.pos2d().bearing_to(aircraft_2.pos2d())
    )

    aircraft_1_sh = get_aircraft_selected_heading(aircraft_1)
    aircraft_2_sh = get_aircraft_selected_heading(aircraft_2)
    angle_diff_ac1_ac2_sh = angle_diff(aircraft_1_sh, aircraft_2_sh)
    # turn_dir_ac1_ac2_sh = left_right_check(
    #    aircraft_1_sh, aircraft_2_sh
    # )
    turn_dir_ac1_ac2_sh = left_right_check(
        aircraft_1_sh, aircraft_1.pos2d().bearing_to(aircraft_2.pos2d())
    )

    dist_future_pos = aircraft_1_cps[1].distance(aircraft_2_cps[1])
    dist_current_pos = aircraft_1_cps[0].distance(aircraft_2_cps[0])

    eps = 0.05
    _dist = dist_future_pos - dist_current_pos
    # threshold distance difference by a small epsilon as a very small
    # numerical difference may result (likely due to floating point
    # overflow) from the operation. this is often experienced in same
    # track scenarios where both aircraft flying at the same speed.
    _dist = 0.0 if abs(_dist) <= eps else _dist

    if _dist < 0.0:
        # both aircraft are drawing closer to each other.
        int_dist = InteractionDistance.CLOSER

        # NOTE: previously, it was assumed only cross and opposite track
        # scenarios had the possibility to have the distance between
        # aircraft reducing over time (collision) and increase after
        # they cross. However, decreasing distance can occur in some same
        # track scenarios (e.g., scenario in which an aircraft behind is
        # faster and is catching up to the aircraft in front). therefore, we
        # can either (i) disable this extra processing (as it would cause
        # both aircraft to face opposite direction in "same track catch
        # up" scenario, while it works for the natural opposite and cross
        # track scenarios), (ii) apply this extra processing to all
        # scenarios. currently favour the former.

        # further process the relative angle such that the aircraft 1
        # points to aircraft 2
        # NOTE disabled, based on the above comment.
        # angle_diff_ac1_ac2 = angle_diff(angle_diff_ac1_ac2, 180.0)
        # turn_dir_ac1_ac2 *= -1.0
    elif _dist == 0.0:
        # both aircraft are either maintaining same distance
        int_dist = InteractionDistance.MAINTAIN
    else:
        # both aircraft are growing farther apart from each other.
        int_dist = InteractionDistance.FARTHER

    return (
        angle_diff_ac1_ac2,
        turn_dir_ac1_ac2,
        angle_diff_ac1_ac2_sh,
        turn_dir_ac1_ac2_sh,
        int_dist,
    )


def get_centreline_distance_diff(
    centre_dist_1: Number, centre_dist_2: Number, category: InteractionCategory
) -> Number:
    """Get the lateral distance between both aircraft.

    Use their route's centreline distance as a proxy and interaction catgory.

    Note, the distance is maximised by placing aircraft on opposite
    sides of the intersecting position along their route.

    Args:
        centre_dist_1: defines the lateral centreline distance from the first
            aircraft's route. the sign of the distance indicates the position
            of the aircraft (either left or right of the centreline).
        centre_dist_2: defines the lateral centreline distance from the second
            aircraft's route. the sign of the distance indicates the position
            of the aircraft (either left or right of the centreline).
        category: defines the interaction category of both aircraft, given by
            one of the following:
            same track, cross track, or opposite track.

    Returns:
        the lateral distance between both aircraft using their centreline distance
        as a proxy.
    """

    pair_distance = 0.0

    if category == InteractionCategory.SAME_TRACK:
        # maximised distance one aircraft is on the left of the
        # route's centreline and the other is on the right.
        pair_distance = abs(centre_dist_1 - centre_dist_2)

    elif category == InteractionCategory.OPPOSITE_TRACK:
        # maximised distance when both aircraft are either on the
        # left side of their route's centreline or on the right.
        pair_distance = abs(centre_dist_1 + centre_dist_2)

    elif (
        category == InteractionCategory.CROSS_TRACK_LEFT
        or category == InteractionCategory.CROSS_TRACK_RIGHT
    ):
        # maximised distance when both aircraft are either on the
        # left side of their route's centreline or on the right.
        pair_distance = abs(centre_dist_1 + centre_dist_2)

    else:
        # default case, no interaction
        # so set the distance to an arbitrarily large number
        pair_distance = 1e3

    return float(pair_distance)


############ core functions below


def lateral_interaction_dist_thresh_heuristic(
    aircraft_1: Aircraft, aircraft_2: Aircraft
) -> Number:
    """Compute the interaction threshold distance b/w aircraft pair.

    Used in safety and conflict resolution to determine the minimum distance
    at which the safety or conflict situation should be resolved.

    Defined using the heuristic function below.
    10 + (20 * (angle between aircraft pair) / 180)

    Args:
        aircraft_1: the first aircraft
        aircraft_2: the second aircraft

    Returns:
        the compute threshold value based on the heuristic formula.
    """

    _diff = angle_diff(aircraft_1.heading, aircraft_2.heading)
    return 10 + (20 * _diff / 180.0)


def lateral_interaction_dist_thresh_by_type(
    aircraft_1: Aircraft,
    aircraft_2: Aircraft,
    interaction_category: InteractionCategory,
    interaction_distance_type: InteractionDistance,
) -> Number:
    """Compute the interaction threshold distance b/w aircraft pair.

    Used in safety and conflict resolution to determine the minimum distance
    at which the safety or conflict situation should be resolved.

    Note:
    If both aircraft are off route (on headings), then the distance can
    be as low as 5 nautical miles irrespective of the interaction class.
    However, a conservative value of 7 nautical miles will be returned
    by this function.

    Otherwise, if one or both aircraft are route following, then one of
    the following below is returned based on interaction class.
    Threshold distance is derived using the interaction category/class:
    Same track: 10 nautical miles
    Cross track: 20 nautical miles
    Opposite track: 30 nautical miles
    No interaction: 0 nautical miles


    Args:
        aircraft_1: the first aircraft
        aircraft_2: the second aircraft
        interaction_category: the interaction category of both aircraft
        interaction_distance_type: the distance type of the interaction:
            reduced distance, maintained distance or growing distance.

    Returns:
        the compute threshold value based on the interaction category.
    """

    if not aircraft_1.on_route and not aircraft_2.on_route:
        distance = 7

    else:
        if interaction_category == InteractionCategory.SAME_TRACK:
            if interaction_distance_type == InteractionDistance.CLOSER:
                speed_diff = abs(aircraft_1.speed_tas - aircraft_2.speed_tas)
                if speed_diff <= 40:
                    distance = 20
                else:  # > 40
                    distance = 30
            else:
                distance = 10

        elif interaction_category == InteractionCategory.CROSS_TRACK_LEFT:
            distance = 20

        elif interaction_category == InteractionCategory.CROSS_TRACK_RIGHT:
            distance = 20

        elif interaction_category == InteractionCategory.OPPOSITE_TRACK:
            distance = 30

        else:  # interaction_category == InteractionCategory.NONE
            # threshold is not need if there's no lateral interaction.
            distance = 0

    return distance


def get_aircraft_interaction_info(
    callsign: str,
    tracked_data: dict[str, ACStateTracker],
    simulator_env: SimulatorEnv,
    rollout_predictor: Predictor,
    sort_by_distance: bool = True,
) -> list[InteractionInfo]:
    """Computes interaction data for an aircraft relative to other aircraft.

    Computes interaction data for an aircraft relative to all other aircraft
    that are actively tracked in a sector.

     The interaction relationship on the track is classed as one of the
     following: same track, cross track or opposite track. See the docstring
     in `get_discrete_track_category(...)` for the track descriptions.

    Args:
        callsign: defines the callsign of the aircraft for which to compute
            its interaction with other aircraft in a sector.
        tracked_data: defines a data store that tracks information about all
            active aircraft in a sector.
        simulator_env: defines the underlying simulator.
        rollout_predictor: defines a trajectory predictor used to simulate a
            forward rollout to get future trajectories of aircraft.
        sort_by_distance: defines whether or not to sort the interaction
            list (to be returned) based on distance between aircraft and the
            other aircraft. Sorted in ascending order.
            Optional, defaults to `True`.

    Returns:
        list of interactions information.
        each interaction contains information about the aircraft interaction
        with a specific other aircraft. the information between the
        aircraft (being analyzed) and another (other) aircraft. the
        information captured are:
        - other aircraft callsign,
        - distance between the aircraft and the other aircraft,
        - bearing to other aircraft from the aircraft,
        - heading (angle) difference between aircraft and other aircraft,
        - turn direction for aircraft to face other aircraft
        - interaction distance type between aircraft and other aircraft,
        - interaction type (same, cross or opposite track)
        - current flight level difference between aircraft
        - selected flight level difference between aircraft
        - speed (true airspeed) difference between aircraft
        - ...

        note: when sort_by_distance is set to `True`, the list is sorted
        based on the distance between the aircraft and the interacting
        aircraft.
    """

    # rollout duration if future trajectory is not present
    # in tracked data. 300 seconds.
    duration = 300

    ac = simulator_env.aircraft[callsign]
    ac_f_traj = f_traj(
        tracked_data.get(callsign, None),
        ac,
        rollout_predictor,
        duration,
        simulator_env,
    )

    # filed route (fr)
    _centre_info = tracked_data[callsign].centreline_info_fr
    ac_centre_dist_fr = _centre_info[0] * _centre_info[1]
    # current route (cr)
    _centre_info = tracked_data[callsign].centreline_info_cr
    ac_centre_dist_cr = _centre_info[0] * _centre_info[1]

    callsign_others = []
    dist_ac_others = []
    bearing_ac_others = []
    angle_diff_ac_others = []
    turn_dir_ac_others = []
    angle_diff_ac_others_sh = []
    turn_dir_ac_others_sh = []
    dist_type_ac_others = []
    track_category_ac_others = []
    fl_diff_ac_others = []
    selected_fl_diff_ac_others = []
    speed_diff_ac_others = []
    centreline_dist_diff_ac_others_fr = []
    centreline_dist_diff_ac_others_cr = []
    lateral_dist_thresh_ac_others = []

    # NOTE: should we consider all aircraft in the simulator_env instead (i.e.,
    # `simulator_env.aircraft`)? rather than only active aircraft in the
    # tracker.
    _callsigns = list(tracked_data.keys())
    _callsigns.remove(callsign)
    for other_callsign in _callsigns:
        # filed route (fr)
        _centre_info = tracked_data[other_callsign].centreline_info_fr
        other_ac_centre_dist_fr = _centre_info[0] * _centre_info[1]
        # current route (cr)
        _centre_info = tracked_data[other_callsign].centreline_info_cr
        other_ac_centre_dist_cr = _centre_info[0] * _centre_info[1]

        # save other aircraft callsign
        callsign_others.append(other_callsign)

        # get other aircraft
        other_ac = simulator_env.aircraft[other_callsign]
        other_ac_f_traj = f_traj(
            tracked_data.get(other_callsign, None),
            other_ac,
            rollout_predictor,
            duration,
            simulator_env,
        )

        # distance aircraft to other aircraft
        ac_dist = ac.pos2d().distance(other_ac.pos2d())
        dist_ac_others.append(ac_dist)

        # bearing from aircraft to other aircraft
        bearing_ac_other = ac.pos2d().bearing_to(other_ac.pos2d())
        bearing_ac_others.append(bearing_ac_other)

        # relative angular diff between aircraft heading and
        # other aircraft heading, turn direction and distance type
        ret = get_continuous_track_info(
            ac, other_ac, ac_f_traj, other_ac_f_traj
        )
        angle_diff_ac_others.append(ret[0])
        turn_dir_ac_others.append(ret[1])
        angle_diff_ac_others_sh.append(ret[2])
        turn_dir_ac_others_sh.append(ret[3])
        dist_type_ac_others.append(ret[4])
        dist_type_ac_other = ret[4]

        # discrete track type between aircraft and other aircraft
        # i.e., cross track or opposite track or same track or undefined
        track_category = get_discrete_track_category(
            ac, other_ac, ac_f_traj, other_ac_f_traj
        )
        track_category_ac_others.append(track_category)

        # flight level difference.
        ## current flight level difference
        fl_diff_ac_others.append(float(ac.fl - other_ac.fl))
        ## selected flight level difference
        selected_fl_diff_ac_others.append(
            float(ac.selected_fl - other_ac.selected_fl)
        )

        # speed difference
        speed_diff_ac_others.append(ac.speed_tas - other_ac.speed_tas)

        # centreline distance difference
        centreline_dist_diff_ac_others_fr.append(
            get_centreline_distance_diff(
                ac_centre_dist_fr, other_ac_centre_dist_fr, track_category
            )
        )
        centreline_dist_diff_ac_others_cr.append(
            get_centreline_distance_diff(
                ac_centre_dist_cr, other_ac_centre_dist_cr, track_category
            )
        )

        # lateral distance threshold for any potential safety violation
        # (if any) to have been resolved. calculated based on interaction
        # type and the route following status of both aircraft
        lateral_dist_thresh_ac_others.append(
            lateral_interaction_dist_thresh_by_type(
                ac,
                other_ac,
                track_category,
                dist_type_ac_other,
            )
        )

    # end of for loop

    # sort other aircraft based on distance from aircraft
    if sort_by_distance:
        indices = sorted(
            list(range(len(dist_ac_others))), key=lambda k: dist_ac_others[k]
        )
    else:
        indices = list(range(len(dist_ac_others)))

    interactions = []
    for idx in indices:
        interactions.append(
            InteractionInfo(
                callsign_others[idx],
                dist_ac_others[idx],
                bearing_ac_others[idx],
                angle_diff_ac_others[idx],
                turn_dir_ac_others[idx],
                angle_diff_ac_others_sh[idx],
                turn_dir_ac_others_sh[idx],
                dist_type_ac_others[idx],
                track_category_ac_others[idx],
                fl_diff_ac_others[idx],
                selected_fl_diff_ac_others[idx],
                speed_diff_ac_others[idx],
                centreline_dist_diff_ac_others_fr[idx],
                centreline_dist_diff_ac_others_cr[idx],
                lateral_dist_thresh_ac_others[idx],
                InteractionRelevance.UNDEFINED,
            )
        )
    return interactions


def filter_interactions_by_filed_route(
    interactions: list[InteractionInfo],
    callsign: str,
    tracked_data: dict[str, ACStateTracker],
    simulator_env: SimulatorEnv,
    sector: str,
    in_sector_fixes_only: bool = False,
    from_next_fix: bool = True,
) -> tuple[list[InteractionInfo], list[int], list[list[IntersectionInfo]]]:
    """
    Filter an aircraft's list of interactions with other aircraft by route.

    Filter conditions, keep interaction (with other aircraft) if:
    i.  the filed route of both aircraft share at least one fix in common.

    Args:
        interactions: captures information about interactions that the
            aircraft have with other aircraft. each item in the list is a
            tuple that contains the interaction information with another
            aircraft. note, the list only captures other aircraft for which
            the aircraft has an interaction (if no interaction exist with
            another aircraft, such aircraft is excluded from the list).
        callsign: defines the identifier of the aircraft for which its
            interactions with other aircraft is defined in the `interaction`
            list.
        tracked_data: defines a data store that tracks information about all
            active aircraft in a sector.
        simulator_env: defines the underlying simulator.
        sector: defines the name of the sector that the aircraft is located.
        in_sector_fixes_only: defines whether the intersecting fixes should
            only be considered when they are within the sector. Defaults
            to `False`.
        from_next_fix: specifies whether to use the entire filed route of
            both aircraft (i.e., if False) or a subset of the filed route
            starting from the next fix of each aircraft (i.e., if True).
            Defaults to True.

    Returns:
        tuple of three elements.
        - the first element is a list that contains a subset of
          `interactions` that satisfies the defined filter conditions.
        - the second element is the index/position of the filtered
          interactions in the original list.
        - the third element is a list (an item for each `other_aircraft` in
          the filtered list), which contains a list of intersecting fixes for
          each corresponding `other aircraft` in the filtered list.
          an intersection is captured as a tuple (fix_name, fix_position).
    """

    places = simulator_env.airspace.fixes.places
    aircraft = simulator_env.aircraft[callsign]

    entry_fix = tracked_data[callsign].entry_coords[sector].fix
    exit_fix = tracked_data[callsign].exit_coords[sector].fix
    aircraft_route_original = aircraft.flight_plan.route.filed

    if from_next_fix:
        _fix = tracked_data[callsign].next_fix_fr
        _fix_idx = aircraft_route_original.index(_fix)
        aircraft_route = set(aircraft_route_original[_fix_idx:])
        del _fix, _fix_idx
    else:
        aircraft_route = set(aircraft_route_original)

    filtered_interactions = []
    indexes = []
    intersections = []

    for idx, other_ac_interaction in enumerate(interactions):
        other_callsign = other_ac_interaction.other_callsign
        other_aircraft = simulator_env.aircraft[other_callsign]
        _route = other_aircraft.flight_plan.route.filed

        if from_next_fix:
            _fix = tracked_data[other_callsign].next_fix_fr
            _fix_idx = _route.index(_fix)
            other_aircraft_route = set(_route[_fix_idx:])
        else:
            other_aircraft_route = set(_route)

        common_fixes = list(aircraft_route.intersection(other_aircraft_route))
        if in_sector_fixes_only:
            common_fixes = filter_route_fixes_in_sector_by_coordination(
                common_fixes, aircraft_route_original, entry_fix, exit_fix
            )

        if len(common_fixes) > 0:
            filtered_interactions.append(other_ac_interaction)
            indexes.append(idx)

            # order the common_fixes based on the aircraft's filed route
            ordered_intersection = [
                _fix for _fix in aircraft_route_original if _fix in common_fixes
            ]
            ordered_intersection = [
                IntersectionInfo(True, places[fix], fix)
                for fix in ordered_intersection
            ]
            intersections.append(ordered_intersection)

    return filtered_interactions, indexes, intersections


def filter_interactions_by_current_route(
    interactions: list[InteractionInfo],
    callsign: str,
    tracked_data: dict[str, ACStateTracker],
    simulator_env: SimulatorEnv,
    sector: str,
    in_sector_fixes_only: bool = False,
    from_next_fix: bool = True,
) -> tuple[list[InteractionInfo], list[int], list[list[IntersectionInfo]]]:
    """
    Filter an aircraft's interactions with other aircraft by current route.

    Filter conditions, keep interaction (with other aircraft) if one of the
    following is True:
    i.  the filed route of both aircraft share at least one fix in common. OR

    ii.  the there's at least one segment in the aircraft current route that
        intersect with a segment in the current route of the other aircraft.

    Args:
        interactions: captures information about interactions that the
            aircraft have with other aircraft. each item in the list is a
            tuple that contains the interaction information with another
            aircraft. note, the list only captures other aircraft for which
            the aircraft has an interaction (if no interaction exist with
            another aircraft, such aircraft is excluded from the list).
        callsign: defines the identifier of the aircraft for which its
            interactions with other aircraft is defined in the `interaction`
            list.
        tracked_data: defines a data store that tracks information about all
            active aircraft in a sector.
        simulator_env: defines the underlying simulator.
        sector: defines the name of the sector that the aircraft is located.
        in_sector_fixes_only: defines whether the intersecting fixes should
            only be considered when they are within the sector. Defaults
            to `False`.
        from_next_fix: specifies whether to use the entire current route of
            both aircraft (i.e., if False) or a subset of the current route
            starting from the next fix of each aircraft (i.e., if True).
            Defaults to True.

    Returns:
        tuple of three elements.
        - the first element is a list that contains a subset of
          `interactions` that satisfies the defined filter conditions.
        - the second element is the index/position of the filtered
          interactions in the original list.
        - the third element is a list (an item for each `other_aircraft` in
          the filtered list), which contains a list of intersecting fixes for
          each corresponding `other aircraft` in the filtered list.
          an intersection is captured as a tuple (fix_name, fix_position).
          note, that if the filter condition (i) is not met, but the filter
          condition (ii) is met, then intersecting position is not a standard
          fix position in the sector. Hence, the name, defined by the custom
          name template `CUSTOM_FIX_BEFORE_X` is used to create a customised
          name for the position (serving as a placeholder).
    """

    airspace = simulator_env.airspace
    places = airspace.fixes.places

    aircraft = simulator_env.aircraft[callsign]
    ac_route = aircraft.flight_plan.route.current
    ac_next_fix = tracked_data[callsign].next_fix_cr
    ac_next_fix_idx = ac_route.index(ac_next_fix) if from_next_fix else 0
    entry_fix = tracked_data[callsign].entry_coords[sector].fix
    exit_fix = tracked_data[callsign].exit_coords[sector].fix

    ac_segments = get_segments_from_route(
        aircraft,
        airspace,
        sector,
        tracked_data[callsign],
        use_filed_route=False,  # current route
        in_sector_fixes_only=in_sector_fixes_only,
        from_next_fix=from_next_fix,
    )

    filtered_interactions = []
    indexes = []
    intersections = []
    for idx, other_ac_interaction in enumerate(interactions):
        other_callsign = other_ac_interaction.other_callsign
        other_aircraft = simulator_env.aircraft[other_callsign]
        other_ac_route = other_aircraft.flight_plan.route.current
        other_ac_next_fix = tracked_data[other_callsign].next_fix_cr
        other_ac_next_fix_idx = (
            other_ac_route.index(other_ac_next_fix) if from_next_fix else 0
        )

        # initialisation
        relevant = False

        route_1 = set(ac_route[ac_next_fix_idx:])
        route_2 = set(other_ac_route[other_ac_next_fix_idx:])
        common_fixes = list(route_1.intersection(route_2))
        if in_sector_fixes_only:
            common_fixes = filter_route_fixes_in_sector_by_coordination(
                common_fixes,
                aircraft.flight_plan.route.filed,
                entry_fix,
                exit_fix,
            )

        if len(common_fixes) > 0:
            relevant = True

            # order the common_fixes based on the aircraft's route
            ordered_intersection = [
                _fix for _fix in ac_route if _fix in common_fixes
            ]
            ordered_intersection = [
                IntersectionInfo(True, places[fix], fix)
                for fix in ordered_intersection
            ]

        elif ac_next_fix_idx == 0 or other_ac_next_fix_idx == 0:
            ordered_intersection = None

            # more (but expensive) checks:
            # useful when one or both aircraft are in the start of their
            # current route (i.e., not yet arrived at the [first] fix being
            # route directed to.
            other_ac_segments = get_segments_from_route(
                other_aircraft,
                airspace,
                sector,
                tracked_data[other_callsign],
                use_filed_route=False,  # current route
                in_sector_fixes_only=in_sector_fixes_only,
                from_next_fix=from_next_fix,
            )
            intersect_exist, location, found_indices = path_intersection(
                ac_segments, other_ac_segments
            )
            if intersect_exist:
                relevant = True
                ac_segment_idx, _ = found_indices

                # aircraft segment that intersect with other aircraft
                found_segment: NamedLine = ac_segments[ac_segment_idx]
                _start_fix, _end_fix = found_segment.get_name()
                _tmp_fix_name = CUSTOM_FIX_BEFORE_X.format(_end_fix)

                ordered_intersection = [
                    IntersectionInfo(intersect_exist, location, _tmp_fix_name)
                ]

            else:
                # final drastic check.
                ret = filter_interactions_by_filed_route(
                    [
                        other_ac_interaction,
                    ],
                    callsign,
                    tracked_data,
                    simulator_env,
                    sector,
                    in_sector_fixes_only=True,
                    from_next_fix=True,
                )

                tmp_interactions, _, tmp_intersect_fixes = ret
                if len(tmp_interactions) == 1:
                    relevant = True
                    ordered_intersection = tmp_intersect_fixes[0]
                else:
                    pass

        if relevant:
            filtered_interactions.append(other_ac_interaction)
            intersections.append(ordered_intersection)
            indexes.append(idx)

    return filtered_interactions, indexes, intersections


def filter_interactions_by_route(
    interactions: list[InteractionInfo],
    callsign: str,
    tracked_data: dict[str, ACStateTracker],
    simulator_env: SimulatorEnv,
    sector: str,
    rollout_predictor: Predictor,
    in_sector_fixes_only: bool = True,
    from_next_fix: bool = True,
    use_filed_route: bool = True,
) -> tuple[list[InteractionInfo], list[int], list[list[IntersectionInfo]]]:
    """
    Filter an aircraft's interactions with other aircraft by route.

    This is a wrapper function for `filter_interactions_by_filed_route` and
    `filter_interactions_by_current_route`. Please see the docstrings for
    both functions to get more information.

    Args:
        interactions: captures information about interactions that the
            aircraft have with other aircraft. each item in the list is a
            tuple that contains the interaction information with another
            aircraft. note, the list only captures other aircraft for which
            the aircraft has an interaction (if no interaction exist with
            another aircraft, such aircraft is excluded from the list).
        callsign: defines the identifier of the aircraft for which its
            interactions with other aircraft is defined in the `interaction`
            list.
        tracked_data: defines a data store that tracks information about all
            active aircraft in a sector.
        simulator_env: defines the underlying simulator.
        sector: defines the name of the sector that the aircraft is located.
        rollout_predictor: defines a trajectory predictor used to simulate a
            forward rollout to get future trajectories of aircraft.
        in_sector_fixes_only: defines whether the intersecting fixes should
            only be considered when they are within the sector. Defaults
            to `True`.
        from_next_fix: specifies whether to use the entire current route of
            both aircraft (i.e., if False) or a subset of the current route
            starting from the next fix of each aircraft (i.e., if True).
            Defaults to True.
        use_filed_route: defines whether to use the aircraft's filed or
            current route. Defaults to `True`.

    Returns:
        Please see the docstrings of the wrapped functions.
    """
    if use_filed_route:
        results = filter_interactions_by_filed_route(
            interactions,
            callsign,
            tracked_data,
            simulator_env,
            sector,
            in_sector_fixes_only,
            from_next_fix,
        )
    else:
        results = filter_interactions_by_current_route(
            interactions,
            callsign,
            tracked_data,
            simulator_env,
            sector,
            in_sector_fixes_only,
            from_next_fix,
        )

        filtered_interactions = []
        indexes = []
        intersections = []

        _cross_track_types = [
            InteractionCategory.CROSS_TRACK_LEFT,
            InteractionCategory.CROSS_TRACK_RIGHT,
        ]
        for interaction, interaction_idx, intersect_fixes in zip(*results):
            if (
                from_next_fix
                and len(intersect_fixes) == 1
                and interaction.track_category_ac_other in _cross_track_types
            ):
                passed, relevant = passed_fixes_intersections(
                    callsign,
                    interaction,
                    intersect_fixes,
                    tracked_data,
                    simulator_env,
                    sector,
                    rollout_predictor,
                    use_filed_route,
                    time_diff=TIME_DIFF,
                )
                if not passed and relevant:
                    filtered_interactions.append(interaction)
                    indexes.append(interaction_idx)
                    intersections.append(intersect_fixes)
            else:
                filtered_interactions.append(interaction)
                indexes.append(interaction_idx)
                intersections.append(intersect_fixes)

        results = (filtered_interactions, indexes, intersections)

    return results


def filter_interactions_by_route_or_heading(
    interactions: list[InteractionInfo],
    callsign: str,
    tracked_data: dict[str, ACStateTracker],
    simulator_env: SimulatorEnv,
    sector: str,
    rollout_predictor: Predictor,
    use_filed_route: bool = True,
) -> tuple[list[InteractionInfo], list[int], list[list[IntersectionInfo]]]:
    """
    Filter an aircraft's list of interactions with other aircraft.

    Filter conditions, keep interaction (with other aircraft) if it is
    relevant (by route similarity) to the aircraft defined by two
    scenarios:

    Scenario 1: when the aircraft and the other aircraft are on route:
    i. check for overlapping fix(es) between the aircraft and the
       other aircraft's filed route, conditioned on their respective
       current position next fix(es) till the end of their route. if
       there's overlap, keep the interaction.

    Scenario 2: when the aircraft or the other aircraft is NOT on route:
    i. draw a vector line (based on current heading) of both aircraft for
       X nautical miles. if there's an intersection between both lines,
       keep the interaction.

    Args:
        interactions: captures information about interactions that the
            aircraft have with other aircraft. each item in the list is a
            tuple that contains the interaction information with another
            aircraft. note, the list only captures other aircraft for which
            the aircraft has an interaction (if no interaction exist with
            another aircraft, such aircraft is excluded from the list).
        callsign: defines the identifier of the aircraft for which its
            interactions with other aircraft is defined in the `interaction`
            list.
        tracked_data: defines a data store that tracks information about all
            active aircraft in a sector.
        simulator_env: defines the underlying simulator.
        sector: defines the name of the sector that the aircraft is located.
        rollout_predictor: defines a trajectory predictor used to simulate a
            forward rollout to get future trajectories of aircraft.
        use_filed_route: defines whether to use the aircraft's filed or
            current route. Defaults to `True`.

    Returns:
        tuple of three elements.
        - the first element is a list that contains a subset of
          `interactions` that satisfies the defined filter conditions.
        - the second element is the index/position of the filtered
          interactions in the original list.
        - the third element is a list (an item for each `other_aircraft` in
          the filtered list), which contains a list of intersecting fixes for
          each corresponding `other aircraft` in the filtered list.
          an intersection information contains the fix name and position.
    """

    # aircraft variables.
    aircraft = simulator_env.aircraft[callsign]
    ac_route_ff = aircraft.on_route

    filtered_interactions = []
    indexes = []
    intersections = []
    for idx, other_ac_interaction in enumerate(interactions):
        other_callsign = other_ac_interaction.other_callsign
        other_aircraft = simulator_env.aircraft[other_callsign]
        other_ac_route_ff = other_aircraft.on_route

        relevant = None
        if ac_route_ff and other_ac_route_ff:
            # intersect_fixes ordered by aircraft's route
            out, indexes, intersect_fixes = filter_interactions_by_route(
                [
                    other_ac_interaction,
                ],
                callsign,
                tracked_data,
                simulator_env,
                sector,
                rollout_predictor,
                in_sector_fixes_only=True,
                from_next_fix=True,
                use_filed_route=use_filed_route,
            )

            if len(out) > 0:
                relevant = True
                intersect_fixes: list[IntersectionInfo] = intersect_fixes[0]

        else:
            # either one or both aircraft are not route following
            # (i.e., instead on heading)

            # segments start from next fix by default in this function
            ret = get_intersection_location(
                callsign,
                tracked_data,
                other_ac_interaction,
                simulator_env,
                sector,
                use_filed_route,
            )
            intersection_info = ret[0]
            found_segment_indices = ret[1]
            aircraft_segments = ret[2]
            other_aircraft_segments = ret[3]

            if intersection_info is not None:
                idx_1, idx_2 = found_segment_indices
                intersect_fixes: list[IntersectionInfo] = [
                    intersection_info,
                ]

                # compute the respective distance of both aircraft to the
                # intersection position (ip).
                ac_dist_ip = 0.0
                for i in range(idx_1):
                    start_pos, end_pos = aircraft_segments[i].get_line()
                    ac_dist_ip += start_pos.distance(end_pos)
                start_pos, _ = aircraft_segments[idx_1]
                end_pos = intersection_info.location
                ac_dist_ip += start_pos.distance(end_pos)

                other_ac_dist_ip = 0.0
                for i in range(idx_2):
                    start_pos, end_pos = other_aircraft_segments[i].get_line()
                    other_ac_dist_ip += start_pos.distance(end_pos)
                start_pos, _ = other_aircraft_segments[idx_2].get_line()
                end_pos = intersection_info.location
                other_ac_dist_ip += start_pos.distance(end_pos)

                # compute the respective time (in minutes) to intersection
                # position (ip) for both aircraft
                ac_time_ip = (ac_dist_ip / aircraft.speed_tas) * 60
                other_ac_time_ip = (
                    other_ac_dist_ip / other_aircraft.speed_tas
                ) * 60

                relevant = abs(ac_time_ip - other_ac_time_ip) < TIME_DIFF

            else:
                ret = path_overlap(aircraft_segments, other_aircraft_segments)
                overlap_exist, found_segment_indices = ret

                if overlap_exist:
                    # the interaction is (or will become) a same track or
                    # opposite track.
                    relevant = True

                    idx_1, idx_2 = found_segment_indices
                    if idx_1 == 0:
                        # the overlap starts from the current segment that
                        # aircraft is in. Use the aircraft position as the
                        # position from where the overlap starts.

                        _tmp_fix_name = CUSTOM_FIX_AT_X.format(
                            CUSTOM_FIX_CURRENT_POS
                        )
                        intersect_fixes: list[IntersectionInfo] = [
                            IntersectionInfo(
                                True, aircraft.pos2d(), _tmp_fix_name
                            ),
                        ]

                    else:
                        # the overlap starts from a segment that is ahead in
                        # the aircraft's route. Use the start position of the
                        # segment as the location where the overlap starts
                        _named_line = aircraft_segments[idx_1]

                        ### this should be a standard fix position.
                        _start_pos = _named_line.get_line()[0]
                        _start_pos_name = _named_line.get_name()[0]

                        intersect_fixes: list[IntersectionInfo] = [
                            IntersectionInfo(True, _start_pos, _start_pos_name),
                        ]

                else:
                    relevant = False

        if relevant:
            filtered_interactions.append(other_ac_interaction)
            indexes.append(idx)
            intersections.append(intersect_fixes)

    return filtered_interactions, indexes, intersections


def filter_interactions_by_current_distance(
    interactions: list[InteractionInfo],
    callsign: str,
    tracked_data: dict[str, ACStateTracker],
    simulator_env: SimulatorEnv,
    distance_threshold: float,
) -> tuple[list[InteractionInfo], list[int]]:
    """
    Filter an aircraft's list of interactions with other aircraft by distance.

    Note: it considers only the distance of the current position (i.e., it
    does not consider the distance of future position of the aircraft).

    Filter conditions, keep interaction (with other aircraft) if:
    i.  the current distance between the current and other aircraft
        is less than a given `threshold`.

    Args:
        interactions: captures information about interactions that the
            aircraft have with other aircraft. each item in the list is a
            tuple that contains the interaction information with another
            aircraft. note, the list only captures other aircraft for which
            the aircraft has an interaction (if no interaction exist with
            another aircraft, such aircraft is excluded from the list).
        callsign: defines the identifier of the aircraft for which its
            interactions with other aircraft is defined in the `interaction`
            list.
        tracked_data: defines a data store that tracks information about all
            active aircraft in a sector.
        simulator_env: defines the underlying simulator.
        distance_threshold: the threshold distance used to check the
            evaluate the condition for filtering out interactions.

    Returns:
        tuple of two elements.
        - list that contains a subset of `interactions` that satisfies the
          defined filter conditions.
        - the indexes/positions of the filtered interactions in the original
          list.
    """

    # get future traj for aircraft
    aircraft = simulator_env.aircraft[callsign]

    # check if the distance between both aircraft goes below the defined
    # threshold (either in the current time or in a future time).
    filtered_interactions = []
    indexes = []

    for idx, other_ac_interaction in enumerate(interactions):
        other_callsign = other_ac_interaction.other_callsign
        other_aircraft = simulator_env.aircraft[other_callsign]

        distance = aircraft.pos2d().distance(other_aircraft.pos2d())

        if distance < distance_threshold:
            filtered_interactions.append(other_ac_interaction)
            indexes.append(idx)

    return filtered_interactions, indexes


def filter_interactions_by_distance(
    interactions: list[InteractionInfo],
    callsign: str,
    tracked_data: dict[str, ACStateTracker],
    simulator_env: SimulatorEnv,
    rollout_predictor: Predictor,
    distance_threshold: float,
) -> tuple[list[InteractionInfo], list[int], list[float]]:
    """
    Filter an aircraft's list of interactions with other aircraft by distance.

    Note: it considers both the distance of the current positions and the
    distance of future positions of the aircraft.

    Filter conditions, keep interaction (with other aircraft) if:
    i.  the current or future distance between the current and other aircraft
        is less than a given `threshold`.

    Args:
        interactions: captures information about interactions that the
            aircraft have with other aircraft. each item in the list is a
            tuple that contains the interaction information with another
            aircraft. note, the list only captures other aircraft for which
            the aircraft has an interaction (if no interaction exist with
            another aircraft, such aircraft is excluded from the list).
        callsign: defines the identifier of the aircraft for which its
            interactions with other aircraft is defined in the `interaction`
            list.
        tracked_data: defines a data store that tracks information about all
            active aircraft in a sector.
        simulator_env: defines the underlying simulator.
        rollout_predictor: defines a trajectory predictor used to simulate a
            forward rollout to get future trajectories of aircraft.
        distance_threshold: the threshold distance used to check the
            evaluate the condition for filtering out interactions.

    Returns:
        tuple of three elements.
        - the first element is a list that contains a subset of `interactions`
          that satisfies the defined filter conditions.
        - the second element is the index/position of the filtered
          interactions in the original list.
        - the third element is a list of distances for each corresponding
          `other aircraft` in the filtered list.
    """

    # get future traj for aircraft
    aircraft = simulator_env.aircraft[callsign]
    # 300 => 300 seconds
    ac_f_traj = f_traj(
        tracked_data.get(callsign, None),
        aircraft,
        rollout_predictor,
        300,
        simulator_env,
    )

    # check if the distance between both aircraft goes below the defined
    # threshold (either in the current time or in a future time).
    filtered_interactions = []
    indexes = []
    distances = []

    for idx, other_ac_interaction in enumerate(interactions):
        lateral_violation = False
        _dists = []

        # get future traj for other aircraft
        other_callsign = other_ac_interaction.other_callsign
        other_aircraft = simulator_env.aircraft[other_callsign]
        other_ac_f_traj = f_traj(
            tracked_data.get(callsign, None),
            other_aircraft,
            rollout_predictor,
            300,
            simulator_env,
        )

        for ac_cp, other_ac_cp in zip(ac_f_traj, other_ac_f_traj):
            distance = ac_cp.distance(other_ac_cp)
            _dists.append(distance)
            if distance < distance_threshold:
                lateral_violation = True
                break

        if lateral_violation:
            filtered_interactions.append(other_ac_interaction)
            indexes.append(idx)
            distances.append(_dists)
    return filtered_interactions, indexes, distances


def filter_interactions_by_distance_type(
    interactions: list[InteractionInfo],
    allowed_distance_types: list[InteractionDistance],
) -> tuple[list[InteractionInfo], list[int]]:
    """
    Filter an aircraft's list of interactions with other aircraft.

    Filter conditions, keep interaction (with other aircraft) if:
    i.  distance between the current and other aircraft is the same
        or reducing as time progresses.

    Args:
        interactions: captures information about interactions that the
            aircraft have with other aircraft. each item in the list is a
            tuple that contains the interaction information with another
            aircraft. note, the list only captures other aircraft for which
            the aircraft has an interaction (if no interaction exist with
            another aircraft, such aircraft is excluded from the list).
        allowed_distance_types: defines the list of distance types allowed
            for each interaction information.

    Returns:
        tuple of two elements.
        - the first element is a list that contains a subset of `interactions`
          that satisfies the defined filter conditions.
        - the second element is the index/position of the filtered
          in the original list.
    """

    filtered_interactions = []
    indexes = []
    for idx, other_ac_interaction in enumerate(interactions):
        if other_ac_interaction.dist_type_ac_other in allowed_distance_types:
            filtered_interactions.append(other_ac_interaction)
            indexes.append(idx)

    return filtered_interactions, indexes


def filter_interactions_by_interaction_type(
    interactions: list[InteractionInfo],
    allowed_interaction_types: list[int],
) -> tuple[list[InteractionInfo], list[int]]:
    """
    Filter an aircraft's list of interactions with other aircraft.

    Filter conditions, keep interaction (with other aircraft) if:
    i.  the interaction type is in `allowed_interaction_types`.

    Note: interaction type can be: same track, cross track, opposite track,
    or no interaction.

    Args:
        interactions: captures information about interactions that the
            aircraft have with other aircraft. each item in the list is a
            tuple that contains the interaction information with another
            aircraft. note, the list only captures other aircraft for which
            the aircraft has an interaction (if no interaction exist with
            another aircraft, such aircraft is excluded from the list).
        allowed_interaction_types: defines the list of interaction types
            allowed for each interaction information.

    Returns:
        tuple of two elements.
        - the first element is a list that contains a subset of `interactions`
          that satisfies the defined filter conditions.
        - the second element is the index/position of the filtered
          interactions in the original list.
    """

    filtered_interactions = []
    indexes = []
    for idx, other_ac_interaction in enumerate(interactions):
        interaction_category = other_ac_interaction.track_category_ac_other
        if interaction_category in allowed_interaction_types:
            filtered_interactions.append(other_ac_interaction)
            indexes.append(idx)

    return filtered_interactions, indexes


def filter_interactions_by_fl_range_overlap(
    interactions: list[InteractionInfo],
    callsign: str,
    tracked_data: dict[str, ACStateTracker],
    simulator_env: SimulatorEnv,
    sector: str,
) -> tuple[list[InteractionInfo], list[int], list[tuple[float, float]]]:
    """
    Filter an aircraft's list of interactions with other aircraft.

    Filter conditions, keep interaction (with other aircraft) if:
    i. and, the flight level range (using current flight level and exit
        flight levels) for both aircraft overlaps.

    Args:
        interactions: captures information about interactions that the
            aircraft have with other aircraft. each item in the list is a
            tuple that contains the interaction information with another
            aircraft. note, the list only captures other aircraft for which
            the aircraft has an interaction (if no interaction exist with
            another aircraft, such aircraft is excluded from the list).
        callsign: defines the identifier of the aircraft for which its
            interactions with other aircraft is defined in the `interaction`
            list.
        tracked_data: defines a data store that tracks information about all
            active aircraft in a sector.
        simulator_env: defines the underlying simulator.
        sector: defines the name of the sector that the aircraft is located.

    Returns:
        tuple of three elements.
        - the first element is a list that contains a subset of `interactions`
          that satisfies the defined filter conditions.
        - the second element is the index/position of the filtered
          interactions in the original list.
        - the third element is a list flight level range tuple (current
          flight level, exit flight level) for each corresponding `other
          aircraft` in the filtered list.
    """

    # the simulator some times store fl as numpy.float64,
    # typecast to regular float
    ac_fl = float(simulator_env.aircraft[callsign].fl)
    ac_exit_fl = float(tracked_data[callsign].exit_coords[sector].fl)
    range_ac = (ac_fl, ac_exit_fl)

    filtered_interactions = []
    ranges_other_ac = []
    indexes = []
    for idx, other_ac_interaction in enumerate(interactions):
        other_callsign = other_ac_interaction.other_callsign
        other_ac_fl = float(simulator_env.aircraft[other_callsign].fl)
        other_ac_exit_fl = tracked_data[other_callsign].exit_coords[sector].fl
        other_ac_exit_fl = float(other_ac_exit_fl)

        range_other_ac = (other_ac_fl, other_ac_exit_fl)
        overlap = flight_level_range_overlap(range_ac, range_other_ac)

        if overlap is True:
            filtered_interactions.append(other_ac_interaction)
            ranges_other_ac.append(range_other_ac)
            indexes.append(idx)

    return filtered_interactions, indexes, ranges_other_ac


def filter_relevant_interactions(
    callsign: str,
    sector: str,
    tracked_data: dict[str, ACStateTracker],
    simulator_env: SimulatorEnv,
    interactions: list[InteractionInfo],
    rollout_predictor: Predictor,
    relevance: InteractionRelevance,
    use_filed_route: bool,
) -> list[InteractionInfo]:
    """Filter aircraft interactions to keep only relevant ones.

    If `relevance` is InteractionRelevance.LEVEL_1, keep interactions
    where:
    (i)   the interaction distance between aircraft pair is reducing or
          being maintained. also, interactions with increasing distance is
          kept *only* if the current distance between aircraft pair is less
          less five nautical miles
    (ii)  a flight route overlap exist between aircraft pair (if they're
          both following their defined route) or if their current headings
          interesects when one of them is not following its defined route.
    (iii) a flight level range overlap occur (range defined using current
          and exit flight level)

    If `relevance` is InteractionRelevance.LEVEL_2, keep interactions
    where:
    (i)   the filed routes of the aircraft pair has overlapping fixes,
          starting from the next fix of each aircraft.
    (ii)  a flight level range overlap occur (range defined using current
          and exit flight level)

    Args:
        callsign: defines the identifier of the aircraft.
        sector: defines the name of the sector that the aircraft is located.
        tracked_data: defines a data store that tracks information about all
            active aircraft in a sector.
        simulator_env: defines the underlying simulator.
        interactions: captures information about interactions that the
            aircraft have with other aircraft. each item in the list is a
            tuple that contains the interaction information with another
            aircraft. note, the list only captures other aircraft for which
            the aircraft has an interaction (if no interaction exist with
            another aircraft, such aircraft is excluded from the list).
        rollout_predictor: defines a trajectory predictor used to simulate
            a forward rollout to get future trajectories of aircraft.
        relevance: defines the category of relevant interactions of
            the subject aircraft to return in the filtered list.
        use_filed_route: defines whether to use the aircraft's filed or
            current route.

    Returns:
        list of relevant traffic or an empty list if no relevant interactions
        were found.
    """

    ####### filter by flight level range overlap
    ret = filter_interactions_by_fl_range_overlap(
        interactions, callsign, tracked_data, simulator_env, sector
    )
    filtered_interactions, _, _ = ret

    ####### filter by relevance level
    if relevance == InteractionRelevance.LEVEL_1:
        ####### filter by route or heading
        # when route following, the aircraft segments (used to determine
        # whether an intersection exists) is generated from the aircraft's
        # its next fix. when on a heading, the segment is drawn from the
        # aircraft's current position.
        ret = filter_interactions_by_route_or_heading(
            filtered_interactions,
            callsign,
            tracked_data,
            simulator_env,
            sector,
            rollout_predictor,
            use_filed_route=use_filed_route,
        )
        filtered_interactions, _, intersections = ret

        for interaction, ac_other_ac_intersections in zip(
            filtered_interactions, intersections
        ):
            # set relevance level
            interaction.relevance = InteractionRelevance.LEVEL_1

            # set intersecting fixes
            interaction.intersections = ac_other_ac_intersections

            # set main intersection
            ac_route_ff = simulator_env.aircraft[callsign].on_route
            other_ac_route_ff = simulator_env.aircraft[
                interaction.other_callsign
            ].on_route

            # if len(ac_other_ac_intersections) == 1 and not (
            #    ac_route_ff and other_ac_route_ff
            # ):
            if len(ac_other_ac_intersections) == 1:
                # this could happen when:
                # - one or both aircraft are on headings
                # - or when there's only one common fix when both aircraft
                #   are route following
                interaction.main_intersection = ac_other_ac_intersections[0]

            else:
                # this should only happen when both aircraft are route
                # following and the intersecting fixes along their route has
                # been returned.
                assert ac_route_ff and other_ac_route_ff

                ret = get_intersection_location_both_route_ff(
                    callsign,
                    interaction.other_callsign,
                    simulator_env,
                    tracked_data,
                    use_filed_route,
                    num_trials=3,
                    trial_duration_mins=10,
                    rollout_predictor=rollout_predictor,
                )
                main_intersection, future_ac_pos, other_ac_pos = ret
                if main_intersection is None:
                    raise ValueError(
                        "`main_intersection` should not be None "
                        "aircraft pair have intersecting fixes"
                    )
                interaction.main_intersection = main_intersection

            interaction.proxy_intersection = None

    elif relevance == InteractionRelevance.LEVEL_2:
        ####### filter by route
        ret = filter_interactions_by_route(
            filtered_interactions,
            callsign,
            tracked_data,
            simulator_env,
            sector,
            rollout_predictor,
            in_sector_fixes_only=True,
            from_next_fix=True,
            use_filed_route=use_filed_route,
        )
        filtered_interactions, _, intersections = ret

        for interaction, ac_other_ac_intersections in zip(
            filtered_interactions, intersections
        ):
            # set relevance level
            interaction.relevance = InteractionRelevance.LEVEL_2

            # set intersecting fixes
            interaction.intersections = ac_other_ac_intersections

            # set main intersection
            # set to None since LEVEL_2 interaction examines only similarity
            # in the route regardless of whether the aircraft path have been
            # vectored to resolve the interactions on heading.
            interaction.main_intersection = None

            if len(interaction.intersections) == 1:
                interaction.proxy_intersection = interaction.intersections[0]
            else:
                ac = simulator_env.aircraft[callsign]
                ac_pos2d = ac.pos2d()
                other_ac = simulator_env.aircraft[interaction.other_callsign]
                other_ac_pos2d = other_ac.pos2d()

                time_diffs_mins = []
                for intersection in interaction.intersections:
                    # consider using route track distance instead of
                    # linear distance
                    dist_1 = ac_pos2d.distance(intersection.location)
                    time_1_mins = (dist_1 / ac.speed_tas) * 60

                    # consider using route track distance instead of
                    # linear distance
                    dist_2 = other_ac_pos2d.distance(intersection.location)
                    time_2_mins = (dist_2 / other_ac.speed_tas) * 60

                    time_diffs_mins.append(abs(time_1_mins - time_2_mins))

                # the intersection with the smallest time (to location)
                # difference is the most relevant intersection.
                idx = time_diffs_mins.index(min(time_diffs_mins))  # argmin
                interaction.proxy_intersection = interaction.intersections[idx]

    else:
        raise ValueError(
            f"`relevance category {relevance} is not supported in this "
            "function."
        )

    return filtered_interactions


def get_aircraft_relevant_interactions(
    callsign: str,
    sector: str,
    tracked_data: dict[str, ACStateTracker],
    simulator_env: SimulatorEnv,
    rollout_predictor: Predictor,
    relevance: InteractionRelevance | None = InteractionRelevance.LEVEL_1,
    use_filed_route: bool = False,
) -> list[InteractionInfo]:
    """For an aircraft, get interactions (with other aircraft) relevant it.

    See docstring in `filter_relevant_interactions(...)` for the definition
    of a relevant interaction for each `relevance` category/type.

    If `relevance` is set to None, the returned interactions list is a
    concatenation the result of LEVEL_1 interactions and LEVEL_2
    interactions.

    Args:
        callsign: defines the identifier of the aircraft.
        sector: defines the name of the sector that the aircraft is located.
        tracked_data: defines a data store that tracks information about all
            active aircraft in a sector.
        simulator_env: defines the underlying simulator.
        rollout_predictor: defines a trajectory predictor used to simulate
            a forward rollout to get future trajectories of aircraft.
        relevance: defines the category of relevant interactions of
            the subject aircraft to return in the filtered list.
            Optional, defaults to InteractionRelevance.LEVEL_1.
        use_filed_route: defines whether to use the aircraft's filed or
            current route. Defaults to False.

    Returns:
        list of relevant traffic or an empty list if no relevant interactions
        were found.
    """

    # get all interactions with the current aircraft
    interactions = get_aircraft_interaction_info(
        callsign, tracked_data, simulator_env, rollout_predictor, True
    )

    # only keep interactions with other aircraft that are still in the
    # sector.
    interactions = [
        interaction
        for interaction in interactions
        if tracked_data[interaction.other_callsign].pos_status
        != PositionStatus.EXIT_REACHED
    ]

    # filter the interactions to only keep relevant ones.
    if relevance is None:
        # return the relevant interactions for each relevance category/type,
        # removing duplicates with a lower priority/category.
        # highest to lowest priority defintion: LEVEL_1, LEVEL_2.
        primary_interactions = filter_relevant_interactions(
            callsign,
            sector,
            tracked_data,
            simulator_env,
            interactions,
            rollout_predictor,
            InteractionRelevance.LEVEL_1,
            use_filed_route,
        )
        callsigns_to_exclude = [
            interaction.other_callsign for interaction in primary_interactions
        ]
        subset_tracked_data = {
            cs: tracked_data[cs]
            for cs in tracked_data.keys()
            if cs not in callsigns_to_exclude
        }
        subset_interactions = [
            interaction
            for interaction in interactions
            if interaction.other_callsign not in callsigns_to_exclude
        ]
        secondary_interactions = filter_relevant_interactions(
            callsign,
            sector,
            subset_tracked_data,
            simulator_env,
            subset_interactions,
            rollout_predictor,
            InteractionRelevance.LEVEL_2,
            use_filed_route,
        )

        filtered_interactions = primary_interactions + secondary_interactions

    else:
        filtered_interactions = filter_relevant_interactions(
            callsign,
            sector,
            tracked_data,
            simulator_env,
            interactions,
            rollout_predictor,
            relevance,
            use_filed_route,
        )

    return filtered_interactions


def get_optimal_unblocked_flight_level(
    callsign: str,
    sector: str,
    tracked_data: dict[str, ACStateTracker],
    simulator_env: SimulatorEnv,
    rollout_predictor: Predictor,
    interactions: list[InteractionInfo] | None = None,
) -> int:
    """Computes the highest/lowest flight level to climb/descend to.

    Computes the highest/lowest flight level to which an aircraft can
    climb/descend. This can be target/exit flight level of the aircraft if
    there are no other aircraft that block its climb/descent path OR an
    intermediate level, which is the highest/lowest unblocked level to which
    the aircraft can climb/descend.

    Blocked levels occur due interactions between the aircraft and other
    aircraft in the sector where the other aircraft occupies a flight level
    through which the aircraft needs to fly through to reach its target/exit
    flight level.

    Args:
        callsign: defines the identifier of the aircraft.
        sector: defines the name of the sector that the aircraft is located.
        tracked_data: defines a data store that tracks information about all
            active aircraft in a sector.
        simulator_env: defines the underlying simulator.
        rollout_predictor: defines a trajectory predictor used to simulate
            a forward rollout to get future trajectories of aircraft.
        interactions: list of relevant interactions or None if there are no
            relevant interactions

    Returns:
        the computed flight level. if there are not blocked flight levels
        between the current flight level of the aircraft and its target/exit
        flight level, then the exit flight level is returned. otherwise, an
        intermediate flight level is returned for the aircraft to
        climb/descend to.
    """

    # the simulator sometimes store fl as numpy.float64,
    # typecast to regular float
    ac_fl = float(simulator_env.aircraft[callsign].fl)
    ac_exit_fl = float(tracked_data[callsign].exit_coords[sector].fl)

    # get interactions relevant to the current aircraft
    if interactions is None:
        interactions = get_aircraft_relevant_interactions(
            callsign,
            sector,
            tracked_data,
            simulator_env,
            rollout_predictor,
            relevance=None,
            use_filed_route=False,
        )

    # filter 1: select only intervation with relevance level 1
    filtered_interactions = [
        interaction
        for interaction in interactions
        if interaction.relevance == InteractionRelevance.LEVEL_1
    ]
    # filter 2: re-filter the original list of interaction and keep
    # an level 2 relevance interaction only if they are close to the subject
    # aircraft.
    for interaction in interactions:
        if (
            interaction.relevance == InteractionRelevance.LEVEL_2
            and interaction.dist_ac_other < MinAircraftSeparation.LATERAL
        ):
            filtered_interactions.append(interaction)
        else:
            continue

    # filter 3: filter the filtered list
    # if the subject aircraft needs to climb, then remove interactions where
    # the subject aircraft can reach its top of ascent before the
    # intersection location.
    if ac_fl < ac_exit_fl:
        tmp_filtered_interactions = []
        for interaction in filtered_interactions:
            _intersection = main_or_proxy_intersection_location(
                callsign, interaction
            )
            ac_status = top_of_ascent_before_intersection(
                callsign,
                simulator_env,
                tracked_data[callsign].dist_to_target_fl,
                _intersection,
                tracked_data[callsign],
                sector,
                interaction_distance=interaction.lateral_dist_thresh_sv,
            )

            if not ac_status:
                tmp_filtered_interactions.append(interaction)
        filtered_interactions = tmp_filtered_interactions

    else:
        # do nothing: keep the current filtered interactions
        pass

    # now optimal unblocked flight level computation based on the filtered
    # interactions.
    other_callsigns = []
    ranges_other_ac = []
    for interac in filtered_interactions:
        other_callsign = interac.other_callsign
        other_aircraft = simulator_env.aircraft[other_callsign]

        other_ac_fl = float(simulator_env.aircraft[other_callsign].fl)
        try:
            other_ac_exit_fl = (
                tracked_data[other_callsign].exit_coords[sector].fl
            )
        except KeyError:
            # this means that aircraft was removed from the tracked data in
            # the previous step, likely due to incorrect sector exit and has
            # been tracked for a number of timestep after the incorrect exit.
            from bluebird_gymnasium.utils.simulator_utils import (
                aircraft_exit_coordination,
            )

            coord = aircraft_exit_coordination(
                other_callsign, simulator_env, sector
            )
            other_ac_exit_fl = coord.fl

        other_ac_exit_fl = float(other_ac_exit_fl)
        range_other_ac = (other_ac_fl, other_ac_exit_fl)

        other_callsigns.append(other_callsign)
        ranges_other_ac.append(range_other_ac)

    optimal_fl = None
    if len(ranges_other_ac) == 0:
        # no conflict. aircraft can climb/descend to exit flight level.
        optimal_fl = ac_exit_fl

    else:
        # conflict. find intermediate flight level level to which
        # aircraft should climb/descend.

        # NOTE: for same track scenario, should the aircraft be kept at their
        # current flight level? this will be apply to both climb and descent
        # situation.

        if ac_fl < ac_exit_fl:  # aircraft needs to climb
            blocked_fls = []
            for other_callsign, other_ac_range in zip(
                other_callsigns, ranges_other_ac
            ):
                other_ac_fl, other_ac_exit_fl = other_ac_range
                other_aircraft = simulator_env.aircraft[other_callsign]
                other_ac_selected_fl = other_aircraft.selected_fl

                if other_ac_fl > ac_fl:  # other aircraft is relevant
                    if other_ac_fl > other_ac_selected_fl:
                        # other aircraft is currently descending. use the
                        # selected fl, but don't capped at aircraft current fl
                        _fl = ac_fl + MinAircraftSeparation.VERTICAL
                        blocked_fls.append(max(_fl, other_ac_selected_fl))

                    else:
                        # other aircraft is ascending or cruising.
                        # use its current fl
                        blocked_fls.append(other_ac_fl)
                else:
                    continue

            # if there are no blocked levels, set climb value as the
            # exit flight level, else set value to a level that is
            # 10 flight levels below the lowest blocked level.
            if len(blocked_fls) > 0:
                blocked_fls = sorted(blocked_fls, reverse=False)
                optimal_fl = blocked_fls[0] - MinAircraftSeparation.VERTICAL
                # round down to the nearest 10s
                optimal_fl = math.floor(optimal_fl / 10) * 10

                # if optimal fl is > than exit fl, clip it to exit fl
                optimal_fl = min(ac_exit_fl, optimal_fl)

            else:
                optimal_fl = ac_exit_fl

        else:  # aircraft needs to descend.
            blocked_fls = []
            for other_callsign, other_ac_range in zip(
                other_callsigns, ranges_other_ac
            ):
                other_ac_fl, other_ac_exit_fl = other_ac_range
                other_aircraft = simulator_env.aircraft[other_callsign]
                other_ac_selected_fl = other_aircraft.selected_fl

                if other_ac_fl < ac_fl:  # other aircraft is relevant
                    if other_ac_fl < other_ac_selected_fl:
                        # other aircraft is currently ascending. use the
                        # selected fl, but don't capped at aircraft current fl
                        _fl = ac_fl - MinAircraftSeparation.VERTICAL
                        blocked_fls.append(max(_fl, other_ac_selected_fl))

                    else:
                        # other aircraft is ascending or cruising.
                        # use its current fl
                        blocked_fls.append(other_ac_fl)
                else:
                    continue

            # if there are no blocked levels, set descent value as the
            # exit flight level, else set value to a level that is
            # 10 flight levels above the highest blocked level.
            if len(blocked_fls) > 0:
                blocked_fls = sorted(blocked_fls, reverse=True)
                optimal_fl = blocked_fls[0] + MinAircraftSeparation.VERTICAL
                # round down to the nearest 10s
                optimal_fl = math.floor(optimal_fl / 10) * 10

                # if optimal fl is < than exit fl, clip it to exit fl
                optimal_fl = max(ac_exit_fl, optimal_fl)

            else:
                optimal_fl = ac_exit_fl

    assert optimal_fl is not None
    return optimal_fl


def get_intersection_location_both_route_ff(
    callsign: str,
    other_callsign: str,
    simulator_env: SimulatorEnv,
    tracked_data: dict[str, ACStateTracker],
    use_filed_route: bool,
    num_trials: int = 1,
    trial_duration_mins: int = 10,
    rollout_predictor: Predictor | None = None,
) -> tuple[IntersectionInfo, tuple[float, float], tuple[float, float]]:
    """Get the intersection location of route following aircraft pair.

    Args:
        callsign: defines the identifier of the target aircraft.
        other_callsign: defines the identifier of the other aircraft.
        simulator_env: defines the underlying simulator.
        num_trials: defines the number of trajectory prediction attempt
            to find the inteserction location. each trial builds on the
            last predicted aircraft positions of the previous trials
            (i.e., the first trial starts with the current aircraft
            positions). Defaults to 1.
        trial_duration: defines the number of minutes to predicted into
            the future per trial. Defaults to 10 minutes.
        rollout_predictor: defines a trajectory predictor used to simulate
            a forward rollout to get future trajectories of aircraft.

    Note: this function truncates the trials once the distance between
        both aircraft starts increasing.

    Returns:
        - the intersection information if at least one intersection exists.
          otherwise None is returned.
        - the last location (lat, lon) on the predicted trajectory for the
          target/subject aircraft.
        - the last location (lat, lon) on the predicted trajectory for the
          other aircraft.

    Raises:
        Exception, if at least one aircraft is *not* route following.
    """

    aircraft = simulator_env.aircraft[callsign]
    other_aircraft = simulator_env.aircraft[other_callsign]

    if not aircraft.on_route or not other_aircraft.on_route:
        raise Exception(
            "Only use this function when both aircraft are route following"
        )

    # use a route following predictor to get a intersection location.
    if rollout_predictor is not None:
        route_ff_predictor = rollout_predictor
    else:
        route_ff_predictor = RouteFollowPredictor(
            dt=6.0,
            fix_proximity_threshold=2.0,
            fixes=simulator_env.airspace.fixes,
        )

    dummy_ac = copy.deepcopy(aircraft)
    dummy_other_ac = copy.deepcopy(other_aircraft)
    found_location = None
    for idx in range(num_trials):
        if idx == 0:
            ac_traj = tracked_data[callsign].future_trajectory
            other_ac_traj = tracked_data[other_callsign].future_trajectory
        else:
            start_time = trial_duration_mins * (idx + 1)
            end_time = start_time + trial_duration_mins
            name = f"extra_tp/{start_time}--{end_time}mins"

            ac_traj = f_traj(
                tracked_data[callsign],
                dummy_ac,
                predictor=route_ff_predictor,
                duration=trial_duration_mins * 60,
                simulator_env=simulator_env,
                trajectory_alias=name,
            )
            other_ac_traj = f_traj(
                tracked_data[callsign],
                dummy_other_ac,
                predictor=route_ff_predictor,
                duration=trial_duration_mins * 60,
                simulator_env=simulator_env,
                trajectory_alias=name,
            )

        dist_increasing = False
        idx_minimum_distance = -1
        minimum_distance = dummy_ac.pos2d().distance(dummy_other_ac.pos2d())
        for idx in range(len(ac_traj)):
            ac_cp = ac_traj[idx]
            other_ac_cp = other_ac_traj[idx]
            current_distance = ac_cp.distance(other_ac_cp)

            if current_distance < minimum_distance:
                idx_minimum_distance = idx
                minimum_distance = current_distance
            else:
                dist_increasing = True
                break

        if dist_increasing:
            if idx_minimum_distance != -1:
                dummy_ac.lat = ac_traj[idx_minimum_distance].lat
                dummy_ac.lon = ac_traj[idx_minimum_distance].lon
                dummy_other_ac.lat = other_ac_traj[idx_minimum_distance].lat
                dummy_other_ac.lon = other_ac_traj[idx_minimum_distance].lon
            else:
                # else do nothing as the current positions of dummy_ac and
                # dummy_other_ac are the locations where their distance is
                # minimum
                pass
            break
        else:
            dummy_ac.lat = ac_traj[-1].lat
            dummy_ac.lon = ac_traj[-1].lon
            dummy_other_ac.lat = other_ac_traj[-1].lat
            dummy_other_ac.lon = other_ac_traj[-1].lon

    found_locations = dummy_ac.pos2d(), dummy_other_ac.pos2d()
    assert found_locations[0].distance(found_locations[1]) == minimum_distance

    if minimum_distance < 1.0:
        # just use the position of the subject aircraft as the distance
        # between aircraft pair is small. avoids extra computation.
        found_location = found_locations[0]
    else:
        # aircraft centreline distance at other aircraft location when
        # the distance is minimum
        dummy_ac.lat = found_locations[1].lat
        dummy_ac.lon = found_locations[1].lon
        ac_centre_dist, _, _ = get_centreline_distance(
            dummy_ac.pos2d(),
            dummy_ac.flight_plan.route.current,
            simulator_env.airspace,
            tracked_data[dummy_ac.callsign].pos_at_last_route_direct,
        )
        # other aircraft centreline distance at aircraft location when
        # the distance is minimum
        dummy_other_ac.lat = found_locations[0].lat
        dummy_other_ac.lon = found_locations[0].lon
        other_ac_centre_dist, _, _ = get_centreline_distance(
            dummy_other_ac.pos2d(),
            dummy_other_ac.flight_plan.route.current,
            simulator_env.airspace,
            tracked_data[dummy_other_ac.callsign].pos_at_last_route_direct,
        )

        if other_ac_centre_dist < ac_centre_dist:
            found_location = found_locations[0]  # use aircraft location
        else:
            found_location = found_locations[1]  # use other aircraft location

    previous_fix, next_fix = get_previous_next_fixes_from_position(
        dummy_ac,
        found_location,
        tracked_data[callsign],
        simulator_env,
        use_filed_route,
        deepcopy_aircraft=False,  # since a copy is being passed
    )
    if previous_fix == next_fix:
        previous_fix = CUSTOM_FIX_AT_X.format(CUSTOM_FIX_CURRENT_POS)
    _tmp_fix_name = CUSTOM_FIX_BETWEEN_X_AND_Y.format(previous_fix, next_fix)
    intersection_info = IntersectionInfo(True, found_location, _tmp_fix_name)

    return (
        intersection_info,
        *found_locations,
    )


def get_intersection_location(
    callsign: str,
    tracked_data: dict[str, ACStateTracker],
    interaction: InteractionInfo,
    simulator_env: SimulatorEnv,
    sector: str,
    use_filed_route: bool = False,
) -> tuple[
    IntersectionInfo | None,
    tuple[int, int] | None,
    list[NamedLine],
    list[NamedLine],
]:
    """Get the first intersection location of an aircraft with another.

    Note:
        - this is useful when at least one aircraft is *not* route following.
          if both are route following, use:
          `get_intersection_location_both_route_ff(...)`
        - start from the next fix of each aircraft.
        - take into account whether each aircraft is on route or heading.

    Args:
        callsign: defines the identifier of the target aircraft
        tracked_data: defines a data store that tracks information about all
            active aircraft in a sector.
        interaction: defines the interaction information with the other
            aircraft.
        simulator_env: defines the underlying simulator.
        sector: defines the name of the sector that the aircraft is located.
        use_filed_route: defines whether to use the aircraft's filed or
            current route. Defaults to `True`.

    Returns:
        a four-element tuple:
        - the intersection information if at least one intersection exists.
          otherwise None is returned.
        - the indices of line (within each path) where the intersection exist
          else None is returned.
        - the path of aircraft 1.
        - the path of aircraft 2.
    """

    ####### aircraft
    aircraft = simulator_env.aircraft[callsign]
    aircraft_segments = get_segments_from_route_or_heading(
        aircraft,
        simulator_env.airspace,
        sector,
        tracked_data[callsign],
        use_filed_route,
        route_in_sector_fixes_only=True,
        route_from_next_fix=True,
    )

    ####### other aircraft
    other_callsign = interaction.other_callsign
    other_aircraft = simulator_env.aircraft[other_callsign]
    other_aircraft_segments = get_segments_from_route_or_heading(
        other_aircraft,
        simulator_env.airspace,
        sector,
        tracked_data[other_callsign],
        use_filed_route,
        route_in_sector_fixes_only=True,
        route_from_next_fix=True,
    )

    ####### get the intersection location
    # at least one of the aircraft is on a heading which implies only
    # one segment. the intersection location can be computed directly
    # without using a trajectory predictor.

    ret = path_intersection(aircraft_segments, other_aircraft_segments)
    intersect_exist, location, found_indices = ret

    if intersect_exist:
        idx_1, _ = found_indices
        start_pos_name, end_pos_name = aircraft_segments[idx_1].get_name()
        _tmp_fix_name = CUSTOM_FIX_BETWEEN_X_AND_Y.format(
            start_pos_name, end_pos_name
        )
        intersection_info = IntersectionInfo(
            intersect_exist, location, _tmp_fix_name
        )
    else:
        intersection_info = None

    return (
        intersection_info,
        found_indices,
        aircraft_segments,
        other_aircraft_segments,
    )


def get_segments_from_route(
    aircraft: Aircraft,
    airspace: Airspace,
    sector: str,
    aircraft_tracked_data: ACStateTracker,
    use_filed_route: bool,
    in_sector_fixes_only: bool,
    from_next_fix: bool,
) -> list[NamedLine]:
    """Get segments of an aircraft based on its filed or current route.

    Args:
        aircraft: defines the target aircraft
        airspace: defines the airspace the aircraft is flying through.
        sector: defines the name of the sector that the aircraft is located.
        aircraft_tracked_data: defines a data store that tracks information
            about the target aircraft.
        use_filed_route: defines whether to use the aircraft's filed or
            current route.
        in_sector_fixes_only: defines whether fixes in sector should be
            considered only, when the aircraft is route following (i.e.,
            not on a heading).
        from_next_fix: specifies whether to use the entire route
            of both aircraft (i.e., if False) or a subset of the route
            starting from the next fix of each aircraft (i.e., if True).

    Returns:
        the list of segments for the aircraft.
    """

    if use_filed_route:
        if from_next_fix:
            # start_from = aircraft_tracked_data.next_fix_fr
            start_from = (aircraft.pos2d(), aircraft_tracked_data.next_fix_fr)
        else:
            start_from = None
    else:
        # ac_next_fix = aircraft_tracked_data.next_fix_cr
        # ac_route = aircraft.flight_plan.route.current
        # if from_next_fix:
        #    ac_next_fix_idx = ac_route.index(ac_next_fix)
        # else:
        #    ac_next_fix_idx = 0

        # if ac_next_fix_idx == 0:
        #    start_from = (
        #        aircraft_tracked_data.pos_at_last_route_direct,
        #        ac_next_fix,
        #    )
        # else:
        #    start_from = ac_next_fix

        ac_route = aircraft.flight_plan.route.current
        if from_next_fix:
            start_from = (aircraft.pos2d(), aircraft_tracked_data.next_fix_cr)
        else:
            start_from = (
                aircraft_tracked_data.pos_at_last_route_direct,
                ac_route[0],
            )

    if use_filed_route:
        route = aircraft.flight_plan.route.filed
    else:
        route = aircraft.flight_plan.route.current

    if in_sector_fixes_only:
        aircraft_segments = get_route_segments_in_sector(
            route,
            airspace,
            sector_name=sector,
            start_from=start_from,
        )
    else:
        aircraft_segments = get_route_segments(
            route,
            airspace,
            start_from=start_from,
        )

    return aircraft_segments


def get_segments_from_route_or_heading(
    aircraft: Aircraft,
    airspace: Airspace,
    sector: str,
    aircraft_tracked_data: ACStateTracker,
    route_use_filed: bool,
    route_in_sector_fixes_only: bool,
    route_from_next_fix: bool,
    heading_segment_length: Number = 100,
) -> list[NamedLine]:
    """Get segments of an aircraft based on its route or heading.

    Note:
    - The returned segment depends on whether the aircraft is currently
      route following or on a heading.

    Args:
        aircraft: defines the target aircraft
        airspace: defines the airspace the aircraft is flying through.
        sector: defines the name of the sector that the aircraft is located.
        aircraft_tracked_data: defines a data store that tracks information
            about the target aircraft.
        route_use_filed: defines whether to use the aircraft's filed or
            current route.
        route_in_sector_fixes_only: defines whether fixes in sector should be
            considered only, when the aircraft is route following (i.e.,
            not on a heading).
        route_from_next_fix: specifies whether to use the entire route
            of both aircraft (i.e., if False) or a subset of the route
            starting from the next fix of each aircraft (i.e., if True).
        heading_segment_length: defines the segment length (distance) when
            the aircraft is on heading (i.e., not route following). Defaults
            to 100 nautical miles.

    Returns:
        the list of segments for the aircraft. note that when the aircraft
        is on a heading, only one segment is returned in the list (which is
        the line from the aircraft position to a forward location, having a
        distance of `heading_segment_length`).
    """

    ac_current_pos = aircraft.pos2d()
    if aircraft.on_route:
        ac_segments = get_segments_from_route(
            aircraft,
            airspace,
            sector,
            aircraft_tracked_data,
            route_use_filed,
            route_in_sector_fixes_only,
            route_from_next_fix,
        )

    else:
        # get segment based on the aircraft's current heading
        ac_future_pos_on_heading = ac_current_pos.forward(
            dist=heading_segment_length, heading=aircraft.heading
        )
        current_pos_name = CUSTOM_FIX_AT_X.format(CUSTOM_FIX_CURRENT_POS)
        future_pos_name = CUSTOM_FIX_AT_X.format(CUSTOM_FIX_FUTURE_POS)
        ac_segments = [
            NamedLine(
                (ac_current_pos, ac_future_pos_on_heading),
                (current_pos_name, future_pos_name),
            )
        ]

    return ac_segments


def main_or_proxy_intersection_location(
    callsign: str, interaction: InteractionInfo
) -> IntersectionInfo:
    """Get the most relevant intersection for an interaction."""

    if interaction.main_intersection is not None:
        assert (
            interaction.relevance == InteractionRelevance.LEVEL_1
            and len(interaction.intersections) > 0
        )
        intersection = interaction.main_intersection

    elif interaction.proxy_intersection is not None:
        assert (
            interaction.relevance == InteractionRelevance.LEVEL_2
            and len(interaction.intersections) > 0
        )
        intersection = interaction.proxy_intersection

    else:
        raise Exception(
            f"{callsign} and {interaction.other_callsign} "
            f"{interaction.relevance} interaction: both `.main_intersection` "
            f"and `.proxy_intersection` cannot be set to None in "
            f"{type(interaction)}."
        )

    return intersection


def top_of_ascent_before_intersection(
    callsign: str,
    simulator_env: SimulatorEnv,
    distance_to_target_fl: Number,
    intersection: IntersectionInfo,
    aircraft_tracked_data: ACStateTracker,
    sector: str,
    interaction_distance: Number = 0,
    uncertainty_distance: Number = 0,
) -> bool:
    """Check whether a climb can be complete before an intersection"""

    aircraft = simulator_env.aircraft[callsign]
    route = aircraft.flight_plan.route.current

    if intersection.location_name in route:
        stop_at = intersection.location_name

    else:
        # custom (fix) location
        previous_fix, next_fix = get_previous_next_fixes_from_position(
            aircraft,
            intersection.location,
            aircraft_tracked_data,
            simulator_env,
            use_filed_route=False,
        )
        stop_at = (intersection.location, previous_fix)

    dist_to_intersection, _ = route_distance_and_time_to_location(
        aircraft,
        stop_at,
        aircraft_tracked_data,
        simulator_env,
        sector,
        use_filed_route=False,
    )

    return (
        distance_to_target_fl + interaction_distance + uncertainty_distance
    ) < dist_to_intersection


def top_of_descent_after_intersection(
    aircraft: Aircraft,
    airspace: Airspace,
    distance_to_target_fl: Number,
    intersection_location: Pos2D,
    exit_position: Pos2D,
    current_route_start_position: Pos2D | None,
    uncertainty_distance: Number = 0,
) -> bool:
    """Check whether top of descent of aircraft occurr after an intersection"""

    # get the track distance to the exit position from the intersection point
    dist_to_exit_from_intersect = distance_to_target_pos_along_route(
        intersection_location,
        exit_position,
        aircraft.flight_plan.route.current,
        airspace,
        current_route_start_position,
    )

    # add an uncertainty buffer to the distance
    # to target fl being checked against.
    return dist_to_exit_from_intersect > (
        distance_to_target_fl + uncertainty_distance
    )


class TrafficMonitor:
    def __init__(self):
        self.store: dict[str, list[InteractionInfo]] = {}

    def reset(self):
        """Reset the traffic monitor store."""
        self.store.clear()

    def update(self, gym_env: BaseEnv) -> None:
        """Update the traffic monitor store."""

        tracked_data = gym_env.get_tracked_aircraft_data()

        # initialise dict to store the extra predicted trajectory
        for callsign in tracked_data.keys():
            if tracked_data[callsign].extra_future_trajectory is None:
                tracked_data[callsign].extra_future_trajectory = {}

        for callsign, _data in tracked_data.items():
            if _data.pos_status == PositionStatus.EXIT_REACHED:
                # once an aircraft has successfully exited a sector, its
                # traffic is no longer monitored.
                self.store[callsign] = []

            else:
                self.store[callsign] = get_aircraft_relevant_interactions(
                    callsign,
                    gym_env.get_active_airspace_sector(),
                    tracked_data,
                    gym_env.get_simulator_env(),
                    gym_env.get_rollout_predictor(),
                    relevance=None,  # return all relevance categories.
                    use_filed_route=False,  # use current route
                )

        # delete the extra predicted trajectory
        for callsign in tracked_data.keys():
            tracked_data[callsign].extra_future_trajectory.clear()

    def get_relevant_traffic(
        self,
        callsign: str,
        relevance: InteractionRelevance | None = None,
        copy: bool = False,
    ) -> list[InteractionInfo]:
        """get relevant traffic for an aircraft."""

        if callsign not in self.store.keys():
            return []

        if copy:
            interactions = copy.deepcopy(self.store[callsign])
        else:
            interactions = self.store[callsign]

        if relevance is not None:
            interactions = [
                interaction
                for interaction in interactions
                if interaction.relevance == relevance
            ]

        return interactions

    def get_relevant_traffic_summary(
        self,
        callsign: str,
        relevance: InteractionRelevance | None,
        copy: bool = False,
    ) -> list[tuple]:
        """get the summary of relevant traffic for an aircraft."""

        interactions = self.get_relevant_traffic(callsign, relevance, copy)
        return [
            (interaction.other_callsign, interaction.intersections)
            for interaction in interactions
        ]

    @classmethod
    def get_aircraft_interaction_info(
        cls: type[Self], callsign: str, gym_env: BaseEnv
    ) -> list[InteractionInfo]:
        tracked_data = gym_env.get_tracked_aircraft_data()
        simulator_env = gym_env.get_simulator_env()
        rollout_predictor = gym_env.get_rollout_predictor()

        return get_aircraft_interaction_info(
            callsign, tracked_data, simulator_env, rollout_predictor, True
        )
