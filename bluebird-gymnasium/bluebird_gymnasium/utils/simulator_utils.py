from __future__ import annotations

import copy
import math
import typing

import numpy as np
from bluebird_dt.core.pos2d import Pos2D
from bluebird_dt.core.pos4d import Pos4D

from bluebird_dt.utility.convert import (
    FL_TO_FT,  # flight level (FL) to feet (FT) converter
    KT_TO_MPS,  # nautical miles per second (knots) to metre per second
    MPS_TO_KT,  # metre per second to nautical miles per second (knots)
    horizontal_tas,
    tas_to_cas,  # true airspeed to calibrated airspeed
)

from bluebird_gymnasium.utils.constants import DEFAULT_RATE_OF_CLIMB_DESCENT
from bluebird_gymnasium.utils.geo_utils import (
    get_centreline_distance,
)

if typing.TYPE_CHECKING:
    from bluebird_dt.core.aircraft import Aircraft
    from bluebird_dt.core.airspace import Airspace
    from bluebird_dt.core.coordination import Coordination
    from bluebird_dt.core.environment import Environment as SimulatorEnv
    from bluebird_dt.predictor import Predictor

Number: typing.TypeAlias = typing.Union[int, float]


def aircraft_prev_next_fixes(
    aircraft: Aircraft,
    simulator_env: SimulatorEnv,
    use_filed_route: bool = True,
    distance_threshold_NMI: Number = 2.0,
) -> tuple[str, str]:
    """Retrieves the previous and the next fix of an aircraft

    Given an aircraft's current position, the previous and next fixes are
    retrieved and returned.

    Args:
        aircraft: the subject aircraft
        simulator_env: the simulator_env instance.
        use_filed_route: defines whether to use the aircraft's filed or
            current route. Defaults to `True`.
        distance_threshold_NMI: threshold used to judge whether an aircraft
            has passed a fix. Nautical miles is the unit of the distance.

    Returns:
        tuple, the previous and the next fix.

    """

    if use_filed_route:
        route = aircraft.flight_plan.route.filed

        # next fix
        next_fix = simulator_env.airspace.closest_forward_fix(
            aircraft,
            distance_threshold_NMI=distance_threshold_NMI,
            route_fixes=route,
        )

    else:
        route = aircraft.flight_plan.route.current

        filed_route = aircraft.flight_plan.route.filed
        if len(set(route).symmetric_difference(set(filed_route))) == 0:
            # current route is the same as filed route
            next_fix = simulator_env.airspace.closest_forward_fix(
                aircraft,
                distance_threshold_NMI=distance_threshold_NMI,
                route_fixes=route,
            )
        else:
            # current route is not the same as filed route.
            # address the special case when the aircraft is yet to reach
            # the first fix of the current route.
            # the `closest_forward_fix` function can sometimes mis-interpret
            # this situation due to sector geometry (labelling the second fix
            # of the current route as the next fix instead of the first fix)
            #
            # hence, this is solved by artificially adding to the route, the
            # previous fix before the first fix in the current route. the
            # previous fix is gotten from the filed route.

            # get the previous fix before the first fix in the current route.
            # note: get it from the filed route.
            _idx = filed_route.index(route[0])
            _fix = filed_route[_idx - 1]

            # append it to the current route
            modified_route = [
                _fix,
            ] + route

            next_fix = simulator_env.airspace.closest_forward_fix(
                aircraft,
                distance_threshold_NMI=distance_threshold_NMI,
                route_fixes=modified_route,
            )

            if next_fix is not None and modified_route.index(next_fix) == 0:
                # if the closest forward fix in the modified route is the
                # first index (i.e., the previous fix from the filed route
                # that was added to the route, then adjust it to point to the
                # actual first fix of the route.
                next_fix = modified_route[1]

    if next_fix is None:
        # assume that aircraft has completed its filed route path
        # and return its last fix as its previous and next fixes.
        return route[-1], route[-1]

    else:
        # previous fix
        next_fix_idx = route.index(next_fix)
        if next_fix_idx == 0:
            # if next fix is the first fix in the route,
            # set previous fix to be same as next fix.
            prev_fix = next_fix

        else:
            prev_fix = route[next_fix_idx - 1]

        return prev_fix, next_fix


def prev_next_fixes(
    callsign: str,
    simulator_env: SimulatorEnv,
    use_filed_route: bool = True,
    distance_threshold_NMI: Number = 1.0,
) -> tuple[str, str]:
    """Retrieves the previous and the next fix of an aircraft

    A wrapper function around `aircraft_prev_next_fixes`.

    Args:
        callsign: the identifier of the aircraft
        simulator_env: the simulator_env instance.
        use_filed_route: defines whether to use the aircraft's filed or
            current route. Defaults to `True`.
        distance_threshold_NMI: threshold used to judge whether an aircraft
            has passed a fix. Nautical miles is the unit of the distance.

    Returns:
        tuple, the previous and the next fix.

    """

    return aircraft_prev_next_fixes(
        simulator_env.aircraft[callsign],
        simulator_env,
        use_filed_route,
        distance_threshold_NMI,
    )


def prev_next_fixes_positions(
    previous_fix: str,
    next_fix: str,
    airspace: Airspace,
    route_start_position: Pos2D | None = None,
) -> tuple[Pos2D, Pos2D]:
    """Get the positions of the previous and next fixes

    Args:
        previous_fix: defines the previous fix name.
        previous_fix: defines the next fix name.
        airspace: defines the airspace which contains the fixes positions.
        route_start_position: defines the start position of the route
            in which the previous and next fixes belong.

    Returns:
        two-element tuple of the previous and next fixes positions.

    Raises:
        ValueError if previous_fix and next_fix have the same value and
        `pos_at_last_route_direct` is None.
    """

    if previous_fix == next_fix:
        if route_start_position is None:
            raise ValueError(
                "`previous_fix` has the same value as `next_fix`. in such "
                "situation, `route_start_position` "
                "should have been set to a Pos2D when the tracked data is "
                "updated in the gymnasium environment being run (in update "
                "dynamic data)"
            )

        # we assume that the last route direct position (which can be an
        # actual route direct action in the scenario or the spawn point of
        # the aircraft the start of the scenario is the previous fix.
        previous_fix_pos = route_start_position
        next_fix_pos = airspace.fixes.places[next_fix]
    else:
        previous_fix_pos = airspace.fixes.places[previous_fix]
        next_fix_pos = airspace.fixes.places[next_fix]

    return previous_fix_pos, next_fix_pos


def predict_trajectory_simple(
    pos: Pos2D, heading: float, distance: Number, num_control_points: int = 10
) -> list[Pos2D]:
    """Compute control points between a position and distance travelled.

    Given a position (e.g., of an aircraft), simulate a forward look-ahead
    into the future) and store a number of control points in between the
    current position and the new position (based on the distance travelled).

    Note, this function accounts for only 2D navigation (latitude and
    longitude), thus lateral. It doesn't take into account the changing
    flight levels (vertical navigation) and lateral speed (i.e., speed_tas)
    if they are required for a given aircraft. When such properties are
    required for the forward rollout, use `predict_trajectory(...)` instead.

    Note, due to its simplicity, this function should be computationally
    faster than `predict_trajectory(...)` which uses trajectory predictors
    in simulation package to simulate a forward rollout.

    Args:
        pos: the current aircraft position.
        heading: the current heading (in degrees) of the aircraft.
        distance: the distance (in nautical miles) ahead to simulate.
        num_control_points: the number of control points (Pos2D) to compute
            Defaults to 10.

    Returns:
        a list of control points.
        The list contains both the start position `pos` and the end position
        (as a result of the `distance` travelled). The minimum number of items
        in the list is 2 (i.e., if `num_control_points` is set to 1, then
        only the start and end positions are stored in the list).
    """
    new_pos = pos.forward(dist=distance, heading=heading)
    ac_cps_lat = np.interp(
        np.arange(0, num_control_points + 1),
        [0, num_control_points],
        [pos.lat, new_pos.lat],
    )
    ac_cps_lon = np.interp(
        np.arange(0, num_control_points + 1),
        [0, num_control_points],
        [pos.lon, new_pos.lon],
    )

    return [Pos2D(lat, lon) for lat, lon in zip(ac_cps_lat, ac_cps_lon)]


def predict_trajectory(
    aircraft: Aircraft,
    predictor: Predictor,
    duration: int = 600,
    curr_time: float = 0.0,
) -> list[Pos4D]:
    """Compute control points of aircraft current and future position.

    Given an aircraft, simulate navigation (a forward look-ahead into
    the future based on the aircraft's parameters such as speed_tas,
    heading,...) using a `predictor`. Get the control points between
    the current position and the future position (based on the distance
    travelled).

    Note, unlike `predict_trajectory_simple`, this function accounts for the
    complete navigation profile, 4D navigation (latitude, longitude, flight
    level and time). Also, this function is more precise than the simple
    version because it uses the aircraft lateral speed (i.e., speed_tas) to
    compute the forward rollout.

    Note, relative to the simple version, this function likely comes at
    a computational cost due to the use of trajectory predictors.
    If control points containing latitude and longitude are needed, without
    need for a precise forward rollout (lateral speed not taken into account),
    then `predict_trajectory_simple(...)` is good enough.

    Args:
        aircraft: the aircraft
        predictor: the predictor to use for forward look-ahead.
        duration: the time (in seconds) to run forward the simulator.
            Defaults to 600 seconds (10 minutes).
        curr_time: the current time in the simulator in which the aircraft is
            navigating. Optional. Defaults to 0.0 (ignore time).

    Returns:
        list, containing the number of control points (bluebird_dt.core.Pos4D).
        Each list element contains: latitude, longitude, flight level,
        and time.

    Raises:
        ValueError: if the list contains no control points. This is caused by
            specifying very small `duration`. In such scenario, the `duration`
            parameter should be increased.
    """

    trajectory = predictor.predict_trajectory(aircraft, duration, curr_time)

    if trajectory is None:
        raise ValueError(
            "No control points to return. Fix: increase `duration` parameter."
        )

    if isinstance(trajectory, list):
        return trajectory
    else:
        return trajectory.control_points


def aircraft_entry_coordination(
    callsign: str, simulator_env: SimulatorEnv, sector_name: str = None
) -> None | Coordination | list[Coordination]:
    """Get the entry coordination of an aircraft.

    Args:
        callsign: the identifier of the aircraft
        simulator_env: the simulator_env instance.
        sector_name: the name of the sector for which to get an entry
            coordination for the aircraft. Optional.
            If set to `None`, the entry coordination data of the aircraft
            is returned across as many sectors as available.

    Returns:
        the aircraft's entry coordination if a specific sector is given.
        otherwise, a list of entry coordinations is returned, capturing the
        aircraft's entry coordinations for all sectors in the airspace.

    Raises:
        ValueError: if `sector_name` is set and the name does not match any
            sector in the airspace.
    """

    aircraft = simulator_env.aircraft[callsign]
    airspace = simulator_env.airspace

    if sector_name is None:
        sectors = set(airspace.sectors.keys())
        ac_coords = [
            simulator_env.entry_coordination(sector_name, callsign)
            for sector_name in sectors
        ]
        return ac_coords

    else:
        if sector_name not in airspace.sectors.keys():
            raise ValueError(
                f"Sector '{sector_name}' does not exist in airspace with "
                f"sectors list: {list(airspace.sectors.keys())}"
            )
        # the method below could return None
        return simulator_env.entry_coordination(sector_name, callsign)


def aircraft_exit_coordination(
    callsign: str, simulator_env: SimulatorEnv, sector_name: str = None
) -> None | Coordination | list[Coordination]:
    """Get the exit coordination of an aircraft.

    Args:
        callsign: the identifier of the aircraft
        simulator_env: the simulator_env instance.
        sector_name: the name of the sector for which to get an exit
            coordination for the aircraft. Optional.
            If set to `None`, the exit coordination data of the aircraft
            is returned across as many sectors as available.

    Returns:
        the aircraft's exit coordination if a specific sector is given.
        otherwise, a list of exit coordinations is returned, capturing the
        aircraft's exit coordinations for all sectors in the airspace.

    Raises:
        ValueError: if `sector_name` is set and the name does not match any
            sector in the airspace.
    """

    aircraft = simulator_env.aircraft[callsign]
    airspace = simulator_env.airspace

    if sector_name is None:
        sectors = set(airspace.sectors.keys())
        ac_coords = [
            simulator_env.exit_coordination(sector_name, callsign)
            for sector_name in sectors
        ]
        return ac_coords
    else:
        if sector_name not in airspace.sectors.keys():
            raise ValueError(
                f"Sector '{sector_name}' does not exist in airspace with "
                f"sectors list: {list(airspace.sectors.keys())}"
            )
        # the method below could return None
        return simulator_env.exit_coordination(sector_name, callsign)


def top_of_ascent(
    aircraft: Aircraft, target_fl: Number, wind: Number = 0
) -> tuple[float, float]:
    """Computes the distance for an aircraft to climb to a target flight level.

    Computes the travel distance (in nautical miles) required for an aircraft
    to reach a target flight level. Note, this is based on a *heuristic*.

    Args:
        aircraft: the aircraft
        target_fl: the target flight level.
        wind: defines the wind component (in knots): tailwind (+)
            or head wind (-).
            Defaults to 0, which indicates no effect of wind.

    Returns:
        two element tuple:
        - the distance (in nautical miles) required for the aircraft to reach
          the target flight level given its current flight level.
        - the time (in minutes) required for the aircraft to reach the target
          flight level given its current flight level.

    Raises:
        ValueError: if the given target flight level is less than or equal to
            the aircraft's current flight level (as this is not an ascent)
    """

    if target_fl <= aircraft.fl:
        raise ValueError(
            "Target flight level is lower than or equal to the aircraft's "
            "current flight level. Hence this is not a ascent but a descent."
        )

    climb_fpm = DEFAULT_RATE_OF_CLIMB_DESCENT  # feet per minute

    vertical_diff = target_fl - aircraft.fl
    vertical_diff *= FL_TO_FT  # convert to feet
    time_mins = vertical_diff / climb_fpm  # time in minutes

    # add an extra 1 minute to account for uncertainty
    time_mins += 1.0

    time_hrs = time_mins / 60  # time in hours

    # ground speed is in nautical miles per hour
    distance = aircraft.ground_speed * time_hrs  # in nautical miles

    return distance, time_mins


def top_of_descent(
    aircraft: Aircraft, target_fl: Number, wind: Number = 0
) -> tuple[float, float]:
    """Computes the distance for an aircraft to reach a target flight level.

    Computes the travel distance (in nautical miles) required for an aircraft
    to reach a target flight level. Note, this is based on a *heuristic*
    defined below as:

    `(height_difference x 3) +/- 10%`
    +10 % if tail wind is present, and -10% if head wind is present.

    E.g., Aircraft with flight level 360 required to descend to target flight
    level 200, the distance to target will be
    360 - 220 = 140 FL. take the most significant non-zero digit (or divide
    by 10). therefore, height_difference becomes 14.
    without wind, the distance is computed as:
    14 x 3 = 42 nautical miles


    Args:
        aircraft: the aircraft
        target_fl: the target flight level.
        wind: defines the wind component (in knots): tailwind (+)
            or head wind (-).
            Defaults to 0, which indicates no effect of wind.

    Returns:
        two element tuple:
        - the distance (in nautical miles) required for the aircraft to reach
          the target flight level given its current flight level.
        - the time (in minutes) required for the aircraft to reach the target
          flight level given its current flight level.

    Raises:
        ValueError: if the given target flight level is greater than or equal
            to the aircraft's current flight level (as this is not a descent).
    """

    if target_fl >= aircraft.fl:
        raise ValueError(
            "Target flight level is higher than or equal to the aircraft's "
            "current flight level. Hence this is not a ascent but a descent."
        )

    # base distance: 3-to-1 rule (3 nm for every 10 flight level to lose)
    fl_diff = abs(aircraft.fl - target_fl) / 10.0
    base_distance = fl_diff * 3

    # wind correction: 1nm for every 10 knots of wind
    # tail wind adds distance (as the aircraft will travel faster)
    # head wind subtracts distance (as the aircraft will travel faster)
    wind_effect = wind / 10

    # speed restriction adjustment: 1 nm for every 10 kts of speed reduction
    # 100 fl speed restriction adjustment
    # if IAS is > 250kts, we need extra distance to slow down before 100fl.
    # TODO: write and use a function to convert tas to ias
    # speed_ias = aircraft.speed_tas
    # deceleration_effect = max(0, (speed_ias - 250) / 10)

    deceleration_effect = 10  # add an approximation of 10 nm

    distance = base_distance + wind_effect + deceleration_effect

    time_mins = (distance / aircraft.ground_speed) / 60

    return distance, time_mins


def distance_time_to_target_fl(
    aircraft: Aircraft, target_fl: Number, wind: Number = 0
) -> tuple[float, float]:
    """Computes the distance for an aircraft to reach a target flight level.

    Computes the travel distance (in nautical miles) required for an aircraft
    to climb or descend to a target flight level.

    Args:
        aircraft: the aircraft
        target_fl: the target flight level.
        wind: defines the wind component (in knots): tailwind (+)
            or head wind (-).
            Defaults to 0, which indicates no effect of wind.

    Returns:
        two element tuple:
        - the distance (in nautical miles) required for the aircraft to reach
          the target flight level given its current flight level.
        - the time (in minutes) required for the aircraft to reach the target
          flight level given its current flight level.
    """

    if aircraft.fl < target_fl:
        # calculate top of ascent
        return top_of_ascent(aircraft, target_fl, wind)

    elif aircraft.fl > target_fl:
        # calculate top of descent
        return top_of_descent(aircraft, target_fl, wind)

    else:
        return 0.0, 0.0


def distance_to_target_pos_along_route(
    start_pos: Pos2D,
    target_pos: Pos2D,
    route: list[str],
    airspace: Airspace,
    route_start_position: Pos2D | None = None,
) -> Number:
    """Computes the distance for an aircraft to reach a target position.

    Computes the distance for an aircraft to reach a target position along
    its route (filed or current).

    Note, the distance computed following the defined route from the
    aircraft's position to the target position.

    Args:
        start_pos: the start position in the route.
        target_pos: the target position.
        route: defines the route.
        airspace: the airspace, which contains the sector, which contains the
            defined route for which to compute the aircraft's distance to the
            target position along the route/track.
        route_start_position: defines a non-standard start position of the
            route. if set, it means that the route starts from a location
            that is not a standard fix (in the airspace). this is useful
            when the `route` passed is the current route of an aircraft.
            if set then, the it is used as the first position in the route,
            before the first fix (standard) position in the route.

    Returns:
        the distance (in nautical miles) required for the aircraft to
        travel to reach the target position given its current position.
        -1 is returned if aircraft has already passed the target
        position.
    """

    # calculate the track distance of the aircraft along its route (i.e.,
    # from the start of the route to the current aircraft position)
    _, _, ac_track_distance = get_centreline_distance(
        start_pos, route, airspace, route_start_position
    )

    # get the track distance to the target location
    # (i.e., from the start of the route)
    _, _, target_track_distance = get_centreline_distance(
        target_pos, route, airspace, route_start_position
    )

    # approximate aircraft distance to target
    lateral_distance = target_track_distance - ac_track_distance
    if lateral_distance < 0:
        # the aircraft has already passed the target position
        # hence, the negative distance.
        return -1
    else:
        return lateral_distance


def time_to_target_pos_along_route(
    aircraft: Aircraft,
    target_pos: Pos2D,
    airspace: Airspace,
    route_start_position: Pos2D | None = None,
) -> Number:
    """Computes the time taken for an aircraft to reach a target position.

    Computes the time taken for an aircraft to reach a target position along
    its route (filed or current).

    Note, the distance computed following the defined route from the
    aircraft's position to the target position.

    Args:
        aircraft: the aircraft.
        target_pos: the target position.
        airspace: the airspace, which contains the sector, which contains the
            defined route for which to compute the aircraft's time to the
            target position along the route/track.
        route_start_position: defines a non-standard start position of the
            route. if set, it means that the route starts from a location
            that is not a standard fix (in the airspace). this is useful
            when the `route` passed is the current route of an aircraft.
            if set then, the it is used as the first position in the route,
            before the first fix (standard) position in the route.

    Returns:
        the time (in minutes) required for the aircraft to get to the
        target position given its current position, lateral speed (true air
        speed) and heading.
        -1 is returned if aircraft has already passed the target
        position.
    """

    # approximate aircraft distance to target
    lateral_distance = distance_to_target_pos_along_route(
        aircraft.pos2d(),
        target_pos,
        aircraft.flight_plan.route.current,
        airspace,
        route_start_position,
    )

    if lateral_distance == -1:
        # the aircraft has already passed the target position
        # hence, the negative distance.
        time_to_target_pos = -1
    else:
        # compute time to the target position from current aircraft position.
        lateral_speed = aircraft.speed_tas  # knots (nautical miles per hour)
        time_to_target_pos = lateral_distance / lateral_speed
        time_to_target_pos *= 60  # convert hour(s) to minutes

    return time_to_target_pos


def basic_distances(
    aircraft: Aircraft,
    airspace: Airspace,
    target_pos2d: Pos2D,
    target_fl: Number,
    current_route_start_position: Pos2D | None = None,
) -> tuple[Number, Number, Number, Number]:
    """Computes different distances to a target position/level for an aircraft.

    Computed lateral distances include:
        - linear distance from the aircraft position to a target position.
        - along filed route/track distance from the aircraft position to a
          target position.
        - along current route/track distance from the aircraft position to a
          target position.
        - distance for an aircraft to ascend or descend from its current
          flight level to a target flight level.

    Args:
        aircraft: the aircraft.
        airspace: the airspace, which contains the sector, which contains the
            defined route for which to compute the aircraft's distance to the
            target position along the route/track.
        target_pos2d: the target position.
        target_fl: the target flight level.
        current_route_start_position: the start position of the current route.
            if set then, the it is used as the first position in the route,
            before the first fix position in the route. this is useful in
            situations when the current route is different from the filed
            route (i.e., a subset of the filed route) due to a route direct
            action in a previous time step.

    Returns:
        a tuple containing the aforementioned distances (in nautical miles).
    """

    # linear distance to target position
    linear_distance_to_pos = aircraft.pos2d().distance(target_pos2d)

    # filed route (fr) track distance to target position
    track_distance_to_pos_fr = distance_to_target_pos_along_route(
        aircraft.pos2d(),
        target_pos2d,
        aircraft.flight_plan.route.filed,
        airspace,
    )
    if track_distance_to_pos_fr == -1:
        track_distance_to_pos_fr = 0

    # current route (cr) track distance to target position
    if aircraft.flight_plan.route.filed == aircraft.flight_plan.route.current:
        track_distance_to_pos_cr = track_distance_to_pos_fr
    else:
        track_distance_to_pos_cr = distance_to_target_pos_along_route(
            aircraft.pos2d(),
            target_pos2d,
            aircraft.flight_plan.route.current,
            airspace,
            current_route_start_position,
        )
        if track_distance_to_pos_cr == -1:
            track_distance_to_pos_cr = 0

    # distance to target flight level
    # compute the lateral travel distance to reach the exit flight level
    dist_to_level, _ = distance_time_to_target_fl(aircraft, target_fl)

    return (
        linear_distance_to_pos,
        track_distance_to_pos_fr,
        track_distance_to_pos_cr,
        dist_to_level,
    )


def get_aircraft_selected_heading(aircraft: Aircraft) -> Number:
    """Return an aircraft's selected heading.

    A useful function as the selected heading is sometimes set to None
    initially by the simulator. In such scenario, the aircraft's current
    heading is used as a proxy for the selected heading.

    Args:
        aircraft: the aircraft for which to get the selected heading

    Returns:
        the selected heading
    """

    if aircraft.selected_instructions.heading is not None:
        return aircraft.selected_instructions.heading
    else:
        return aircraft.heading


def get_aircraft_selected_flight_level(aircraft: Aircraft) -> Number:
    """Return an aircraft's selected flight level.

    if the selected flight level is set to None (this could happen at the
    initial start of a simulation), then the aircraft's current flight
    level is used as a proxy.

    Args:
        aircraft: the aircraft for which to get the selected flight level

    Returns:
        the selected flight level
    """

    if aircraft.selected_fl is not None:
        return aircraft.selected_fl

    elif aircraft.selected_instructions.fl is not None:
        return aircraft.selected_instructions.fl

    else:
        return aircraft.fl


def infer_aircraft_speed(
    aircraft: Aircraft, rollout_predictor: Predictor
) -> tuple[float, float]:
    """Infer aircraft true airspeed and groundspeed using a trajectory predictor

    Args:
        aircraft: defines the aircraft for which to infer its speed.
        rollout_predictor: defines the trajectory predictor to be used in
            inferring the true airspeed.

    Return:
        two element tuple:
        - the inferred true airspeed of the aircraft (in knots).
        - the inferred ground speed of the aircraft (in knots).
    """

    predicted_aircraft = rollout_predictor.predict_aircraft(
        aircraft, delta_t=12, deepcopy_aircraft=True
    )
    return predicted_aircraft.speed_tas, predicted_aircraft.ground_speed


def get_aircraft_selected_cas(aircraft: Aircraft) -> Number:
    """Get the selected calibrated airspeed (cas) of the aircraft.

    if the selected speed (CAS) is set to None (this could happen at the
    initial start of a simulation), then the CAS is calculated from the
    aircraft's current true airspeed (TAS) and flight level.

    Args:
        aircraft: defines the aircraft for which to calculated the calibrated
            airspeed.

    Return:
        float, the selected calibrated airspeed of the aircraft (in knots).
    """

    if aircraft.selected_instructions.cas is not None:
        ## first check if it can be recovered from the selected instruction
        current_cas = aircraft.selected_instructions.cas
    else:
        ## else compute it from the current tas
        tas_KT = aircraft.speed_tas  # tas in knots (nautical miles per hour)

        tas_MPS = tas_KT * KT_TO_MPS  # tas in metre per second

        current_cas_MPS = tas_to_cas(aircraft.fl, tas_MPS, delta_T=0.0)
        current_cas = current_cas_MPS * MPS_TO_KT  # cas in knots

    return current_cas


def east_north_ground_speed(
    callsign: str, simulator_env: SimulatorEnv
) -> tuple[float, float]:
    """Get the east ground speed and north ground speed.

    The east and north ground speed are computed only if wind information
    is available in the simulator. if there's no wind information, the
    aircraft's ground speed is returned as the east and north ground speed.

    Args
        callsign: the identifier of the aircraft
        simulator_env: the simulator_env instance.

    Returns:
        tuple containing the east ground speed and the north ground speed.
    """

    aircraft = simulator_env.aircraft[callsign]

    if simulator_env.wind_field is None:
        # no wind information. assume east and
        # north ground speed as the same.
        east_ground_speed = aircraft.ground_speed
        north_ground_speed = aircraft.ground_speed
    else:
        # compute wind vector based on aircraft flight level and position
        wind_vector = simulator_env.wind_field.get_wind_vector(
            flight_level=aircraft.fl,
            latitude=aircraft.lat,
            longitude=aircraft.lon,
        )
        horizontal_tas_kts = horizontal_tas(aircraft.speed_tas, aircraft.vertical_speed)
        # copied from bluebird_dt.utility.convert.ground_speed_from_tas
        east_ground_speed = (
            wind_vector.u_comp * MPS_TO_KT
            + horizontal_tas_kts * math.sin(math.radians(aircraft.heading))
        )
        north_ground_speed = (
            wind_vector.v_comp * MPS_TO_KT
            + horizontal_tas_kts * math.cos(math.radians(aircraft.heading))
        )

    return east_ground_speed, north_ground_speed


def get_n_forward_fixes(
    route: list[str], start_from: str, n: int
) -> list[str | None]:
    """Get the next forward N fixes in a route.

    If `N` is greater than the available forward fixes in the route, then
    `None` is used to fill remaining positions of the list.

    Args:
        route: the route from which to return the fixes.
        start_from: the fix from which to start counting forward.
        n: the number for forward fixes to return (which includes
            `start_from`)

    Return:
        the list of N forward fixes

    """

    fixes = []
    fix_idx = route.index(start_from)

    for i in range(n):
        _idx = fix_idx + i
        if _idx < len(route):
            fixes.append(route[_idx])
        else:
            fixes.append(None)

    return fixes
