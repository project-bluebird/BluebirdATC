from __future__ import annotations

import typing

import numpy as np

# simulation package
from bluebird_dt.core.pos2d import Pos2D
from bluebird_dt.core.pos3d import Pos3D
from bluebird_dt.core.pos4d import Pos4D
from bluebird_dt.core.sector import Sector
from bluebird_dt.utility import convert
from bluebird_dt.utility.geometry import (
    find_all_boundary_intersections,
)
from bluebird_dt.utility.geometry import (
    line_intersection as _intersection,
)

# shapely: a dependency of the simulation package.
from shapely.geometry import Point

try:
    from bluebird_dt.utility.constants import (
        R_E_WGS84 as EARTH_RADIUS_IN_METERS,
    )
except ImportError:
    from bluebird_dt.utility.constants import R_E as EARTH_RADIUS_IN_METERS

from bluebird_gymnasium.utils.constants import (
    CUSTOM_FIX_AFTER_X,
    CUSTOM_FIX_BEFORE_X,
)
from bluebird_gymnasium.utils.types import (
    LineIntersection,
    NamedLine,
    Quadrant,
    QuadrantBound,
    TurnDirection,
)

if typing.TYPE_CHECKING:
    from bluebird_dt.core.aircraft import Aircraft
    from bluebird_dt.core.airspace import Airspace
    from bluebird_dt.core.area import Area

    from bluebird_gymnasium.utils.types import Line, Number


def left_right_check(reference: float, target: float) -> int:
    """Compute the direction of move from reference to target angle.

    For example, it is used to:
    (i)  compute the handedness to lateral route deviation.
    (ii) compute direction of travel from bearing of aircraft to a
         fix/other-aircraft (reference) to an aircraft heading (target)
         or vice versa.

    Args:
        reference: bearing of position 1 (in degrees).
        target: bearing of position 2 (in degrees).

    Return:
        the direction of turn, encoded as an ternary:
         0: no turn
         1: the reference would need to turn right to the target bearing.
        -1: the reference would need to turn left to the target bearing.
    """
    direction = None
    if reference == target:
        direction = TurnDirection.NO_TURN
    elif (reference < target and target - reference < 180.0) or (  # 180 degrees
        reference > target and reference - target > 180.0  # 180 degrees
    ):
        direction = TurnDirection.RIGHT
    else:
        direction = TurnDirection.LEFT
    return direction


def segment_in_sector_intersect(
    segment: Line | NamedLine,
    airspace: Airspace,
    sector_name: str,
    epsilon: float = 1e-10,
) -> bool:
    """Determines whether a segment/line within strictly in a sector.

    This does the check by:
    - checking that both the start and end position of the segment/line
      are within the sector. Otherwise, False is returned.
    - determining that there are no intersections between the segment and the
      sector boundary. If an intersection exist, False is returned.

    Note: this function fails in some instances due to sector specific
    geometry. In a situation where the segment starts and ends within the
    segment but has a portion that goes out of the sector, the function
    returns `True` which is incorrect. Therefore, use
    `segment_in_sector_interp` instead which is computationally more expensive
    but handles all edge cases correctly.

    Args:
        segment: defines the line segment to check.
        airspace: defines the airspace the aircraft is flying through.
        sector_name: defines the name of the sector in the airspace.
        epsilon: defines a threshold that is used when checking if a position
            is within the sector. It serves as a tolerance value for
            positions that are close to the sector's boundary.

    Returns:
        `True` if the segment is strictly within the sector. Otherwise, if
        the segment is either outside the sector or crosses from within the
        sector into outside the sector, then `False` is returned.
    """

    sector_obj = airspace.sectors[sector_name]
    # create a new sector where both the main volumes and conditional volumes
    # are combined.
    if sector_obj.conditional_volume_dict is not None:
        main_sector_volumes = sector_obj.volumes
        cond_sector_volumes = list(sector_obj.conditional_volume_dict.values())
        _volumes = main_sector_volumes + cond_sector_volumes
        sector_obj = Sector(_volumes)

    start_pos = segment[0]
    end_pos = segment[1]

    if sector_obj.contains_laterally(start_pos, epsilon=epsilon) and sector_obj.contains_laterally(
        end_pos, epsilon=epsilon
    ):
        intersections = find_all_boundary_intersections(start_pos.location, end_pos.location, sector_obj)

        if len(intersections) == 0:
            status = True

        elif len(intersections) == 1:
            c, d, u = intersections[0]
            _location = c + u * (d - c)
            intersection_position = Pos2D.from_array(_location)

            if start_pos.distance(intersection_position) < epsilon or end_pos.distance(intersection_position) < epsilon:
                status = True
            else:
                status = False

        else:
            status = False

    else:
        status = False

    return status


def segment_in_sector_interp(
    segment: Line | NamedLine,
    airspace: Airspace,
    sector_name: str,
    epsilon: float = 1e-10,
    num_interp_positions: int = 5,
) -> bool:
    """Determines whether a segment within strictly in a sector.

    Does the check by generating interpolated positions on the segment/line
    and checking if each position is within the sector. If any point is
    outside the sector, the segment is deemed to be either outside the sector
    or cross over the sector from inside to outside.

    Args:
        segment: defines the line segment to check.
        airspace: defines the airspace the aircraft is flying through.
        sector_name: defines the name of the sector in the airspace.
        epsilon: defines a threshold that is used when checking if a position
            is within the sector. It serves as a tolerance value for
            positions that are close to the sector's boundary.
        num_interp_positions: defines the number of positions/points to
            interpolate between the start and end positions of the segment.

    Returns:
        `True` if the segment is strictly within the sector. Otherwise, if
        the segment is either outside the sector or crosses from within the
        sector into outside the sector, then `False` is returned.
    """

    sector_obj = airspace.sectors[sector_name]
    # create a new sector where both the main volumes and conditional volumes
    # are combined.
    if sector_obj.conditional_volume_dict is not None:
        main_sector_volumes = sector_obj.volumes
        cond_sector_volumes = list(sector_obj.conditional_volume_dict.values())
        _volumes = main_sector_volumes + cond_sector_volumes
        sector_obj = Sector(_volumes)

    start_pos = segment[0]
    end_pos = segment[1]
    cps_lat = np.interp(
        np.arange(0, num_interp_positions + 1),
        [0, num_interp_positions],
        [start_pos.lat, end_pos.lat],
    )
    cps_lon = np.interp(
        np.arange(0, num_interp_positions + 1),
        [0, num_interp_positions],
        [start_pos.lon, end_pos.lon],
    )

    in_sector = True
    for lat, lon in zip(cps_lat, cps_lon, strict=True):
        pos = Pos2D(lat=lat, lon=lon)
        if not sector_obj.contains_laterally(pos, epsilon=epsilon):
            in_sector = False
            break

    return in_sector


def get_route_segments_in_sector(
    route: list[str],
    airspace: Airspace,
    sector_name: str,
    start_from: str | tuple[Pos2D, str] | None = None,
    stop_at: str | tuple[Pos2D, str] | None = None,
) -> list[NamedLine]:
    """Get route segments/lines for segments within a sector

    Returns route segments for the given route, consisting of tuples marking
    the start and end in Pos2D.

    Note that for a segment where one fix is within the sector and the other
    is outside the sector, it is still considered as an "in sector" segment.

    Args:
        route: the given route for which to generate the segments
        airspace: the airspace which contains the route.
        start_from: defines the location from where to start the first segment.
            if set as a string, then it defines a fix name on the route,
            from which to start. if set as a tuple it defines a custom
            (non-standard fix) position from which to start and the name of
            the next/forward (standard) fix on the route (i.e., the custom
            position occurs *before* the standard fix). otherwise, if `None`
            is specified, then the entire route is used.
            Defaults to `None` which indicate a start from the first fix in
            the route.
        stop_at: defines the location to stop the last segment. if set as a
            string, then it defines a fix name on the route at which to end.
            if set as tuple, it defines a custom (non-standard fix) position
            at which to end, and the closest previous fix before the custom
            position (i.e., the custom position occurs *after* the standard
            fix).
            Defaults to `None` which indicates a stop at the last fix in
            the route.
        sector_name: defines the name of the sector in the airspace.

    Note: if `stop_at` is set (not None) and the `start_from` appears after
    after `stop_at` in the route, then the returned segment is truncated at
    the `stop_at`.

    Returns:
        list of segments, generated based on the route in the airspace. note
        that each segment/line is defined by its start and end position.

    Raises:
        ValueError: if one of the following is True
            - the fix defined in `start_from` (if set) is not in the route.
            - the fix defined in `stop_at` (if set) is not in the route.
            - the given `stop_at` fix appears after the `start_from` fix in
              the route.
    """

    places = airspace.fixes.places
    route_positions = [places[fix] for fix in route]

    ####### filter by end/stop position.
    if isinstance(stop_at, str):
        try:
            stop_idx = route.index(stop_at)
        except ValueError as err:
            raise ValueError(f"stop fix {stop_at} is not in the route: {route}.") from err

        route = route[: stop_idx + 1]
        route_positions = route_positions[: stop_idx + 1]

    elif isinstance(stop_at, tuple):
        end_position = stop_at[0]
        fix_before_end_pos = stop_at[1]

        try:
            stop_idx = route.index(fix_before_end_pos)
        except ValueError as err:
            raise ValueError(f"stop fix {fix_before_end_pos} is not in the route: {route}.") from err

        route = route[: stop_idx + 1]
        route_positions = route_positions[: stop_idx + 1]

        # make a custom fix at this end position and add it to the
        # end of the route.
        _name = CUSTOM_FIX_AFTER_X.format(fix_before_end_pos)
        route = [*route, _name]
        route_positions = [*route_positions, end_position]
    # else: it should be None; do nothing.

    ####### filter by start position.
    if isinstance(start_from, str):
        # check that `start_from` hasn't been truncated from the route. this
        # could occur if `stop_at` is set and appears before `start_from` in
        # the unprocessed route.
        try:
            idx = route.index(start_from)
        except ValueError as err:
            raise ValueError(
                f"start fix {start_from} is either not in the route: {route} "
                f"or appears after the fix defined `stop_at` {stop_at}."
            ) from err

        route = route[idx:]
        route_positions = route_positions[idx:]

    elif isinstance(start_from, tuple):
        start_position = start_from[0]
        fix_after_start_pos = start_from[1]

        # check that `fix_after_start_pos` hasn't been truncated from the
        # route. this could occur if `stop_at` is set and appears before
        # `start_from` in the unprocessed route.
        try:
            idx = route.index(fix_after_start_pos)
        except ValueError as err:
            raise ValueError(
                f"start fix {fix_after_start_pos} is either not in the "
                f"route: {route} or appears after the fix defined in "
                f"`stop_at` {stop_at}."
            ) from err

        route = route[idx:]
        route_positions = route_positions[idx:]

        # make a custom fix at this start position and add it to the
        # beginning of the route.
        _name = CUSTOM_FIX_BEFORE_X.format(fix_after_start_pos)
        route = [_name, *route]
        route_positions = [start_position, *route_positions]
    # else: it should be None; do nothing.

    ####### generate the segments.
    sector_obj = airspace.sectors[sector_name]
    segments = []
    for i in range(len(route_positions) - 1):
        fix_name = route[i]
        fix_loc = route_positions[i]

        next_fix_name = route[i + 1]
        next_fix_loc = route_positions[i + 1]

        if sector_obj.contains_laterally(fix_loc) or sector_obj.contains_laterally(next_fix_loc):
            segments.append(NamedLine((fix_loc, next_fix_loc), (fix_name, next_fix_name)))

    return segments


def get_route_segments(
    route: list[str],
    airspace: Airspace,
    start_from: str | tuple[Pos2D, str] | None = None,
    stop_at: str | tuple[Pos2D, str] | None = None,
) -> list[NamedLine]:
    """Get route segments/lines

    Returns route segments for the given route, consisting of tuples marking
    the start and end in Pos2D; Used for finding closest segments, finding
    lateral route deviations etc.

    Args:
        route: the given route for which to generate the segments
        airspace: the airspace which contains the route.
        start_from: defines the location from where to start the first segment.
            if set as a string, then it defines a fix name on the route,
            from which to start. if set as a tuple it defines a custom
            (non-standard fix) position from which to start and the name of
            the next/forward (standard) fix on the route (i.e., the custom
            position occurs *before* the standard fix). otherwise, if `None`
            is specified, then the entire route is used.
            Defaults to `None` which indicate a start from the first fix in
            the route.
        stop_at: defines the location to stop the last segment. if set as a
            string, then it defines a fix name on the route at which to end.
            if set as tuple, it defines a custom (non-standard fix) position
            at which to end, and the closest previous fix before the custom
            position (i.e., the custom position occurs *after* the standard
            fix).
            Defaults to `None` which indicates a stop at the last fix in
            the route.

    Note: if `stop_at` is set (not None) and the `start_from` appears after
    after `stop_at` in the route, then the returned segment is truncated at
    the `stop_at`.

    Returns:
        list of segments, generated based on the route in the airspace. note
        that each segment/line is defined by its start and end position.

    Raises:
        ValueError: if one of the following is True
            - the fix defined in `start_from` (if set) is not in the route.
            - the fix defined in `stop_at` (if set) is not in the route.
            - the given `stop_at` fix appears after the `start_from` fix in
              the route.
    """

    places = airspace.fixes.places
    route_positions = [places[fix] for fix in route]

    ####### filter by end/stop position.
    if isinstance(stop_at, str):
        try:
            stop_idx = route.index(stop_at)
        except ValueError as err:
            raise ValueError(f"stop fix {stop_at} is not in the route.") from err

        route = route[: stop_idx + 1]
        route_positions = route_positions[: stop_idx + 1]

    elif isinstance(stop_at, tuple):
        end_position = stop_at[0]
        fix_before_end_pos = stop_at[1]

        try:
            stop_idx = route.index(fix_before_end_pos)
        except ValueError as err:
            raise ValueError(f"stop fix {fix_before_end_pos} is not in the route.") from err

        route = route[: stop_idx + 1]
        route_positions = route_positions[: stop_idx + 1]

        # make a custom fix at this end position and add it to the
        # end of the route.
        _name = CUSTOM_FIX_AFTER_X.format(fix_before_end_pos)
        route = [*route, _name]
        route_positions = [*route_positions, end_position]
    # else: it should be None; do nothing.

    ####### filter by start position.
    if isinstance(start_from, str):
        # check that `start_from` hasn't been truncated from the route. this
        # could occur if `stop_at` is set and appears before `start_from` in
        # the unprocessed route.
        try:
            idx = route.index(start_from)
        except ValueError as err:
            raise ValueError(
                f"start fix {start_from} is either not in the route or appears after the fix defined in `stop_at`."
            ) from err

        route = route[idx:]
        route_positions = route_positions[idx:]

    elif isinstance(start_from, tuple):
        start_position = start_from[0]
        fix_after_start_pos = start_from[1]

        # check that `fix_after_start_pos` hasn't been truncated from the
        # route. this could occur if `stop_at` is set and appears before
        # `start_from` in the unprocessed route.
        try:
            idx = route.index(fix_after_start_pos)
        except ValueError as err:
            raise ValueError(
                f"start fix {fix_after_start_pos} is either not in the route "
                "or appears after the fix defined in `stop_at`."
            ) from err

        route = route[idx:]
        route_positions = route_positions[idx:]

        # make a custom fix at this start position and add it to the
        # beginning of the route.
        _name = CUSTOM_FIX_BEFORE_X.format(fix_after_start_pos)
        route = [_name, *route]
        route_positions = [start_position, *route_positions]
    # else: it should be None; do nothing.

    ####### generate the segments.
    segments = []
    for i in range(len(route_positions) - 1):
        fix_name = route[i]
        fix_loc = route_positions[i]

        next_fix_name = route[i + 1]
        next_fix_loc = route_positions[i + 1]

        segments.append(NamedLine((fix_loc, next_fix_loc), (fix_name, next_fix_name)))

    return segments


def get_centreline_distance(
    position: Pos2D,
    route: list[str],
    airspace: Airspace,
    route_start_position: Pos2D | None = None,
) -> tuple[float, int, float]:
    """Compute a position's distance from the nearest point on a route.

    This is usually used to compute an aircraft's distance from its route
    centreline -- given the aircraft's position and its route segment.

    Compute centreline Distance: Used in metrics as a proxy for leaving
    controlled airspace

    Args:
        position: defines the position for which to compute the distance from
            the defined route.
        route: defines the route -- a list of (standard) fixes. this could be
            the filed route or current route of an aircraft.
        airspace: the airspace which contains the route.
        route_start_position: defines a non-standard start position of the
            route. if set, it means that the route starts from a location
            that is not a standard fix (in the airspace). this is useful
            when the `route` passed is the current route of an aircraft.
            if set then, the it is used as the first position in the route,
            before the first fix (standard) position in the route.

    Returns:
        tuple containing three elements:
        - the centreline distance,
        - direction of turn (deviation) from centreline (encoded as a
          ternary value),
        - track distance from the start of the route to the position's
          approximated (nearest) location on the route.
    """

    start_from = (route_start_position, route[0]) if route_start_position is not None else None

    named_lines = get_route_segments(route, airspace, start_from=start_from)
    segments = [nline.get_line() for nline in named_lines]

    return _get_centreline_distance(position, segments)


# helper function.
def _get_centreline_distance(position: Pos2D, segments: list[Line]) -> tuple[float, int, float]:
    """Compute a position's distance from the nearest point on a route.

    Args:
        position: defines the position for which to compute the distance from
            the defined route.
        segments: the list of segments/lines based on the route. note that
            each segment/line is defined by its start and end position.

    Returns:
        tuple containing three elements:
        - the centreline distance,
        - direction of turn (deviation) from centreline (encoded as a
          ternary value),
        - track distance from the start of the route to the position's
          approximated (nearest) location on the route.
    """

    R = EARTH_RADIUS_IN_METERS * convert.M_TO_NMI  # Earth's radius in nm
    seg_distances = []
    left_right = []
    seg_lengths = []
    along_seg_dist = []
    passed_segments = []
    for idx, segment in enumerate(segments):
        p1, p2 = segment
        seg_length = p1.distance(p2)
        seg_lengths.append(seg_length)
        p3 = position

        bearing_p1_p2 = p1.bearing_to(p2)
        bearing_p1_p3 = p1.bearing_to(p3)
        turn_direction = left_right_check(bearing_p1_p3, bearing_p1_p2)
        bearing_p1_p2 *= convert.DEG_TO_RAD
        bearing_p1_p3 *= convert.DEG_TO_RAD

        dist_p1_p3 = p1.distance(p3)
        delta_h = abs(bearing_p1_p3 - bearing_p1_p2)

        if delta_h > np.pi:
            delta_h = 2 * np.pi - delta_h

        if delta_h > (np.pi / 2):
            # the position is behind this segment
            dxa = dist_p1_p3
            along_seg_dist.append(0)

        else:
            dxt = np.arcsin(np.sin(p1.distance(p3) / R) * np.sin(bearing_p1_p3 - bearing_p1_p2)) * R
            dist_p1_p2 = p1.distance(p2)
            dist_p1_p4 = np.arccos(np.cos(dist_p1_p3 / R) / np.cos(dxt / R)) * R
            p4 = p1.forward(dist_p1_p4, bearing_p1_p2)  # noqa: F841
            if dist_p1_p4 > dist_p1_p2:
                # the position ahead of (has passed) this segment
                dxa = p2.distance(p3)
                along_seg_dist.append(float(dxa))

                if len(passed_segments) == 0:
                    passed_segments.append(idx)
                elif (passed_segments[-1] + 1) == idx:
                    # passed segments is consecutive. the correct logic
                    passed_segments.append(idx)
                else:
                    # passed segments is not consecutive. this is incorrect
                    # such situation may occur due to the geometry of the route
                    # (i.e., if there's a partial u-turn in the future segment
                    # relative to the given position, then the function's logic
                    # indicates that the future segment has been passed. this
                    # is incorrect and manifests as a non-consecutive index in
                    # the list of passed segments. so ignore it.
                    pass
            else:
                dxa = abs(dxt)
                along_seg_dist.append(float(dist_p1_p4))

        seg_distances.append(float(dxa))
        left_right.append(turn_direction)

    # find the segment index of the smallest centreline distance across
    # segments, excluding the segments that is behind the position.
    index_min = int(np.argmin(seg_distances))
    if index_min in passed_segments and index_min != len(segments) - 1:
        # the segment with the smallest distance has been passed;
        # use the next segment
        index_min += 1

    # calc along track dist
    along_track_dist = 0
    for i in range(0, index_min):
        along_track_dist += seg_lengths[i]
    along_track_dist += along_seg_dist[index_min]

    final_turn_dir = left_right[index_min]
    return min(seg_distances), final_turn_dir, float(along_track_dist)


def angle_diff(a: float, b: float) -> float:
    """Compute obtuse angle difference between two angles

    Args:
        a: angle or bearing of point 1 (in degrees).
        b: angle or bearing of point 2 (in degrees).

    Return:
        the computed angle difference (in degrees).
    """
    diff = abs(a - b)
    if diff >= 180:
        diff = 360 - diff
    return diff


def angle_in_range(angle_deg: float, range_deg: tuple[float, float]) -> bool:
    """Check if an angle is within a specified angular range [val1, val2).

    It works regardless of the position of the min/max value in
    the range tuple.

    Args:
        angle_deg: specifies the angle to check (in degrees).
        range_deg: specifies the range as a 2-element tuple (each element is
            a float, in degrees).

    Returns:
        bool, `True` if angle is within the specified range, otherwise,
        `False is returned.

    Raise ValueError for incorrect input when:
        - the two angles in `range_deg` have the same.
        - any angular value (i.e., `angle_deg` or the angles in `range_deg`)
          is not within [0.0, 360.0].
    """

    if not (0.0 <= angle_deg <= 360.0):
        raise ValueError(
            "`angle_deg` should be within [0.0, 360.0]",
        )
    if not (0.0 <= range_deg[0] <= 360.0):
        raise ValueError(
            "`range_deg[0]` should be within [0.0, 360.0]",
        )
    if not (0.0 <= range_deg[1] <= 360.0):
        raise ValueError(
            "`range_deg[1]` should be within [0.0, 360.0]",
        )

    if range_deg[0] == range_deg[1]:
        raise ValueError(
            "elements in `range_deg` cannot be the same",
        )
    if range_deg[0] < range_deg[1]:
        return angle_deg >= range_deg[0] and angle_deg <= range_deg[1]

    # range_deg[0] > range_deg[1]
    return (angle_deg >= range_deg[0] and angle_deg <= 360.0) or (angle_deg >= 0.0 and angle_deg <= range_deg[1])


def project_x_from_range_to_range(
    x: Number,
    source_range: tuple[Number, Number],
    target_range: tuple[Number, Number],
) -> Number:
    """Project value within a source range to its equivalent in a target range

    Note: the ranges are expected to be in ascending order (i.e., [min, max]).
    If any of them is in descending order (i.e., [max, min]), it is rearranged
    ascending order.

    Args:
        x: defines the value to project
        source_range: defines the source range.
        target_range: defines the target range.

    Returns:
        the projected value in the target range.

    Raises:
        ValueError, if the value to project is not within the source range.
    """

    sr = sorted(source_range)
    tr = sorted(target_range)

    if sr[0] <= x <= sr[1]:
        # linear interpolation formula
        a, b = sr
        c, d = tr

        projected_x = c + (((x - a) * (d - c)) / (b - a))

    else:
        raise ValueError(f"Number {x} is not in the source range: {source_range}")

    return projected_x


def at_exit_window(
    aircraft: Aircraft,
    distance_to_exit_along_track: float,
    exit_window: Line | NamedLine,
) -> bool:
    """Check whether the aircraft has arrived at its sector's exit.

    Args:
        aircraft: the aircraft for which to check if it is at the exit window.
        distance_to_exit_along_track: the aircraft distance to the sector's
            exit following the aircraft's track (route).
        exit_window: defines the exit window/segment/line of the aircraft. it
            is specified as a two-element tuple of positions. the line
            specifies the maximum lateral deviation from the aircraft's
            `exit_position`.

    Returns:
        bool. `True` if the aircraft has arrived at (or nearby) the exit
        position and it's within the defined exit window bound, `False`
        otherwise.
    """

    DISTANCE_THRESHOLD = 1.0  # nautical miles

    if distance_to_exit_along_track > DISTANCE_THRESHOLD:
        # not yet at exit position (and window region)
        return False
    aircraft_position = aircraft.pos2d()
    position_1 = exit_window[0]
    position_2 = exit_window[1]
    full_exit_width = position_1.distance(position_2)

    return (
        aircraft_position.distance(position_1) < full_exit_width
        and aircraft_position.distance(position_2) < full_exit_width
    )


def nearest_360_boundary_position(aircraft: Aircraft, sector: Sector) -> tuple[Pos2D, float, float]:
    """Find the point on a sector boundary nearest to an aircraft.

    Looking 360 degrees around an aircraft's current position, find the
    point (laterally) on a sector boundary nearest to an aircraft, and
    compute some metrics.

    Args:
        aircraft: the aircraft for which run the nearest boundary check.
        sector: defines the sector.

    Returns:
        three-element tuple:
        - the first is the lat/lon (Pos2D) position on the sector boundary
          which the aircraft is closest to
        - the second is the distance of the aircraft (position) to the
          sectory boundary position,
        - third is the bearing from the aircraft position to the sector
          boundary position.
    """

    sector_lateral_boundary: Area = sector.boundary()

    aircraft_position = aircraft.pos2d()
    exterior = sector_lateral_boundary.boundary.exterior

    dist_euclid = exterior.project(Point(aircraft_position.lon, aircraft_position.lat))
    nearest_point = sector_lateral_boundary.boundary.exterior.interpolate(dist_euclid)

    nearest_point_latlon = np.array([nearest_point.y, nearest_point.x])  # format: (lat, lon)
    nearest_point_pos2d = Pos2D.from_array(nearest_point_latlon)

    _distance = aircraft_position.distance(nearest_point_pos2d)

    _bearing = aircraft_position.bearing_to(nearest_point_pos2d)

    return nearest_point_pos2d, _distance, _bearing


def nearest_forward_boundary_position(
    aircraft: Aircraft,
    sector: Sector,
    next_fix_pos: Pos2D,
    exit_fix_pos: Pos2D,
    route_distance_to_exit_fix: Number,
    on_heading_forward_distance: Number = 150,
) -> tuple[Pos2D | None, Number | None]:
    """Get intersection between an aircraft's forward pathand a sector boundary

    This takes into account whether the aircraft is route following or on a
    heading.

    Note:
    - if route following, only the line path from the aircraft's current
      position to the next fix is checked. If the path does not intersect
      with the sector boundary, then we assume that the rest of the route
      will continue within the sector until the exit fix location is reached.
      hence, the boundary intersection is set as the exit fix position.
      otherwise, if an intersection location is returned (based on the
      current path to the next fix), it's either the next fix is the exit
      fix or the aircraft may have been issued a route_direct_to action which
      skipped some intermediate fix and the new path goes out of the sector.
    - if on heading, the line is drawn from the aircraft's position based on
      its current heading (with an arbitrarily chosen large distance; 150nm).
      if an intersection exist, this could be a location that does not fit
      into the aircraft's exit fix, or it could be the correct exit fix
      (or within the correct exit fix window). If no intersection exist (
      None returned), it implies that the sector is longer than the 150nm
      distance chosen. Consider calling the function again with an increased
      value passed for `on_heading_forward_distance`.

    Args:
        aircraft: defines the aircraft.
        sector: defines the sector.
        next_fix_pos: defines the location of the aircraft's next fix.
        exit_fix_pos: defines the location of the aircraft's exit fix.
        route_distance_to_exit_fix: defines the aircraft's current track
            distance to its exit fix position. This is used when the
            aircraft is route following and its current segment does not
            intersect with the sector boundary. Hence, the returned distance
            from this function is set to this value.
        on_heading_forward_distance: defines the distance to project the
            aircraft's position if it is on heading. the aircraft's current
            position and its projected future position becomes the line
            path used to assess whether a sector intersection exists.
            Defaults to 150 nauticla miles.

    Returns:
        the intersection location or `None` if no intersection was found.
    """

    ac_pos = aircraft.pos2d()

    if aircraft.on_route:
        trajectory = [ac_pos, next_fix_pos]
        pos, dist = nearest_traj_boundary_position(trajectory, sector)

        if pos is None:
            # indicates that the aircraft current route segment does not go
            # out of the sector (this could happen if clearances are issued to
            # skip fixes via route_direct_to command). therefore, if it
            # continues the follow route, the boundary point should be the
            # exit fix.
            pos = exit_fix_pos

            # in this scenario, the distance to the exit fix position is
            # the track/route distance of the aircraft from its current
            # position to the exit fix position.
            dist = route_distance_to_exit_fix

    else:
        future_pos = ac_pos.forward(dist=on_heading_forward_distance, heading=aircraft.heading)
        trajectory = [ac_pos, future_pos]

        pos, dist = nearest_traj_boundary_position(trajectory, sector)

    return pos, dist


def nearest_traj_boundary_position(
    trajectory: list[Pos2D | Pos3D | Pos4D], sector: Sector
) -> tuple[Pos2D | None, Number | None]:
    """Locate the intersection between a trajectory and a sector boundary

    Args:
        trajectory: defines a list of control points.
        sector: defines the sector.

    Returns:
        a two-element tuple
        - the intersection location or `None` if no intersection was found.
        - the distance from the first/start location of the trajectory to
          the nearest intersection location.

        Note, if no intersection is found, then the tuple (None, None) is
        returned.
    """
    from shapely import LineString, MultiPoint, Point

    sector_lateral_boundary: Area = sector.boundary()

    # shapely LineString intersections
    ret, _ = sector_lateral_boundary.intersection(trajectory)

    if isinstance(ret, Point):
        intersections = np.concatenate([arr.tolist() for arr in ret.xy]).reshape(1, -1)

    elif isinstance(ret, MultiPoint):
        intersections = np.stack([np.concatenate([arr.tolist() for arr in p.xy]) for p in ret.geoms])

    elif isinstance(ret, LineString):
        intersections = np.stack([arr.tolist() for arr in ret.xy]).T

    else:
        raise ValueError(f"Intersection output of type: {type(ret)} is not supported.")

    if intersections.shape[0] == 0:
        # no intersections
        pos = None
        dist = None

    elif intersections.shape[0] == 1:
        # only one intersection.
        lat = intersections[0, 1]
        lon = intersections[0, 0]
        pos = Pos2D(lat=lat, lon=lon)

        # distance from the first location in the trajectory to the
        # intersection location
        dist = trajectory[0].distance(pos)

    else:
        # more than one intersection.
        intersection_locations = [
            Pos2D(lat=intersections[i, 1], lon=intersections[i, 0]) for i in range(intersections.shape[0])
        ]

        # get the intersection location closest to the starting location
        # of the trajectory
        source_location = trajectory[0]

        distances = np.asarray([source_location.distance(_location) for _location in intersection_locations])

        dist = distances.min().item()
        idx = distances.argmin().item()
        pos = intersection_locations[idx]

    return pos, dist


def get_quadrant(angle: float) -> Quadrant:
    """Compute the quadrant to which an angle belongs

    Args:
        angle: defines the angle for which to compute the quadrant

    Returns:
        the quadrant to which the angle belongs.
    """

    assert 0.0 <= angle <= 360.0, "`angle` should be in range [0, 360]"

    if 0.0 <= angle <= 90.0:
        return Quadrant.Q1

    if 90.0 < angle <= 180.0:
        return Quadrant.Q2

    if 180.0 < angle <= 270.0:
        return Quadrant.Q3

    return Quadrant.Q4


def quadrant_range(angle: float) -> tuple[float, float]:
    """Computes the lower and upper quadrant bounds of an angle

    Args:
        angle (float): defines the angle for which to compute the lower
            and upper quadrant bounds

    Returns:
        tuple, which contains the lower and upper quadrant bounds of the
        angle.
    """

    return QuadrantBound[get_quadrant(angle)]


def passed_location(location: Pos2D, aircraft_cps: list[Pos4D]) -> bool:
    """Determine whether an aircraft has passed a given location/position.

    Args:
        location: the target location to check against.
        aircraft_cps: defines the aircraft future trajectory as list of
            points/positions.

    Returns:
        bool, True if the position has been passed. False, otherwise.
    """

    d0 = aircraft_cps[0].distance(location)
    d1 = aircraft_cps[1].distance(location)

    return d0 <= d1


############ fixes and position checks in a given sector.
def filter_positions_in_sector(
    positions: list[Pos2D],
    airspace: Airspace,
    sector_name: str,
    epsilon: float = 1e-10,
) -> list[Pos2D]:
    """Filter a list of 2d positions: keep only positions in a given sector.

    Args:
        positions: the list of 2d positions to process.
        airspace: defines the airspace which contains the given sector.
        sector (str): defines the name of the sector in the airspace. used
            to retrieve the sector that the positions are checked against.
        epsilon (float): defines the tolerance value.

    Returns:
        a list of the positions that are contained in the sector. note that
        an empty list is returned if None of the positions are in the sector.
    """

    sector_obj = airspace.sectors[sector_name]
    return [p for p in positions if sector_obj.contains_laterally(p, epsilon=epsilon)]


def positions_in_sector(
    positions: list[Pos2D],
    airspace: Airspace,
    sector_name: str,
    epsilon: float = 1e-10,
) -> list[bool]:
    """Assess a list of 2d positions to check whether they're in a sector.

    Args:
        positions: the list of 2d positions to process.
        airspace: defines the airspace which contains the given sector.
        sector (str): defines the name of the sector in the airspace.
            used to retrieve the sector that the fixes are checked against.
        epsilon (float): defines the tolerance value.

    Returns:
        a list of the booleans that indicate the status of a corresponding
        position at the given index.
    """

    sector_obj = airspace.sectors[sector_name]
    return [sector_obj.contains_laterally(p, epsilon=epsilon) for p in positions]


def filter_fixes_in_sector(
    fixes: list[str],
    airspace: Airspace,
    sector_name: str,
    epsilon: float = 1e-10,
) -> list[str]:
    """Filter a list of fixes: keep fixes located within a given sector.

    Args:
        fixes: the list of fixes positions to process.
        airspace: defines the airspace which contains the given sector.
        sector (str): defines the name of the sector in the airspace. used
            to retrieve the sector that the positions are checked against.
        epsilon (float): defines the tolerance value.

    Returns:
        a list of the fixes that are contained in the sector. note that
        an empty list is returned if None of the fixes are in the sector.
    """

    places = airspace.fixes.places
    positions = [places[fix] for fix in fixes]
    sector_obj = airspace.sectors[sector_name]

    filtered_fixes = []
    for idx, fix in enumerate(fixes):
        p = positions[idx]

        if sector_obj.contains_laterally(p, epsilon=epsilon):
            filtered_fixes.append(fix)

    return filtered_fixes


def fixes_in_sector(
    fixes: list[str],
    airspace: Airspace,
    sector_name: str,
    epsilon: float = 1e-10,
) -> list[bool]:
    """Assess a list of fixes to check whether they're in a sector.

    Args:
        fixes: the list of fixes positions to process.
        airspace: defines the airspace which contains the given sector.
        sector (str): defines the name of the sector in the airspace. used
            to retrieve the sector that the positions are checked against.
        epsilon (float): defines the tolerance value.

    Returns:
        a list of the booleans that indicate the status of a corresponding
        fix at the given index.
    """

    places = airspace.fixes.places
    positions = [places[fix] for fix in fixes]
    sector_obj = airspace.sectors[sector_name]

    return [sector_obj.contains_laterally(p, epsilon=epsilon) for p in positions]


def filter_route_fixes_in_sector_by_coordination(
    query_fixes: list[str],
    filed_route: list[str],
    entry_fix: str,
    exit_fix: str,
) -> tuple[str]:
    """Filter route fixes: keep fixes located within a given sector.

    Args:
        query_fixes: defines the list of fixes to check whether they are in
            the sector.
        filed_route: defines the filed roue (flight plan) of an aircraft.
        entry_fix: defines the entry fix of the aircraft.
        exit_fix: defines the exit fix of the aircraft.

    Returns:
        a tuple of the fixes that are contained in the sector. note that
        an empty tuple is returned if None of the fixes are in the sector.

    Raises:
        - ValueError, if the entry_fix or the exit_fix is not found in the
          filed_route.
    """

    start_idx = filed_route.index(entry_fix)
    stop_idx = filed_route.index(exit_fix) + 1
    in_sector_fixes = filed_route[start_idx:stop_idx]

    return tuple(fix for fix in query_fixes if fix in in_sector_fixes)


############ line and path intersections
def line_intersection(line_1: Line | NamedLine, line_2: Line | NamedLine) -> tuple[LineIntersection, Pos2D | None]:
    """Determine whether two lines/segments intersect and the location.

    Note that potential future intersection (if the lines are extended) is
    also taken into account.

    Args:
        line_1: defines the start and end positions of the first line.
        line_2: defines the start and end positions of the second line.

    Returns:
        the result of the intersection check, a two-element tuple,
        - a ternary integer value, defined by the possible values below:
            0, if there exist no intersection between the lines
               (even if they are extended)
            1, if there exist an intersection between the lines.
            2, if there exist no intersection between the lines, but an
               intersection can occur if the lines are extended (probable).
        - the location of the intersection if it exists. otherwise, it is set
          to `None`.
    """

    t, u = _intersection(
        line_1[0].location,
        line_1[1].location,
        line_2[0].location,
        line_2[1].location,
    )

    ret = None
    location = None

    if t == np.inf and u == np.inf:
        # no intersection
        ret = LineIntersection.NONE
        location = None

    elif (0 <= t <= 1) and (0 <= u <= 1):
        ret = LineIntersection.EXIST
        c = line_2[0].location
        d = line_2[1].location
        location = c + u * (d - c)
        location = Pos2D.from_array(location)

    else:
        ret = LineIntersection.PROBABLE
        location = None

    return (ret, location)


def line_intersection_any(target_line: Line | NamedLine, other_lines: list[Line | NamedLine]) -> bool:
    """Determine whether a line intersect with any line in a list of lines.

    Args:
        target_line: defines the start and end positions of the target line.
        other_lines: defines a list of other lines to check the target line
            against.

    Returns:
        bool, `True` if the target line intersects with at least one line
        in the batch. otherwise, `False` is returned.
    """

    status = False
    for other_line in other_lines:
        intersect_status, _ = line_intersection(target_line, other_line)
        if intersect_status == LineIntersection.EXIST:
            status = True
            break

    return status


def line_intersection_batch(
    target_line: Line | NamedLine, other_lines: list[Line | NamedLine]
) -> list[tuple[LineIntersection, Pos2D | None]]:
    """Batch version of `line_intersection(...)`.

    Compares a target line against a list of other lines to check if there
    is an intersection with each line.

    Args:
        target_line: defines the start and end positions of the target line.
        other_lines: defines a list of other lines to check the target line
            against.

    Returns:
        list of intersection status, one per line in `other_line`. each
        intersection status defined in the return type of `line_intersection`
    """

    return [line_intersection(target_line, other_line) for other_line in other_lines]


def path_intersection(
    lines_1: list[Line | NamedLine],
    lines_2: list[Line | NamedLine],
) -> tuple[bool, Pos2D | None, tuple[int, int] | None]:
    """Get the first intersecting location between two paths.

    A path is a collection of segments/lines.
    Note: this is a generalised version of `line_intersection_any(...)`
    where there are multiple target lines.

    Args:
        lines_1: defines the list of lines for the first path.
        lines_2: defines the list of lines for the second path.

    Returns:
        a three-element tuple which contains:
        - the intersection status: `True` if at least one intersection exist.
          otherwise, `False` is returned.
        - the intersection location if at least one intersection exists (i.e,
          when the intersection status is True).
          otherwise None is returned.
        - a two-element tuple which contains the indexes of the segment
          within each path where the intersection occurs (if an intersection
          exists; otherwise, None is returned).
    """

    intersection_exist = False
    intersection_location = None
    found_indices = None
    for i, line_1 in enumerate(lines_1):
        for j, line_2 in enumerate(lines_2):
            intersect_status, location = line_intersection(line_1, line_2)

            if intersect_status == LineIntersection.EXIST:
                intersection_exist = True
                intersection_location = location
                found_indices = (i, j)
                break

        if intersection_exist:
            break

    return intersection_exist, intersection_location, found_indices


############ line parallel, colinearity and overlap.
def line_parallel(line_1: Line | NamedLine, line_2: Line | NamedLine, epsilon: float = 1e-2) -> bool:
    """Determine whether two lines/segments are parallel.

    Determines the parallelism by
    (i)  checking whether two lines have the same gradient/slope, approximated
         using the angular equality of the lines. for each line, the angle is
         computed as bearing from the start to the finish position of the line
         line.
    (ii) and, checking whether the lines are *latitudinally* separated.

    Args:
        line_1: defines the start and end positions of the first line.
        line_2: defines the start and end positions of the second line.
        epsilon: defines the threshold/tolerance for comparing the lines.
            if the difference between the angles of the lines is less than
            epsilon, then they're considered parallel.

    Returns:
        bool, the parallel status which is True if both lines are parallel.
        Otherwise, False is returned.
    """

    # step 1: check that the two lines have the same (or similar) angles.
    line_1_angle = line_1[0].bearing_to(line_1[1])
    line_2_angle = line_2[0].bearing_to(line_2[1])
    angular_diff = angle_diff(line_1_angle, line_2_angle)

    line_2_angle_reverse = angle_diff(line_2_angle, 180)
    angular_diff2 = angle_diff(line_1_angle, line_2_angle_reverse)

    if angular_diff <= epsilon or angular_diff2 <= epsilon:
        # step 2: check for latitudinal separation
        bearing = line_1[0].bearing_to(line_2[0])

        parallel_status = angle_diff(bearing, line_1_angle) > epsilon

    else:
        parallel_status = False

    return parallel_status


def line_colinear(line_1: Line | NamedLine, line_2: Line | NamedLine, epsilon: float = 1e-2) -> bool:
    """Determine whether two lines/segments are colinear.

    Determines the colinearity by:
    (i)  checking whether two lines have the same gradient/slope, approximated
         using the angular equality of the lines. for each line, the angle is
         computed as bearing from the start to the finish position of the line
         line.
    (ii) and, checking whether the lines are *longitudinally* separated or
         directly overlapping.

    Note: longitudinally separated lines will overlap if the lines are
    extended to infinity.

    Args:
        line_1: defines the start and end positions of the first line.
        line_2: defines the start and end positions of the second line.
        epsilon: defines the threshold/tolerance for comparing the lines.
            if the difference between the angles of the lines is less than
            epsilon, then they're considered parallel.

    Returns:
        bool, the colinear status which is True if both lines are colinear.
        Otherwise, False is returned.
    """

    # step 1: check that the two lines have the same (or similar) angles.
    line_1_angle = line_1[0].bearing_to(line_1[1])
    line_2_angle = line_2[0].bearing_to(line_2[1])
    angular_diff = abs(line_1_angle - line_2_angle)

    line_2_angle_reverse = (line_2_angle + 180) % 360
    angular_diff_reverse = abs(line_1_angle - line_2_angle_reverse)

    if angular_diff <= epsilon or angular_diff_reverse <= epsilon:
        # step 2: check for longitudinal separation or overlap
        bearing = line_1[0].bearing_to(line_2[0])

        forward_reference = line_1_angle
        reverse_reference = (forward_reference + 180) % 360
        colinear_status = abs(bearing - forward_reference) <= epsilon or abs(bearing - reverse_reference) <= epsilon

    else:
        colinear_status = False

    return colinear_status


def line_overlap(line_1: Line | NamedLine, line_2: Line | NamedLine, epsilon: float = 1e-2) -> bool:
    """Determine whether two colinear lines/segments are directly overlapping.

    Args:
        line_1: defines the start and end positions of the first line.
        line_2: defines the start and end positions of the second line.
        epsilon: defines the threshold/tolerance for comparing overlap.

    Returns:
        bool, the overlapping status which is True if both lines colinear.
        and they overlap given the current length of the lines. Otherwise,
        False is returned.
    """

    # step 1: check that both lines are colinear
    if not line_colinear(line_1, line_2, epsilon):
        # raise Exception("The lines are not colinear.")
        return False

    # step 2: check for overlap given the current length of both lines
    # (without any further extensions to them).
    angle_l1_start_l2_start = line_1[0].bearing_to(line_2[0])
    angle_l1_end_l2_start = line_1[1].bearing_to(line_2[0])

    angle_l1_start_l2_end = line_1[0].bearing_to(line_2[1])
    angle_l1_end_l2_end = line_1[1].bearing_to(line_2[1])

    fwd_ref = line_1[0].bearing_to(line_1[1])  # forward reference
    rev_ref = (fwd_ref + 180) % 360  # reverse reference

    # forward reference check: line 1 start, end to line 2 start
    f_l1_start_l2_start = abs(angle_l1_start_l2_start - fwd_ref) <= epsilon
    f_l1_end_l2_start = abs(angle_l1_end_l2_start - fwd_ref) <= epsilon

    # forward reference check: line 1 start, end to line 2 end
    f_l1_start_l2_end = abs(angle_l1_start_l2_end - fwd_ref) <= epsilon
    f_l1_end_l2_end = abs(angle_l1_end_l2_end - fwd_ref) <= epsilon

    # reverse reference check: line 1 start, end to line 2 start
    r_l1_start_l2_start = abs(angle_l1_start_l2_start - rev_ref) <= epsilon
    r_l1_end_l2_start = abs(angle_l1_end_l2_start - rev_ref) <= epsilon

    # reverse reference check: line 1 start, end to line 2 end
    r_l1_start_l2_end = abs(angle_l1_start_l2_end - rev_ref) <= epsilon
    r_l1_end_l2_end = abs(angle_l1_end_l2_end - rev_ref) <= epsilon

    no_overlap = (f_l1_start_l2_start and f_l1_end_l2_start and f_l1_start_l2_end and f_l1_end_l2_end) or (
        r_l1_start_l2_start and r_l1_end_l2_start and r_l1_start_l2_end and r_l1_end_l2_end
    )

    return not no_overlap


def path_overlap(
    lines_1: list[Line | NamedLine],
    lines_2: list[Line | NamedLine],
    epsilon: float = 1e-2,
) -> tuple[bool, tuple[int, int] | None]:
    """Determine whether two paths overlap at any line/segment.

    A path is a collection of segments/lines.

    Args:
        lines_1: defines the list of lines for the first path.
        lines_2: defines the list of lines for the second path.
        epsilon: defines the threshold/tolerance for comparing overlap.

    Returns:
        a two-element tuple which contains:
        - boolean that indicates whether at least one overlap exists.
        - a two-element tuple which contains the indexes of the segment
          within each path where the overlap occurs.
    """

    overlap_status = False
    found_indices = None

    for i, line_1 in enumerate(lines_1):
        for j, line_2 in enumerate(lines_2):
            if line_overlap(line_1, line_2, epsilon):
                overlap_status = True
                found_indices = (i, j)
                break

        if overlap_status:
            break

    return overlap_status, found_indices
