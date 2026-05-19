from __future__ import annotations

import math
import typing

import numpy as np

if typing.TYPE_CHECKING:
    from bluebird_dt.core.airspace import Airspace
    from bluebird_dt.core.pos2d import Pos2D
    from bluebird_dt.core.sector import Sector


from bluebird_dt.utility.constants import R_E_WGS84


def centre(a: Pos2D, b: Pos2D) -> Pos2D:
    """
    Find the centre of two locations.

    Parameters
    ----------
    a: Pos2D
        A point with latitude and longitude.
    b: Pos2D
        A point with latitude and longitude.

    Returns
    ----------
    Pos2D
    """
    from bluebird_dt.core.pos2d import Pos2D

    lat = (a.lat + b.lat) * 0.5
    lon = (a.lon + b.lon) * 0.5

    return Pos2D(lat, lon)


def rotate(pos: Pos2D, theta: float) -> Pos2D:
    """
    Rotate the given position around the origin by the given angle [rad].
    Parameters
    ----------
    a: Pos2D
        Point with latitude and longitude.
    theta: float
        Angle in radians.
    Returns
    ----------
    Pos2D
    """
    from bluebird_dt.core.pos2d import Pos2D

    theta *= -np.pi / 180.0

    lat = (pos.lat * np.cos(theta)) - (pos.lon * np.sin(theta))
    lon = (pos.lat * np.sin(theta)) + (pos.lon * np.cos(theta))

    return Pos2D(lat, lon)


def rotate_about(pos: Pos2D, theta: float, pivot: Pos2D) -> Pos2D:
    """
    Rotate the given position around a pivot by the given angle [rad].

    Parameters
    ----------
    a: Pos2D
        Point with latitude and longitude.
    theta: float
        Angle in radians.
    pivot: Pos2D
        Pivot point with latitude and longitude.

    Returns
    ----------
    Pos2D
    """
    from bluebird_dt.core.pos2d import Pos2D

    theta *= -np.pi / 180.0

    lat = (((pos.lat - pivot.lat) * np.cos(theta)) - ((pos.lon - pivot.lon) * np.sin(theta))) + pivot.lat
    lon = (((pos.lat - pivot.lat) * np.sin(theta)) + ((pos.lon - pivot.lon) * np.cos(theta))) + pivot.lon

    return Pos2D(lat, lon)


def line_intersection(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    c: np.ndarray | list[float],
    d: np.ndarray | list[float],
    equal_grad_tol: float = 1e-10,
) -> tuple[float, float]:
    """
    Returns how far along the line that goes though a->b the intersection
    with a line that goes through c->d occurs.
    Returns two values, `t` and `u`, which define the intersection as:
        1. a + t * (b - a)
        2. t * b + (1 - t) * a
        3. c + u * (d - c)
        4. u * b + (1 - u)
    - if t and u are both within [0, 1] then the line segments a->b and c->d
      intersect one another
    - if only t is within [0, 1] then, if the line c->d is extended,
      it will intersect with the line segment a->b
    - if only u is within [0, 1] then, if the line a->b is extended,
      it will intersect with the line segment c->d
    If the lines are parallel, i.e. they have the same gradient, the method
    returns np.inf, np.inf
    Parameters
    ----------
    a, b: np.ndarray, shape (2, )
        Two ends of the first line segment in the form [lat, lon] each
    c, d: np.ndarray, shape (2, )
        Two ends of the second line segment in the form [lat, lon] each
    equal_grad_tol: float, default = 1e-10
        If the gradient of a -> b and c -> d has an absolute difference of less
        than this value they are considered equal.
    Returns
    ----------
    Tuple[float, float]
        `t` and `u` values respectively, which can be used to compute the
        intersection point as described above (e.g., as `c + u * (d - c)`)
    """

    # line 1: a -> b = (x1, y1) -> (x2, y2)
    # line 2: c -> d = (x3, y3) -> (x4, y4)
    x1, y1 = a
    x2, y2 = b
    x3, y3 = c
    x4, y4 = d

    # first, check if the gradient of the lines are equal, if so they either
    # overlap or never intersect
    ab_dx, ab_dy = (x2 - x1), (y2 - y1)
    cd_dx, cd_dy = (x4 - x3), (y4 - y3)

    if ab_dx == 0 and cd_dx == 0:
        equal_grad = True
    elif (ab_dx == 0 and cd_dx != 0) or (ab_dx != 0 and cd_dx == 0):
        equal_grad = False
    else:
        grad_ab = ab_dy / ab_dx
        grad_cd = cd_dy / cd_dx

        equal_grad = np.abs(grad_ab - grad_cd) < equal_grad_tol

    # if the gradient's aren't equal, they must intersect somewhere along their
    # total lines.
    if not equal_grad:
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        # the denominator can only be zero at this point if a == c and b == d
        if denom != 0:
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom

            return t, u

    # they must have the same gradient
    return np.inf, np.inf


def find_all_boundary_intersections(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    airspace_or_sector_or_area: Airspace | Sector | Area,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """
    Find all intersections between the Airspace/Sector/Area boundary and a line a->b.

    Parameters
    ----------
    a, b: np.ndarray, shape (2, )
        Two ends of the line segment in the form [lat, lon] each.

    Returns
    ----------
    List[Tuple[np.ndarray, np.ndarray, float]]
        List of intersection points given in terms of [c, d, u] values.
        - c, d are points [lat, lon] representing consecutive coordinates of the Airspace border
        - u is a float within [0, 1]
        - the line a->b intersects c->d at the location: c + u * (d - c)
        - see `line_intersection()` function for more detail
    """
    from bluebird_dt.core.area import Area # avoid circular import

    # extract the coordinates of the polygon -- note that the
    # attribute .xy consists of two rows: lons and lats
    if isinstance(airspace_or_sector_or_area, Area):
        v_lons, v_lats = airspace_or_sector_or_area.boundary.exterior.coords.xy  # type: ignore
    else:
        v_lons, v_lats = airspace_or_sector_or_area.boundary().boundary.exterior.coords.xy  # type: ignore

    intersections: list[tuple[np.ndarray, np.ndarray, float]] = []

    # for each pair of consecutive coordinates of the polygon's border
    for c_lon, c_lat, d_lon, d_lat in zip(v_lons[:-1], v_lats[:-1], v_lons[1:], v_lats[1:], strict=False):
        # find if the line ab = a -> b intersects the boundary line
        # cd = (c1_lat, c1_lon) -> (c2_lat, c2_lon)
        c = np.array([c_lat, c_lon])
        d = np.array([d_lat, d_lon])

        t, u = line_intersection(a, b, c, d)

        # if both values are not in [0, 1], then they do not intersect
        if not ((0 <= t <= 1) and (0 <= u <= 1)):
            continue

        # otherwise the line ab intersects cd at the location: c + u * (d - c)
        intersections.append((c, d, u))

    return intersections


def nearest_point_on_line_segment(
    Q: np.ndarray, a: np.ndarray, b: np.ndarray, truncate_to_segment: bool = True
) -> np.ndarray:
    """
    Calculates nearest point on the line segment a -> b for each query point
    in Q.

    Parameters
    ----------
    Q : np.ndarray, shape (N, 2) or (2, )
        Point(s) from which to find the nearest points on the line segment.
    a, b: np.ndarray, shape (2, )
        Two ends of the line segment
    truncate_to_segment: bool, default is True
        Whether to clip to nearest point on the segment (i.e. in [0, 1])

    Returns
    ----------
    np.ndarray, shape (N, 2)
        Closest point on the line segment to each of the N locations.
    """

    if Q.ndim == 1:
        Q = np.reshape(Q, (1, 2))

    # the vector b -> a and the vectors Q -> a
    ab = np.subtract(b, a)
    Qa = np.subtract(Q, a)

    # project Q onto the line that goes though b -> a, such that they are
    # defined as a + alpha * (b - a)
    alphas = ab @ Qa.T / (ab @ ab)

    if truncate_to_segment:
        # clip to nearest point on the segment (i.e. in [0, 1])
        alphas = np.clip(alphas, 0, 1)

    # projected points such that Q[i] is projected to P[i]
    return a + alphas[:, np.newaxis] * ab


def nearest_point_on_3D_rectangle(Q: np.ndarray, origin: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Calculates the nearest position on the face of a rectangle defined in 3D
    for an arbitrary number of 3D query points (Q). The rectangle is defined
    as an origin coordinate and the two corners of the rectangle that connect
    to it. Graphical example:
        D ---- C    origin: A, p1: B, p2: D
        |      |    origin: B, p1: C, p2: A
        |      |    origin: C, p1: D, p2: B
        A------B    origin: D, p1: A, p2: C

    Parameters
    ----------
    Q : np.ndarray, shape (N, 3) or (3, )
        Points from which to find the nearest points on the rectangle face.
    origin: np.ndarray, shape (3, )
        3D Origin of the polygon face
    p1, p2: np.ndarray, shape (3, )
        Lower-right (p1) and upper-left (p2) points of the polygon face.
        The lines from origin -> p1 and origin -> p2 define the lower-left
        two edges of the rectangle.

    Returns
    ----------
    np.ndarray, shape (N, 3)
        Closest point on the 3D rectangle to each of the N locations.

    Notes
    ----------
        Based on: https://stackoverflow.com/a/44824522/2161490
    """
    if Q.ndim == 1:
        Q = np.reshape(Q, (1, 3))

    # turn the points into vectors coming from the origin
    vx = np.subtract(p1, origin)
    vy = np.subtract(p2, origin)
    vQ = np.subtract(Q, origin[np.newaxis, :])

    # project Q onto each vector. the calculated amount reflects how
    # 'far along' the vector Q is, where values of [0, 1] are somewhere on the
    # line defined by the vector, and values outside of this range are not on
    # the line (i.e. outside of the rectangle)
    tx = np.dot(vQ, vx) / np.dot(vx, vx)
    ty = np.dot(vQ, vy) / np.dot(vy, vy)

    # clip the projection to the rectangle (i.e., the range [0, 1]), so that we
    # can find the nearest point within the rectangle to Q
    tx = np.clip(tx, 0.0, 1.0)
    ty = np.clip(ty, 0.0, 1.0)

    # project back onto the original coordinate system.
    P = tx[:, np.newaxis] * vx[np.newaxis, :]
    P += ty[:, np.newaxis] * vy[np.newaxis, :]
    P += origin[np.newaxis, :]

    return P


def get_perpendicular_line(
    start_pos: Pos2D,
    end_pos: Pos2D,
    airspace: Airspace | None = None,
    target_sector: str | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Given a path start_pos -> end_pos, return perpendicular line centred on end_pos.
    If Airspace is provided, ensure the line is within its boundary.
    Note that this function assumes Cartesian coordinates - it will not be precise in
    non-Cartesian projections (though may be close enough for some purposes).

    Parameters
    ----------
    start_pos: Pos2D
        2D start position of path.
    end_pos: Pos2D
        2D end position of path.
    airspace: optional Airspace
        Airspace to truncate the line to.
    target_sector: str, optional
        Sector to truncate perpendicular line to.

    Returns
    ----------
    Tuple(np.ndarray, np.ndarray, float)
        tuple of (c, d, u), defining line c -> d:  c + u * (d - c)
    """

    dlat = end_pos.lat - start_pos.lat
    dlon = end_pos.lon - start_pos.lon

    if dlat == 0.0 and dlon == 0.0:
        raise ValueError("start and end points must be different")

    # Perpendicular direction to (dlat, dlon) is (-dlon, dlat).
    # Normalise to get a unit vector, so that output line has length 1
    perp = np.array([-dlon, dlat])
    perp = perp / np.linalg.norm(perp)

    end_point = np.array([end_pos.lat, end_pos.lon])
    c = end_point - 0.5 * perp
    d = end_point + 0.5 * perp
    u = 0.5

    if airspace is not None:
        # we need to extend the points to the be outside the
        # sector to find both points on the line that intersects c and d
        # at the sector boundary (so alpha = 0 will [at most] hit the
        # sector boundary at c, and alpha = 1 at d).

        # we can do this by calculating the intersection of the line that
        # goes through c and d to each of the boundary polygon edges
        v_lons, v_lats = airspace.boundary(target_sector).boundary.exterior.coords.xy  # type: ignore
        intersections = []

        # for each pair of consecutive coordinates (a to b) of the polygon
        for a_lon, a_lat, b_lon, b_lat in zip(v_lons[:-1], v_lats[:-1], v_lons[1:], v_lats[1:], strict=False):
            # find the line that goes through c and d and intersects a->b
            a = np.array([a_lat, a_lon])
            b = np.array([b_lat, b_lon])

            t, v = line_intersection(c, d, a, b)

            # edge case: the lines are parallel, which means that the
            # fix must lie on the sector boundary. Therefore, the fix's
            # location is the intersection point, i.e. it is located
            # at c + 0.5 * (d - c)
            if not np.isfinite(t) and not np.isfinite(v):
                t = 0.5

            # the point of intersection will occur along a->b at
            # a + v * (b - a), if and only if v is in [0, 1]
            # see geometry.line_intersection() docstring for more info
            # otherwise we skip
            elif not (0 <= v <= 1):
                continue

            # store the intersect point along the line that goes through
            # c->d
            intersections.append(t)

        # finally, we could have the case where there are >2 intersections
        # due to a non-compact sector shape (i.e. extending one direction
        # could intersect more than once). so, find the two amounts that
        # are the closest to the midpoint (0.5), either side, i.e. the one
        # the least larger than 0.5 and the one the least smaller than 0.5
        # note that this only needs to be carried out if 0.5 is not in
        # the list of intersections, otherwise we can carry on without
        # adjusting the intersection location parametrisation
        intersections = np.asarray(intersections, dtype="float")
        intersections = np.unique(intersections)

        # first, check to see if the midpoint is a valid intersection
        # and select it if so
        if 0.5 in intersections:
            u = 0.5

        # check if there is only one intersection, in which case we
        # use it because we have nothing else to compare to
        elif len(intersections) == 1:
            u = intersections[0]

        # otherwise we can do this by subtracting the mid point (0.5)
        # and finding the smallest positive value, this will be the one
        # closest to 0.5 that is larger than it. we can repeat to find
        # the least negative value, this will be closest to 0.5 but
        # smaller than it
        else:
            # make values <= 0.5 infinite, so the smallest value left
            # will be that of the smallest one that is greater than 0.5
            isubhigh = np.copy(intersections)
            isubhigh[isubhigh <= 0.5] = np.inf
            tmax = intersections[np.argmin(isubhigh)]

            # repeat in the other direction
            isublow = np.copy(intersections)
            isublow[isublow >= 0.5] = -np.inf
            tmin = intersections[np.argmax(isublow)]

            # create the new points and readjust where the midpoint is
            newc = c + tmin * np.subtract(d, c)
            newd = c + tmax * np.subtract(d, c)
            newu = (0.5 - tmin) / (tmax - tmin)

            c, d, u = newc, newd, newu

    return (c, d, u)


def haversine(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    EARTH_RADIUS_IN_METERS: float = R_E_WGS84,
) -> float:
    """
    Great circle distance between two points on Earth using Haversine formula.

    Notes:
    - Earth radius taken from: https://en.wikipedia.org/wiki/World_Geodetic_System#WGS_84
    - Formulae taken from: https://www.movable-type.co.uk/scripts/latlong.html

    Parameters
    ----------
    lat1: float
        Latitude of first point.
    lon1: float
        Longitude of first point.
    lat2: float
        Latitude of second point.
    lon2: float
        Longitude of second point.
    EARTH_RADIUS_IN_METERS: float
        The Earth's radius in meters, as per WGS84 coordinate system.

    Returns
    ----------
    float
        Great circle distance between the two points.
    """

    PIOVER180 = np.pi / 180

    φ1 = lat1 * PIOVER180
    φ2 = lat2 * PIOVER180
    Δφ = (lat2 - lat1) * PIOVER180
    Δλ = (lon2 - lon1) * PIOVER180

    half_Δφ_sin_sq = math.sin(Δφ / 2) ** 2
    half_Δλ_sin_sq = math.sin(Δλ / 2) ** 2

    a = half_Δφ_sin_sq + (math.cos(φ1) * math.cos(φ2) * half_Δλ_sin_sq)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return EARTH_RADIUS_IN_METERS * c


def points_same_side_of_line(
    p: np.ndarray,
    q: np.ndarray,
    line_coords: tuple[np.ndarray, np.ndarray],
) -> bool:
    """
    Check if p and q are on the same side of the line defined by line_coords.

    Note that it is assumed that p, q, and the line coords are all 2D positions in [lat, lon] coordinates.

    Parameters
    ----------
    p: np.ndarray
        Location in [lat, lon] format to check if is on the same side of the line as q.
        If p and q are on different sides, check if p is below distance threshold from the line.
    q: np.ndarray
        Location in [lat, lon] format to check if it is on the same side of the line as p.
    line_coords: Tuple[np.ndarray, np.ndarray]
        Tuple of two locations in [lat, lon] format that define end points of the line.

    Returns
    ----------
    bool
        True if p and q are on the same side of the line, False otherwise.
    """

    # the line a -> b
    a, b = line_coords

    # get the vectors from a to b, from a to p and from a to q, e.g.:
    #      a          a
    #     /|\         |\ \
    #    p | q   OR   | p q
    #      b          b
    ab = b - a
    ap = p - a
    aq = q - a

    # get the cross product of the vectors (2D vectors therefore result is a single value)
    # AB X AP and AB X AQ will have the same sign if p and q are on the same side of the line a->b
    # cross_ab_ap = np.cross(ab, ap)
    # cross_ab_aq = np.cross(ab, aq)
    cross_ab_ap = ab[..., 0] * ap[..., 1] - ab[..., 1] * ap[..., 0]
    cross_ab_aq = ab[..., 0] * aq[..., 1] - ab[..., 1] * aq[..., 0]

    # if the product of the cross is positive then p and q are on the same side of the line a->b
    product_of_cross = float(cross_ab_ap * cross_ab_aq)

    return product_of_cross > 0
