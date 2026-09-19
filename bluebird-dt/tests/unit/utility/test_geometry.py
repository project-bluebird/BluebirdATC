import numpy as np
import pytest

from bluebird_dt.core import Pos2D
from bluebird_dt.utility.geometry import (
    centre,
    get_perpendicular_line,
    line_intersection,
    nearest_point_on_3D_rectangle,
    nearest_point_on_line_segment,
    points_same_side_of_line,
    rotate,
    rotate_about,
)

@pytest.mark.parametrize("start_lat, start_lon, end_lat, end_lon", [
    # fixed latitude
    (0., 1., 0., 2.),
    # small change in latitude
    (0., 1., 0.001, 2.),
    # fixed longitude
    (0., 0., 1., 0.),
    # small change in longitude
    (0., 0., 1., 0.001),
    # diagonal
    (-1., -1., 1., 1.),
    # another diagonal
    (1., -1., -1., 1.),
    # diagonal paths north and south of the equator
    (50.6, -3.6, 50.8, -3.4),
    (-50.6, -3.6, -50.8, -3.4),
])
def test_get_perpendicular_line(start_lat, start_lon, end_lat, end_lon):
    """
    Check that the lines form a right angle when latitude and longitude are treated as a flat grid.
    """
    start_pos = Pos2D(lat=start_lat, lon=start_lon)
    end_pos = Pos2D(lat=end_lat, lon=end_lon)
    c, d, u = get_perpendicular_line(start_pos, end_pos)
    direction = end_pos.location - start_pos.location
    assert np.dot(direction, d - c) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(d - c) == pytest.approx(1.0)
    assert u == 0.5
    assert c + u * (d - c) == pytest.approx(end_pos.location)


def test_get_perpendicular_line_start_end_points_equal():
    start_pos = Pos2D(50., 1.)
    end_pos = Pos2D(50., 1.)
    with pytest.raises(ValueError):
        c, d, u = get_perpendicular_line(start_pos, end_pos)


def test_nearest_point_on_line_segment():
    """
    Test the functionality of nearest_point_on_line_segment():
        - Given points on the line, return the same points on the line
        - Given points off the line, return the correct closest points
    """

    # assume a horizontal line segment from a->b
    a = np.array([0, 0])
    b = np.array([0, 1])

    # points exactly on the line -> return those points
    Q = np.array([[0, 0], [0, 0.5], [0, 1]])
    P = nearest_point_on_line_segment(Q, a, b)
    assert np.array_equal(P, Q)

    # points on a parallel line -> return same y values but line x values
    Q = np.array([[1, 0], [1, 0.5], [1, 1]])
    P = nearest_point_on_line_segment(Q, a, b)
    assert np.array_equal(P, np.array([[0, 0], [0, 0.5], [0, 1]]))

    # points on a perpendicular line --> should get same answer for each
    Q = np.array([[1, 0.5], [0, 0.5], [-1, 0.5]])
    P = nearest_point_on_line_segment(Q, a, b)
    assert np.array_equal(P, np.array([[0, 0.5], [0, 0.5], [0, 0.5]]))

    # points on a perpendicular line that doesn't intersect -> should get the line start point for each
    Q = np.array([[1, -1], [0, -1], [-1, -1]])
    P = nearest_point_on_line_segment(Q, a, b)
    assert np.array_equal(P, np.array([[0, 0], [0, 0], [0, 0]]))


def test_nearest_point_on_3D_rectangle():
    """
    Test the functionality of nearest_point_on_3D_rectangle():
        - Giving the corners and midpoint of the rectangle returns the nearest
          point as the same corner/midpoint.
        - Points outside the rectangle return the correct closest point on the
          rectangle itself.
    """
    # assume some nice simple 3D face of a rectangle defined on
    # [1, 1, 0] to [2, 2, 0]

    origin = np.array([1, 1, 0])
    p1 = np.array([1, 2, 0])
    p2 = np.array([2, 1, 0])

    query_locations = np.array(
        [
            # corners of the rectangle
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            # midpoint
            [1.5, 1.5, 0.0],
            # points away from the rectangle
            [3.0, 3.0, 11.0],
            [1.5, 1.5, 20.0],
            [1.5, 20.0, 0.0],
        ]
    )

    target_locations = np.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            # midpoint
            [1.5, 1.5, 0.0],
            # away
            [2.0, 2.0, 0.0],
            [1.5, 1.5, 0.0],
            [1.5, 2.0, 0.0],
        ]
    )

    for query_location, target_location in zip(query_locations, target_locations, strict=False):
        closest_location = nearest_point_on_3D_rectangle(query_location, origin, p1, p2)

        assert np.allclose(closest_location, target_location)


@pytest.mark.parametrize(
    "line",
    [
        (np.array([0, -1]), np.array([0, 1])),  # x = 0
        (np.array([-1, 0]), np.array([1, 0])),  # y = 0
        (np.array([-1, -1]), np.array([1, 1])),  # y = x
        (np.array([-1, 1]), np.array([1, -1])),  # y = -x
    ],
)
def test_points_same_side_of_line(line: tuple[np.ndarray, np.ndarray]):
    p = np.array([1, 2])
    q = np.array([10, 20])
    r = np.array([-10, -20])

    assert points_same_side_of_line(p, q, line)
    assert not points_same_side_of_line(p, r, line)
    assert not points_same_side_of_line(q, r, line)


def test_centre():
    """
    Test midpoint of two locations calculated correctly
    """
    a = Pos2D(lat=1.0, lon=2.0)
    b = Pos2D(lat=3.0, lon=6.0)
    mid = centre(a, b)
    assert mid.lat == 2.0
    assert mid.lon == 4.0


def test_rotate():
    """
    Test point rotations
    """
    p = Pos2D(lat=1.0, lon=0.0)
    rotated = rotate(p, 90.0)
    assert np.isclose(rotated.lat, 0.0, atol=1e-6)
    assert np.isclose(rotated.lon, -1.0, atol=1e-6)

    pivot = Pos2D(lat=1.0, lon=0.0)
    p2 = Pos2D(lat=2.0, lon=0.0)
    rotated_about = rotate_about(p2, 90.0, pivot)
    assert np.isclose(rotated_about.lat, 1.0, atol=1e-6)
    assert np.isclose(rotated_about.lon, -1.0, atol=1e-6)


def test_line_intersection():
    """
    Test line intersections for both parallel_and_crossing
    """
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.0, 1.0])
    d = np.array([1.0, 1.0])
    t, u = line_intersection(a, b, c, d)
    # Parallel lines to expect no intersection i.e. infinity
    assert np.isinf(t) and np.isinf(u)

    c2 = np.array([0.5, -1.0])
    d2 = np.array([0.5, 1.0])
    t2, u2 = line_intersection(a, b, c2, d2)
    # Lines should cross at (0.5, 0.5)
    assert np.isclose(t2, 0.5, atol=1e-6)
    assert np.isclose(u2, 0.5, atol=1e-6)
