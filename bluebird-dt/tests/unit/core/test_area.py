import pytest
import numpy as np

from bluebird_dt.core.area import Area
from bluebird_dt.core.pos2d import Pos2D


def test_init_exceptions():
    """Area must be initialised with a list of length >= 3."""

    with pytest.raises(ValueError):
        Area([])

    with pytest.raises(ValueError):
        Area([1, 2])

    with pytest.raises(ValueError):
        Area([Pos2D(0, 0), Pos2D(0, 1)])


@pytest.mark.parametrize(
    "airspace_routes,sector_name,outside_fixes,boundary_fixes,inner_fixes",
    [
        ("generate_i", "sector_i", ["FIRE", "SPIRIT"], ["EARTH", "AIR"], ["WATER"]),
        ("generate_x", "sector_x", ["LIMBO", "SANTA", "SIN", "SIREN"], ["DEMON", "GATES", "WITCH", "HAUNT"], ["ABYSS"]),
        ("generate_y", "sector_y", ["GOD", "GHOST", "DECAN"], ["CANON", "BISHP"], ["TRI"]),
    ],
)
def test_contains(airspace_routes, sector_name, outside_fixes, boundary_fixes, inner_fixes, request):
    """
    Method returns True for Pos2D inside the Area or on the boundary. It
    returns False for Pos2D outside the Area.
    """

    airspace, routes = request.getfixturevalue(airspace_routes)
    area: Area = airspace.sectors[sector_name].volumes[0].area
    fixes = airspace.fixes.places
    for fix in outside_fixes:
        assert not area.contains(fixes[fix], epsilon=1e-4)
    for fix in boundary_fixes:
        assert area.contains(fixes[fix], epsilon=1e-4)
    for fix in inner_fixes:
        assert area.contains(fixes[fix], epsilon=1e-4)


@pytest.mark.parametrize(
    "airspace_routes,sector_name,outside_fixes,boundary_fixes,inner_fixes",
    [
        ("generate_i", "sector_i", ["FIRE", "SPIRIT"], ["EARTH", "AIR"], ["WATER"]),
        ("generate_x", "sector_x", ["LIMBO", "SANTA", "SIN", "SIREN"], ["DEMON", "GATES", "WITCH", "HAUNT"], ["ABYSS"]),
        ("generate_y", "sector_y", ["GOD", "GHOST", "DECAN"], ["CANON", "BISHP"], ["TRI"]),
    ],
)
def test_on_boundary(airspace_routes, sector_name, outside_fixes, boundary_fixes, inner_fixes, request):
    """
    Method returns True for Pos2D on the boundary and False for Pos2D inside
    or outside the Area.
    """

    airspace, routes = request.getfixturevalue(airspace_routes)
    area = airspace.sectors[sector_name].volumes[0].area
    fixes = airspace.fixes.places
    for fix in outside_fixes:
        assert not area.on_boundary(fixes[fix])
    for fix in boundary_fixes:
        assert area.on_boundary(fixes[fix])
    for fix in inner_fixes:
        assert not area.on_boundary(fixes[fix])


@pytest.mark.parametrize(
    "airspace_routes,sector_name,outside_fixes,boundary_fixes,inner_fixes",
    [
        ("generate_i", "sector_i", ["FIRE", "SPIRIT"], ["EARTH", "AIR"], ["WATER"]),
        ("generate_x", "sector_x", ["LIMBO", "SANTA", "SIN", "SIREN"], ["DEMON", "GATES", "WITCH", "HAUNT"], ["ABYSS"]),
        ("generate_y", "sector_y", ["GOD", "GHOST", "DECAN"], ["CANON", "BISHP"], ["TRI"]),
    ],
)
def test_centre(airspace_routes, sector_name, outside_fixes, boundary_fixes, inner_fixes, request):
    """
    Check that we get the correct centre position for each artificial sector
    """

    airspace, _ = request.getfixturevalue(airspace_routes)
    area = airspace.sectors[sector_name].volumes[0].area
    centre = area.centre
    assert isinstance(centre, Pos2D)
    # x and i sectors should be centered on 0,0
    if sector_name in ["sector_i", "sector_x"]:
        expected_centre = np.array([0., 0.])
        assert np.allclose(centre.location, expected_centre)


def test_distance(generate_i):
    """
    Test the distance method by verifying that the distance from the center of the I-sector to the nearest boundary is half the sector width.  This works because:
        - generate_i create sector_i centred on 0,0
        - the Water fix is positioned at 0,0
        - SectorI has width=60NMI, height=120NMI.
        - the nearest boundary from the center is half the width.
    """
    airspace, _ = generate_i
    point = airspace.fixes.places["WATER"]
    area = airspace.sectors["sector_i"].volumes[0].area

    expected_distance = 30.0
    area_distance = area.distance(point)
    assert area_distance == pytest.approx(expected_distance, abs=0.5)


def test_area_json_round_trip(tmp_path):
    """
    Test converting Area objects to and from json works correctly
    """
    boundary = [
        Pos2D(0.0, 0.0),
        Pos2D(0.0, 1.0),
        Pos2D(1.0, 1.0),
        Pos2D(0.0, 0.0),
    ]
    area = Area(boundary)
    loaded = Area.from_json(area.to_json())
    assert loaded == area
    assert loaded.centre == area.centre

    path = tmp_path / "area.json"
    area.save(str(path))
    reloaded = Area.load(str(path))
    assert reloaded == area


@pytest.mark.parametrize(
    ("point", "expected_segments"),
    [
        pytest.param(
            Pos2D(0.5, 2.0),
            [{(1.0, 1.0), (1.0, 0.0)}],
            id="outside_east_edge",
        ),
        pytest.param(
            Pos2D(0.5, -1.0),
            [{(0.0, 0.0), (0.0, 1.0)}],
            id="outside_west_edge",
        ),
        pytest.param(
            Pos2D(2.0, 0.5),
            [{(0.0, 1.0), (1.0, 1.0)}],
            id="outside_north_edge",
        ),
        pytest.param(
            Pos2D(-1.0, 0.5),
            [{(0.0, 0.0), (1.0, 0.0)}],
            id="outside_south_edge",
        ),
        pytest.param(
            Pos2D(0.5, 0.8),
            [{(1.0, 1.0), (1.0, 0.0)}],
            id="inside_nearest_east_edge",
        ),
        pytest.param(
            Pos2D(0.5, 1.0),
            [{(1.0, 1.0), (1.0, 0.0)}],
            id="point_on_edge",
        ),
        pytest.param(
            Pos2D(1.0, 1.0),
            [
                {(1.0, 1.0), (1.0, 0.0)},
                {(0.0, 1.0), (1.0, 1.0)},
            ],
            id="point_on_vertex_tie",
        ),
    ],
)
def test_nearest_segment_to_point(point, expected_segments):
    """
    Check nearest_segment_to_point over multiple edge, interior, boundary, and tie cases.

    Note that Pos2D accepts (lat, lon) whereas nearest_segment_to_point returns (lon, lat),
    so the expected coords are reversed compared to the Pos2D inputs.
    """
    boundary = [
        Pos2D(0.0, 0.0),
        Pos2D(0.0, 1.0),
        Pos2D(1.0, 1.0),
        Pos2D(1.0, 0.0),
        Pos2D(0.0, 0.0),
    ]
    area = Area(boundary)

    segment = area.nearest_segment_to_point(point)
    assert segment is not None
    # Treating lat/lons as xy coords the above area is a unit square on the equator.
    # Segment endpoints are compared as sets because direction is not significant here.
    assert set(segment) in expected_segments

def test_nearest_segment_has_polygon_interior_on_left() -> None:
    area = Area(
        boundary=[
            Pos2D(0.0, 0.0),
            Pos2D(0.0, 1.0),
            Pos2D(1.0, 1.0),
            Pos2D(1.0, 0.0),
        ]
        
    )

    segment = area.nearest_segment_to_point(Pos2D(lat=0.5, lon=1.1))

    assert segment is not None

    start, end = segment

    # Cross product of the segment direction with a vector towards a point
    # inside the polygon. Positive means that point lies to the left.
    interior = (0.5, 0.5)
    cross_product = (
        (end[0] - start[0]) * (interior[1] - start[1])
        - (end[1] - start[1]) * (interior[0] - start[0])
    )

    assert cross_product > 0
