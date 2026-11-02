import copy
from collections import OrderedDict
from datetime import timezone

import numpy as np
import pytest

from bluebird_dt.airspace_generator import SectorI
from bluebird_dt.core import Action, Aircraft, Airspace, Area, Environment, Fixes, FlightPlan, Pos2D, Route, Sector, Volume
from bluebird_dt.core.airway import Airway
from bluebird_dt.scenario_manager import Custom

from bluebird_dt.utility.geometry import centre


def test_init_exceptions():
    """
    Airspace must be initiated with a dictionary of >= 1 Sectors.
    """

    with pytest.raises(ValueError):
        Airspace({}, Fixes({}))


@pytest.mark.parametrize(
    ("airspace_routes", "sector_name"),
    [
        ("generate_i", "sector_i"),
        ("generate_x", "sector_x"),
        ("generate_y", "sector_y"),
        ("generate_thunderdome", "thunderdome"),
    ],
)
def test_boundary_single_volume(airspace_routes, sector_name, request):
    """
    For Airspaces composed of a single Volume, the Airspace boundary is the same as the
    Volume Area boundary. This is a simple check that we haven't rotated the coordinates.
    """

    airspace, _ = request.getfixturevalue(airspace_routes)
    assert airspace.boundary().boundary == airspace.sectors[sector_name].volumes[0].area.boundary


def test_boundary_multiple_volumes(generate_two_sector):
    """
    Test method works for Airspaces composed of multiple Sectors/Volumes.
    """

    airspace, _ = generate_two_sector
    sector1_poly = airspace.sectors["sector_i1"].volumes[0].area.boundary
    sector2_poly = airspace.sectors["sector_i2"].volumes[0].area.boundary

    assert airspace.boundary().boundary == sector1_poly.union(sector2_poly, grid_size=0.00062)


def test_bandbox_sectors(generate_two_sector):
    """
    Check that the new, combined airspace object has the expected properties:
        - name
        - volumes
        - fixes
    """

    airspace, _ = generate_two_sector
    new_airspace = copy.deepcopy(airspace)
    new_airspace.bandbox_sectors({"combined": ["sector_i1", "sector_i2"]})

    assert len(new_airspace.sectors) == 1
    assert "combined" in new_airspace.sectors
    assert len(new_airspace.sectors["combined"].volumes) == 2

    new_volumes = [vol.data() for vol in new_airspace.sectors["combined"].volumes]
    assert airspace.sectors["sector_i1"].volumes[0].data() in new_volumes
    assert airspace.sectors["sector_i2"].volumes[0].data() in new_volumes
    assert airspace.fixes.data() == new_airspace.fixes.data()


def test_split_sector(generate_two_sector):
    """
    Create a bandbox then check that it can be split back into the original sectors.
    """
    airspace, _ = generate_two_sector
    airspace.bandbox_sectors({"combined": ["sector_i1", "sector_i2"]})
    assert "combined" in airspace.airspace_configuration

    airspace_config = copy.deepcopy(airspace.airspace_configuration)
    airspace.split_sector("NON_EXISTENT")
    assert airspace.airspace_configuration == airspace_config

    airspace.split_sector("combined")
    assert "combined" not in airspace.airspace_configuration
    assert "sector_i1" in airspace.airspace_configuration
    assert "sector_i2" in airspace.airspace_configuration


def test_list_individual_sectors(generate_two_sector):
    """
    Test that list_individual_sectors returns the correct list of sectors
    """
    airspace, _ = generate_two_sector
    assert all(k in airspace.list_individual_sectors() for k,_ in airspace.sectors.items())

def test_list_all_sector_sets(generate_two_sector):
    """
    Test list all sector sets. Build sets from airspace configuration and check list_all_sector_sets returns the same
    """
    airspace, _ = generate_two_sector
    sector_sets = airspace.list_all_sector_sets()
    assert all(set(airspace.airspace_configuration[sector]) in sector_sets for sector in airspace.sectors)

def test_sector_name_from_list_of_individual_sectors(generate_two_sector):
    airspace, _ = generate_two_sector

    # Extract the sector config and use it to build mock inputs
    # We need to use sectors that are actually in the airspace for this test
    sector_list = list(airspace.airspace_configuration.values())
    first_sector_name = list(airspace.airspace_configuration)[0]
    first_sector_list = sector_list[0]

    # Test the method for a single sector
    sector_name = airspace.sector_name_from_list_of_individual_sectors(first_sector_list)
    assert sector_name is first_sector_name

    # Now set up a bandboxed sector and test the correct name returned
    mock_bandbox_name = "combined"
    mock_bandboxed_sectors = [x[0] for x in airspace.airspace_configuration.values()]
    airspace.bandbox_sectors({ mock_bandbox_name: mock_bandboxed_sectors })
    sector_name = airspace.sector_name_from_list_of_individual_sectors(mock_bandboxed_sectors)
    assert sector_name is mock_bandbox_name

    # Now test None returned for non existent sectors
    sector_name = airspace.sector_name_from_list_of_individual_sectors(['Fictional', 'Imaginary'])
    assert sector_name is None

def test_get_containing_bandboxed_sector(generate_two_sector):
    """
    Test that get_containing_bandboxed_sector returns the correct parent sector by checking:
    - Returns sector name for non bandboxed sector
    - Returns correct parent name for previously injected bandboxing
    - Returns background for non existent sector
    """
    airspace, _ = generate_two_sector

    # Find a valid (non bandboxed) sector and test that the same name is returned
    first_sector_name = list(airspace.airspace_configuration)[0]
    containing_sector = airspace.get_containing_bandboxed_sector(first_sector_name)
    assert containing_sector == first_sector_name

     # Now set up a bandboxed sector and test the correct name returned
    mock_bandbox_name = "combined"
    mock_bandboxed_sectors = [x[0] for x in airspace.airspace_configuration.values()]
    airspace.bandbox_sectors({ mock_bandbox_name: mock_bandboxed_sectors })
    containing_sector = airspace.get_containing_bandboxed_sector(first_sector_name)
    assert containing_sector == mock_bandbox_name

    # Now test background returned for non existent sectors
    fictional_sector_name = 'Fictional'
    containing_sector = airspace.get_containing_bandboxed_sector(fictional_sector_name)
    assert containing_sector == 'background'

def test_distance(generate_i):
    """
    Test that airspace distance matches sector distance for a single-sector airspace.
    This tests wiring.  Test of the same name in test_area tests the calculation more explicitly.
    """
    airspace, _ = generate_i
    point = airspace.fixes.places["WATER"]

    airspace_distance = airspace.distance(point)
    sector_distance = airspace.sectors["sector_i"].distance(point)

    assert airspace_distance == pytest.approx(sector_distance)
    assert airspace_distance > 0.0


@pytest.mark.parametrize(
    ("airspace_routes", "outside_fixes", "boundary_fixes", "inner_fixes"),
    [
        ("generate_i", ["FIRE", "SPIRIT"], ["EARTH", "AIR"], ["WATER"]),
        ("generate_x", ["LIMBO", "SANTA", "SIN", "SIREN"], ["DEMON", "GATES", "WITCH", "HAUNT"], ["ABYSS"]),
        ("generate_y", ["GOD", "GHOST", "DECAN"], ["CANON", "BISHP"], ["TRI"]),
    ],
)
def test_boundary_fixes(airspace_routes, outside_fixes, boundary_fixes, inner_fixes, request):
    """
    Test method returns all Fixes on the Airspace boundary (and no other).
    """

    airspace, routes = request.getfixturevalue(airspace_routes)
    for fix in outside_fixes:
        assert fix not in airspace.boundary_fixes().places
    for fix in boundary_fixes:
        assert fix in airspace.boundary_fixes().places
    for fix in inner_fixes:
        assert fix not in airspace.boundary_fixes().places


@pytest.mark.parametrize(
    ("airspace_routes", "route_boundary_fixes"),
    [
        ("generate_i", [["EARTH", "AIR"], ["EARTH", "AIR"]]),
        (
            "generate_x",
            [
                ["WITCH", "DEMON"],
                ["WITCH", "GATES"],
                ["WITCH", "HAUNT"],
                ["HAUNT", "GATES"],
                ["HAUNT", "DEMON"],
                ["HAUNT", "WITCH"],
                ["GATES", "HAUNT"],
                ["GATES", "DEMON"],
                ["GATES", "WITCH"],
                ["DEMON", "WITCH"],
                ["DEMON", "GATES"],
                ["DEMON", "HAUNT"],
            ],
        ),
        (
            "generate_y",
            [
                ["CANON", "SON"],
                ["CANON", "SON"],
                ["BISHP", "SON"],
                ["BISHP", "SON"],
                ["BISHP", "CANON"],
                ["CANON", "BISHP"],
            ],
        ),
    ],
)
def test_route_boundary_fixes(airspace_routes, route_boundary_fixes, request):
    """
    Test method returns only the two Route boundary Fixes.
    """

    airspace, routes = request.getfixturevalue(airspace_routes)

    for i, route in enumerate(routes):
        route_fixes = airspace.route_boundary_fixes(route).places
        for fix in route_fixes:
            assert fix in route_boundary_fixes[i]


@pytest.mark.parametrize(
    ("airspace_routes", "fixes"),
    [
        ("generate_i", ["EARTH", "AIR"]),
        (
            "generate_x",
            [
                "WITCH",
                "WITCH",
                "WITCH",
                "HAUNT",
                "HAUNT",
                "HAUNT",
                "GATES",
                "GATES",
                "GATES",
                "DEMON",
                "DEMON",
                "DEMON",
            ],
        ),
        ("generate_y", ["CANON", "SON", "BISHP", "SON", "BISHP", "CANON"]),
    ],
)
def test_route_entry_fix(airspace_routes, fixes, request):
    """
    Test method returns the Route entry Fix.
    """

    airspace, routes = request.getfixturevalue(airspace_routes)
    for i, route in enumerate(routes):
        assert airspace.route_entry_fix(route) == airspace.fixes.places[fixes[i]]


@pytest.mark.parametrize(
    ("airspace_routes", "fixes"),
    [
        ("generate_i", ["AIR", "EARTH"]),
        (
            "generate_x",
            [
                "DEMON",
                "GATES",
                "HAUNT",
                "GATES",
                "DEMON",
                "WITCH",
                "HAUNT",
                "DEMON",
                "WITCH",
                "WITCH",
                "GATES",
                "HAUNT",
            ],
        ),
        ("generate_y", ["SON", "CANON", "SON", "BISHP", "CANON", "BISHP"]),
    ],
)
def test_route_exit_fix(airspace_routes, fixes, request):
    """
    Test method returns the Route exit Fix.
    """

    airspace, routes = request.getfixturevalue(airspace_routes)
    for i, route in enumerate(routes):
        assert airspace.route_exit_fix(route) == airspace.fixes.places[fixes[i]]


@pytest.mark.parametrize(
    "airspace_routes", ["generate_i", "generate_x", "generate_y", "generate_two_sector"]
)
def test_rotate_all_airspaces(airspace_routes, request):
    """
    Test that rotate() can be applied to any example Airspace.
    """

    for angle in [0, 45, 90, 360]:
        airspace, _ = request.getfixturevalue(airspace_routes)
        airspace.rotate(angle)
        assert isinstance(airspace, Airspace)


def test_rotate_one_sector(generate_i):
    """
    Test that rotate() works as expected in a 1 Sector Airspace.
    """

    airspace_original, _ = generate_i
    airspace_original_area = airspace_original.boundary()

    # rotating 360 should not change anything
    airspace_translate = copy.deepcopy(airspace_original)
    airspace_translate.rotate(360)
    airspace_translate_area = airspace_translate.boundary()

    assert airspace_original_area.boundary == airspace_translate_area.boundary

    # rotating by any other degree should change Airspace boundary
    airspace_translate = copy.deepcopy(airspace_original)
    airspace_translate.rotate(45)
    airspace_translate_area = airspace_translate.boundary()

    assert airspace_original_area.boundary != airspace_translate_area.boundary

    # internal fixes are rotated with the Airspace and stay inside the boundary
    for fix in ["EARTH", "WATER", "AIR"]:
        assert airspace_translate_area.contains(airspace_translate.fixes.places[fix])


def test_rotate_two_sector(generate_two_sector):
    """
    Test that rotate() works as expected in a 2 Sector Airspace.
    """

    airspace_original, _ = generate_two_sector
    airspace_original_area = airspace_original.boundary()

    # rotating 360 shouldn't change anything
    airspace_translate = copy.deepcopy(airspace_original)
    airspace_translate.rotate(360)
    airspace_translate_area = airspace_translate.boundary()

    assert (
        airspace_original.sectors["sector_i1"].volumes[0].area.boundary
        == airspace_translate.sectors["sector_i1"].volumes[0].area.boundary
    )
    assert (
        airspace_original.sectors["sector_i2"].volumes[0].area.boundary
        == airspace_translate.sectors["sector_i2"].volumes[0].area.boundary
    )
    assert airspace_original_area.boundary == airspace_translate_area.boundary

    # rotating by any other degree should change Airspace boundary
    airspace_translate = copy.deepcopy(airspace_original)
    airspace_translate.rotate(90)
    airspace_translate_area = airspace_translate.boundary()

    assert (
        airspace_original.sectors["sector_i1"].volumes[0].area.boundary
        != airspace_translate.sectors["sector_i1"].volumes[0].area.boundary
    )
    assert (
        airspace_original.sectors["sector_i2"].volumes[0].area.boundary
        != airspace_translate.sectors["sector_i2"].volumes[0].area.boundary
    )
    assert airspace_original_area.boundary != airspace_translate_area.boundary

    # internal fixes are rotated with the Airspace and stay inside the boundary
    for fix_name, pos in airspace_translate.fixes.places.items():
        # have two external fixes
        if fix_name not in ["FIX0", "FIX6"]:
            assert airspace_translate_area.contains(pos)


@pytest.mark.parametrize(
    "airspace_routes", ["generate_i", "generate_x", "generate_y", "generate_two_sector"]
)
def test_scale_all_airspaces(airspace_routes, request):
    """
    Test that scale() can be applied to any example Airspace.
    """

    for factor in [0.1, 0.25, 1.0, 5.0]:
        airspace, _ = request.getfixturevalue(airspace_routes)
        airspace.scale(factor)
        assert isinstance(airspace, Airspace)


def test_scale_one_sector(generate_i):
    """
    Test that scale() works as expected in a 1 Sector Airspace.
    """

    airspace_original, _ = generate_i
    airspace_original_area = airspace_original.boundary()

    # scaling by 1 shouldn't change anything
    airspace_translate = copy.deepcopy(airspace_original)
    airspace_translate.scale(1.0)
    airspace_translate_area = airspace_translate.boundary()

    assert airspace_original_area.boundary == airspace_translate_area.boundary

    # applying any other scale should change Airspace boundary
    airspace_translate = copy.deepcopy(airspace_original)
    airspace_translate.scale(0.5)
    airspace_translate_area = airspace_translate.boundary()

    assert airspace_original_area.boundary != airspace_translate_area.boundary

    # internal fixes are scaled with the Airspace and stay inside the boundary
    for fix in ["EARTH", "WATER", "AIR"]:
        assert airspace_translate_area.contains(airspace_translate.fixes.places[fix])


def test_scale_two_sector(generate_two_sector):
    """
    Test that scale() works as expected in a 2 Sector Airspace.
    """

    airspace_original, _ = generate_two_sector
    airspace_original_area = airspace_original.boundary()

    # scaling by 1 shouldn't change anything
    airspace_translate = copy.deepcopy(airspace_original)
    airspace_translate.scale(1.0)
    airspace_translate_area = airspace_translate.boundary()

    assert (
        airspace_original.sectors["sector_i1"].volumes[0].area.boundary
        == airspace_translate.sectors["sector_i1"].volumes[0].area.boundary
    )
    assert (
        airspace_original.sectors["sector_i2"].volumes[0].area.boundary
        == airspace_translate.sectors["sector_i2"].volumes[0].area.boundary
    )
    assert airspace_original_area.boundary == airspace_translate_area.boundary

    # applying any other scale should change Airspace boundary
    airspace_translate = copy.deepcopy(airspace_original)
    airspace_translate.scale(2.0)
    airspace_translate_area = airspace_translate.boundary(precision=0)

    assert (
        airspace_original.sectors["sector_i1"].volumes[0].area.boundary
        != airspace_translate.sectors["sector_i1"].volumes[0].area.boundary
    )
    assert (
        airspace_original.sectors["sector_i2"].volumes[0].area.boundary
        != airspace_translate.sectors["sector_i2"].volumes[0].area.boundary
    )
    assert airspace_original_area.boundary != airspace_translate_area.boundary

    # internal fixes are scaled with the Airspace and stay inside the boundary
    for fix_name, pos in airspace_translate.fixes.places.items():
        # have two external fixes
        if fix_name not in ["FIX0", "FIX6"]:
            assert airspace_translate_area.contains(pos)


@pytest.mark.parametrize(
    "airspace_routes", ["generate_i", "generate_x", "generate_y", "generate_two_sector"]
)
def test_forwardall_airspaces(airspace_routes, request):
    """
    Test that forward() can be applied to any example Airspace.
    """

    for distance, heading in [(60, 45), (300, 0), (0, 0), (105, 70)]:
        airspace, _ = request.getfixturevalue(airspace_routes)
        airspace.forward(distance=distance, heading=heading)
        assert isinstance(airspace, Airspace)


def test_translate_one_sector(generate_i):
    """
    Test that translate() works as expected in a 1 Sector Airspace.
    """

    airspace_original, _ = generate_i
    airspace_original_area = airspace_original.boundary()

    # offsetting by 0 shouldn't change anything
    airspace_translate = copy.deepcopy(airspace_original)
    airspace_translate.forward(distance=0.0, heading=0.0)
    airspace_translate_area = airspace_translate.boundary()

    assert airspace_original_area.boundary == airspace_translate_area.boundary

    # applying any non-zero offset should shift Airspace boundary
    airspace_translate = copy.deepcopy(airspace_original)
    airspace_translate.forward(distance=1.0, heading=0.0)
    airspace_translate_area = airspace_translate.boundary()

    assert airspace_original_area.boundary != airspace_translate_area.boundary

    # internal fixes are moved with the Airspace and stay inside the boundary
    for fix in ["EARTH", "WATER", "AIR"]:
        assert airspace_translate_area.contains(airspace_translate.fixes.places[fix])


def test_translate_two_sector(generate_two_sector):
    """
    Test that translate() works as expected in 2 Sector Airspace.
    """

    airspace_original, _ = generate_two_sector
    airspace_original_area = airspace_original.boundary()

    # offsetting by 0 shouldn't change anything
    airspace_translate = copy.deepcopy(airspace_original)
    airspace_translate.forward(distance=0.0, heading=0.0)
    airspace_translate_area = airspace_translate.boundary()

    assert (
        airspace_original.sectors["sector_i1"].volumes[0].area.boundary
        == airspace_translate.sectors["sector_i1"].volumes[0].area.boundary
    )
    assert (
        airspace_original.sectors["sector_i2"].volumes[0].area.boundary
        == airspace_translate.sectors["sector_i2"].volumes[0].area.boundary
    )
    assert airspace_original_area.boundary == airspace_translate_area.boundary

    # applying any non-zero offset should shift Airspace boundary
    airspace_translate = copy.deepcopy(airspace_original)
    airspace_translate.forward(distance=1.0, heading=0.0)
    airspace_translate_area = airspace_translate.boundary()

    assert (
        airspace_original.sectors["sector_i1"].volumes[0].area.boundary
        != airspace_translate.sectors["sector_i1"].volumes[0].area.boundary
    )
    assert (
        airspace_original.sectors["sector_i2"].volumes[0].area.boundary
        != airspace_translate.sectors["sector_i2"].volumes[0].area.boundary
    )
    assert airspace_original_area.boundary != airspace_translate_area.boundary

    # internal fixes are moved with the Airspace and stay inside the boundary
    for fix_name, pos in airspace_translate.fixes.places.items():
        # have two external fixes
        if fix_name not in ["FIX0", "FIX6"]:
            assert airspace_translate_area.contains(pos)


@pytest.mark.parametrize(
    ("airspace_routes", "fixes", "true_lims"),
    [("generate_i", ["WATER"], [[60, 400]]), ("generate_two_sector", ["FIX5", "FIX1"], [[60, 400], [100, 300]])],
)
def test_find_fl_lim(airspace_routes, fixes, true_lims, request):
    """
    Test the correct FL limits are returned.
    This depends on which Sector/Volume the position is in.
    """

    airspace, _ = request.getfixturevalue(airspace_routes)
    for i, fix in enumerate(fixes):
        pos = airspace.fixes.places[fix]
        FL_lims = airspace.find_fl_lim(pos)
        assert len(FL_lims) == 2
        assert FL_lims[0] == true_lims[i][0]
        assert FL_lims[1] == true_lims[i][1]


def test_get_airspace_bounds():
    """
    Test the correct airspace bounds are returned by get_airspace_bounds().
    """

    # airspace 1 degree by 4 degrees (60 NMI ~= 1 deg), centred on (0, 0)
    width = 60.0
    height = 60.0 * 4
    flight_levels = (200, 400)

    sector_i_gen = SectorI(width=width, height=height, fl_limits=flight_levels)
    airspace, _routes = sector_i_gen.generate_airspace()

    expected_lb = np.array([-0.5, -2, 200.0])
    expected_ub = np.array([0.5, 2, 400.0])

    lb, ub = airspace.get_bounds()

    assert np.allclose(lb, expected_lb, atol=0.005)
    assert np.allclose(ub, expected_ub, atol=0.005)


def test_get_exit_point(generate_two_sector):
    """
    The exit point is either:
    - an intersection with the airspace boundary
    - the last point on route if it is inside the sector
    The method can be applied to the whole airspace or a specific sector.
    An error is raised if the route does not intersect with the sector/airspace.
    """

    # sector_i1 span: lon = [-0.5, 0.5], lats = [-2, 0], fl = [60, 400]
    # sector_i2 span: lon = [-0.5, 0.5], lats = [0, 2], fl = [100, 300]
    # routes[0]: FIX0 -> FIX6, sector_i2 -> sector_i1
    # routes[1]: FIX6 -> FIX0, sector_i1 -> sector_i2
    airspace, routes = generate_two_sector
    assert isinstance(airspace, Airspace)

    # 1: return point on airspace/sector boundary (whether there is a fix there or not)

    # create two aircraft, one at the start of each route
    pos1 = airspace.fixes.places["FIX0"]
    pos2 = airspace.fixes.places["FIX6"]
    # most of the values here are placeholders - all that matters is lat/lon and FlightPlan.route
    aircraft1 = Aircraft(pos1.lat, pos1.lon, 200, 0, FlightPlan(routes[0]), "AIR0")
    aircraft2 = Aircraft(pos2.lat, pos2.lon, 200, 0, FlightPlan(routes[1]), "AIR1")

    # boundary fixes exist (FIX1 and FIX5) - method returns those
    a1_exit_pos = airspace.get_exit_point(aircraft1)
    a2_exit_pos = airspace.get_exit_point(aircraft2)
    assert airspace.fixes.places["FIX5"].lat == pytest.approx(a1_exit_pos.lat, abs=0.005)
    assert airspace.fixes.places["FIX1"].lat == pytest.approx(a2_exit_pos.lat, abs=0.005)

    # only the lat is changing on this route, lon of all points is 0.0
    assert np.isclose(a1_exit_pos.lon, 0.0)
    assert np.isclose(a2_exit_pos.lon, 0.0)

    # we can also get the exit window for a given sector rather than the whole airspace
    # expected behaviour:
    # - the inner boundary point is returned as exit for first sector on route
    # - the airspace exit point is returned for second sector on route
    # the boundary point between the two sectors is at [0.0, 0.0]
    a1_exit_i1_pos = airspace.get_exit_point(aircraft1, "sector_i1")
    a1_exit_i2_pos = airspace.get_exit_point(aircraft1, "sector_i2")
    a2_exit_i1_pos = airspace.get_exit_point(aircraft2, "sector_i1")
    a2_exit_i2_pos = airspace.get_exit_point(aircraft2, "sector_i2")

    assert airspace.fixes.places["FIX5"].lat == pytest.approx(a1_exit_i1_pos.lat, abs=0.005)
    assert a1_exit_i2_pos.lat == pytest.approx(0.0, abs=0.005)

    assert a2_exit_i1_pos.lat == pytest.approx(0.0, abs=0.005)
    assert airspace.fixes.places["FIX1"].lat == pytest.approx(a2_exit_i2_pos.lat, abs=0.005)

    # we should get the same results as above even if we remove the boundary fixes from
    # the route i.e., the method should find the same locations
    non_boundary_fixes = [f"FIX{i}" for i in [0, 2, 4, 6]]
    aircraft1 = Aircraft(pos1.lat, pos1.lon, 200, 0, FlightPlan(Route(non_boundary_fixes)), "AIR0")
    aircraft2 = Aircraft(pos2.lat, pos2.lon, 200, 0, FlightPlan(Route(non_boundary_fixes[::-1])), "AIR1")

    a1_exit_pos = airspace.get_exit_point(aircraft1)
    a2_exit_pos = airspace.get_exit_point(aircraft2)
    a1_exit_i1_pos = airspace.get_exit_point(aircraft1, "sector_i1")
    a1_exit_i2_pos = airspace.get_exit_point(aircraft1, "sector_i2")
    a2_exit_i1_pos = airspace.get_exit_point(aircraft2, "sector_i1")
    a2_exit_i2_pos = airspace.get_exit_point(aircraft2, "sector_i2")

    assert airspace.fixes.places["FIX5"].lat == pytest.approx(a1_exit_pos.lat, abs=0.005)
    assert airspace.fixes.places["FIX1"].lat == pytest.approx(a2_exit_pos.lat, abs=0.005)
    assert a1_exit_pos.lon == 0.0
    assert a2_exit_pos.lon == 0.0

    assert airspace.fixes.places["FIX5"].lat == pytest.approx(a1_exit_i1_pos.lat, abs=0.005)
    assert a1_exit_i2_pos.lat == pytest.approx(0.0, abs=0.005)

    assert a2_exit_i1_pos.lat == pytest.approx(0.0, abs=0.005)
    assert airspace.fixes.places["FIX1"].lat == pytest.approx(a2_exit_i2_pos.lat, abs=0.005)

    # 2: shorten the route to end inside the airspace and return the last route point

    # start with route ending in the second sector on route
    aircraft1 = Aircraft(pos1.lat, pos1.lon, 200, 0, FlightPlan(Route([f"FIX{i}" for i in [0, 2, 4]])), "AIR0")
    aircraft2 = Aircraft(pos2.lat, pos2.lon, 200, 0, FlightPlan(Route([f"FIX{i}" for i in [6, 4, 2]])), "AIR1")
    assert airspace.get_exit_point(aircraft1).lat == airspace.fixes.places["FIX4"].lat
    assert airspace.get_exit_point(aircraft1, "sector_i1").lat == airspace.fixes.places["FIX4"].lat
    assert airspace.get_exit_point(aircraft2).lat == airspace.fixes.places["FIX2"].lat
    assert airspace.get_exit_point(aircraft2, "sector_i2").lat == airspace.fixes.places["FIX2"].lat

    # the exit for first on route sector is still the boundary point
    assert airspace.get_exit_point(aircraft1, "sector_i2").lat == pytest.approx(0.0, abs=0.005)
    assert airspace.get_exit_point(aircraft2, "sector_i1").lat == pytest.approx(0.0, abs=0.005)

    # now end route in the first on route sector
    aircraft1 = Aircraft(pos1.lat, pos1.lon, 200, 0, FlightPlan(Route([f"FIX{i}" for i in [0, 2]])), "AIR0")
    aircraft2 = Aircraft(pos2.lat, pos2.lon, 200, 0, FlightPlan(Route([f"FIX{i}" for i in [6, 4]])), "AIR1")
    assert airspace.get_exit_point(aircraft1).lat == airspace.fixes.places["FIX2"].lat
    assert airspace.get_exit_point(aircraft1, "sector_i2").lat == airspace.fixes.places["FIX2"].lat
    assert airspace.get_exit_point(aircraft2).lat == airspace.fixes.places["FIX4"].lat
    assert airspace.get_exit_point(aircraft2, "sector_i1").lat == airspace.fixes.places["FIX4"].lat

    # in this case the route does not intersect in any way with the second on route sector
    with pytest.raises(ValueError):
        airspace.get_exit_point(aircraft1, "sector_i1")

    with pytest.raises(ValueError):
        airspace.get_exit_point(aircraft2, "sector_i2")


def test_get_entry_point(generate_two_sector):
    """
    The entry point is an intersection with the airspace boundary.
    The method can be applied to the whole airspace or a specific sector.
    An error is raised if the route does not intersect with the sector/airspace
    or if the aircraft is already inside (this includes being on the boundary).
    """

    # sector_i1 span: lon = [-0.5, 0.5], lats = [-2, 0], fl = [60, 400]
    # sector_i2 span: lon = [-0.5, 0.5], lats = [0, 2], fl = [100, 300]
    # routes[0]: FIX0 -> FIX6, sector_i2 -> sector_i1
    # routes[1]: FIX6 -> FIX0, sector_i1 -> sector_i2
    airspace, routes = generate_two_sector

    # 1: return point on airspace/sector boundary (whether there is a fix there or not)

    # create two aircraft, one at the start of each route
    pos1 = airspace.fixes.places["FIX0"]
    pos2 = airspace.fixes.places["FIX6"]
    # most of the values here are placeholders - all that matters is lat/lon and FlightPlan.route
    route_1, route_2 = routes
    aircraft1 = Aircraft(pos1.lat, pos1.lon, 200, 0, FlightPlan(route_1), "AIR0")
    aircraft2 = Aircraft(pos2.lat, pos2.lon, 200, 0, FlightPlan(route_2), "AIR1")

    a1_entry_pos = airspace.get_entry_point(aircraft1)
    a2_entry_pos = airspace.get_entry_point(aircraft2)

    # boundary fixes exist (FIX1 and FIX5) - method returns those
    assert airspace.fixes.places["FIX1"].lat == pytest.approx(a1_entry_pos.lat, abs=0.005)
    assert airspace.fixes.places["FIX5"].lat == pytest.approx(a2_entry_pos.lat, abs=0.005)

    # only the lat is changing on this route, lon of all points is 0.0
    assert np.isclose(a1_entry_pos.lon, 0.0)
    assert np.isclose(a2_entry_pos.lon, 0.0)

    # we can also get the entry window for a given sector rather than the whole airspace
    # expected behaviour:
    # - the airspace entry point is returned for first sector on route
    # - the inner boundary point is returned as entry for second sector on route
    # the boundary point between the two sectors is at [0.0, 0.0]
    a1_entry_i1_pos = airspace.get_entry_point(aircraft1, "sector_i1")
    a1_entry_i2_pos = airspace.get_entry_point(aircraft1, "sector_i2")
    a2_entry_i1_pos = airspace.get_entry_point(aircraft2, "sector_i1")
    a2_entry_i2_pos = airspace.get_entry_point(aircraft2, "sector_i2")

    assert pytest.approx(a1_entry_i1_pos.lat, abs=0.005) == 0.0
    assert airspace.fixes.places["FIX1"].lat == pytest.approx(a1_entry_i2_pos.lat, abs=0.005)

    assert airspace.fixes.places["FIX5"].lat == pytest.approx(a2_entry_i1_pos.lat, abs=0.005)
    assert pytest.approx(a2_entry_i2_pos.lat, abs=0.005) == 0.0

    # we should get the same results as above even if we remove the boundary fixes from
    # the route i.e., the method should find the same locations
    non_boundary_fixes = [f"FIX{i}" for i in [0, 2, 4, 6]]
    aircraft1 = Aircraft(pos1.lat, pos1.lon, 200, 0, FlightPlan(Route(non_boundary_fixes)), "AIR0")
    aircraft2 = Aircraft(pos2.lat, pos2.lon, 200, 0, FlightPlan(Route(non_boundary_fixes[::-1])), "AIR1")

    a1_entry_pos = airspace.get_entry_point(aircraft1)
    a2_entry_pos = airspace.get_entry_point(aircraft2)
    a1_entry_i1_pos = airspace.get_entry_point(aircraft1, "sector_i1")
    a1_entry_i2_pos = airspace.get_entry_point(aircraft1, "sector_i2")
    a2_entry_i1_pos = airspace.get_entry_point(aircraft2, "sector_i1")
    a2_entry_i2_pos = airspace.get_entry_point(aircraft2, "sector_i2")

    assert airspace.fixes.places["FIX1"].lat == pytest.approx(a1_entry_pos.lat, abs=0.005)
    assert airspace.fixes.places["FIX5"].lat == pytest.approx(a2_entry_pos.lat, abs=0.005)
    assert a1_entry_pos.lon == 0.0
    assert a2_entry_pos.lon == 0.0

    assert pytest.approx(a1_entry_i1_pos.lat, abs=0.005) == 0.0
    assert airspace.fixes.places["FIX1"].lat == pytest.approx(a1_entry_i2_pos.lat, abs=0.005)

    assert airspace.fixes.places["FIX5"].lat == pytest.approx(a2_entry_i1_pos.lat, abs=0.005)
    assert pytest.approx(a2_entry_i2_pos.lat, abs=0.005) == 0.0

    # 2: aircraft is on the boundary or inside the airspace, return current position
    # first, place on boundary
    pos1 = airspace.fixes.places["FIX1"]
    pos2 = airspace.fixes.places["FIX5"]
    aircraft1 = Aircraft(pos1.lat, pos1.lon, 200, 0, FlightPlan(Route([f"FIX{i}" for i in [3, 4, 5, 6]])), "AIR0")
    aircraft2 = Aircraft(pos2.lat, pos2.lon, 200, 0, FlightPlan(Route([f"FIX{i}" for i in [3, 2, 1, 0]])), "AIR1")

    assert airspace.get_entry_point(aircraft1).lat == pos1.lat
    assert airspace.get_entry_point(aircraft2).lat == pos2.lat

    # second, place inside
    pos1 = airspace.fixes.places["FIX2"]
    pos2 = airspace.fixes.places["FIX4"]
    aircraft1 = Aircraft(pos1.lat, pos1.lon, 200, 0, FlightPlan(Route([f"FIX{i}" for i in [3, 4, 5, 6]])), "AIR0")
    aircraft2 = Aircraft(pos2.lat, pos2.lon, 200, 0, FlightPlan(Route([f"FIX{i}" for i in [3, 2, 1, 0]])), "AIR1")

    assert airspace.get_entry_point(aircraft1).lat == pos1.lat
    assert airspace.get_entry_point(aircraft2).lat == pos2.lat

@pytest.mark.parametrize("airspace_routes", ["generate_y", "generate_x", "generate_i", "generate_i_bent_route"])
def test_closest_forward_fix(airspace_routes, request):
    """
    Test the method returns the closest fix on Route in the direction of the exit Fix at various positions.
    """

    # set up
    airspace, routes = request.getfixturevalue(airspace_routes)
    env = Custom(1, airspace=airspace, routes=routes).create_env_manager().environment
    aircraft = env.aircraft["AIR0"]

    # 1. test from different positions on Route (exactly between Fixes)
    # --> expect to always return the Fix right ahead
    assert aircraft.flight_plan is not None
    route_fixes = aircraft.flight_plan.route.filed
    start_fix = airspace.fixes.places[route_fixes[0]]
    entry_fix = airspace.fixes.places[route_fixes[1]]
    central_fix = airspace.fixes.places[route_fixes[2]]
    exit_fix = airspace.fixes.places[route_fixes[3]]
    final_fix = airspace.fixes.places[route_fixes[4]]

    # move Aircraft between starting and entry Fix
    new_pos = centre(start_fix, entry_fix)
    aircraft.lat = new_pos.lat
    aircraft.lon = new_pos.lon
    assert airspace.closest_forward_fix(aircraft, 0) == route_fixes[1]

    # move Aircraft between entry and central Fix
    new_pos = centre(entry_fix, central_fix)
    aircraft.lat = new_pos.lat
    aircraft.lon = new_pos.lon
    assert airspace.closest_forward_fix(aircraft, 0) == route_fixes[2]

    # move Aircraft between central and exit Fix
    new_pos = centre(central_fix, exit_fix)
    aircraft.lat = new_pos.lat
    aircraft.lon = new_pos.lon
    assert airspace.closest_forward_fix(aircraft, 0) == route_fixes[3]

    # move Aircraft between exit and final Fix
    new_pos = centre(exit_fix, final_fix)
    aircraft.lat = new_pos.lat
    aircraft.lon = new_pos.lon
    assert airspace.closest_forward_fix(aircraft, 0) == route_fixes[4]

    # Test no matching fixes clause which previously threw a Type Error
    route_fixes.append("RHINO")
    try:
        airspace.closest_forward_fix(aircraft,0, route_fixes=route_fixes)
    except Exception as e:
        pytest.fail(f"Unexpected exception raised: {e}")

@pytest.fixture
def generate_i_air0_south(generate_i) -> tuple[Airspace, Aircraft]:
    """Generate Sector I airspace with aircraft AIR0 travelling south"""
    # Build I scenario
    airspace, routes = generate_i
    env = Custom(1, airspace=airspace, routes=routes, aircraft_on_route=True).create_env_manager().environment

    # Designate a test aircraft AIR0
    air0 = env.aircraft["AIR0"]
    # Ensure we're travelling South (direction determined randomly)
    assert air0.flight_plan is not None
    air0.flight_plan.route.filed = ["SPIRIT", "AIR", "WATER", "EARTH", "FIRE"]
    air0.flight_plan.route.current = air0.flight_plan.route.filed

    return airspace, air0


def test_closest_forward_fix_longitude_offset(generate_i_air0_south):
    """Test closest_forward_fix when parallel to route but outside threshold"""
    airspace, air0 = generate_i_air0_south

    # Place AIR0 just North of WATER's latitude, but offset to the East so we're
    # outside the distance threshold - next fix WATER.
    water = airspace.fixes.places["WATER"]
    air0.lat = water.lat + 0.01
    air0.lon = water.lon + 1.0
    distance_threshold = 10.0

    closest_fix_out = airspace.closest_forward_fix(air0, distance_threshold)
    assert closest_fix_out == "WATER"


def test_closest_forward_fix_close_to_fix(generate_i_air0_south):
    """Test closest_forward_fix within the threshold-distance of a fix"""
    airspace, air0 = generate_i_air0_south

    # Place AIR0 within the distance_threshold of WATER - should now count as
    # past WATER, next fix EARTH.
    water = airspace.fixes.places["WATER"]
    air0.lat = water.lat + 0.05
    air0.lon = water.lon + 0.05
    distance_threshold = 10.0

    closest_fix_out = airspace.closest_forward_fix(air0, distance_threshold)
    assert closest_fix_out == "EARTH"


def test_closest_forward_fix_before_first_leg(generate_i):
    """Test forward_fix identifies being before the first leg"""
    airspace, routes = generate_i
    em = Custom(2, airspace=airspace, routes=routes).create_env_manager()

    aircraft = em.environment.aircraft["AIR0"]
    spirit = airspace.fixes.places["SPIRIT"]

    filed = ["SPIRIT", "AIR", "WATER", "EARTH", "FIRE"]
    assert aircraft.flight_plan is not None
    aircraft.flight_plan.route.filed = filed
    aircraft.flight_plan.route.current = filed

    # Put aircraft before SPIRIT
    aircraft.lat, aircraft.lon = spirit.location
    aircraft.lat += 1.0

    closest_fix_out = airspace.closest_forward_fix(aircraft, 0)
    assert closest_fix_out == "SPIRIT"


def test_closest_forward_fix_after_last_leg(generate_i_air0_south):
    """Test forward_fix identifies being after the last leg (route finished)"""
    airspace, aircraft = generate_i_air0_south

    fire = airspace.fixes.places["FIRE"]

    # Put aircraft after FIRE (last fix)
    aircraft.lat, aircraft.lon = fire.location
    aircraft.lat -= 1.0

    closest_fix_out = airspace.closest_forward_fix(aircraft, 0)
    assert closest_fix_out is None


def test_closest_forward_fix_near_last_fix(generate_i_air0_south):
    """Test forward_fix identifies being near the last fix (route finished)"""
    airspace, aircraft = generate_i_air0_south

    fire = airspace.fixes.places["FIRE"]

    # Put aircraft near to FIRE (last fix)
    aircraft.lat, aircraft.lon = fire.location
    aircraft.lat += 0.1
    proximity_threshold_NMI = 10.0
    assert aircraft.pos2d().distance(fire) < proximity_threshold_NMI

    closest_fix_out = airspace.closest_forward_fix(aircraft, proximity_threshold_NMI)
    assert closest_fix_out is None


def test_closest_forward_fix_looping_bug(generate_thunderdome):
    """Demonstrate fixed bug in closest_forward_fix when a route looped back on itself"""
    airspace, routes = generate_thunderdome
    env = Custom(1, airspace=airspace, routes=routes).create_env_manager().environment

    # Create a route which loops back on itself, with the aircraft currently at
    # BOND3.
    # The thunderdome scenario is somewhat random, but using the boundary points
    # seems to be consistent.
    looping_aircraft = env.aircraft["AIR0"]
    looping_route = Route(["PORT4", "BOND4", "BOND0", "BOND1", "BOND3", "PORT3"])
    bond3 = airspace.fixes.places["BOND3"]
    assert looping_aircraft.flight_plan is not None
    looping_aircraft.flight_plan.route = looping_route
    looping_aircraft.lat, looping_aircraft.lon = bond3.location

    closest_fix_out = airspace.closest_forward_fix(looping_aircraft, 0)
    assert closest_fix_out == "PORT3"


def test_next_fix_basic_filed_behaviour(generate_i_air0_south):
    """Basic Route progression status for an in-progress route"""
    airspace, air0 = generate_i_air0_south

    # Place AIR0 within the distance_threshold of WATER - should now count as
    # past WATER, next fix EARTH.
    water = airspace.fixes.places["WATER"]
    air0.lat = water.lat + 0.05
    air0.lon = water.lon + 0.05
    distance_threshold = 10.0
    air0.next_fix_index = 3

    air0.update_route_status(airspace, distance_threshold_NMI=distance_threshold)
    route_status = OrderedDict(air0.route_status)

    expected_route_status = OrderedDict(
        {
            "SPIRIT": "passed",
            "AIR": "passed",
            "WATER": "passed",
            "EARTH": "next",
            "FIRE": "",
        }
    )
    assert route_status == expected_route_status


def test_off_route_filed_behaviour(generate_i_air0_south):
    """Basic Route progression status whilst off route"""
    airspace, air0 = generate_i_air0_south

    # Place AIR0 within the distance_threshold of WATER - should now count as
    # past WATER.
    water = airspace.fixes.places["WATER"]
    air0.lat = water.lat + 0.05
    air0.lon = water.lon + 0.05
    distance_threshold = 10.0
    air0.on_route = False
    air0.next_fix_index = 0

    air0.update_route_status(airspace, distance_threshold_NMI=distance_threshold)
    route_status = OrderedDict(air0.route_status)

    expected_route_status = OrderedDict(
        {
            "SPIRIT": "passed",
            "AIR": "passed",
            "WATER": "passed",
            "EARTH": "",
            "FIRE": "",
        }
    )
    assert route_status == expected_route_status


def test_next_fix_before_progressed(generate_i_air0_south):
    """Route progression status for a route not-yet started"""
    airspace, air0 = generate_i_air0_south

    # Place AIR0  away from any fixes.  First fix is SPIRIT (index 0).
    air0.lat = -10.0
    air0.lon = -10.0
    distance_threshold = 10.0
    air0.next_fix_index = 0

    air0.update_route_status(airspace, distance_threshold_NMI=distance_threshold)
    route_status = OrderedDict(air0.route_status)

    expected_route_status = OrderedDict(
        {
            "SPIRIT": "next",
            "AIR": "",
            "WATER": "",
            "EARTH": "",
            "FIRE": "",
        }
    )
    assert route_status == expected_route_status


def test_next_fix_last_fix_progression(generate_i_air0_south):
    """Route progression going towards the last fix in the filed route"""
    airspace, air0 = generate_i_air0_south

    air0.lat, air0.lon = airspace.fixes.places["EARTH"].location
    distance_threshold = 1.0
    air0.next_fix_index = 4
    air0.update_route_status(airspace, distance_threshold_NMI=distance_threshold)
    route_status = OrderedDict(air0.route_status)

    expected_route_status = OrderedDict(
        {
            "SPIRIT": "passed",
            "AIR": "passed",
            "WATER": "passed",
            "EARTH": "passed",
            "FIRE": "next",
        }
    )
    assert route_status == expected_route_status


def test_progression_after_last_fix(generate_i_air0_south):
    """Route progression having passed all fixes in the filed route"""
    airspace, air0 = generate_i_air0_south

    air0.lat, air0.lon = airspace.fixes.places["FIRE"].location
    distance_threshold = 1.0
    air0.next_fix_index = None
    air0.update_route_status(airspace, distance_threshold_NMI=distance_threshold)
    route_status = OrderedDict(air0.route_status)

    expected_route_status = OrderedDict(
        {
            "SPIRIT": "passed",
            "AIR": "passed",
            "WATER": "passed",
            "EARTH": "passed",
            "FIRE": "passed",
        }
    )
    assert route_status == expected_route_status


def test_route_progression_skipping_fixes(generate_i_air0_south):
    """From first fix, route-direct causes fixes to be marked as "to-skip"

    Firstly at SPIRIT:

    | SPIRIT | AIR  | WATER | EARTH | FIRE |
    | passed | next |       |       |      |

    Following a route-direct to EARTH, we should get

    | SPIRIT | AIR  | WATER | EARTH | FIRE |
    | passed | skip | skip  | next  |      |

    And finally, moving to EARTH should result in all passed except for FIRE:

    | SPIRIT | AIR    | WATER  | EARTH  | FIRE |
    | passed | passed | passed | passed | next |

    The route-direct testing is replicated in test_progression_route_directs.
    """
    airspace, air0 = generate_i_air0_south
    distance_threshold = 10

    # Initial state
    air0.lat, air0.lon = airspace.fixes.places["SPIRIT"].location
    air0.next_fix_index = 1
    air0.update_route_status(airspace, distance_threshold_NMI=distance_threshold)

    status = [OrderedDict(air0.route_status)]
    expected_status_0 = OrderedDict({"SPIRIT": "passed", "AIR": "next", "WATER": "", "EARTH": "", "FIRE": ""})
    assert status[0] == expected_status_0

    # Apply route-direct to EARTH
    env = Environment(time=0.0, airspace=airspace, aircraft={"AIR0": air0})
    route_direct_action = Action("AIR0", "route_direct_to", "EARTH")

    air0.pilot.receive_actions([route_direct_action], env)
    air0.pilot.process_actions(env)
    air0.update_route_status(airspace, distance_threshold_NMI=distance_threshold)
    status.append(OrderedDict(air0.route_status))

    expected_status_1 = OrderedDict(
        {"SPIRIT": "passed", "AIR": "skipping", "WATER": "skipping", "EARTH": "next", "FIRE": ""}
    )
    assert status[1] == expected_status_1

    # Move to EARTH
    air0.lat, air0.lon = airspace.fixes.places["EARTH"].location
    air0.next_fix_index += 1

    air0.update_route_status(airspace, distance_threshold_NMI=distance_threshold)
    status.append(OrderedDict(air0.route_status))

    expected_status_2 = OrderedDict(
        {"SPIRIT": "passed", "AIR": "passed", "WATER": "passed", "EARTH": "passed", "FIRE": "next"}
    )
    assert status[2] == expected_status_2


@pytest.mark.parametrize(
    ("initial_state", "route_direct_value", "expected_state"),
    [
        pytest.param(
            {"SPIRIT": "passed", "AIR": "next", "WATER": "", "EARTH": "", "FIRE": ""},
            "EARTH",
            {"SPIRIT": "passed", "AIR": "skipping", "WATER": "skipping", "EARTH": "next", "FIRE": ""},
            id="skipping fixes",
        ),
        pytest.param(
            {"SPIRIT": "passed", "AIR": "passed", "EARTH": "next", "FIRE": ""},
            ["WATER", "EARTH"],
            {"SPIRIT": "passed", "AIR": "passed", "WATER": "next", "EARTH": "", "FIRE": ""},
            id="add single extra fix next",
        ),
        pytest.param(
            {"SPIRIT": "passed", "WATER": "next", "EARTH": "", "FIRE": ""},
            ["AIR", "EARTH"],
            {"SPIRIT": "passed", "WATER": "skipping", "AIR": "next", "EARTH": "", "FIRE": ""},
            id="add single extra fix later with later skipping",
        ),
        pytest.param(
            {"SPIRIT": "passed", "FIRE": "next"},
            ["AIR", "EARTH", "FIRE"],
            {"SPIRIT": "passed", "AIR": "next", "EARTH": "", "FIRE": ""},
            id="add multiple extra fixes before resume",
        ),
        pytest.param(
            {"SPIRIT": "passed", "AIR": "next", "EARTH": "", "FIRE": ""},
            ["WATER", "EARTH"],
            {"SPIRIT": "passed", "AIR": "skipping", "WATER": "next", "EARTH": "", "FIRE": ""},
            id="add single extra fix out of spatial order",
        ),
        pytest.param(
            {"SPIRIT": "passed", "WATER": "next", "FIRE": ""},
            ["AIR", "WATER", "EARTH", "FIRE"],
            {"SPIRIT": "passed", "AIR": "next", "WATER": "", "EARTH": "", "FIRE": ""},
            id="add_multiple_extra_fixes_including_filed_fixes_before_resume",
        ),
    ],
)

def test_progression_route_directs(generate_i_air0_south, initial_state, route_direct_value, expected_state):
    """Test the state of the route status following a route-direct action"""
    airspace, air0 = generate_i_air0_south
    distance_threshold = 10

    # Initial state - AIR0 at SPIRIT and remove AIR and EARTH from routes
    initial_fixes = list(initial_state)
    starting_fix = [k for k, v in initial_state.items() if v == "passed"][-1]
    next_fix = next(k for k, v in initial_state.items() if v == "next")
    next_fix_idx = initial_fixes.index(next_fix)

    air0.lat, air0.lon = airspace.fixes.places[starting_fix].location
    air0.flight_plan.route.filed = initial_fixes
    air0.flight_plan.route.current = initial_fixes
    air0.update_route_status(airspace, distance_threshold_NMI=distance_threshold)
    air0.next_fix_index = next_fix_idx
    assert OrderedDict(air0.route_status) == OrderedDict(initial_state)

    # Route direct to extra fixes
    env = Environment(time=0.0, airspace=airspace, aircraft={"AIR0": air0})
    route_direct_action = Action("AIR0", "route_direct_to", route_direct_value)
    air0.pilot.receive_actions([route_direct_action], env)
    air0.pilot.process_actions(env)
    air0.update_route_status(airspace, distance_threshold_NMI=distance_threshold)

    post_route_direct_state = OrderedDict(air0.route_status)
    assert post_route_direct_state == OrderedDict(expected_state)

def test_airway_generation(generate_i):
    airspace, routes = generate_i

    env = Custom(1, airspace=airspace, routes=routes).create_env_manager().environment
    sector = env.airspace.sectors["sector_i"]

    fixes_in_sector = [val for key, val in env.airspace.fixes.places.items() if key in ["AIR", "WATER", "EARTH"]]
    fixes_not_in_sector = [
        val for key, val in env.airspace.fixes.places.items() if key not in ["AIR", "WATER", "EARTH"]
    ]

    aircraft = env.aircraft["AIR0"]
    assert aircraft.flight_plan is not None

    aircraft.flight_plan.unexpanded_route = "FIRE UL1 SPIRIT"

    airways: list[Airway] = airspace.flight_plan_airways(aircraft.flight_plan)

    assert len(airways) == 1
    assert airways[0].identifier == "UL1"

    for airway in airways:
        volumes = Sector(airway.volumes())
        fixes = airway.coords()

        for fix in fixes:
            assert volumes.contains(fix.pos3d(150))

        # Get all the airway legs which are within a volume of airspace
        volumes = Sector(airway.volumes(inside_sector=sector))

        for fix in fixes_in_sector:
            assert volumes.contains(fix.pos3d(150))

        # Get all the airway legs which are outside a volume of airspace
        volumes = Sector(airway.volumes(outside_sector=sector))

    for fix in fixes_not_in_sector:
        assert volumes.contains(fix.pos3d(150))


def test_expand_bandbox_sector_and_get_containing(simple_airspace):
    """
    Expand bandbox sectors and resolve containing sectors, including fallbacks.
    """
    coords = [
        Pos2D(lat=0.0, lon=0.0),
        Pos2D(lat=0.0, lon=1.0),
        Pos2D(lat=1.0, lon=1.0),
        Pos2D(lat=1.0, lon=0.0),
        Pos2D(lat=0.0, lon=0.0),
    ]
    sector2 = Sector([Volume(Area(coords), 0, 100)])
    simple_airspace.sectors["S2"] = sector2
    simple_airspace._individual_sectors["S2"] = sector2
    simple_airspace._airspace_configuration["S2"] = ["S2"]

    simple_airspace.bandbox_sectors({"SB": ["S1", "S2"]})

    assert simple_airspace.expand_bandbox_sector("SB") == ["S1", "S2"]
    assert simple_airspace.get_containing_bandboxed_sector("S1") == "SB"
    assert simple_airspace.get_containing_bandboxed_sector("SB") == "SB"
    assert simple_airspace.get_containing_bandboxed_sector("UNKNOWN") == "background"


def test_combine_two_airspaces_prefixes_fixes(simple_airspace):
    """
    Ensure combined airspaces preserve sectors and prefix fixes from the second airspace.
    """
    airspace1 = simple_airspace
    airspace2 = copy.deepcopy(airspace1)
    airspace2.sectors = {"S2": next(iter(airspace2.sectors.values()))}
    airspace2._individual_sectors = airspace2.sectors.copy()
    airspace2._airspace_configuration = {"S2": ["S2"]}
    airspace2.fixes = Fixes({"F2": Pos2D(lat=2.0, lon=2.0)})

    combined = Airspace.combine_two_airspaces(airspace1, airspace2, new_sector_2_name="S2X", prefix="P_")

    assert set(combined.sectors.keys()) == {"S1", "S2X"}
    assert "FIX1" in combined.fixes.places
    assert "P_F2" in combined.fixes.places


def test_is_obtuse_angle_helper():
    """
    Validate obtuse-angle detection for a simple colinear case.
    """
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([2.0, 0.0])
    assert Airspace._is_obtuse_angle(a, b, c)
