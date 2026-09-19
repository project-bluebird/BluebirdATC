from unittest.mock import Mock

import numpy as np
import pytest

from bluebird_dt.airspace_generator.artificial_airspace import ArtificialAirspace
from bluebird_dt.scenario_manager import Infinite, Tactical


@pytest.fixture(params=[0.0, 50.716667, -50.716667, 75.0])
def lateral_airspace(request):
    return ArtificialAirspace("x", origin=(-3.533333, request.param)).generate_airspace()


@pytest.mark.parametrize("manager_class", [Infinite, Tactical])
@pytest.mark.parametrize("sector_type", ["i", "x", "y"])
@pytest.mark.parametrize("latitude", [0.0, 50.716667, -50.716667, 75.0])
def test_lateral_headings_are_perpendicular(manager_class, sector_type, latitude):
    airspace, routes = ArtificialAirspace(sector_type, origin=(-3.533333, latitude)).generate_airspace()
    if manager_class is Infinite:
        manager = Infinite(airspace=airspace, routes=routes)
        headings = manager.setup_lateral_offset_headings()
    else:
        manager = Tactical(1, airspace=airspace, routes=routes)
        headings = manager.set_up_lateral_start_points(airspace)

    assert set(headings) == {route.filed[0] for route in routes}
    for route in routes:
        first, second = (airspace.fixes.places[name] for name in route.filed[:2])
        route_bearing = first.bearing_to(second)
        relative_headings = sorted((heading - route_bearing) % 360 for heading in headings[route.filed[0]])
        assert relative_headings == pytest.approx([90.0, 270.0], abs=1e-8)
        assert all(0.0 <= heading < 360.0 for heading in headings[route.filed[0]])


@pytest.mark.parametrize("manager_class", [Infinite, Tactical])
def test_lateral_headings_reject_coincident_fixes(manager_class, generate_i):
    airspace, routes = generate_i
    first, second = (airspace.fixes.places[name] for name in routes[0].filed[:2])
    second.lat, second.lon = first.lat, first.lon

    with pytest.raises(ValueError, match="start and end points must be different"):
        if manager_class is Infinite:
            Infinite(airspace=airspace, routes=routes)
        else:
            Tactical(1, airspace=airspace, routes=routes).set_up_lateral_start_points(airspace)


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("distance", [0.0, 10.0])
def test_tactical_lateral_spawn(lateral_airspace, side, distance, monkeypatch):
    airspace, routes = lateral_airspace
    manager = Tactical(1, airspace=airspace, routes=routes, lateral_offset=(distance, distance))
    monkeypatch.setattr(np.random, "choice", lambda headings: headings[side])

    for route in routes:
        first, second = (airspace.fixes.places[name] for name in route.filed[:2])
        position = manager.stochastic_start_pos(airspace, route)

        assert first.distance(position) == pytest.approx(distance, abs=1e-8)
        if distance:
            angle = (first.bearing_to(position) - first.bearing_to(second)) % 360
            assert angle == pytest.approx([90.0, 270.0][side], abs=1e-8)


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("aircraft_on_route", [False, True])
def test_infinite_lateral_spawn(lateral_airspace, side, aircraft_on_route):
    airspace, routes = lateral_airspace
    manager = Infinite(airspace=airspace, routes=routes, aircraft_on_route=aircraft_on_route)
    # Choose which side of the route to test, and use fixed choices for the route and altitude.
    manager.rng = Mock()
    manager.rng.choice.side_effect = lambda values: values[side % len(values)]

    for route in routes:
        # Use a speed of 400 knots and place the aircraft 10 nautical miles to the side.
        manager.rng.uniform.side_effect = [400.0, 10.0]
        aircraft, _, _ = manager.create_aircraft_with_coordinations(
            [route], callsign="TEST", spawn_distance_behind_fix=0.0
        )
        first, second = (airspace.fixes.places[name] for name in route.filed[:2])
        position = aircraft.pos2d()

        assert first.distance(position) == pytest.approx(0.0 if aircraft_on_route else 10.0, abs=1e-8)
        assert aircraft.on_route == aircraft_on_route
        if not aircraft_on_route:
            angle = (first.bearing_to(position) - first.bearing_to(second)) % 360
            assert angle == pytest.approx([90.0, 270.0][side], abs=1e-8)
