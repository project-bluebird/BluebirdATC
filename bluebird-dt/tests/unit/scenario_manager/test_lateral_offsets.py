from unittest.mock import Mock

import numpy as np
import pytest

from bluebird_dt.airspace_generator.artificial_airspace import ArtificialAirspace
from bluebird_dt.core import Airspace, Route
from bluebird_dt.scenario_manager import Custom, Infinite


@pytest.fixture(params=[0.0, 50.716667, -50.716667, 75.0])
def lateral_airspace(request):
    return ArtificialAirspace("x", origin=(-3.533333, request.param)).generate_airspace()

@pytest.mark.parametrize("distance", [0.0, 5.0, 10.0])
def test_custom_lateral_spawn(lateral_airspace: tuple[Airspace, list[Route]], distance: float):
    airspace, routes = lateral_airspace
    for route in routes:
        sim = Custom(num_aircraft=1, airspace=airspace, routes=[route], lateral_offset=(distance, distance), lateral_buffer_distance=40).to_simulator()

        first, second = (airspace.fixes.places[name] for name in route.filed[:2])
        position = sim.manager.environment.aircraft["AIR0"].pos2d()

        assert first.distance(position) == pytest.approx(distance, abs=1e-6)
        if distance:
            angle = (first.bearing_to(position) - first.bearing_to(second)) % 360
            assert angle % 180.0 == pytest.approx(90.0, abs=1e-6)


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("aircraft_on_route", [False, True])
def test_infinite_lateral_spawn(lateral_airspace: tuple[Airspace, list[Route]], side: int, aircraft_on_route: bool):
    airspace, routes = lateral_airspace
    manager = Infinite(airspace=airspace, routes=routes, aircraft_on_route=aircraft_on_route)
    # Choose which side of the route to test, and use fixed choices for the route and altitude.
    manager.rng = Mock()
    manager.rng.choice.side_effect = lambda values: values[side % len(values)]

    for route in routes:
        # Use a speed of 400 knots and place the aircraft 10 nautical miles to the side.
        manager.rng.uniform.side_effect = [400.0, 10.0]
        aircraft, _, _ = manager.spawn_aircraft(
            [route], callsign="TEST", spawn_distance_behind_fix=0.0
        )
        first, second = (airspace.fixes.places[name] for name in route.filed[:2])
        position = aircraft.pos2d()

        assert first.distance(position) == pytest.approx(0.0 if aircraft_on_route else 10.0, abs=1e-6)
        assert aircraft.on_route == aircraft_on_route
        if not aircraft_on_route:
            angle = (first.bearing_to(position) - first.bearing_to(second)) % 360
            assert angle == pytest.approx([90.0, 270.0][side], abs=1e-6)
