"""Tests for route fix sequencing in the shared turn model (Predictor.update_position_with_turn_model)."""

import pytest

from bluebird_dt.core import Aircraft, Fixes, FlightPlan, Pos2D, Route
from bluebird_dt.predictor import LinearPredictor

# An approach with a 98 degree turn at the second to last fix and only 4 nmi of final leg
# to run, which an aircraft with a large turn radius cannot fly.
SHARP_APPROACH = {
    "ENTRY": Pos2D(2.33235, -2.433833),
    "TURN": Pos2D(2.526833, -2.638767),
    "CORNER": Pos2D(2.597317, -2.649033),
    "END": Pos2D(2.597217, -2.581667),
}


def fly(
    route: list[str], fixes: dict[str, Pos2D], heading: float, cas: float, rate_of_turn: float, seconds: int = 3600
) -> Aircraft:
    """Fly an aircraft from the first fix of its route until it stops route following.

    Speeds are held at ``cas`` and the rate of turn is fixed, so the turn radius is exactly
    ``cas / (radians(rate_of_turn) * 3600)`` and the test does not depend on performance data.
    """

    start = fixes[route[0]]
    aircraft = Aircraft(
        lat=start.lat,
        lon=start.lon,
        fl=220.0,
        heading=heading,
        flight_plan=FlightPlan(Route(route)),
        callsign="TEST1",
        selected_fl=220,
        aircraft_type="A320",
        rate_of_turn=rate_of_turn,
    )
    aircraft.next_fix_index = 1
    aircraft.on_route = True
    aircraft.cleared_instructions.cas = cas
    aircraft.selected_instructions.cas = cas

    predictor = LinearPredictor(dt=1.0, fix_proximity_threshold=2.0, fixes=Fixes(fixes), use_cas_as_tas=True)

    for _ in range(seconds):
        predictor.predict_aircraft(aircraft, 1.0, deepcopy_aircraft=False)
        if not aircraft.on_route:
            break

    return aircraft


def test_aircraft_does_not_orbit_an_unflyable_last_fix():
    """
    An aircraft that cannot make the turn onto its last fix stops route following rather than circling it.
    """

    # 450 kt at 1.5 deg/s gives a ~4.8 nmi turn radius against a 4 nmi final leg.
    aircraft = fly(["ENTRY", "TURN", "CORNER", "END"], SHARP_APPROACH, heading=313.0, cas=450.0, rate_of_turn=1.5)

    assert not aircraft.on_route, "aircraft is still route following - it is orbiting the last fix"
    assert aircraft.next_fix_index is None

    # It should give up near the last fix, not part way down the approach.
    assert Pos2D(aircraft.lat, aircraft.lon).distance(SHARP_APPROACH["END"]) < 15.0


def test_aircraft_that_can_make_the_last_turn_still_reaches_the_fix():
    """
    The same approach flown with a turn radius that fits is unaffected: the aircraft reaches the fix.
    """

    # 250 kt at 6 deg/s gives a ~0.7 nmi turn radius, comfortably inside the final leg.
    aircraft = fly(["ENTRY", "TURN", "CORNER", "END"], SHARP_APPROACH, heading=313.0, cas=250.0, rate_of_turn=6.0)

    assert not aircraft.on_route
    assert Pos2D(aircraft.lat, aircraft.lon).distance(SHARP_APPROACH["END"]) <= 2.0


@pytest.mark.parametrize("leg_nmi", [10.0, 40.0])
def test_straight_in_last_fix_still_uses_the_proximity_threshold(leg_nmi: float):
    """
    An aircraft tracking straight at its last fix ends its route within the proximity threshold, as before.
    """

    fixes = {"START": Pos2D(2.0, -2.0), "END": Pos2D(2.0 + leg_nmi / 60.0, -2.0)}

    aircraft = fly(["START", "END"], fixes, heading=0.0, cas=450.0, rate_of_turn=1.5)

    assert not aircraft.on_route
    assert Pos2D(aircraft.lat, aircraft.lon).distance(fixes["END"]) <= 2.0
