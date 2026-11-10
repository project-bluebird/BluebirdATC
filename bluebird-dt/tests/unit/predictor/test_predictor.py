"""Tests for the shared turn model (Predictor.update_position_with_turn_model)."""

import pytest

from bluebird_dt.core import Action, Aircraft, Environment, Fixes, FlightPlan, Pos2D, Route
from bluebird_dt.predictor import LinearPredictor
from bluebird_dt.utility.clearance import which_way

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


def fly_heading_clearance(
    environment: Environment, heading: float, kind: str, value: float | tuple[float, str], seconds: int
) -> list[float]:
    """Clear AIR0, flying ``heading`` at 450 kt and 1.5 deg/s with no wind, and return its heading after each second."""

    aircraft = environment.aircraft["AIR0"]
    aircraft.heading = heading
    aircraft.rate_of_turn = 1.5
    aircraft.cleared_instructions.cas = 450.0
    aircraft.selected_instructions.cas = 450.0
    aircraft.pilot.process_lateral_actions(Action("AIR0", kind, value), environment)

    predictor = LinearPredictor(
        dt=1.0, fix_proximity_threshold=2.0, fixes=environment.airspace.fixes, use_cas_as_tas=True
    )

    headings = []
    for _ in range(seconds):
        predictor.predict_aircraft(aircraft, 1.0, deepcopy_aircraft=False)
        headings.append(aircraft.heading)

    return headings


@pytest.mark.parametrize(("heading", "change", "after_one_second"), [(100.0, 200.0, 101.5), (90.0, -180.0, 88.5)])
def test_change_heading_by_turns_the_way_of_its_sign(
    heading: float, change: float, after_one_second: float, generate_simple_environment: Environment
):
    """
    change_heading_by turns right for a positive change and left for a negative one, even by 180 degrees or more.
    """

    headings = fly_heading_clearance(generate_simple_environment, heading, "change_heading_by", change, seconds=140)

    assert headings[0] == pytest.approx(after_one_second)
    assert headings[-1] == pytest.approx((heading + change) % 360)


@pytest.mark.parametrize("target", [101.0, 110.0])
def test_left_turn_to_a_heading_just_right_goes_the_long_way(target: float, generate_simple_environment: Environment):
    """
    A left turn onto a heading a little to the right turns all the way round, rather than stopping at once.
    """

    headings = fly_heading_clearance(
        generate_simple_environment, 100.0, "change_heading_to_by_direction", (target, "left"), seconds=250
    )

    assert headings[0] == pytest.approx(98.5)
    assert headings[-1] == pytest.approx(target)


@pytest.mark.parametrize(("heading", "target"), [(270.0, 90.0), (90.0, 270.0)])
def test_reversal_turns_the_way_the_clearance_says(
    heading: float, target: float, generate_simple_environment: Environment
):
    """
    On a turn of exactly 180 degrees the aircraft turns the way the clearance phraseology says.
    """

    headings = fly_heading_clearance(generate_simple_environment, heading, "change_heading_to", target, seconds=1)

    turned = "right" if (headings[0] - heading) % 360 < 180 else "left"
    assert turned == which_way(heading, target)
