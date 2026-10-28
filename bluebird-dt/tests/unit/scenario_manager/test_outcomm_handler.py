import pytest

from bluebird_dt.airspace_generator import SectorI
from bluebird_dt.airspace_generator.artificial_airspace import ArtificialAirspace
from bluebird_dt.scenario_manager.outcomm_handler import ExitGate, OutcommHandler
from bluebird_dt.core import Aircraft, Coordination, FlightPlan, Pos2D, Route


@pytest.fixture
def gate() -> ExitGate:
    """Create a north-south gate around longitude 0."""
    return ExitGate.new(
        start_position=Pos2D(lon=0.0, lat=-0.1),
        end_position=Pos2D(lon=0.0, lat=0.1),
    )


def test_is_crossed_when_moving_left_to_right(gate: ExitGate) -> None:
    """Aircraft crossing from the permitted side should cross the gate."""
    previous = Pos2D(lon=-0.1, lat=0.0)
    current = Pos2D(lon=0.1, lat=0.0)

    assert gate.is_crossed(previous, current)


def test_is_not_crossed_when_moving_right_to_left(gate: ExitGate) -> None:
    """Crossing the gate in the opposite direction should not be permitted."""
    previous = Pos2D(lon=0.1, lat=0.0)
    current = Pos2D(lon=-0.1, lat=0.0)

    assert not gate.is_crossed(previous, current)


def test_is_not_crossed_when_movement_does_not_intersect_gate(
    gate: ExitGate,
) -> None:
    """Movement entirely on one side of the gate should not cross it."""
    previous = Pos2D(lon=-0.2, lat=0.0)
    current = Pos2D(lon=-0.1, lat=0.0)

    assert not gate.is_crossed(previous, current)


def test_is_not_crossed_outside_gate_segment(gate: ExitGate) -> None:
    """Crossing the gate's infinite line outside the segment is not a crossing."""
    previous = Pos2D(lon=-0.1, lat=0.2)
    current = Pos2D(lon=0.1, lat=0.2)

    assert not gate.is_crossed(previous, current)


def test_is_not_crossed_when_movement_ends_on_gate(gate: ExitGate) -> None:
    """Merely reaching the gate is not considered crossing it."""
    previous = Pos2D(lon=-0.1, lat=0.0)
    current = Pos2D(lon=0.0, lat=0.0)

    assert not gate.is_crossed(previous, current)


def test_flipping_gate_reverses_crossing_direction(gate: ExitGate) -> None:
    """Flipping the gate should reverse its permitted crossing direction."""
    previous = Pos2D(lon=-0.1, lat=0.0)
    current = Pos2D(lon=0.1, lat=0.0)

    assert gate.is_crossed(previous, current)
    assert not gate.flip().is_crossed(previous, current)

def test_crosses_very_near_start_of_gate(gate: ExitGate) -> None:
    assert gate.is_crossed(
        previous_position=Pos2D(lon=-0.001, lat=-0.1 + 1e-6),
        current_position=Pos2D(lon=0.001, lat=-0.1 + 1e-6),
    )


def test_crosses_very_near_end_of_gate(gate: ExitGate) -> None:
    assert gate.is_crossed(
        previous_position=Pos2D(lon=-0.001, lat=0.1 - 1e-6),
        current_position=Pos2D(lon=0.001, lat=0.1 - 1e-6),
    )


def test_does_not_cross_just_past_start_of_gate(gate: ExitGate) -> None:
    assert not gate.is_crossed(
        previous_position=Pos2D(lon=-0.001, lat=-0.1 - 1e-6),
        current_position=Pos2D(lon=0.001, lat=-0.1 - 1e-6),
    )


def test_does_not_cross_just_past_end_of_gate(gate: ExitGate) -> None:
    assert not gate.is_crossed(
        previous_position=Pos2D(lon=-0.001, lat=0.1 + 1e-6),
        current_position=Pos2D(lon=0.001, lat=0.1 + 1e-6),
    )


def test_flip_reverses_gate(gate: ExitGate) -> None:
    flipped = gate.flip()

    assert flipped.start_position == gate.end_position
    assert flipped.end_position == gate.start_position


def test_flip_twice_returns_original_gate(gate: ExitGate) -> None:
    assert gate.flip().flip() == gate


def test_oriented_for_approach_keeps_gate_when_approaching_from_west(
    gate: ExitGate,
) -> None:
    oriented = gate.oriented_for_approach(
        Pos2D(lon=-0.1, lat=0.0),
    )

    assert oriented == gate


def test_oriented_for_approach_flips_gate_when_approaching_from_east(
    gate: ExitGate,
) -> None:
    oriented = gate.oriented_for_approach(
        Pos2D(lon=0.1, lat=0.0),
    )

    assert oriented == gate.flip()


@pytest.mark.parametrize(
    "approach_position",
    [
        Pos2D(lon=0.0, lat=0.0),
        Pos2D(lon=0.0, lat=-0.2),
        Pos2D(lon=0.0, lat=0.2),
    ],
)
def test_oriented_for_approach_raises_when_collinear(
    gate: ExitGate,
    approach_position: Pos2D,
) -> None:
    with pytest.raises(
        ValueError,
        match="route approach position is collinear with the gate",
    ):
        gate.oriented_for_approach(approach_position)

@pytest.mark.parametrize(
    "previous_position,current_position,outcomms",
    [
        # Perpendicular crossing near eastern edge
        (
            Pos2D(lon=0.0167, lat=0.033273597890004086 - 1e-6),
            Pos2D(lon=0.0167, lat=0.033273597890004086 + 1e-6),
            True
        ),

        # Perpendicular crossing near western edge
        (
            Pos2D(lon=-0.0167, lat=0.033273597890004086 - 1e-6),
            Pos2D(lon=-0.0167, lat=0.033273597890004086 + 1e-6),
            True
        ),

        # Very close to eastern endpoint
        (
            Pos2D(lon=0.0167489211451249 - 1e-6, lat=0.033273597890004086 - 1e-6),
            Pos2D(lon=0.0167489211451249 - 1e-6, lat=0.033273597890004086 + 1e-6),
            True
        ),

        # Very close to western endpoint
        (
            Pos2D(lon=-0.016748921145137624 + 1e-6, lat=0.033273597890004086 - 1e-6),
            Pos2D(lon=-0.016748921145137624 + 1e-6, lat=0.033273597890004086 + 1e-6),
            True
        ),

        # Diagonal crossing near eastern edge: southwest -> northeast
        (
            Pos2D(lon=0.0167 - 1e-6, lat=0.033273597890004086 - 1e-6),
            Pos2D(lon=0.0167 + 1e-6, lat=0.033273597890004086 + 1e-6),
            True
        ),

        # Opposite diagonal near eastern edge: southeast -> northwest
        (
            Pos2D(lon=0.0167 + 1e-6, lat=0.033273597890004086 - 1e-6),
            Pos2D(lon=0.0167 - 1e-6, lat=0.033273597890004086 + 1e-6),
            True
        ),

        # Diagonal crossing near western edge
        (
            Pos2D(lon=-0.0167 - 1e-6, lat=0.033273597890004086 - 1e-6),
            Pos2D(lon=-0.0167 + 1e-6, lat=0.033273597890004086 + 1e-6),
            True
        ),

        # Shallow-angle crossing near eastern edge
        (
            Pos2D(lon=0.0166, lat=0.033273597890004086 - 1e-7),
            Pos2D(lon=0.0168, lat=0.033273597890004086 + 1e-7),
            True
        ),

        # Shallow-angle crossing near western edge
        (
            Pos2D(lon=-0.0168, lat=0.033273597890004086 - 1e-7),
            Pos2D(lon=-0.0166, lat=0.033273597890004086 + 1e-7),
            True
        ),
        # Reversed shallow-angle crossing near western edge
        (
            Pos2D(lon=-0.0166, lat=0.033273597890004086 + 1e-7),
            Pos2D(lon=-0.0168, lat=0.033273597890004086 - 1e-7),
            False
        ),
    ],
)
def test_outcomm_handler_near_edge(
        previous_position: Pos2D, 
        current_position: Pos2D, 
        outcomms: bool,
        ) -> None:
    airspace, _ = SectorI(2, 4, (200, 300)).generate_airspace()
    sector = airspace.sectors.get("sector_i")

    assert sector is not None

    aircraft = Aircraft(current_position.lat, current_position.lon, 250, 360, FlightPlan(Route(["FIRE", "EARTH", "WATER", "AIR", "SPIRIT"])), "AIR01", current_sector="sector_i")
    coordination = Coordination(callsign="AIR01", from_sector="sector_i", to_sector="background", fl=250, fix="AIR", direction="Horizontal")

    outcomm_handler = OutcommHandler()

    outcomm_handler._previous_positions = {"AIR01": previous_position}
    outcomm_handler._previously_aircraft_in_sector = {"sector_i": {"AIR01"}}

    assert (not outcomms) ^ outcomm_handler._should_outcom_aircraft(aircraft, previous_position, sector, "sector_i", coordination, airspace.fixes)

