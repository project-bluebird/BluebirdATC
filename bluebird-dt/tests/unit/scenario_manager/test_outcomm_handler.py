import pytest

from bluebird_dt.scenario_manager.outcomm_handler import ExitGate
from bluebird_dt.core import Pos2D


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
