from . import core_conftest as vt
import pytest
import numpy as np

from bluebird_dt.core.pos2d import Pos2D
from bluebird_dt.core.pos3d import Pos3D
from bluebird_dt.core.pos4d import Pos4D
from bluebird_dt.utility import convert

lat_lon_fl_time_exceptions = vt.lat_lon_fl_time_exceptions


@pytest.mark.parametrize(
    """lat_exc,
    long_exc,
    FL_exc,
    time_exc""",
    lat_lon_fl_time_exceptions,
)
def test_init_exceptions(lat_exc, long_exc, FL_exc, time_exc) -> None:
    """Test for exceptions in the latitude, longitude, flight level, and time input with known exceptions"""
    with pytest.raises(ValueError):
        Pos4D(lat_exc, long_exc, FL_exc, time_exc)


Pos4D_examples = vt.Pos4D_examples


@pytest.mark.parametrize(
    """latitude,
    longitude,
    flight_level,
    time,
    str_coords,
    precision,
    round_lat,
    round_lon,
    round_fl,
    round_time""",
    Pos4D_examples,
)
def test_Pos4D(
    latitude, longitude, flight_level, time, str_coords, precision, round_lat, round_lon, round_fl, round_time
) -> None:
    """
    Test functionality of pos4d methods
    A range of longitude latitude flight level and time combinations have been chosen
    """
    ip = Pos4D(latitude, longitude, flight_level, time)

    # Check from_str
    # Check coordinates input are correctly converted to decimal degrees
    sip = Pos4D.from_str(str_coords)
    assert (sip.lat, sip.lon, sip.fl, sip.time) == (latitude, longitude, flight_level, time)

    # Check __str__
    # Check input coordinates are written to 'human readable' form
    assert str(ip) == str_coords

    # Check the rounding
    rip = Pos4D.__round__(ip, precision)
    assert (rip.lat, rip.lon, rip.fl, rip.time) == (round_lat, round_lon, round_fl, round_time)

    # Get a pos2D object from a pos4D object
    p4d2p2d = ip.pos2d()
    assert isinstance(p4d2p2d, Pos2D)

    # Get a pos3D object from a pos4D object
    p4d2p3d = ip.pos3d()
    assert isinstance(p4d2p3d, Pos3D)


Pos4D_incorrect_string_input = vt.Pos4D_incorrect_string_input


@pytest.mark.parametrize(
    """input_string""",
    Pos4D_incorrect_string_input,
)
def test_incorrect_str(input_string) -> None:
    """
    Test the behaviour of the pos3d string methods with known incorrect strings
    """
    with pytest.raises(ValueError):
        Pos3D.from_str(str(input_string))


def test_pos4d_json_round_trip() -> None:
    """
    Check that from and to json work correctly
    """
    pos = Pos4D(10.0, -2.0, 120.0, 50.0)
    as_json = pos.to_json()
    restored = Pos4D.from_json(as_json)
    assert restored == pos
    
def test_pos4d_from_array() -> None:
    with pytest.raises(ValueError):
        Pos4D.from_array(np.array([1.0, 2.0, 3.0]))


def test_pos4d_initialisation() -> None:
    """
    Test exception thrown for wrong number of inputs
    """
    with pytest.raises(TypeError):
        Pos4D(1.0, 2.0, 3.0)


def test_pos4d_save_load_round_trip(tmp_path) -> None:
    """
    Test save and load methods work as expected
    """
    pos = Pos4D(10.0, -2.0, 120.0, 50.0)
    path = tmp_path / "pos4d.json"
    pos.save(str(path))

    loaded = Pos4D.load(str(path))
    assert loaded == pos


@pytest.mark.parametrize(
    ("start", "end", "speed", "expected_interval"),
    [
        pytest.param(
            Pos4D(0.0, 0.0, 200.0, 0.0),
            Pos4D(0.0, 0.0, 200.0, 0.0),
            120.0,
            0.0,
            id="same_point",
        ),
        pytest.param(
            Pos4D(0.0, 0.0, 200.0, 0.0),
            Pos4D(0.0, 1.0, 200.0, 0.0),
            120.0,
            0.5,
            id="one_degree_east_at_equator",
        ),
        pytest.param(
            Pos4D(0.0, 0.0, 200.0, 0.0),
            Pos4D(1.0, 0.0, 200.0, 0.0),
            120.0,
            0.4975,
            id="one_degree_north",
        ),
        pytest.param(
            Pos4D(0.0, 0.0, 200.0, 0.0),
            Pos4D(0.0, 2.0, 200.0, 0.0),
            240.0,
            0.5,
            id="two_degrees_east_faster_speed",
        ),
    ],
)
def test_interval(start: Pos4D, end: Pos4D, speed: float, expected_interval: float) -> None:
    """
    Test interval calculations for multiple known distance and speed combinations.
    """
    assert start.interval(end, speed=speed) == pytest.approx(expected_interval, abs=1e-3)

@pytest.mark.parametrize(
    "heading, expected_lat_sign, expected_lon_sign",
    [
        (0.0, 1, 0),   # north
        (90.0, 0, 1),  # east
    ],
  )
def test_forward(heading: float, expected_lat_sign: int, expected_lon_sign: int) -> None:
    """
    Call forward() with various inputs and check expected results obtained
    """

    pos = Pos4D(0.0, 0.0, 200.0, 0.0)
    moved = pos.forward(dist=60.0, heading=90.0, speed=120.0)
    # time should always advance by 0.5 hours at 120 kts
    assert moved.fl == pos.fl
    assert moved.time == pytest.approx(0.5 * convert.HRS_TO_SEC)

    moved = pos.forward(dist=60.0, heading=heading, speed=120.0)
    assert moved.time == pytest.approx(0.5 * convert.HRS_TO_SEC, abs=1e-6)

    # check directionality
    if expected_lat_sign:
        assert (moved.lat - pos.lat) * expected_lat_sign > 0
    if expected_lon_sign:
        assert (moved.lon - pos.lon) * expected_lon_sign > 0


# Testing forward, bearing, and distance
Pos4D_forward_bearing_distance_interval = vt.Pos4D_forward_bearing_distance_interval
