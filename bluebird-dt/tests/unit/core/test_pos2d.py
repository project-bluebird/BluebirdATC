import pytest
from . import core_conftest as vt

from bluebird_dt.core.pos2d import Pos2D
from bluebird_dt.core.pos3d import Pos3D
from bluebird_dt.core.pos4d import Pos4D

# Tests for Pos2D

lat_lon_exceptions = vt.lat_lon_exceptions

@pytest.mark.parametrize(
    """lat_exc,
    long_exc""",
    lat_lon_exceptions,
)
def test_init_exceptions(lat_exc, long_exc) -> None:
    """Test for exceptions in the latitude and longitude input with known exceptions"""
    with pytest.raises(ValueError):
        Pos2D(lat_exc, long_exc)


Pos2D_examples = vt.Pos2D_examples


@pytest.mark.parametrize(
    """latitude,
    longitude,
    str_coords,
    precision,
    round_lat,
    round_lon""",
    Pos2D_examples,
)
def test_Pos2D(
    latitude,
    longitude,
    str_coords,
    precision,
    round_lat,
    round_lon,
) -> None:
    """
    Test functionality of pos2d methods
    A range of longitude and latitude combinations have been chosen
    """
    ip = Pos2D(latitude, longitude)

    # Check from_str
    # Check coordinates input are correctly converted to decimal degrees
    sip = Pos2D.from_str(str_coords)
    assert (sip.lat, sip.lon) == (latitude, longitude)

    # Check __str__
    # Check input coordinates are written to 'human readable' form
    assert str(ip) == str_coords

    # Check the rounding
    rip = Pos2D.__round__(ip, precision)
    assert (rip.lat, rip.lon) == (round_lat, round_lon)

    # Check the pos3d method returns a Pos3D object
    p2d2p3d = ip.pos3d(140)
    assert isinstance(p2d2p3d, Pos3D)

    # Check the pos4d method returns a Pos4D object
    p2d2p4d = p2d2p3d.pos4d(10)
    assert isinstance(p2d2p4d, Pos4D)


incorrect_str = vt.incorrect_string_input


@pytest.mark.parametrize(
    """input_string""",
    incorrect_str,
)
def test_incorrect_str(input_string) -> None:
    """
    Test the behaviour of the pos2d string methods with known incorrect strings
    """
    with pytest.raises(ValueError):
        Pos2D.from_str(str(incorrect_str))


bearing_dist = vt.bearing_and_distance


@pytest.mark.parametrize(
    """lat_A,
    lon_A,
    lat_B,
    lon_B,
    bear_A2B,
    bear_B2A,
    distance""",
    bearing_dist,
)
def test_pos2D_forward_bearing_distance(
    lat_A,
    lon_A,
    lat_B,
    lon_B,
    bear_A2B,
    bear_B2A,
    distance,
) -> None:
    """
    Test functionality of pos2d methods
    A range of longitude and latitude combinations have been chosen
    The bearings, heading, and distances have been independently checked
    """
    A = Pos2D(lat_A, lon_A)
    B = Pos2D(lat_B, lon_B)
    C = A.forward(distance, bear_A2B)
    D = B.forward(distance, bear_B2A)
    assert A.bearing_to(B) == pytest.approx(bear_A2B)
    assert B.bearing_to(A) == pytest.approx(bear_B2A)
    assert A.distance(B) == pytest.approx(distance)
    assert A.distance(B) == B.distance(A)
    assert (C.lat, C.lon) == (pytest.approx(B.lat), pytest.approx(B.lon))
    assert (D.lat, D.lon) == (pytest.approx(A.lat), pytest.approx(A.lon))
