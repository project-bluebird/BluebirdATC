from . import core_conftest as vt
import pytest

from bluebird_dt.core.pos2d import Pos2D
from bluebird_dt.core.pos3d import Pos3D
from bluebird_dt.core.pos4d import Pos4D

lat_lon_fl_exceptions = vt.lat_lon_fl_exceptions


@pytest.mark.parametrize(
    """lat_exc,
    long_exc,
    FL_exc""",
    lat_lon_fl_exceptions,
)
def test_init_exceptions(lat_exc, long_exc, FL_exc) -> None:
    """Test for exceptions in the latitude, longitude, and flight level input with known exceptions"""
    with pytest.raises(ValueError):
        Pos3D(lat_exc, long_exc, FL_exc)


Pos3D_examples = vt.Pos3D_examples


@pytest.mark.parametrize(
    """latitude,
    longitude,
    flight_level,
    str_coords,
    precision,
    round_lat,
    round_lon,
    round_fl""",
    Pos3D_examples,
)
def test_Pos3D(
    latitude,
    longitude,
    flight_level,
    str_coords,
    precision,
    round_lat,
    round_lon,
    round_fl,
) -> None:
    """
    Test functionality of pos3d methods
    A range of longitude latitude and flight level combinations have been chosen
    """

    ip = Pos3D(latitude, longitude, flight_level)

    # Check from_str
    # Check coordinates input are correctly convened to decimal degrees
    sip = Pos3D.from_str(str_coords)
    assert (sip.lat, sip.lon, sip.fl) == (latitude, longitude, flight_level)

    # Check __str__
    # Check input coordinates are written to 'human readable' form
    assert str(ip) == str_coords

    # Check the rounding
    rip = Pos3D.__round__(ip, precision)
    assert (rip.lat, rip.lon, rip.fl) == (round_lat, round_lon, round_fl)

    # Get a pos2D object from a pos3D object
    p3d2p2d = ip.pos2d()
    assert isinstance(p3d2p2d, Pos2D)

    # Get a pos4D object from a pos3D object
    p3d2p4d = ip.pos4d(10)
    assert isinstance(p3d2p4d, Pos4D)

    # Get a pos3d from a pos2d object and flight level
    p2d_ip = Pos2D(latitude, longitude)
    p3dfrom2d = p2d_ip.pos3d(flight_level)
    assert str(p3dfrom2d) == str_coords


Pos3D_incorrect_string = vt.Pos3D_incorrect_string_input


@pytest.mark.parametrize(
    """input_string""",
    Pos3D_incorrect_string,
)
def test_incorrect_str(input_string) -> None:
    """
    Test the behaviour of the pos3d string methods with known incorrect strings
    """
    with pytest.raises(ValueError):
        Pos3D.from_str(str(input_string))


# Testing forward, bearing, and distance
Pos3D_forward_bearing_and_distance = vt.Pos3D_forward_bearing_and_distance


@pytest.mark.parametrize(
    """
    latA,
    lonA,
    flA,
    latB,
    lonB,
    flB,
    bear_A2B,
    bear_B2A,
    distance
    """,
    Pos3D_forward_bearing_and_distance,
)
def test_pos3D_forward_bearing_distance(
    latA,
    lonA,
    flA,
    latB,
    lonB,
    flB,
    bear_A2B,
    bear_B2A,
    distance,
) -> None:
    """
    Test functionality of pos3d methods
    A range of longitude latitude and flight level combinations have been chosen
    The bearings, heading, and distances have been independently checked
    """
    A = Pos3D(latA, lonA, flA)
    B = Pos3D(latB, lonB, flB)
    C = A.forward(distance, bear_A2B)
    D = B.forward(distance, bear_B2A)

    assert A.distance(B) == pytest.approx(distance)
    assert A.distance(B) == B.distance(A)
    assert A.bearing_to(B) == pytest.approx(bear_A2B)
    assert B.bearing_to(A) == pytest.approx(bear_B2A)
    assert (C.lat, C.lon) == (pytest.approx(B.lat), pytest.approx(B.lon))
    assert (D.lat, D.lon) == (pytest.approx(A.lat), pytest.approx(A.lon))
