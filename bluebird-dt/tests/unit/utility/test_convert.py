import os
import time

import pytest

from bluebird_dt.core.wind import WindVector
from bluebird_dt.utility import convert


def test_time_str() -> None:
    """
    Test ISO string to Posix seconds conversion function
    """
    assert convert.string_to_timestamp("1970-01-01T00:00:02") == 2.0
    assert convert.string_to_timestamp("1980-03-24T12:50:33") == 322750233.0
    assert convert.string_to_timestamp("2023-01-01T03:12:44.456772") == 1672542764.456772


def test_string_to_timestamp_assumes_utc() -> None:
    """
    Naive ISO strings (with no timezone offset) must be interpreted as UTC, regardless
    of the host's local timezone. Forcing a non-UTC TZ guards the explicit
    ``tzinfo=timezone.utc`` in the implementation: a naive-local interpretation would
    give a different epoch.
    """
    original_tz = os.environ.get("TZ")
    try:
        os.environ["TZ"] = "America/New_York"
        time.tzset()
        assert convert.string_to_timestamp("1970-01-01T00:00:00") == 0.0
        assert convert.string_to_timestamp("2023-01-01T03:12:44.456772") == 1672542764.456772
    finally:
        if original_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_tz
        time.tzset()


def test_pressure_from_fl() -> None:
    """
    Test flight level to pressure conversion function
    """
    press_low = convert.pressure_from_fl(flight_level=120)
    press_high = convert.pressure_from_fl(flight_level=380)
    assert press_low == pytest.approx(64440.83)
    assert press_high == pytest.approx(20646.15)


def test_temperature_from_fl() -> None:
    """
    Test temperature at two different flight_levels in a non ISA
    atmosphere.
    """
    temp_low = convert.temperature_from_fl(flight_level=120, delta_T=10.0)
    temp_high = convert.temperature_from_fl(flight_level=380, delta_T=10.0)
    assert temp_low == pytest.approx(274.3756)
    assert temp_high == pytest.approx(226.6500)


def test_density_from_fl() -> None:
    """
    Test density at two different flight_levels in a non ISA
    atmosphere.
    """
    rho_low = convert.density_from_fl(flight_level=120, delta_T=10.0)
    rho_high = convert.density_from_fl(flight_level=380, delta_T=10.0)
    assert rho_low == pytest.approx(0.8181892)
    assert rho_high == pytest.approx(0.3173375)


def test_geopot_altitude_from_fl() -> None:
    """
    Test geopotential altitude at two different flight_levels in a
    non ISA atmosphere.
    """
    alt_low = convert.geopot_altitude_from_fl(flight_level=120, delta_T=10.0, delta_P=300.0)
    alt_high = convert.geopot_altitude_from_fl(flight_level=380, delta_T=10.0, delta_P=300.0)
    assert alt_low == pytest.approx(3815.885)
    assert alt_high == pytest.approx(11609.28)


def test_geodet_altitude_from_fl() -> None:
    """
    Test geodetic altitude at two different flight_levels in a
    non ISA atmosphere.
    """
    alt_low = convert.geodet_altitude_from_fl(flight_level=120, delta_T=10.0, delta_P=300.0)
    alt_high = convert.geodet_altitude_from_fl(flight_level=380, delta_T=10.0, delta_P=300.0)
    assert alt_low == pytest.approx(3818.188)
    assert alt_high == pytest.approx(11630.63)


def test_cas_to_tas() -> None:
    """
    Tests conversion of calibrated air speed to true air speed.
    """
    true_air_speed = convert.cas_to_tas(flight_level=200.0, calibrated_air_speed=200, delta_T=10.0)
    assert true_air_speed == pytest.approx(267.9501)


def test_tas_to_cas() -> None:
    """
    Tests conversion of true air speed to calibrated air speed.
    """
    calibrated_air_speed = convert.tas_to_cas(flight_level=200.0, true_air_speed=267.95, delta_T=10.0)
    assert calibrated_air_speed == pytest.approx(200.0000)


def test_tas_to_mach() -> None:
    """
    Tests conversion of true air speed to mach.
    """
    mach = convert.tas_to_mach(flight_level=200.0, true_air_speed=280.0, delta_T=10.0)
    assert mach == pytest.approx(0.8686823)


def test_mach_to_tas() -> None:
    """
    Tests conversion of true air speed to mach.
    """
    tas = convert.mach_to_tas(flight_level=200.0, mach=0.8686823, delta_T=10.0)
    assert tas == pytest.approx(280.0000)


def test_mach_cas_trans_altitude() -> None:
    """
    Tests method to find transition altitude for given mach number and calibrated
    air speed
    """
    # Below the tropopause
    transition_pressure_altitude = convert.mach_cas_trans_altitude(mach=0.75, calibrated_air_speed=200.0, delta_T=10.0)
    assert transition_pressure_altitude == pytest.approx(4483.489)

    # Above the tropopause
    transition_pressure_altitude = convert.mach_cas_trans_altitude(mach=0.90, calibrated_air_speed=150.0, delta_T=10.0)
    assert transition_pressure_altitude == pytest.approx(11498.10)


def test_horizontal_tas() -> None:
    """
    Tests method to find horizontal true air speed form the total true air speed,
    and the vertical speed
    """
    assert convert.horizontal_tas(tas=240.0, vertical_speed=1500.0) == pytest.approx(239.5425)


@pytest.mark.parametrize(
    ("horizontal_tas", "heading", "wind_speed_kts", "wind_direction", "ground_speed", "ground_track_angle"),
    [
        (240.0, 0.0, 30.0, 0.0, 210, 0.0),
        (240.0, 0.0, 30.0, 180.0, 270, 360.0),
        (240.0, 330.0, 30.0, 330.0, 210, 330.0),
        (240.0, 330.0, 30.0, 150.0, 270, 330.0),
        (240.0, 330.0, 30.0, 270.0, 226.4950, 336.5868),
        (240.0, 165.0, 45.0, 25.0, 275.9920, 171.0160),
    ],
)
def test_ground_speed_from_tas(
    horizontal_tas: float,
    heading: float,
    wind_speed_kts: float,
    wind_direction: float,
    ground_speed: float,
    ground_track_angle: float,
) -> None:
    """
    Tests method for converting ground to tas
    """
    wind_vector = WindVector.from_polar(wind_speed=wind_speed_kts * convert.KT_TO_MPS, wind_direction=wind_direction)
    assert convert.ground_speed_from_tas(
        horizontal_tas=horizontal_tas, heading=heading, wind_vector=wind_vector
    ) == pytest.approx((ground_speed, ground_track_angle))


def test_ground_speed_from_tas_no_wind() -> None:
    """
    With no wind, ground speed equals horizontal TAS and ground track angle equals heading.
    """
    ground_speed, ground_track_angle = convert.ground_speed_from_tas(
        horizontal_tas=240.0, heading=123.0, wind_vector=None
    )
    assert ground_speed == pytest.approx(240.0)
    assert ground_track_angle == pytest.approx(123.0)


@pytest.mark.parametrize(
    ("ground_track_angle", "horizontal_tas", "wind_speed_kts", "wind_direction", "heading"),
    [
        (0.0, 240.0, 30.0, 0.0, 360.0),
        (0.0, 240.0, 30.0, 180.0, 0.0),
        (330.0, 240.0, 30.0, 330.0, 330.0),
        (330.0, 240.0, 30.0, 150.0, 330.0),
        (336.0, 240.0, 30.0, 270.0, 329.4429),
        (171.0, 240.0, 45.0, 25.0, 164.9816),
    ],
)
def test_heading_from_ground_track(
    ground_track_angle: float, horizontal_tas: float, wind_speed_kts: float, wind_direction: float, heading: float
) -> None:
    """
    Tests method for determining required heading for a given ground track angle in the presence of wind
    """
    wind_vector = WindVector.from_polar(wind_speed=wind_speed_kts * convert.KT_TO_MPS, wind_direction=wind_direction)
    assert convert.heading_from_ground_track(
        ground_track_angle=ground_track_angle, horizontal_tas=horizontal_tas, wind_vector=wind_vector
    ) == pytest.approx(heading)


def test_heading_from_ground_track_no_wind() -> None:
    """
    With no wind, the required heading equals the requested ground track angle.
    """
    assert convert.heading_from_ground_track(
        ground_track_angle=123.0, horizontal_tas=240.0, wind_vector=None
    ) == pytest.approx(123.0)


def test_heading_from_ground_track_wind_too_large() -> None:
    """
    When the wind is too strong to achieve the requested ground track angle, the
    aircraft flies directly into the wind (heading equals the direction the wind is
    coming from). Here a 150 m/s wind against 240 kt TAS makes the correction angle
    unachievable (|sin(correction_angle)| > 1).
    """
    wind_vector = WindVector.from_polar(wind_speed=150.0, wind_direction=270.0)
    assert convert.heading_from_ground_track(
        ground_track_angle=0.0, horizontal_tas=240.0, wind_vector=wind_vector
    ) == pytest.approx(wind_vector.direction_wind_from)

def test_heading_from_ground_track_at_correction_limit() -> None:
    """
    At the boundary where the crosswind component exactly equals the TAS, the correction
    angle is exactly +/-90 degrees and the ground track is *still achievable* -- so the
    strict ``> 1.0`` guard must NOT fall back to flying into the wind.

    Ground track due north with an eastward crosswind equal to the TAS requires a heading
    of due west (270), cancelling the crosswind, regardless of the (along-track) northward
    wind component. A non-zero northward component is included specifically so the result
    differs from ``direction_wind_from`` (which the fallback branch would return).
    """
    tas = 240.0
    component = tas * convert.KT_TO_MPS  # eastward crosswind == |TAS| => |sin(correction)| == 1 exactly
    wind_vector = WindVector(u_comp=component, v_comp=component)
    assert convert.heading_from_ground_track(
        ground_track_angle=0.0, horizontal_tas=tas, wind_vector=wind_vector
    ) == pytest.approx(270.0)