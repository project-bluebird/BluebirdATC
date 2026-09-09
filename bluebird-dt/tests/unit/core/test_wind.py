import json
import math
import os
import shutil

import numpy as np
import pytest
import scipy
from pydantic import ValidationError

from bluebird_dt.core.wind import WindField, WindVector
from bluebird_dt.utility.paths import ROOT_DIR


def test_wind_vector() -> None:
    """
    Test wind vector class objects
    """
    # Create a wind vector by specifying the components
    u_comp = 20.0
    v_comp = -20.0
    wind_vector = WindVector(u_comp=u_comp, v_comp=v_comp)

    # Check components retrieval, wind speed and direction calculation yield correct results
    assert wind_vector.u_comp == u_comp
    assert wind_vector.v_comp == v_comp
    assert wind_vector.speed == math.sqrt(u_comp**2 + v_comp**2)
    assert wind_vector.direction_wind_from == 315.0
    assert wind_vector.direction_wind_to == (315.0 + 180.0) % 360

    # Create a wind vector given wind speed and direction
    wind_speed = 50.0
    wind_direction = 60.0
    wind_vector = WindVector.from_polar(
        wind_speed=wind_speed, wind_direction=wind_direction
    )
    assert wind_vector.speed == pytest.approx(wind_speed)
    assert wind_vector.direction_wind_from == pytest.approx(wind_direction)
    assert wind_vector.direction_wind_to == pytest.approx(
        (wind_direction + 180.0) % 360
    )
    assert wind_vector.u_comp == pytest.approx(-0.5 * math.sqrt(3.0) * wind_speed)
    assert wind_vector.v_comp == pytest.approx(-0.5 * wind_speed)


def test_wind_field_instantiation_exception() -> None:
    """
    Test wind instantiation exceptions
    """
    # Instantiation with negative wind speed
    with pytest.raises(ValueError):
        WindField.uniform(
            wind_speed=-20.0,
            wind_direction=100.0,
            min_lat=-10.0,
            max_lat=20,
            min_lon=10,
            max_lon=30,
            no_grid_points=20,
        )
    # Instantiation with out of range wind direction
    with pytest.raises(ValueError):
        WindField.uniform(
            wind_speed=20.0,
            wind_direction=380.0,
            min_lat=-10.0,
            max_lat=20,
            min_lon=10,
            max_lon=30,
            no_grid_points=20,
        )
    with pytest.raises(ValueError):
        WindField.uniform(
            wind_speed=20.0,
            wind_direction=-100.0,
            min_lat=-10.0,
            max_lat=20,
            min_lon=10,
            max_lon=30,
            no_grid_points=20,
        )
    # Instantiation with latitudes out of range [-90, 90]
    with pytest.raises(ValueError):
        WindField.uniform(
            wind_speed=20.0,
            wind_direction=100.0,
            min_lat=-100.0,
            max_lat=20,
            min_lon=10,
            max_lon=30,
            no_grid_points=20,
        )
    with pytest.raises(ValueError):
        WindField.uniform(
            wind_speed=20.0,
            wind_direction=100.0,
            min_lat=110.0,
            max_lat=20,
            min_lon=10,
            max_lon=30,
            no_grid_points=20,
        )
    with pytest.raises(ValueError):
        WindField.uniform(
            wind_speed=20.0,
            wind_direction=100.0,
            min_lat=-10.0,
            max_lat=95,
            min_lon=10,
            max_lon=30,
            no_grid_points=20,
        )
    # Instantiation with longitudes out of range [-180, 180]
    with pytest.raises(ValueError):
        WindField.uniform(
            wind_speed=20.0,
            wind_direction=100.0,
            min_lat=-10.0,
            max_lat=20,
            min_lon=-185,
            max_lon=30,
            no_grid_points=20,
        )
    with pytest.raises(ValueError):
        WindField.uniform(
            wind_speed=20.0,
            wind_direction=100.0,
            min_lat=-10.0,
            max_lat=20,
            min_lon=30,
            max_lon=360,
            no_grid_points=20,
        )
    with pytest.raises(ValueError):
        WindField.uniform(
            wind_speed=20.0,
            wind_direction=100.0,
            min_lat=-10.0,
            max_lat=20,
            min_lon=10,
            max_lon=-200,
            no_grid_points=20,
        )
    # Instantiation with maximum lat-lon being smaller than minimum values
    # Instantiation with latitudes out of range [-90, 90]
    with pytest.raises(ValueError):
        WindField.uniform(
            wind_speed=20.0,
            wind_direction=100.0,
            min_lat=-10.0,
            max_lat=-20,
            min_lon=10,
            max_lon=30,
            no_grid_points=20,
        )
    with pytest.raises(ValueError):
        WindField.uniform(
            wind_speed=20.0,
            wind_direction=100.0,
            min_lat=-10.0,
            max_lat=20,
            min_lon=10,
            max_lon=5,
            no_grid_points=20,
        )

    # Trying to instantiate with less than two grid points along lat-lon axes:
    with pytest.raises(ValueError):
        WindField.uniform(
            wind_speed=20.0,
            wind_direction=100.0,
            min_lat=-10.0,
            max_lat=20,
            min_lon=10,
            max_lon=100,
            no_grid_points=1,
        )

    # Instantiation of artificial with negative pressure levels
    with pytest.raises(ValueError):
        WindField.artificial(
            wind_speed=20.0,
            wind_direction=100.0,
            pressure_array=np.linspace(-500, 500, 10),
            min_lat=-10.0,
            max_lat=20,
            min_lon=10,
            max_lon=30,
            no_grid_points=20,
        )


def test_artificial_wind_field() -> None:
    """
    Test artificial wind instantiation method yields correct results
    """
    pressure_array = np.array(
        [100000, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 7000, 3000]
    )
    min_lat = -10.0
    max_lat = 20
    min_lon = 10
    max_lon = 30
    no_grid_points = 20
    wind_field = WindField.artificial(
        wind_speed=20.0,
        wind_direction=100.0,
        pressure_array=pressure_array,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        no_grid_points=no_grid_points,
    )

    # Check lat-lon grid minimum and maximum limits are correct
    assert np.min(wind_field.lat_array) == min_lat
    assert np.max(wind_field.lat_array) == max_lat
    assert np.min(wind_field.lon_array) == min_lon
    assert np.max(wind_field.lon_array) == max_lon

    # Check lat and lon arrays are the right size
    assert wind_field.lat_array.size == no_grid_points
    assert wind_field.lon_array.size == no_grid_points

    # Check all u and v wind values have the correct values. Values checked here
    # are for the special 30-60-90 degree triangle in trigonometry, as applied
    # in all 4 quadrants.
    wind_speed = 100.0
    wind_direction = 30.0
    wind_field = WindField.artificial(
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        pressure_array=pressure_array,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        no_grid_points=no_grid_points,
    )
    assert np.all(wind_field.u_comp == pytest.approx(-0.5 * wind_speed))
    assert np.all(
        wind_field.v_comp == pytest.approx(-0.5 * wind_speed * math.sqrt(3.0))
    )

    wind_speed = 100.0
    wind_direction = 60.0
    wind_field = WindField.artificial(
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        pressure_array=pressure_array,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        no_grid_points=no_grid_points,
    )
    assert np.all(
        wind_field.u_comp == pytest.approx(-0.5 * wind_speed * math.sqrt(3.0))
    )
    assert np.all(wind_field.v_comp == pytest.approx(-0.5 * wind_speed))

    wind_speed = 100.0
    wind_direction = 120.0
    wind_field = WindField.artificial(
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        pressure_array=pressure_array,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        no_grid_points=no_grid_points,
    )
    assert np.all(
        wind_field.u_comp == pytest.approx(-0.5 * wind_speed * math.sqrt(3.0))
    )
    assert np.all(wind_field.v_comp == pytest.approx(0.5 * wind_speed))

    wind_speed = 100.0
    wind_direction = 150.0
    wind_field = WindField.artificial(
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        pressure_array=pressure_array,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        no_grid_points=no_grid_points,
    )
    assert np.all(wind_field.u_comp == pytest.approx(-0.5 * wind_speed))
    assert np.all(wind_field.v_comp == pytest.approx(0.5 * wind_speed * math.sqrt(3.0)))

    wind_speed = 100.0
    wind_direction = 210.0
    wind_field = WindField.artificial(
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        pressure_array=pressure_array,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        no_grid_points=no_grid_points,
    )
    assert np.all(wind_field.u_comp == pytest.approx(0.5 * wind_speed))
    assert np.all(wind_field.v_comp == pytest.approx(0.5 * wind_speed * math.sqrt(3.0)))

    wind_speed = 100.0
    wind_direction = 240.0
    wind_field = WindField.artificial(
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        pressure_array=pressure_array,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        no_grid_points=no_grid_points,
    )
    assert np.all(wind_field.u_comp == pytest.approx(0.5 * wind_speed * math.sqrt(3.0)))
    assert np.all(wind_field.v_comp == pytest.approx(0.5 * wind_speed))

    wind_speed = 100.0
    wind_direction = 300.0
    wind_field = WindField.artificial(
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        pressure_array=pressure_array,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        no_grid_points=no_grid_points,
    )
    assert np.all(wind_field.u_comp == pytest.approx(0.5 * wind_speed * math.sqrt(3.0)))
    assert np.all(wind_field.v_comp == pytest.approx(-0.5 * wind_speed))

    wind_speed = 100.0
    wind_direction = 330.0
    wind_field = WindField.artificial(
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        pressure_array=pressure_array,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        no_grid_points=no_grid_points,
    )
    assert np.all(wind_field.u_comp == pytest.approx(0.5 * wind_speed))
    assert np.all(
        wind_field.v_comp == pytest.approx(-0.5 * wind_speed * math.sqrt(3.0))
    )

    wind_speed = np.linspace(0, 120, 6)
    wind_direction = np.linspace(0, 300, 6)
    pressure_array = np.linspace(9000, 0, 6)
    wind_field = WindField.artificial(
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        pressure_array=pressure_array,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        no_grid_points=no_grid_points,
    )
    expected_u_values = np.array([0, -20.78, -41.57, 0, 83.14, 103.92])
    expected_v_values = np.array([0, -12, 24, 72, 48, -60])
    generate_expected_comp = lambda values: np.tile( # noqa: E731
        values, (no_grid_points, no_grid_points, 1) # noqa: E731
    ).transpose(2, 0, 1)  # noqa: E731
    expected_u_comp = generate_expected_comp(expected_u_values)
    expected_v_comp = generate_expected_comp(expected_v_values)

    assert np.allclose(wind_field.u_comp, expected_u_comp, atol=1e-2)
    assert np.allclose(wind_field.v_comp, expected_v_comp, atol=1e-2)


def test_save_load_artificial() -> None:
    """
    Test saving then loading artificial wind yields identical results
    """
    pressure_array = np.array(
        [90000, 80000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 7000, 3000]
    )
    min_lat = -10.0
    max_lat = 20
    min_lon = 10
    max_lon = 30
    no_grid_points = 20
    orig_wind_field = WindField.artificial(
        wind_speed=20.0,
        wind_direction=100.0,
        pressure_array=pressure_array,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        no_grid_points=no_grid_points,
    )

    test_data_dir = os.path.join(
        ROOT_DIR.joinpath("tests"), "test_data", "artificial_save_load_test"
    )
    test_filename = os.path.join(test_data_dir, "artificial_wind_field.json")

    # make test directory
    os.makedirs(test_data_dir, exist_ok=True)

    # save wind field
    orig_wind_field.save(test_filename)

    # reload wind field
    new_wind_field = WindField.load(test_filename)

    # Check original and serialised/deserialised wind fields are identical
    assert new_wind_field == orig_wind_field

    # delete the test directory files
    shutil.rmtree(test_data_dir)


def test_save_load_uniform() -> None:
    """
    Test saving then loading wind created using uniform constructor yields identical results
    """
    min_lat = -10.0
    max_lat = 20
    min_lon = 10
    max_lon = 30
    no_grid_points = 20
    orig_wind_field = WindField.uniform(
        wind_speed=20.0,
        wind_direction=100.0,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        no_grid_points=no_grid_points,
    )

    test_data_dir = os.path.join(
        ROOT_DIR.joinpath("tests"), "test_data", "artificial_save_load_test"
    )
    test_filename = os.path.join(test_data_dir, "uniform_wind_field.json")

    # make test directory
    os.makedirs(test_data_dir, exist_ok=True)

    # save wind field
    orig_wind_field.save(test_filename)

    # reload wind field
    new_wind_field = WindField.load(test_filename)

    # Check original and serialised/deserialised wind fields are identical
    assert new_wind_field == orig_wind_field

    # delete the test directory files
    shutil.rmtree(test_data_dir)


def test_uniform_wind_field() -> None:
    """
    Test uniform wind instantiation method yields wind field
    """
    wind_field = WindField.uniform(
        wind_speed=20.0,
        wind_direction=100.0,
    )

    assert isinstance(wind_field, WindField)


def test_from_json():
    """
    Inject an invalid payload and check it's rejected
    """

    bad_payload = {"method": "unknown", "args": {}}
    with pytest.raises(ValidationError):
        WindField.from_json(json.dumps(bad_payload))


def test_get_wind_vector_using_plevels():
    """
    Create an artificial wind field and instantiate a wind vector with pressure levels
    """
    wind_field = WindField.artificial(
        wind_speed=10.0,
        wind_direction=90.0,
        pressure_array=[100.0, 200.0],
        min_lat=0.0,
        max_lat=1.0,
        min_lon=0.0,
        max_lon=1.0,
        no_grid_points=2,
        interpolation_method="nearest",
    )
    wind_vector = wind_field.get_wind_vector_using_plevels(
        pressure_level=1000.0, latitude=10.0, longitude=10.0
    )
    assert isinstance(wind_vector, WindVector)


def test_get_wind_vector_invalid_interpolation_method():
    """
    Create an artificial wind field and check invalid interpolation method gets rejected.
    """
    wind_field = WindField.artificial(
        wind_speed=10.0,
        wind_direction=90.0,
        pressure_array=[100.0, 200.0],
        min_lat=0.0,
        max_lat=1.0,
        min_lon=0.0,
        max_lon=1.0,
        no_grid_points=2,
        interpolation_method="nearest",
    )
    wind_field.interpolation_method = "invalid"
    with pytest.raises(ValueError):
        wind_field.get_wind_vector_using_plevels(pressure_level=150.0, latitude=0.5, longitude=0.5)


def test_fl_interpolation_mode():
    """
    Create an artificial wind field and a midpoint vector and check vector components are as expected
    """
    wind_field = WindField.artificial(
        wind_speed=[10.0, 20.0],
        wind_direction=[90.0, 180.0],
        pressure_array=[200.0, 100.0],
        min_lat=0.0,
        max_lat=1.0,
        min_lon=0.0,
        max_lon=1.0,
        no_grid_points=2,
        interpolation_method="fl_interpolation",
    )
    # Create a wind vector at the midpoint of the wind field
    wind_vector = wind_field.get_wind_vector_using_plevels(
        pressure_level=150.0, latitude=0.5, longitude=0.5
    )
    # We can interpolate the midpoint expected values from the supplied mins & maxs
    expected_speed = 15.0
    expected_dir = 135.0
    expected_u = expected_speed * math.sin(math.radians(expected_dir + 180.0))
    expected_v = expected_speed * math.cos(math.radians(expected_dir + 180.0))
    assert wind_vector.u_comp == pytest.approx(expected_u, abs=1e-6)
    assert wind_vector.v_comp == pytest.approx(expected_v, abs=1e-6)


def test_fl_interpolation_requires_wind_speed_and_direction() -> None:
    """
    A WindField with interpolation_method='fl_interpolation' must carry both
    wind_speed and wind_direction. Constructing or deserialising one without
    them must raise a ValidationError.
    """
    common_kwargs = dict(
        u_comp=np.zeros((2, 2, 2)),
        v_comp=np.zeros((2, 2, 2)),
        pressure_array=np.array([100.0, 200.0]),
        lat_array=np.array([0.0, 1.0]),
        lon_array=np.array([0.0, 1.0]),
        interpolation_method="fl_interpolation",
    )

    # Both missing.
    with pytest.raises(ValidationError):
        WindField(**common_kwargs)

    # Only wind_direction missing.
    with pytest.raises(ValidationError):
        WindField(**common_kwargs, wind_speed=[10.0, 20.0])

    # Only wind_speed missing.
    with pytest.raises(ValidationError):
        WindField(**common_kwargs, wind_direction=[90.0, 180.0])

    # Deserialisation path enforces the same invariant.
    payload = {
        "u_comp": np.zeros((2, 2, 2)).tolist(),
        "v_comp": np.zeros((2, 2, 2)).tolist(),
        "pressure_array": [100.0, 200.0],
        "lat_array": [0.0, 1.0],
        "lon_array": [0.0, 1.0],
        "interpolation_method": "fl_interpolation",
        "wind_speed": None,
        "wind_direction": None,
    }
    with pytest.raises(ValidationError):
        WindField.from_json(json.dumps(payload))

    # Other interpolation methods are unaffected by the optional fields being missing.
    WindField(**{**common_kwargs, "interpolation_method": "nearest"})


def get_wind_vector_testing_implementation(
    pressure_levels,
    latitudes,
    longitudes,
    wind_u_array,
    wind_v_array,
    target_p_level,
    target_lat,
    target_lon,
):
    """Independent implementation for finding the wind vector at a target
    pressure level and location. To be used for comparisons with the Starling WindField implementation"""

    # Prepare interpolation functions
    interp_u = scipy.interpolate.RegularGridInterpolator(
        (pressure_levels, latitudes, longitudes),
        wind_u_array,
        bounds_error=False,
        fill_value=None,
    )

    interp_v = scipy.interpolate.RegularGridInterpolator(
        (pressure_levels, latitudes, longitudes),
        wind_v_array,
        bounds_error=False,
        fill_value=None,
    )

    # Interpolate at the specified pressure level, latitude, and longitude
    point = np.array([target_p_level, target_lat, target_lon])
    wind_u_interp = interp_u(point)
    wind_v_interp = interp_v(point)

    return wind_u_interp[0], wind_v_interp[0]


def _make_artificial_wind_field(
    interpolation_method: str = "nearest",
) -> WindField:
    pressure_array = np.array(
        [90000, 80000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 7000, 3000]
    )
    return WindField.artificial(
        wind_speed=20.0,
        wind_direction=100.0,
        pressure_array=pressure_array,
        min_lat=-10.0,
        max_lat=20,
        min_lon=10,
        max_lon=30,
        no_grid_points=20,
        interpolation_method=interpolation_method,
    )


def test_to_from_json_roundtrip_artificial() -> None:
    """
    A WindField produced via ``artificial`` should round-trip through
    ``to_json`` / ``from_json`` preserving all model data.
    """
    orig = _make_artificial_wind_field()

    payload = orig.to_json()

    # Output must be valid JSON.
    assert isinstance(payload, str)
    parsed = json.loads(payload)
    assert {"u_comp", "v_comp", "pressure_array", "lat_array", "lon_array", "interpolation_method", "wind_speed", "wind_direction"} <= parsed.keys()

    restored = WindField.from_json(payload)

    assert restored == orig

    # Restored arrays must be numpy arrays (the field_validator should convert lists back).
    assert isinstance(restored.u_comp, np.ndarray)
    assert isinstance(restored.v_comp, np.ndarray)
    assert isinstance(restored.pressure_array, np.ndarray)
    assert isinstance(restored.lat_array, np.ndarray)
    assert isinstance(restored.lon_array, np.ndarray)
    assert isinstance(restored.wind_speed, np.ndarray) or isinstance(restored.wind_speed, float) or restored.wind_speed is None
    assert isinstance(restored.wind_direction, np.ndarray) or isinstance(restored.wind_direction, float) or restored.wind_direction is None


def test_to_from_json_roundtrip_uniform() -> None:
    """
    A WindField produced via ``uniform`` should round-trip through
    ``to_json`` / ``from_json``.
    """
    orig = WindField.uniform(
        wind_speed=20.0,
        wind_direction=100.0,
        min_lat=-10.0,
        max_lat=20,
        min_lon=10,
        max_lon=30,
        no_grid_points=20,
    )

    restored = WindField.from_json(orig.to_json())

    assert restored == orig


def test_to_from_json_double_roundtrip_is_stable() -> None:
    """
    Round-tripping twice should produce the same JSON payload, ensuring deterministic output.
    """
    orig = _make_artificial_wind_field()

    first_payload = orig.to_json()
    restored = WindField.from_json(first_payload)
    second_payload = restored.to_json()

    assert first_payload == second_payload
