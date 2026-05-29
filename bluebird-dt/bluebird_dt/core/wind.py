from __future__ import annotations

import math
import typing

import numpy as np
import typing_extensions
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from typing_extensions import override

from bluebird_dt.logger import logger
from bluebird_dt.mixin import Comparison
from bluebird_dt.utility import convert


class WindVector(BaseModel, Comparison):
    """
    A class for wind velocity vectors
    """

    u_comp: float
    v_comp: float

    @classmethod
    def from_polar(cls: type[WindVector], wind_speed: float, wind_direction: float) -> WindVector:
        """
        Instantiate a wind vector from the wind speed and a wind direction

        Parameters
        ----------
        wind_speed : float
            The wind speed (in m/s)
        wind_direction : float
            Direction of the wind (in degrees east of north). Note this is the conventional
            definition of direction - i.e. the heading FROM which the wind is blowing
        """
        u_comp = wind_speed * math.sin(math.radians(wind_direction + 180))
        v_comp = wind_speed * math.cos(math.radians(wind_direction + 180))
        return cls(u_comp=u_comp, v_comp=v_comp)

    @property
    def speed(self) -> float:
        """
        Getter for magnitude of the wind vector

        Returns
        -------
        Float :
            Wind speed, i.e. magnitude of wind vector (in m/s)
        """
        return math.sqrt(self.u_comp**2 + self.v_comp**2)

    @property
    def direction_wind_from(self) -> float:
        """
        Getter for direction of the wind.

        Direction of the wind (in degrees east of north). Note this is the CONVENTIONAL
        definition of direction - i.e. the heading FROM which the wind is blowing.

        Returns
        -------
        Float :
            Direction the wind (in degrees east of north) is coming FROM.
        """
        return math.degrees(math.atan2(self.u_comp, self.v_comp) + math.pi) % 360.0

    @property
    def direction_wind_to(self) -> float:
        """
        Getter for direction of the wind.

        Direction of the wind (in degrees east of north). Note this is VECTOR
        definition of direction - i.e. the heading TO which the wind is blowing.

        Returns
        -------
        Float :
            Direction the wind (in degrees east of north) is going TO.
        """
        return math.degrees(math.atan2(self.u_comp, self.v_comp)) % 360.0


class WindField(BaseModel):
    """
    A class that holds a wind field

    Parameters
    ----------
    u_comp: np.ndarray
        3D array containing u (eastward wind) component values. The three coordinate axis are
        [pressure_level, latitude, longitude] and in that order. Units are m/s
    v_comp: np.ndarray
        3D array containing v (northward wind) component values. The three coordinate axis are
        [pressure_level, latitude, longitude] and in that order. Units are m/s.
    pressure_array: np.ndarray, list
        1D array containing pressure level grid coordinate values. Units are Pa.
    lat_array: np.ndarray, list
        1D array containing latitude grid coordinate values. Units are degrees.
    lon_array: np.ndarray, list
        1D array containing longitude grid coordinate values. Units are degrees.
    interpolation_method: str
        The interpolation method used to estimate wind vector at a given point.
        Can be either "trilinear" or "nearest".

    """

    # This line allows us to use numpy arrays as fields.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    u_comp: np.ndarray | list
    v_comp: np.ndarray | list
    pressure_array: np.ndarray | list[float]
    lat_array: np.ndarray | list[float]
    lon_array: np.ndarray | list[float]
    interpolation_method: typing.Literal["trilinear", "nearest", "fl_interpolation"] = "nearest"

    wind_speed: np.ndarray | list[float] | float | None = None
    wind_direction: np.ndarray | list[float] | float | None = None

    # Validation and serialisation methods for numpy arrays, since these aren't natively supported by Pydantic
    @field_validator("u_comp", "v_comp", "pressure_array", "lat_array", "lon_array")
    def convert_to_array(cls, v: list | np.ndarray) -> np.typing.NDArray[np.float64]:
        return np.asarray(v, dtype=np.float64)

    @field_validator("wind_speed", "wind_direction")
    def convert_optional_to_array(cls, v: list | np.ndarray | float | None) -> np.typing.NDArray[np.float64] | None:
        return None if v is None else np.asarray(v, dtype=np.float64)

    @field_serializer("u_comp", "v_comp", "pressure_array", "lat_array", "lon_array")
    def serialise_ndarray(self, v: np.ndarray | list) -> list:
        return v.tolist() if isinstance(v, np.ndarray) else v

    @field_serializer("wind_speed", "wind_direction")
    def serialise_optional_ndarray(self, v: np.ndarray | list | float | None) -> list | float | None:
        if v is None:
            return None
        return v.tolist() if isinstance(v, np.ndarray) else v

    @property
    def pressure_min(self) -> float:
        return np.min(self.pressure_array)

    @property
    def lat_min(self) -> float:
        return np.min(self.lat_array)

    @property
    def lon_min(self) -> float:
        return np.min(self.lon_array)

    @property
    def pressure_max(self) -> float:
        return np.max(self.pressure_array)

    @property
    def lat_max(self) -> float:
        return np.max(self.lat_array)

    @property
    def lon_max(self) -> float:
        return np.max(self.lon_array)

    @property
    def pressure_array_size(self) -> int:
        return len(self.pressure_array)

    @property
    def lat_array_size(self) -> int:
        return len(self.lat_array)

    @property
    def lon_array_size(self) -> int:
        return len(self.lon_array)

    @property
    def pressure_indexing(self) -> float:
        return self.pressure_array[-1] - self.pressure_array[0]

    @property
    def lat_indexing(self) -> float:
        return self.lat_array[-1] - self.lat_array[0]

    @property
    def lon_indexing(self) -> float:
        return self.lon_array[-1] - self.lon_array[0]

    @override
    def __eq__(self, other: object) -> bool:
        assert isinstance(other, WindField)
        return (
            (self.u_comp == other.u_comp).all()
            and (self.v_comp == other.v_comp).all()
            and (self.pressure_array == other.pressure_array).all()
            and (self.lat_array == other.lat_array).all()
            and (self.lon_array == other.lon_array).all()
            and self.interpolation_method == other.interpolation_method
            and self.pressure_min == other.pressure_min
            and self.lat_min == other.lat_min
            and self.lon_min == other.lon_min
            and self.pressure_max == other.pressure_max
            and self.lat_max == other.lat_max
            and self.lon_max == other.lon_max
            and self.pressure_array_size == other.pressure_array_size
            and self.lat_array_size == other.lat_array_size
            and self.lon_array_size == other.lon_array_size
            and self.pressure_indexing == other.pressure_indexing
            and self.lat_indexing == other.lat_indexing
            and self.lon_indexing == other.lon_indexing
            and np.array_equal(self.wind_speed, other.wind_speed)
            and np.array_equal(self.wind_direction, other.wind_direction)
        )

    @classmethod
    def uniform(
        cls: type[WindField],
        wind_speed: float,
        wind_direction: float,
        min_lat: float = -90.0,
        max_lat: float = 90.0,
        min_lon: float = -180.0,
        max_lon: float = 180.0,
        no_grid_points: int = 2,
        interpolation_method: typing.Literal["trilinear", "nearest"] = "nearest",
    ) -> WindField:
        """
        A class method for instantiating a uniform wind field

        Parameters
        ----------
        wind_speed : float
            The wind speed (in m/s)
        wind_direction : float
            The wind direction (in degrees). Note this is the conventional wind direction definition, i.e.
            the direction FROM which the wind is blowing. I.e. this is 180 degrees from the direction of
            the wind vector.
        min_lat : float
            The minimum grid latitude (in degrees)
        max_lat : float
            The maximum grid latitude (in degrees)
        min_lon : float
            The minimum grid longitude (in degrees)
        max_lon : float
            The maximum grid longitude (in degrees)
        no_grid_points : int
            The number of grid points in latitude and longitude
        interpolation_method: str
            The interpolation method used to estimate wind vector at a given point.
            Can be either "trilinear" or "nearest".

        Returns
        -------
        Wind :
            A uniform wind field
        """

        pressure_array = np.array([102000, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 7000, 3000])

        return cls.artificial(
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            pressure_array=pressure_array,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            no_grid_points=no_grid_points,
            interpolation_method=interpolation_method,
        )

    @classmethod
    def artificial(
        cls,
        wind_speed: np.ndarray | list | float,
        wind_direction: np.ndarray | list | float,
        pressure_array: np.ndarray | list,
        min_lat: float,
        max_lat: float,
        min_lon: float,
        max_lon: float,
        no_grid_points: int,
        interpolation_method: typing.Literal["trilinear", "nearest", "fl_interpolation"] = "nearest",
    ) -> typing_extensions.Self:
        """
        A class method for instantiating an artificial wind field

        Parameters
        ----------
        wind_speed : np.ndarray, list, float
            1D array, or constant value, of wind speed (in m/s)
        wind_direction : np.ndarray, list, float
            1D array, or constant value, of wind direction (in degrees).
        pressure_array : np.ndarray, list
            1D array of pressure levels (in Pa)
        min_lat : float
            The minimum grid latitude (in degrees)
        max_lat : float
            The maximum grid latitude (in degrees)
        min_lon : float
            The minimum grid longitude (in degrees)
        max_lon : float
            The maximum grid longitude (in degrees)
        no_grid_points : int
            The number of grid points in latitude and longitude
        interpolation_method: str
            The interpolation method used to estimate wind vector at a given point.
            Can be either "trilinear" or "nearest".

        Returns
        -------
        Wind :
            An artificial wind field
        """
        # ensure relevant parameters are numpy arrays
        wind_speed = np.asarray(wind_speed)
        wind_direction = np.asarray(wind_direction)
        pressure_array = np.asarray(pressure_array)

        if np.any(wind_speed < 0):
            raise ValueError("Wind speeds must be positive.")

        if np.any(wind_direction < 0) or np.any(wind_direction >= 360):
            raise ValueError("Wind direction must be in the range [0, 360) degrees.")

        if np.any(pressure_array < 0):
            raise ValueError("Pressure must be positive.")

        if min_lat < -90.0 or min_lat > 90.0 or max_lat < -90.0 or max_lat > 90.0:
            raise ValueError("Latitudes must be in the range [-90, 90]")

        if max_lat <= min_lat:
            raise ValueError("Maximum latitude must be larger than the minimum latitude")

        if min_lon < -180.0 or min_lon > 180.0 or max_lon < -180.0 or max_lon > 180.0:
            raise ValueError("Longitudes must be in the range [-180, 180]")

        if max_lon <= min_lon:
            raise ValueError("Maximum longitude must be larger than the minimum longitude")

        if no_grid_points < 2:
            raise ValueError("Minimum number of grid points along lat-lon axes should be 2")

        lat_array = np.linspace(min_lat, max_lat, no_grid_points)
        lon_array = np.linspace(min_lon, max_lon, no_grid_points)

        # The wind vector direction in radians
        theta = np.radians((wind_direction + 180) % 360)

        shape = (len(pressure_array), len(lat_array), len(lon_array))

        theta = np.asarray(theta)
        wind_speed = np.asarray(wind_speed)

        # Compute u and v components
        u_comp = np.sin(theta)[..., np.newaxis, np.newaxis] * wind_speed[..., np.newaxis, np.newaxis]
        v_comp = np.cos(theta)[..., np.newaxis, np.newaxis] * wind_speed[..., np.newaxis, np.newaxis]

        # Broadcast to the desired shape
        u_comp = np.broadcast_to(u_comp, shape)
        v_comp = np.broadcast_to(v_comp, shape)

        return cls(
            u_comp=u_comp,
            v_comp=v_comp,
            pressure_array=pressure_array,
            lat_array=lat_array,
            lon_array=lon_array,
            interpolation_method=interpolation_method,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
        )

    def data(self) -> dict:
        """
        Return a dictionary representation of the wind field.

        Returns
        -------
        dict :
            The wind field model as a dictionary (arrays serialised to nested lists).
        """
        return self.model_dump()

    @classmethod
    def load(cls: type[WindField], filename: str) -> WindField | None:
        """
        Construct a new wind field instance from a json file.

        Parameters
        ----------
        filename: str
            Path to a JSON file with a wind field instantiation definition in a dictionary format.

        Returns
        -------
        Wind
            A wind object
        """
        with open(filename) as fd:
            wind_json = fd.read()
            return cls.from_json(wind_json)

    def save(self, filename: str):
        """
        Write the wind field instantiation definition to a JSON file.

        Parameters
        ----------
        filename : str
            Path to file.
        """
        with open(filename, "w") as fd:
            fd.write(self.to_json())

    def to_json(self) -> str:
        """
        Serialise the full wind field (arrays included) to a JSON string via Pydantic.

        Returns
        -------
        str :
            A JSON string of the full wind field model.
        """
        return self.model_dump_json()

    @classmethod
    def from_json(cls, s: str) -> typing_extensions.Self | None:
        """
        Construct a new wind field instance from a JSON string matching the WindField model schema.

        Parameters
        ----------
        s : str
            JSON string previously produced by :meth:`to_json` (or otherwise matching the
            WindField model schema).

        Returns
        -------
        WindField | None :
            A wind field reconstructed from the serialised data, or ``None`` if validation fails.
        """
        return cls.model_validate_json(s)

    def get_wind_vector_using_plevels(self, pressure_level: float, latitude: float, longitude: float) -> WindVector:
        """
        Interpolate a wind field at a given latitude, longitude and
        an atmospheric pressure level using trilinear interpolation

        Parameters
        ----------
        pressure_level : float
            The pressure level (in Pa)
        latitude : float
            The latitude (in degrees)
        longitude : float
            The longitude (in degrees)

        Returns
        -------
        WindVector :
            Estimate of the wind vector at the 3D location
        """

        point_outside_grid = (
            pressure_level < self.pressure_min
            or pressure_level > self.pressure_max
            or latitude < self.lat_min
            or latitude > self.lat_max
            or longitude < self.lon_min
            or longitude > self.lon_max
        )

        if point_outside_grid:
            logger.warning(
                f"""Requested point outside wind field grid. Pressure (in Pa) must be in\
                range [{self.pressure_min}, {self.pressure_max}], latitude (in deg.)\
                must be in range [{self.lat_min} {self.lat_max}] and longitude (in deg.)\
                must be in range [{self.lon_min}, {self.lon_max}]. Values provided were\
                [pressure_level: {pressure_level}, latitude: {latitude}, longitude: {longitude}].
                Using values at grid boundary""",
                stacklevel=2,
            )
            pressure_level = pressure_level if pressure_level < self.pressure_max else self.pressure_max
            pressure_level = pressure_level if pressure_level > self.pressure_min else self.pressure_min
            latitude = latitude if latitude < self.lat_max else self.lat_max
            latitude = latitude if latitude > self.lat_min else self.lat_min
            longitude = longitude if longitude < self.lon_max else self.lon_max
            longitude = longitude if longitude > self.lon_min else self.lon_min

        pressure_index, lat_index, lon_index = self.locate_cell(pressure_level, latitude, longitude)

        match self.interpolation_method:
            case "nearest":
                u_comp_value = self.u_comp[pressure_index, lat_index, lon_index]
                v_comp_value = self.v_comp[pressure_index, lat_index, lon_index]

            case "trilinear":
                # Determine relative coordinates of the point in the cell, and their compliments.
                f_press = (pressure_level - self.pressure_array[pressure_index]) / (
                    self.pressure_array[pressure_index + 1] - self.pressure_array[pressure_index]
                )
                f_lat = (latitude - self.lat_array[lat_index]) / (
                    self.lat_array[lat_index + 1] - self.lat_array[lat_index]
                )
                f_lon = (longitude - self.lon_array[lon_index]) / (
                    self.lon_array[lon_index + 1] - self.lon_array[lon_index]
                )
                f_press_comp = 1 - f_press
                f_lat_comp = 1 - f_lat
                f_lon_comp = 1 - f_lon

                # Interpolation weights for trilinear interpolation
                weights = np.array(
                    [
                        f_press_comp * f_lat_comp * f_lon_comp,
                        f_press * f_lat_comp * f_lon_comp,
                        f_press_comp * f_lat * f_lon_comp,
                        f_press * f_lat * f_lon_comp,
                        f_press_comp * f_lat_comp * f_lon,
                        f_press * f_lat_comp * f_lon,
                        f_press_comp * f_lat * f_lon,
                        f_press * f_lat * f_lon,
                    ]
                )

                # The wind values at cell corners. Take note the ordering of this array must correspond to
                # the weight coefficients ordering in the weights array above.
                u_comp_at_cell_corners = np.array(
                    [
                        self.u_comp[pressure_index, lat_index, lon_index],
                        self.u_comp[pressure_index + 1, lat_index, lon_index],
                        self.u_comp[pressure_index, lat_index + 1, lon_index],
                        self.u_comp[pressure_index + 1, lat_index + 1, lon_index],
                        self.u_comp[pressure_index, lat_index, lon_index + 1],
                        self.u_comp[pressure_index + 1, lat_index, lon_index + 1],
                        self.u_comp[pressure_index, lat_index + 1, lon_index + 1],
                        self.u_comp[pressure_index + 1, lat_index + 1, lon_index + 1],
                    ]
                )

                v_comp_at_cell_corners = np.array(
                    [
                        self.v_comp[pressure_index, lat_index, lon_index],
                        self.v_comp[pressure_index + 1, lat_index, lon_index],
                        self.v_comp[pressure_index, lat_index + 1, lon_index],
                        self.v_comp[pressure_index + 1, lat_index + 1, lon_index],
                        self.v_comp[pressure_index, lat_index, lon_index + 1],
                        self.v_comp[pressure_index + 1, lat_index, lon_index + 1],
                        self.v_comp[pressure_index, lat_index + 1, lon_index + 1],
                        self.v_comp[pressure_index + 1, lat_index + 1, lon_index + 1],
                    ]
                )

                u_comp_value = np.dot(weights, u_comp_at_cell_corners)
                v_comp_value = np.dot(weights, v_comp_at_cell_corners)

            case "fl_interpolation":
                apply_shear = False

                pressure_array = self.pressure_array

                wind_spd_raw = self.wind_speed.tolist() if isinstance(self.wind_speed, np.ndarray) else self.wind_speed
                wind_dir_raw = (
                    self.wind_direction.tolist() if isinstance(self.wind_direction, np.ndarray) else self.wind_direction
                )

                # Find the index of the nearest pressure level at or below the target
                # and obtain matching wind speed and direction
                if apply_shear:
                    valid_index = np.where(pressure_array < pressure_level)[0][0]
                    wind_spd_interp = wind_spd_raw[valid_index]
                    wind_dir_interp = wind_dir_raw[valid_index]

                # Interpolate based on the altitude, compute wind swing and linearly interpolate
                # speed and direction
                else:
                    for i in range(1, len(wind_dir_raw)):
                        if wind_dir_raw[i - 1] > 270.0 and wind_dir_raw[i] < 90.0:
                            # wind is veering (direction going clockwise with altitude)
                            wind_dir_raw[i] = wind_dir_raw[i] + 360.0

                    wind_spd_interp = np.interp(pressure_level, np.flip(pressure_array), np.flip(wind_spd_raw))
                    wind_dir_interp = np.interp(pressure_level, np.flip(pressure_array), np.flip(wind_dir_raw))

                u_comp_value = wind_spd_interp * math.sin(math.radians(wind_dir_interp + 180.0))
                v_comp_value = wind_spd_interp * math.cos(math.radians(wind_dir_interp + 180.0))

            case _:
                raise ValueError(f"Invalid interpolation method: {self.interpolation_method}")

        return WindVector(u_comp=u_comp_value, v_comp=v_comp_value)

    def get_wind_vector(self, flight_level: float, latitude: float, longitude: float) -> WindVector:
        """
        Interpolate a wind field at a given latitude, longitude and flight_level

        Parameters
        ----------
        flight_level : float
            The flight level
        latitude : float
            The latitude (in degrees)
        longitude : float
            The longitude (in degrees)

        Returns
        -------
        WindVector :
            Estimate of the wind vector at the 3D location
        """
        return self.get_wind_vector_using_plevels(
            pressure_level=convert.pressure_from_fl(flight_level=flight_level),
            latitude=latitude,
            longitude=longitude,
        )

    def locate_cell(self, pressure_level: float, latitude: float, longitude: float) -> tuple[int, int, int]:
        """
        Determine the lowest coordinate indices of grid cell containing interpolation point.

        It is here assumed the grid is uniform in latitude-longitude plane, hence making the determination
        of latitude and longitude cell indices of order O(1).

        In the vertical dimension (pressure level) the pressure level array is searched stepwise to determine
        the correct vertical index.

        Note: The assumption here is the point is located on or within the boundaries of the grid!

        Parameters
        ----------
        pressure_level : float
            Pressure (in Pa) of interpolation point
        latitude : float
            latitude (in degrees) of interpolation point
        longitude : float
            longitude (in degrees) of interpolation point

        Returns
        -------
        Tuple :
            Tuple of three integers which are the lowest coordinate indices of cell containing interpolation point
        """

        # Loop through pressure level dimension to find the lowest index of interval that contain the point.
        # Note if the point is located at the upper level boundary, the cell index will correspond to the
        # second last level as it should.
        for pressure_index in range(self.pressure_array_size):
            if (self.pressure_array[pressure_index] - pressure_level) * self.pressure_indexing > 0:
                break
        pressure_index -= 1

        # Find the cell latitude index. Note to cater for the points located on the extreme far
        # boundary, limit the index to the index of the far boundary cells.
        ratio = (latitude - self.lat_array[0]) / (self.lat_array[-1] - self.lat_array[0])
        lat_index = int(np.floor(ratio * (self.lat_array_size - 1)))
        lat_index = min(lat_index, self.lat_array_size - 2)

        # Find the cell longitude index. Note to cater for the points located on the extreme far
        # boundary, limit the index to the index of the far boundary cells.
        ratio = (longitude - self.lon_array[0]) / (self.lon_array[-1] - self.lon_array[0])
        lon_index = int(np.floor(ratio * (self.lon_array_size - 1)))
        lon_index = min(lon_index, self.lon_array_size - 2)

        return pressure_index, lat_index, lon_index
