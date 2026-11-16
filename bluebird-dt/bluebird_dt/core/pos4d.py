from __future__ import annotations

import json
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, field_validator
from typing_extensions import override

from bluebird_dt.core.pos2d import Pos2D
from bluebird_dt.core.pos3d import Pos3D
from bluebird_dt.mixin import Comparison
from bluebird_dt.utility import convert


class Pos4D(BaseModel, Comparison):
    """
    Four-dimensional (latitude, longitude, flight level, time) control-point.
    """

    lat: float
    lon: float
    fl: float
    time: float

    @field_validator("lat")
    def validate_lat(cls, value: float) -> float:
        if not -90 <= value <= 90:
            raise ValueError("Latitude must be in the range [-90, +90] degrees.")
        return value

    @field_validator("lon")
    def validate_lon(cls, value: float) -> float:
        if not -180 <= value <= 180:
            raise ValueError("Longitude must be in the range [-180, +180] degrees.")
        return value

    @field_validator("time")
    def validate_time(cls, value: float) -> float:
        if value < 0:
            raise ValueError("Time must be non-negative")
        return value

    # For compatibility with old code, constructor that instantiates this class without keyword args
    @overload
    def __init__(self, lat: float, lon: float, fl: float, time: float) -> None: ...

    @overload
    def __init__(self, *, lat: float, lon: float, fl: float, time: float) -> None: ...

    def __init__(self, *args: float, **kwargs: Any) -> None:
        """
        Construct a new instance.

        Parameters
        ----------
        lat: float
            The latitude of the control point in degrees.
        lon: float
            The longitude of the control point in degrees.
        fl: float
            The altitude of the control point in flight levels (FL).
        time: float
            The time of the control point in seconds (UNIX/Posix time).

        Examples
        --------
        >>> pos = Pos4D(50.7, -3.5, 120, 300.0)
        """
        if args and not kwargs:
            if len(args) != 4:
                raise TypeError("Pos4D() takes exactly 4 positional arguments: lat, lon, fl, time")
            kwargs = {"lat": args[0], "lon": args[1], "fl": args[2], "time": args[3]}
        super().__init__(**kwargs)

    @property
    def location(self) -> NDArray[np.float]:
        return np.array([self.lat, self.lon, self.fl, self.time])

    @staticmethod
    def from_str(s: str) -> Pos4D:
        """
        Construct a new instance from a string representation.

        Parameters
        ----------
        s: str
            A string representation of Pos4D.

        Returns
        ----------
        Pos4D

        Examples
        --------
        >>> Pos4D.from_str("10.0S 5.0E 120FL 1970-01-01T02:35:42")
        """

        s.strip()
        [lat, lon, fl, time_str] = s.split()

        return Pos3D.from_str(f"{lat} {lon} {fl}").pos4d(convert.string_to_timestamp(time_str))

    @staticmethod
    def from_array(a: np.ndarray) -> Pos4D:
        """
        Construct a new instance from a numpy array.

        Parameters
        ----------
        a: np.ndarray
            A (4,) numpy array containing the latitude, longitude, flight level and time.

        Returns
        ----------
        Pos4D
        """

        if a.shape != (4,):
            raise ValueError(f"Expected array shape (4,), given: {a.shape}")

        lat, lon, fl, time = a
        return Pos4D(lat=lat, lon=lon, fl=fl, time=time)

    @staticmethod
    def from_json(s: str) -> Pos4D:
        """
        Construct a new instance from a json representation.

        Parameters
        ----------
        s: str
            A json representation of Pos4D.location array [lat, lon, fl, time]

        Returns
        ----------
        Pos4D

        Examples
        ----------
        >>> Pos4D.from_json("[1.0, 0.0, 200, 60.5]")
        """

        data = json.loads(s)
        return Pos4D.from_array(np.array(data))

    def to_json(self) -> str:
        """
        Serialise the instance to JSON string.

        Returns
        ----------
        str
        """

        return json.dumps([self.lat, self.lon, self.fl, self.time])

    @staticmethod
    def load(filename: str) -> Pos4D:
        """
        Construct a new instance from a file.

        Parameters
        ----------
        filename: str
            Path to a JSON file with a Pos4D definition in a dictionary format.

        Returns
        ----------
        Pos4D
        """

        with open(filename) as fd:
            return Pos4D.from_json(fd.read())

    def save(self, filename: str):
        """
        Write the instance to a file.

        Parameters
        ----------
        filename: str
            Path to file.
        """

        with open(filename, "w") as fd:
            fd.write(self.to_json())

    @override
    def __str__(self) -> str:
        """
        Create a human-readable representation of the instance.

        Returns
        ----------
        str
            A string representation of Pos4D (e.g. "10.0S 5.0E 120FL 1970-01-01T02:35:42.000000")
        """

        return f"{self.pos3d()} {convert.timestamp_to_string(self.time)}"

    def __round__(self, precision: int) -> Pos4D:
        """
        Round the position elements to the given precision.

        Parameters
        ----------
        precision: int
            The precision in decimal digits.

        Returns
        ----------
        Pos4D
        """

        return Pos4D.from_array(np.round([self.lat, self.lon, self.fl, self.time], precision))

    def pos2d(self) -> Pos2D:
        """
        Get the two-dimensional location of the position.

        Returns
        ----------
        Pos2D
        """

        return Pos2D(lat=self.lat, lon=self.lon)

    def pos3d(self) -> Pos3D:
        """
        Get the three-dimensional location of the position.

        Returns
        ----------
        Pos3D
        """

        return Pos3D(lat=self.lat, lon=self.lon, fl=self.fl)

    def forward(self, dist: float, heading: float, speed: float) -> Pos4D:
        """
         Determine the four-dimensional control-point
         a given distance away [nmi] following a given heading [deg] at a given constant speed [kts].
         Flight level is maintained.

        Parameters
         ----------
         dist: float
             The distance to be used in nmi.
         heading: float
             The heading to be used in degrees.
         speed: float
             The speed to be used in knots.

         Returns
         ----------
         Pos4D
        """

        proj_pos2d = self.pos2d().forward(dist, heading)

        return Pos4D(
            lat=proj_pos2d.lat,
            lon=proj_pos2d.lon,
            fl=self.fl,
            time=self.time + (dist / speed) * convert.HRS_TO_SEC,
        )

    def bearing_to(self, other: Pos2D | Pos3D | Pos4D) -> float:
        """
        Calculate the bearing [deg] to another location.

        Parameters
        ----------
        other: Union[Pos2D, Pos3D, Pos4D]
            The other location to be used.

        Returns
        ----------
        float
            Bearing in degrees.
        """

        if isinstance(other, Pos3D | Pos4D):
            other = other.pos2d()
        return self.pos2d().bearing_to(other)

    def distance(self, other: Pos2D | Pos3D | Pos4D) -> float:
        """
         Calculate the geodesic distance [nmi] to another position.
         Flight level is not considered.

        Parameters
         ----------
         other: Union[Pos2D, Pos3D, Pos4D]
             The other location to be used.

         Returns
         ----------
         float
             Distance in nautical miles.
        """

        # same as `self.pos2d().distance(other.pos2d())`, without building the
        # two intermediate Pos2D objects.
        return Pos2D.geo_helper.distance(lat=self.lat, lon=self.lon, lat_origin=other.lat, lon_origin=other.lon)

    def interval(self, other: Pos4D, speed: float) -> float:
        """
        Calculate the time interval [sec] to another position given a constant speed [kts].
        Flight level is considered.

        Parameters
        ----------
        other: Pos4D
            The other location to be used.
        speed: the speed to be travelled in knots

        Returns
        ----------
        float
            The time travel in seconds to travel the distance given the speed.
        """

        return self.distance(other) / speed

    @staticmethod
    def interpolate_points(time: float, initial: Pos4D, final: Pos4D) -> Pos4D:
        """
        Given two Pos4D points, and a time between the two
        returns the linear interpolation of their point at that time

        Parameters
        ----------
        time: float
            The requested time for the new Pos4D point
        initial: Pos4D
            The coordinates and flight level at time before the requested time
        final: Pos4D
            The coordinates and flight level at time after the requested time

        Returns
        ----------
        Pos4D
            A linear interpolation of the coordinates and flight level
            at the requested time
        """
        lat = np.interp(time, [initial.time, final.time], [initial.lat, final.lat])
        lon = np.interp(time, [initial.time, final.time], [initial.lon, final.lon])
        fl = np.interp(time, [initial.time, final.time], [initial.fl, final.fl])
        return Pos4D(lat=lat, lon=lon, fl=fl, time=time)

    @override
    def __eq__(self, other: Pos4D) -> bool:
        """
        Check if two positions are equal.

        Parameters
        ----------
        other: Pos4D
            The other position to be used.

        Returns
        ----------
        bool
        """
        return np.allclose(
            [self.lat, self.lon, self.fl, self.time],
            [other.lat, other.lon, other.fl, other.time],
        )

    @override
    def __hash__(self) -> int:
        """
        Compute the hash value of the position.
        """
        return hash((self.lat, self.lon, self.fl, self.time))
