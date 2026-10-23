from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar, overload

import geojson
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, field_validator
from typing_extensions import override

from bluebird_dt.mixin import Comparison
from bluebird_dt.utility.geo_helper import GeoHelper

# avoid circular imports
if TYPE_CHECKING:
    from bluebird_dt.core.pos3d import Pos3D
    from bluebird_dt.core.pos4d import Pos4D


class Pos2D(BaseModel, Comparison):
    """
    Two-dimensional (latitude, longitude) location on the globe.
    """

    geo_helper: ClassVar[GeoHelper] = GeoHelper()
    lat: float
    lon: float

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

    # For compatibility with old code, constructor that instantiates this class without keyword args
    @overload
    def __init__(self, lat: float, lon: float) -> None: ...

    @overload
    def __init__(self, *, lat: float, lon: float) -> None: ...

    def __init__(self, *args: float, **kwargs: Any) -> None:
        if args and not kwargs:
            if len(args) != 2:
                raise TypeError("Pos2D() takes exactly 2 positional arguments: lat, lon")

            kwargs = {"lat": args[0], "lon": args[1]}
        super().__init__(**kwargs)

    @property
    def location(self) -> NDArray[np.float64]:
        return np.array([self.lat, self.lon])

    @staticmethod
    def from_str(s: str) -> Pos2D:
        """
        Construct a new instance from a string representation.

        Parameters
        ----------
        s: str
            A string representation of Pos2D.

        Returns
        --------
        Pos2D

        Examples
        --------
        >>> Pos2D("10.0S 5.0E")
        """

        s.strip()
        [a, b] = s.split()

        lat = float(a[:-1])
        lon = float(b[:-1])

        if a[-1] == "S":
            lat *= -1
        elif a[-1] != "N":
            raise ValueError(f"Invalid latitude string: {s}")

        if b[-1] == "W":
            lon *= -1
        elif b[-1] != "E":
            raise ValueError(f"Invalid longitude string: {s}")

        return Pos2D(lat=lat, lon=lon)

    @staticmethod
    def from_list(a: list[float]) -> Pos2D:
        if len(a) != 2:
            raise ValueError(f"Expected list of length 2, given: {a}")

        return Pos2D(lat=a[0], lon=a[1])

    @staticmethod
    def from_array(a: np.ndarray) -> Pos2D:
        """
        Construct a new instance from a numpy array.

        Parameters
        ----------
        a: np.ndarray
            A (2,) numpy array containing the latitude and longitude.

        Returns
        --------
        Pos2D
        """

        if a.shape != (2,):
            raise ValueError(f"Expected array shape (2, ), given: {a.shape}")

        lat, lon = a
        return Pos2D(lat=lat, lon=lon)

    @staticmethod
    def from_json(s: str) -> Pos2D:
        """
        Construct a new instance from a json representation.

        Parameters
        ----------
        s: str
            A json representation of Pos2D array [lat, lon]

        Returns
        --------
        Pos2D

        Examples
        ----------
        >>> Pos2D.from_json("[1.0, 0.0]")
        """

        data = json.loads(s)
        return Pos2D.from_array(np.array(data))

    def to_json(self) -> str:
        """
        Serialise the instance to JSON string.

        Returns
        --------
        str
        """

        return json.dumps([self.lat, self.lon])

    @staticmethod
    def load(filename: str) -> Pos2D:
        """
        Construct a new instance from a file.

        Parameters
        ----------
        filename: str
            Path to a JSON file with a Pos2D definition in a dictionary format.

        Returns
        --------
        Pos2D
        """

        with open(filename) as fd:
            return Pos2D.from_json(fd.read())

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
        --------
        str
            A string representation of Pos2D (e.g. "10.0S 5.0E")
        """

        v = "N"
        if self.lat < 0:
            v = "S"

        h = "E"
        if self.lon < 0:
            h = "W"

        return f"{abs(self.lat)}{v} {abs(self.lon)}{h}"

    def __round__(self, precision: int) -> Pos2D:
        """
        Round the location elements to the given precision.

        Parameters
        ----------
        precision: int
            The precision in decimal digits.

        Returns
        --------
        Pos2D
        """

        return Pos2D(round(self.lat, precision), round(self.lon, precision))

    def pos3d(self, fl: float) -> Pos3D:
        """
        Create a three-dimensional position of the location with a given flight level [fl].

        Parameters
        ----------
        fl: float
            The flight level (FL) to be used for the 3D location.

        Returns
        --------
        Pos3D
        """

        from .pos3d import Pos3D

        return Pos3D(lat=self.lat, lon=self.lon, fl=fl)

    def pos4d(self, fl: float, time: float) -> Pos4D:
        """
        Create a four-dimensional control-point of the location with a given flight level [fl] and time [sec].

        Parameters
        ----------
        fl: float
            The flight level (FL) to be used for the 4D location.
        time: float
            The time in seconds to be used for the 4D location.

        Returns
        --------
        Pos4D
        """

        from .pos4d import Pos4D

        return Pos4D(lat=self.lat, lon=self.lon, fl=fl, time=time)

    def forward(self, dist: float, heading: float) -> Pos2D:
        """
        Determine the two-dimensional location
        a given distance away [nmi] following a given heading [deg].

        Parameters
        ----------
        dist: float
            The distance to be used in nautical miles.
        heading: float
            The heading to be used in degrees.

        Returns
        --------
        Pos2D
        """

        # method normalizes heading and transforms distance to metres
        lon, lat = self.geo_helper.forward(self.lon, self.lat, heading=heading, distance=dist)

        return Pos2D(lat=lat, lon=lon)

    def bearing_to(self, other: Pos2D) -> float:
        """
        Calculate the bearing [deg] to another location.

        Parameters
        ----------
        other: Pos2D
            The other location to be used.

        Returns
        --------
        float
            Bearing in degrees.
        """

        # the origin point has to go second
        return self.geo_helper.bearing_to(lat=other.lat, lon=other.lon, lat_origin=self.lat, lon_origin=self.lon)

    def distance(self, other: Pos2D) -> float:
        """
        Calculate the geodesic distance [nmi] to another location.

        Parameters
        ----------
        other: Pos2D
            The other location to be used.

        Returns
        --------
        float
            Distance in nautical miles.
        """

        return self.geo_helper.distance(lat=self.lat, lon=self.lon, lat_origin=other.lat, lon_origin=other.lon)

    @override
    def __eq__(self, other: Pos2D) -> bool:
        """
        Check if two positions are equal.

        Parameters
        ----------
        other: Pos2D
            The other position to be used.

        Returns
        --------
        bool
        """
        return np.allclose([self.lat, self.lon], [other.lat, other.lon])

    @override
    def __hash__(self) -> int:
        """
        Compute the hash value of the position.
        """
        return hash((self.lat, self.lon))

    @staticmethod
    def from_geojson(point: geojson.Point) -> Pos2D:
        """
        Constructor of a Pos2D from a Geojson Point's coordinates.

        This function performs no conversion of coordinate systems.

        Parameters
        ----------
        point: geojson.Point
            The geojson Point from which the object is to be created.
        """
        if not isinstance(point, geojson.Point):
            raise TypeError(f"Expected a geojson.Point, got {type(point)}")

        coords = point["coordinates"]

        return Pos2D(lat=coords[1], lon=coords[0])
