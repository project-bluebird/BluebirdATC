from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from itertools import repeat
from typing import TYPE_CHECKING, Any, overload

import numpy as np
import numpy.typing as npt
from pyproj import Geod, Proj
from shapely.geometry import MultiPoint, Point
from shapely.ops import transform

from bluebird_dt.utility import convert

if TYPE_CHECKING:
    from bluebird_dt.core.pos2d import Pos2D


class GeoHelper:
    """
    A helper class to deal with geographical computations.

    Attributes
    ----------
    origin : Tuple[float, float]
        Origin of the geographical computations.
    ellipsoid : str
        The ellipsoid model used for calculations.
    projection : Proj
        Projection model used for computations.
    geod : Geod
        Geodetic model used for computations.

    Methods
    -------
    inverse_projection(
            points: Union[List[Tuple[float, float]], Tuple[float, float]],
            ndigits: int = 4
    ) -> Union[List[Pos2D], Pos2D]
        Returns the inverse projection of the provided points.
    forward(x, y, z=None, distance=0.0, heading=0.0) -> Tuple[Any, ...]
        Computes and returns the forward geographic computation.
    distance(lat: float, lon: float, lat_origin: float = 0.0, lon_origin: float = 0.0) -> float
        Returns the distance between the two provided points.
    """

    def __init__(self, origin: tuple[float, float] = (0.0, 0.0), ellipsoid: str = "WGS84"):
        """
        Parameters
        ----------
        origin : Tuple[float, float], optional
            Origin of the geographical computations given in (lat, lon), default is (0.0, 0.0).
        ellipsoid : str, optional
            The ellipsoid model used for calculations, defaults to "WGS84".
        """
        self.origin = origin
        self.ellipsoid = ellipsoid

        proj_string = (
            f"+proj=stere +lat_0={self.origin[0]} +lon_0={self.origin[1]} +k_0=1 "
            f"+x_0=0 +y_0=0 +ellps={self.ellipsoid} +units=kmi +no_defs "
        )

        self.projection = Proj(proj_string)
        self.geod = Geod(ellps=ellipsoid)

    @overload
    def inverse_projection(self, points: list[tuple[float, float]], ndigits: int | None = None) -> list[Pos2D]: ...

    @overload
    def inverse_projection(self, points: tuple[float, float], ndigits: int | None = None) -> Pos2D: ...

    def inverse_projection(
        self,
        points: list[tuple[float, float]] | tuple[float, float],
        ndigits: int | None = None,
    ) -> list[Pos2D] | Pos2D:
        """
        Returns the inverse projection of the provided points (from x,y coordinates to longitude,latitude).

        Parameters
        ----------
        points : Union[List[Tuple[float, float]], Tuple[float, float]]
            List of points or a single point to calculate the inverse projection.
        ndigits : int, optional
            Number of decimal places to round the result to (no rounding by default).

        Returns
        -------
        Union[List[Pos2D], Pos2D]
            The inverse projection of the provided points.

        Raises
        ------
        TypeError
            If points is not a List of Tuples or a Tuple.
        """
        from bluebird_dt.core.pos2d import Pos2D

        # points error handling
        if not isinstance(points, list | tuple):
            raise TypeError("Points must be a list of tuples or a tuple.")

        if isinstance(points, list):
            if not all(isinstance(point, tuple) for point in points):
                raise TypeError("All elements in the list must be tuples.")

            # Create a MultiPoint object from the list of points
            geom = MultiPoint(points)

            # Perform the inverse transformation on the MultiPoint object
            transformed_geom = transform(partial(self.projection, inverse=True), geom)

            # Return a list of the transformed (and optionally rounded points)
            pts = [Pos2D(point.x, point.y) for point in transformed_geom.geoms]

            if ndigits is not None:
                pts = [round(pt, ndigits=ndigits) for pt in pts]

            return pts

        # If points is a single tuple, perform the inverse transformation and rounding
        # Transform the point using the projection
        point = transform(partial(self.projection, inverse=True), Point(points))

        # Return the transformed (and optionally rounded point)
        p = Pos2D(point.x, point.y)

        if ndigits is not None:
            p = round(p, ndigits=ndigits)

        return p

    @overload  # (x, y) -> (x, y)
    def forward(
        self,
        x: float,
        y: float,
        z: None = None,
        distance: float = 0.0,
        heading: float = 0.0,
    ) -> tuple[float, float]: ...

    @overload  # (x, y, z) -> (x, y, z)
    def forward(
        self, x: float, y: float, z: float, distance: float = 0.0, heading: float = 0.0
    ) -> tuple[float, float, float]: ...

    @overload  # (x, ...), (y, ...) -> (x, ...), (y, ...)
    def forward(
        self,
        x: Iterable[float],
        y: Iterable[float],
        z: None = None,
        distance: float = 0.0,
        heading: float = 0.0,
    ) -> tuple[Iterable[float], Iterable[float]]: ...

    @overload  # [(x, ...), (y, ...), z] -> [(x, ...), (y, ...), z]
    def forward(
        self,
        x: Iterable[float],
        y: Iterable[float],
        z: float,
        distance: float = 0.0,
        heading: float = 0.0,
    ) -> tuple[Iterable[float], Iterable[float], float]: ...

    @overload  # ((x, ...), (y, ...), (z, ...)) -> ((x, ...), (y, ...), (z, ...))
    def forward(
        self,
        x: Iterable[float],
        y: Iterable[float],
        z: Iterable[float],
        distance: float = 0.0,
        heading: float = 0.0,
    ) -> tuple[Iterable[float], Iterable[float], Iterable[float]]: ...

    def forward(
        self,
        x: float | Iterable[float],
        y: float | Iterable[float],
        z: float | Iterable[float] | None = None,
        distance: float = 0.0,
        heading: float = 0.0,
    ) -> tuple[Any, ...]:
        """
        Performs a forward geodetic computation. Given a (2D or 3D) location(s), it returns
        a new location that is a given distance away following a given heading.

        Parameters
        ----------
        x : float, iterable
            x-coordinates.
        y : float, iterable
            y-coordinates.
        z : float, iterable, optional
            z-coordinates (default is None).
        distance : float, optional
            Distance to project (default is 0.0) in NMI.
        heading : float, optional
            Heading to project (default is 0.0) in degrees.

        Returns
        -------
        Tuple[Any, ...]
            The forward geographic computation.
        """
        # If distance is zero, return the original coordinates
        if distance == 0.0:
            if z is None:
                return (x, y)
            return (x, y, z)

        # Normalize the heading and convert the distance to meters
        if isinstance(x, float):
            headings = heading % 360.0
            distances = distance * convert.NMI_TO_M
        else:
            headings = tuple(repeat(heading % 360.0, len(x)))
            distances = tuple(repeat(distance * convert.NMI_TO_M, len(x)))

        # Perform the forward geodetic computation
        p_x, p_y, _ = self.geod.fwd(x, y, headings, distances)

        # Return the computed coordinates, including z if it is specified
        if z is None:
            return (p_x, p_y)

        return (p_x, p_y, z)

    @overload
    def distance(self, lat: float, lon: float, lat_origin: float, lon_origin: float) -> float: ...

    @overload
    def distance(
        self,
        lat: npt.NDArray[np.floating],
        lon: npt.NDArray[np.floating],
        lat_origin: npt.NDArray[np.floating],
        lon_origin: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]: ...

    def distance(
        self,
        lat: float | npt.NDArray[np.floating],
        lon: float | npt.NDArray[np.floating],
        lat_origin: float | npt.NDArray[np.floating],
        lon_origin: float | npt.NDArray[np.floating],
    ) -> float | npt.NDArray[np.floating]:
        """
        Returns the haversine distance between the two provided points (lat, lon) -> (lat_origin, lon_origin) in
        nautical miles, or the distance between the two provided arrays of points.

        Parameters
        ----------
        lat : float, npt.NDArray[np.floating]
            Latitude of the first point(s).
        lon : float, npt.NDArray[np.floating]
            Longitude of the first point(s).
        lat_origin : float, npt.NDArray[np.floating]
            Latitude of the origin point(s).
        lon_origin : float, npt.NDArray[np.floating]
            Longitude of the origin point(s).

        Returns
        -------
        float
            The distance between the two points or arrays of points in nautical miles.
        """
        # Compute the geodesic distance
        _, _, dist_M = self.geod.inv(lon, lat, lon_origin, lat_origin)

        # Convert the distance to nautical miles and return it
        return dist_M * convert.M_TO_NMI

    @overload
    def bearing_to(self, lat: float, lon: float, lat_origin: float, lon_origin: float) -> float: ...

    @overload
    def bearing_to(
        self,
        lat: npt.NDArray[np.floating],
        lon: npt.NDArray[np.floating],
        lat_origin: npt.NDArray[np.floating],
        lon_origin: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]: ...

    def bearing_to(
        self,
        lat: float | npt.NDArray[np.floating],
        lon: float | npt.NDArray[np.floating],
        lat_origin: float | npt.NDArray[np.floating],
        lon_origin: float | npt.NDArray[np.floating],
    ) -> float | npt.NDArray[np.floating]:
        """
        Returns the bearing between the two provided points (lat, lon) -> (lat_origin, lon_origin) or
        the bearing between the two provided arrays of points.

        Parameters
        ----------
        lat : float, npt.NDArray[np.floating]
            Latitude of the first point(s).
        lon : float, npt.NDArray[np.floating]
            Longitude of the first point(s).
        lat_origin : float, npt.NDArray[np.floating]
            Latitude of the origin point(s).
        lon_origin : float, npt.NDArray[np.floating]
            Longitude of the origin point(s).

        Returns
        -------
        float
            The bearing between the two points or arrays of points.
        """
        # Compute the geodesic distance
        _, back_azimuth, _ = self.geod.inv(lon, lat, lon_origin, lat_origin)

        return back_azimuth % 360.0
