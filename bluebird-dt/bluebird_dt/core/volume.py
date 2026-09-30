from __future__ import annotations

import json

from shapely.geometry import Polygon
from typing_extensions import override

from bluebird_dt.core.area import Area
from bluebird_dt.core.pos4d import Pos2D, Pos3D, Pos4D
from bluebird_dt.mixin import Comparison
from bluebird_dt.utility.geo_helper import GeoHelper


class Volume(Comparison):
    """
    Fundamental unit of three-dimensional space.
    """

    def __init__(
        self,
        area: Area,
        min_fl: int,
        max_fl: int,
        sector_name: str | None = None,
        description: str | None = None,
        airspace_id: str | None = None,
    ):
        """
        Construct a new instance.

        Parameters
        ----------
        area: Area
            The horizontal cross-section of the Volume (2D).
        min_fl: int
            Minimum flight level at which Aircraft are allowed to fly within the Volume.
        max_fl: int
            Maximum flight level at which Aircraft are allowed to fly within the Volume.
        sector_name: str | None, optional
            The sector to which the Volume belongs. Default is None.
        description: str | None, optional
            A description of the Volume. Default is None.
        airspace_id: str | None, optional
            The airspace ID to which the Volume belongs. In the case that the Volume is part of a conditional route
            structure, then `airspace_id` will contain `FIX1_FIX2`, corresponding to the two fixes that define the
            conditional route.
            TODO: This will eventually be changed to an Airway name, e.g. `UN9`, once the conditional route structure
            has been extended to AC.
            Default is None.
        """
        if min_fl < 0:
            raise ValueError("Minimum flight level must be non-negative.")


        self.area = area
        self.min_fl = min_fl
        self.max_fl = max_fl
        self.sector_name = sector_name
        self.description = description
        self.airspace_id = airspace_id

    @staticmethod
    def from_json(s: str) -> Volume:
        """
        Construct a new instance from JSON representation.

        Parameters
        ----------
        s: str
            A string representation of the Volume in a JSON structure.

        Returns
        --------
        Volume

        Examples
        --------
        >>> Volume.from_json('''
            >>>  {
            >>>    "area": [
            >>>        "50.0N 4.0W",
            >>>        "50.0N 3.0W",
            >>>        "51.0N 3.0W",
            >>>        "51.0N 4.0W"
            >>>    ],
            >>>    "min_fl": 0.0,
            >>>    "max_fl": 200.0
            >>> }''')
        """

        data = json.loads(s)

        area = Area.from_json(json.dumps(data["area"]))
        min_fl = int(data["min_fl"])
        max_fl = int(data["max_fl"])
        sector_name = data.get("sector_name", None)
        description = data.get("description", None)
        airspace_id = data.get("airspace_id", None)

        return Volume(area, min_fl, max_fl, sector_name, description, airspace_id)

    @staticmethod
    def from_polygon(p: Polygon, min_fl: int, max_fl: int) -> Volume:
        """
        Construct a new instance from a Shapely Polygon.

        Parameters
        ----------
        p: Polygon
            The polygon representing the horizontal cross-section of the Volume (2D). It is assumed that the polygon is
            closed and the coordinates are defined such that x is longitude and y is latitude.
        min_fl: int
            Minimum flight level at which Aircraft are allowed to fly within the Volume.
        max_fl: int
            Maximum flight level at which Aircraft are allowed to fly within the Volume.

        Returns
        --------
        Volume
        """

        lons, lats = p.exterior.coords.xy  # stored as lon, lat
        pos2d_list = [Pos2D(point[0], point[1]) for point in zip(lats, lons, strict=True)]
        area = Area(pos2d_list)
        return Volume(area, min_fl, max_fl)

    @staticmethod
    def load(filename: str) -> Volume:
        """
        Construct a new instance from a file.

        Parameters
        ----------
            Path to a JSON file with a Volume definition.

        Returns
        --------
        Volume
        """

        with open(filename) as fd:
            return Volume.from_json(fd.read())

    def data(self) -> dict:
        """
        Create a dictionary with key/value pairs representing the Volume data.

        Returns
        --------
        dict
        """

        return {
            "area": self.area.data(),
            "min_fl": self.min_fl,
            "max_fl": self.max_fl,
            "sector_name": self.sector_name,
            "description": self.description,
            "airspace_id": self.airspace_id,
        }

    def to_json(self) -> str:
        """
        Serialise the instance to JSON string.

        Returns
        --------
        str
        """

        return json.dumps(self.data(), indent=4)

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

    def contains(self, position: Pos3D, epsilon: float = 1e-10) -> bool:
        """
        Determine if a given position is contained within the Volume.

        Parameters
        ----------
        position: Pos3D
            A latitude, longitude, altitude position.
        epsilon: float = 1e-10
            The tolerance value.

        Returns
        --------
        bool
        """

        if position.fl < self.min_fl or self.max_fl < position.fl:
            return False

        return self.area.contains(position.pos2d(), epsilon=epsilon)

    def contains_laterally(self, position: Pos2D, epsilon: float = 1e-10) -> bool:
        """
        Determine if a given position is contained laterally within the volume.
        In other words, this disregards the vertical dimension.

        Parameters
        ----------
        position: Pos2D
            A latitude and longitude position.
        epsilon: float = 1e-10
            The tolerance value.

        Returns
        ----------
        bool
        """

        return self.area.contains(position, epsilon=epsilon)

    def boundary_crossing_points(self, trajectory: list[Pos4D]) -> list[Pos2D]:
        """
        Find all the boundary crossing points of the
        given aircraft Trajectory with the Volume.

        Parameters
        ----------
        trajectory : Trajectory
            An aircraft trajectory

        Returns
        ----------
        crossing_pts: list[Pos2D]
            List of boundary crossing points
        """

        intersection_pts, _ = self.area.intersection(trajectory)

        return [Pos2D(y, x) for (x, y) in zip(*intersection_pts.xy, strict=False)]

    @property
    def centre(self) -> Pos3D:
        """
        The geometric centre of the Volume.

        Returns
        -------
        Pos2D
        """

        centre = self.area.centre
        mid_fl = (self.min_fl + self.max_fl) / 2

        return centre.pos3d(mid_fl)

    def distance(self, point: Pos2D) -> float:
        """
        The lateral distance [nmi] between a point and the closest point on the Volume boundary.

        Parameters
        ----------
        point: Pos2D
            The position to calculate the shortest distance to

        Returns
        ----------
        float
            Distance in nmi
        """

        # Get the distance from a point to the nearest point on the polygon in nautical miles
        return self.area.distance(point)

    @override
    def __eq__(self, other: Volume) -> bool:
        """
        Check if two volumes are equal.

        Parameters
        ----------
        other: Volume
            The other volume to be used.

        Returns
        --------
        bool
        """

        return self.area == other.area and self.min_fl == other.min_fl and self.max_fl == other.max_fl

    @override
    def __hash__(self) -> int:
        """
        Hash the Volume.

        Returns
        --------
        int
        """

        return hash((self.area, self.min_fl, self.max_fl))

    @staticmethod
    def from_shapely_polygon(polygon: Polygon, fl_limits: tuple[int, int], geo_helper: GeoHelper) -> Volume:
        return Volume(
            Area(geo_helper.inverse_projection([(x, y) for (x, y) in zip(*polygon.exterior.xy, strict=False)])),
            min(fl_limits),
            max(fl_limits),
        )
