from __future__ import annotations

import json
import math

import numpy as np
from geopy.distance import geodesic
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.multipoint import MultiPoint
from typing_extensions import override

from bluebird_dt.core.pos2d import Pos2D
from bluebird_dt.core.pos3d import Pos3D
from bluebird_dt.core.pos4d import Pos4D
from bluebird_dt.mixin import Comparison
from bluebird_dt.utility.geometry import nearest_point_on_line_segment


class Area(Comparison):
    """
    Fundamental unit of two-dimensional space.
    """

    def __init__(self, boundary: list[Pos2D]):
        """
        Construct a new instance.

        Parameters
        ----------
        boundary:
            The list of 2D points that define the Area boundary.
        """

        if len(boundary) < 3:
            raise ValueError("Boundary must contain at least three locations.")

        self.boundary = Polygon([[location.lon, location.lat] for location in boundary])

        # Delete the last vertex if it's the same as the first
        eps = 1e-3
        if math.fabs(boundary[0].lat - boundary[-1].lat) < eps and math.fabs(boundary[0].lon - boundary[-1].lon) < eps:
            self.boundary_vertices = boundary[:-1]
        else:
            self.boundary_vertices = boundary

    @staticmethod
    def from_json(s: str) -> Area:
        """
        Construct a new instance from a string in JSON format.

        Parameters
        ----------
        s: str
            A string representation of an Area in JSON/list format.

        Returns
        --------
        Area

        Examples
        ----------
        >>> Area.from_json(
        >>>     '''
        >>>     [
        >>>         [52.2069, 0.1713],
        >>>         [50.7350, -3.4153],
        >>>         [51.4700, -0.4543],
        >>>         [50.9515, -1.3577]
        >>>     ]
        >>>     '''
        >>>     )
        """

        boundary = [Pos2D.from_list(location) for location in json.loads(s)]

        return Area(boundary)

    @staticmethod
    def load(filename: str) -> Area:
        """
        Construct a new instance from a file.

        Parameters
        ----------
        filename: str
            Path to a JSON file with an Area definition in a list format.

        Returns
        --------
        Area
        """

        with open(filename) as fd:
            return Area.from_json(fd.read())

    def data(self) -> list[list[float]]:
        """
        Create a list of strings representing the Area data.

        Returns
        --------
        list[list[float]]
        """

        return [[p[1], p[0]] for p in self.boundary.exterior.coords][:-1]

    def to_json(self) -> str:
        """
        Serialise the instance to a JSON string.

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

    def contains(self, position: Pos2D | Pos3D | Pos4D, epsilon: float = 1e-10) -> bool:
        """
        Determine if a given position is contained within the Area
        including the boundary (within epsilon tolerance).

        Parameters
        ----------
        position: Union[Pos2D, Pos3D, Pos4D]
            A point with latitude and longitude (any additional data is disregarded).
        epsilon: float = 1e-10
            The tolerance value.

        Returns
        --------
        bool
        """

        return self.boundary.buffer(epsilon).intersects(Point(position.lon, position.lat))

    def on_boundary(self, position: Pos2D | Pos3D | Pos4D, epsilon: float = 0.08) -> bool:
        """
        Determine if the given location is on the Area boundary (within epsilon tolerance).

        Parameters
        ----------
        position: Union[Pos2D, Pos3D, Pos4D]
            A point with latitude and longitude (any additional data is disregarded).
        epsilon: float = 0.08
            The tolerance value. ~5NMI

        Returns
        ----------
        bool
        """

        # NOTE: self.boundary is a Polygon, to get the boundary (LineString) you have to call self.boundary.boundary
        return self.boundary.boundary.distance(Point(position.lon, position.lat)) < epsilon

    def intersection(self, trajectory: list[Pos4D]) -> tuple[Point | MultiPoint | LineString, LineString]:
        """
        Find all intersections between the Area Polygon's LineString
        and an aircraft trajectory (converted to LineString).

        Parameters
        ----------
        trajectory : list[Pos4D]
            A Trajectory of an aircraft

        Returns
        ----------
        tuple (r1, r2)
            r1: A shapely Point or MultiPoint or empty LineString that define the intersection points
            r2: A shapely LineString that represents the trajectory
        """

        trajectory_line_2d = LineString([Point(c.lon, c.lat) for c in trajectory])
        return self.boundary.boundary.intersection(trajectory_line_2d), trajectory_line_2d

    @property
    def centre(self) -> Pos2D:
        """
        The geometric centre of the Area.

        Returns
        -------
        Pos2D
        """

        centroid = self.boundary.centroid
        return Pos2D(centroid.y, centroid.x)

    def distance(self, point: Pos2D) -> float:
        """
        Calculate lateral distance [nmi] between a point and the closest point on the area boundary.

        Parameters
        ----------
        point: Pos2D
            The position to calculate the shortest distance to

        Returns
        ----------
        float
            Distance in nmi
        """
        # Get the nearest point on the polygon to the point
        # TODO: I think the below calculation is not 100% valid as the nearest point is
        # found BEFORE taking account of the lat/lon scaling
        exterior = self.boundary.exterior
        nearest_point = exterior.interpolate(exterior.project(Point(point.lon, point.lat)))

        # calculate distance in nautical miles and return it
        return geodesic((point.lat, point.lon), (nearest_point.y, nearest_point.x)).nm

    @override
    def __eq__(self, other: Area) -> bool:
        """
        Check if two Areas are equal.

        Parameters
        ----------
        other: Areas
            The other area to be used.

        Returns
        ----------
        bool
        """

        return all(a == b for a, b in zip(self.boundary_vertices, other.boundary_vertices, strict=False))

    @override
    def __hash__(self) -> int:
        """
        Compute the hash value of the Area.
        """
        return hash(tuple(self.boundary_vertices))

    def nearest_segment_to_point(self, point: Pos2D) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """
        Find the nearest segment of the area to a point.

        This function relies on the projection of WGS coordinates on a flat plane to calculate the closes point of
        a segment to the point.
        This assumption should have limited impact to the solution.

        Parameters
        ----------
        point: Pos2D
            The reference point to find the segment closest to it.

        Returns
        -------
        tuple[tuple[float, float], tuple[float, float]]
            The two points, as (longitude, latitude), defining the limits of the

        """
        polygon_coordinates = self.boundary.exterior.coords

        min_distance = float("inf")
        nearest_segment = None

        for i in range(len(polygon_coordinates) - 1):
            point_on_segment = nearest_point_on_line_segment(
                np.array([point.lon, point.lat]),
                np.array(polygon_coordinates[i]),
                np.array(polygon_coordinates[i + 1]),
                truncate_to_segment=True,
            )[0]
            dist = point.distance(Pos2D(point_on_segment[1], point_on_segment[0]))

            if dist < min_distance:
                min_distance = dist
                nearest_segment = (polygon_coordinates[i], polygon_coordinates[i + 1])

        return nearest_segment
