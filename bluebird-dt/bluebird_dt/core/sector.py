from __future__ import annotations

import json
import typing
from itertools import pairwise

import numpy as np
from shapely import unary_union

from bluebird_dt.core.area import Area
from bluebird_dt.core.pos2d import Pos2D
from bluebird_dt.core.pos3d import Pos3D
from bluebird_dt.core.pos4d import Pos4D
from bluebird_dt.core.volume import Volume
from bluebird_dt.logger import logger
from bluebird_dt.mixin import Comparison

if typing.TYPE_CHECKING:
    from bluebird_dt.core import Aircraft


class Sector(Comparison):
    """
    Represents a sector of airspace, defined primarily by a collection of `Volume` objects.

    A Sector models a controlled region of airspace. It consists of:
    1. A list of `Volume` objects defining its primary 3D spatial extent.
    2. An optional list of `Volume` objects defining its Area of Responsibility (AoR), which represents the airspace a
       controller manages, potentially differing from the sector's physical boundaries.
    3. An optional dictionary mapping specific route segments (fix pairs) to conditional `Volume` objects, used for
       specialized routing rules.


    Attributes
    ----------
    volumes : list[Volume]
        A list of `Volume` objects defining the primary 3D space of the sector.
        These volumes are typically adjacent but non-overlapping.
    area_of_responsibility : list[Volume] | None
        An optional list of `Volume` objects defining the sector's area of responsibility. Defaults to None.
    conditional_volume_dict : dict[str, Volume] | None
        An optional dictionary mapping standardized fix-pair strings (e.g., "FIXA_FIXB") to `Volume` objects
        representing conditional routes. Defaults to None.
    """

    def __init__(
        self,
        volumes: list[Volume],
        area_of_responsibility: list[Volume] | None = None,
        conditional_volume_dict: dict[str, Volume] | None = None,
    ):
        """
        Initializes a Sector instance.

        Parameters
        ----------
        volumes : list[Volume]
            A non-empty list of three-dimensional `Volume` objects defining the sector's primary spatial extent. These
            are assumed to be adjacent and non-overlapping.
        area_of_responsibility : list[Volume] | None, optional
            An optional list of `Volume` objects representing the sector's area of responsibility. Defaults to None.
        conditional_volume_dict : dict[str, Volume] | None, optional
            An optional dictionary mapping standardized fix-pair strings (e.g., "FIXA_FIXB") to `Volume` objects
            representing conditional routes. Defaults to None.

        Raises
        ------
        ValueError
            If the provided `volumes` list is empty.
        """

        if len(volumes) == 0:
            raise ValueError("Sector must contain at least one Volume.")

        self.volumes = volumes  # Note that the order of Volumes should not matter.
        self.area_of_responsibility = area_of_responsibility

        # lazily-cached bounds storage
        self._bounds: None | tuple[np.ndarray, np.ndarray] = None

        # mapping of conditional route fix pairs (as sorted strings like "FIXA_FIXB") to their corresponding Volume
        self.conditional_volume_dict = conditional_volume_dict

    @staticmethod
    def from_json(s: str) -> Sector:
        """
        Construct a new instance from JSON representation.

        Parameters
        ----------
        s: str
            A string representation of a Sector in a JSON structure.

        Returns
        --------
        Sector

        Examples
        ----------
        >>> Sector.from_json(
        >>> '''
        >>> [
        >>>    {
        >>>        "area": [
        >>>            "50.0N 4.0W",
        >>>            "50.0N 3.0W",
        >>>            "51.0N 3.0W",
        >>>            "51.0N 4.0W"
        >>>        ],
        >>>        "min_fl": 0.0,
        >>>        "max_fl": 200.0
        >>>    },
        >>>    {
        >>>        "area": [
        >>>            "50.0N 4.0W",
        >>>            "50.0N 3.0W",
        >>>            "51.0N 3.0W",
        >>>            "51.0N 4.0W"
        >>>        ],
        >>>        "min_fl": 200.0,
        >>>        "max_fl": 400.0
        >>>    }
        >>> ]
        >>> '''
        >>> )
        """

        data = json.loads(s)

        volumes = [Volume.from_json(json.dumps(volume)) for volume in data["volumes"]]
        area_of_responsibility = (
            [Volume.from_json(json.dumps(volume)) for volume in data["area_of_responsibility"]]
            if data.get("area_of_responsibility", None) is not None
            else None
        )

        if data.get("conditional_volume_dict", None) is None:
            conditional_volume_dict = None
        else:
            conditional_volume_dict = {
                pair_string: Volume.from_json(json.dumps(volume_data))
                for pair_string, volume_data in data["conditional_volume_dict"].items()
            }

        return Sector(volumes, area_of_responsibility, conditional_volume_dict)

    @staticmethod
    def load(filename: str) -> Sector:
        """
        Construct a new instance from a file.

        Parameters
        ----------
        filename: str
            Path to a JSON file with a Sector definition.

        Returns
        --------
        Sector
        """

        with open(filename) as fd:
            return Sector.from_json(fd.read())

    def data(self) -> dict[str, typing.Any]:
        """
        Get the data as a serialisable list.

        Create a dictionary with key/value pairs representing the Sector data.

        Returns
        --------
        dict
        """

        data_volumes = [volume.data() for volume in self.volumes]
        data_area_of_responsibility = (
            [volume.data() for volume in self.area_of_responsibility]
            if self.area_of_responsibility is not None
            else None
        )
        data_conditional_volume_dict = (
            {fix_pair: volume.data() for fix_pair, volume in self.conditional_volume_dict.items()}
            if self.conditional_volume_dict is not None
            else None
        )

        return {
            "volumes": data_volumes,
            "area_of_responsibility": data_area_of_responsibility,
            "conditional_volume_dict": data_conditional_volume_dict,
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

    def contains(self, position: Pos3D | Pos4D, epsilon: float = 1e-10) -> bool:
        """
        Determine if a given position is contained within the Sector.

        Parameters
        ----------
        position: Union[Pos3D, Pos4D]
            A point with latitude, longitude and flight level (any additional data is disregarded).
        epsilon: float = 1e-10
            The tolerance value.

        Returns
        ----------
        bool
        """
        if isinstance(position, Pos4D):
            position = position.pos3d()

        return any(volume.contains(position, epsilon=epsilon) for volume in self.volumes)

    def contains_laterally(self, position: Pos2D | Pos3D | Pos4D, epsilon: float = 1e-10) -> bool:
        """
        Returns True if the position is in the Sector laterally, otherwise False.
        In other words, this disregards the vertical dimension.

        Parameters
        ----------
        position: Union[Pos2D, Pos3D, Pos4D]
            A point with latitude and longitude (any additional data is disregarded).
        epsilon: float = 1e-10
            The tolerance value.

        Returns
        ----------
        bool
        """

        return any(volume.contains_laterally(position, epsilon=epsilon) for volume in self.volumes)

    def get_volume(self, position: Pos3D | Pos4D, epsilon: float = 1e-10) -> Volume | None:
        """
        Returns the sector volume that contains the given position, otherwise None.

        Parameters
        ----------
        position: Union[Pos3D, Pos4D]
            A point with latitude, longitude and altitude (any additional data is disregarded).
        epsilon: float = 1e-10
            The tolerance value.

        Returns
        ----------
        bool
        """

        return next(
            (volume for volume in self.volumes if volume.contains(position, epsilon=epsilon)),
            None,
        )

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the lower and upper [lon, lat, fl] bounds of the Sector. More
        specifically, we calculate the smallest/largest possible of lon/lat/fl
        given every Volume.

        Returns
        ----------
        (np.ndarray, np.ndarray), shapes (3, ) and (3, )
            Lower and upper [lon, lat, fl] bounds of the Sector.
        """

        # if we haven't calculated the bounds before, cache their values
        if self._bounds is None:
            # initial the min/max values as the largest/smallest they could be
            min_fl = min_lat = min_lon = np.inf
            max_fl = max_lat = max_lon = -np.inf

            # extract each volume in the sector
            for volume in self.volumes:
                # compare the min/max fls of the volume to those stored
                min_fl = np.minimum(min_fl, volume.min_fl)
                max_fl = np.maximum(max_fl, volume.max_fl)

                # extract its lon/lats
                lons, lats = volume.area.boundary.exterior.coords.xy  # type:ignore

                # compare the min/max lat and lons to those stored
                min_lat = np.minimum(min_lat, np.min(lats))
                max_lat = np.maximum(max_lat, np.max(lats))

                min_lon = np.minimum(min_lon, np.min(lons))
                max_lon = np.maximum(max_lon, np.max(lons))

            # combine them into lower and upper bounds
            min_vals = np.array([min_lon, min_lat, min_fl])
            max_vals = np.array([max_lon, max_lat, max_fl])

            self._bounds = min_vals, max_vals

        # return the cached bounds' values
        return self._bounds

    def boundary(self, precision: float = 0.00062) -> Area:
        """
        Return the combined Area of the Sector i.e., the union of all its Volumes.

        Parameters
        ----------
        precision: float, optional
            The level of precision used to create the union of the sectors.
            Default value of 0.00062 corresponds to up to 70 metres depending on latitude
            See grid_size in https://shapely.readthedocs.io/en/latest/reference/shapely.unary_union.html
            for more information.


        Returns
        ----------
        Area
        """
        polygons = [volume.area.boundary for volume in self.volumes]

        poly = unary_union(polygons, grid_size=precision)

        # NOTE: Shapely returns lon,lat and it is Pos2D(lat,lon)
        coords = [Pos2D(lat, lon) for lon, lat in poly.exterior.coords]  # type:ignore

        return Area(coords)

    @property
    def centre(self) -> Pos3D:
        """
        The geometric centre of the Sector.

        Returns
        -------
        Pos3D
        """

        centroid = self.boundary().boundary.centroid
        max_fl = max(vol.max_fl for vol in self.volumes)
        min_fl = min(vol.min_fl for vol in self.volumes)
        mid_fl = (min_fl + max_fl) / 2

        return Pos3D(centroid.y, centroid.x, mid_fl)

    def distance(self, point: Pos2D) -> float:
        """
        The lateral distance [nmi] between a point and the closest point on the Sector boundary.

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
        return self.boundary().distance(point)

    def get_conditional_volumes_for_aircraft(self, aircraft: Aircraft) -> dict[str, Volume]:
        """
        Finds conditional volumes applicable to an aircraft's filed route segments.

        This method checks the aircraft's filed flight plan route against the sector's pre-defined conditional volumes.
        Conditional volumes are associated with specific route segments (pairs of fixes). The method identifies which
        segments of the aircraft's route correspond to defined conditional volumes and returns them.

        The keys in the returned dictionary are strings representing the fix pair, sorted alphabetically and joined by
        an underscore (e.g., "FIXA_FIXB").

        Parameters
        ----------
        aircraft : Aircraft
            The aircraft whose filed flight plan route will be checked.

        Returns
        -------
        dict[str, Volume]
            A dictionary where keys are sorted fix-pair strings (e.g., "FIXA_FIXB") from the aircraft's route, and
            values are the corresponding `Volume` objects. Returns an empty dictionary if the sector has no conditional
            volumes, the aircraft lacks a flight plan, or none of the aircraft's route segments match any conditional
            volume keys.

        Warns
        -----
            If the provided `aircraft` does not have a flight plan.
        """

        if self.conditional_volume_dict is None:
            return {}

        if aircraft.flight_plan is None:
            logger.warning(
                "Aircraft has no flight plan. Cannot determine conditional volumes.",
                stacklevel=2,
            )
            return {}

        # grab the filed route and turn it into a list of tuples of sorted fix pairs
        filed_route = aircraft.flight_plan.route.filed
        fix_keys = [f"{fix1}_{fix2}" if fix1 <= fix2 else f"{fix2}_{fix1}" for (fix1, fix2) in pairwise(filed_route)]

        # get the conditional route volume for each fix pair, if it exists
        return {pair: self.conditional_volume_dict[pair] for pair in fix_keys if pair in self.conditional_volume_dict}
