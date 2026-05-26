from __future__ import annotations

import json
import typing
from functools import partial
from itertools import chain, pairwise

import numpy as np
import typing_extensions
from shapely import unary_union
from shapely.affinity import rotate, scale
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.collection import GeometryCollection
from shapely.ops import transform

from bluebird_dt.core.aircraft import Aircraft
from bluebird_dt.core.airway import Airway
from bluebird_dt.core.area import Area
from bluebird_dt.core.fixes import Fixes
from bluebird_dt.core.flight_plan import FlightPlan
from bluebird_dt.core.pos2d import Pos2D
from bluebird_dt.core.pos3d import Pos3D
from bluebird_dt.core.pos4d import Pos4D
from bluebird_dt.core.route import Route
from bluebird_dt.core.sector import Sector
from bluebird_dt.core.volume import Volume
from bluebird_dt.logger import logger
from bluebird_dt.mixin import Comparison
from bluebird_dt.utility.geo_helper import GeoHelper
from bluebird_dt.utility.geometry import (
    find_all_boundary_intersections,
    get_perpendicular_line,
    nearest_point_on_line_segment,
)


class Airspace(Comparison):
    """
    Union of adjacent Sectors and nearby Fixes.
    """

    def __init__(
        self,
        sectors: dict[str, Sector],
        fixes: Fixes,
        airways: dict[str, Airway] | None = None,
    ):
        """
        Construct a new instance.

        Parameters
        ----------
        sectors: Dict[Sector]
            A dictionary where keys are the names of the Sectors (in string format) and values are
            Sector objects. The union of these Sectors constitutes the Airspace.
        fixes: Fixes
            A set of named locations within the Airspace.
        airways : dict[str, Airway] | None, optional
            A dictionary where keys are airway identifiers (strings) and values are the corresponding `Airway` objects.
            If None, an empty dictionary is used. Defaults to None.
        """

        if airways is None:
            airways = {}

        if len(sectors) == 0:
            raise ValueError("Airspace must contain at least one Sector.")

        self.sectors = sectors
        self._individual_sectors = sectors.copy()  # shallow copy of dict

        self.fixes = fixes
        self.geo_helper = GeoHelper()

        # dictionary defining the mapping from names to lists of sectors
        # also includes lists of a single sector, e.g. {"13": ["13"]}
        # it is not bandboxed at initialisation so each mapping is to a single sector only
        self._airspace_configuration = {sec_name: [sec_name] for sec_name in sectors}

        self.airways = airways

    @classmethod
    def from_json(cls, s: str) -> typing_extensions.Self:
        """
        Construct a new instance from a string in JSON format.

        Parameters
        ----------
        s: str
            A string representation of an Airspace in a JSON/dictionary structure.

        Returns
        --------
        Airspace

        Examples
        ----------
        >>> Airspace.from_json(
        >>>     '''
        >>>     {
        >>>         "sectors": {
        >>>             "devon": [
        >>>                 {
        >>>                     "area": ["50.0N 4.0W", "50.0N 3.0W", "51.0N 3.0W", "51.0N 4.0W"],
        >>>                     "min_fl": 0.0,
        >>>                     "max_fl": 200.0
        >>>                 },
        >>>                 {
        >>>                     "area": ["50.0N 4.0W", "50.0N 3.0W", "51.0N 3.0W", "51.0N 4.0W"],
        >>>                     "min_fl": 201.0,
        >>>                     "max_fl": 400.0
        >>>                 }
        >>>             ]
        >>>         },
        >>>         "fixes": {
        >>>             "places": {
        >>>                 "EXTER": "50.7351N 3.4153W",
        >>>                 "HTHRW": "51.4702N 0.4479W",
        >>>                 "STHPN": "50.9515N 1.3577W"
        >>>             }
        >>>         },
        >>>         "airspace_configuration": {"devon": ["devon"]},
        >>>         "individual_sectors": {
        >>>             "devon": [
        >>>                 {
        >>>                     "area": ["50.0N 4.0W", "50.0N 3.0W", "51.0N 3.0W", "51.0N 4.0W"],
        >>>                     "min_fl": 0.0,
        >>>                     "max_fl": 200.0
        >>>                 },
        >>>                 {
        >>>                     "area": ["50.0N 4.0W", "50.0N 3.0W", "51.0N 3.0W", "51.0N 4.0W"],
        >>>                     "min_fl": 201.0,
        >>>                     "max_fl": 400.0
        >>>                 }
        >>>             ]
        >>>         }
        >>>     }
        >>>     '''
        >>>     )
        """

        data = json.loads(s)

        sectors = {name: Sector.from_json(json.dumps(sector)) for name, sector in data["sectors"].items()}
        fixes = Fixes.model_validate(data["fixes"])

        # Create an airspace assuming no bandboxing of sectors.
        airspace = cls(sectors, fixes)

        if data.get("airspace_configuration") is not None:
            airspace._airspace_configuration = dict(data["airspace_configuration"].items())
        else:
            logger.debug(
                "No airspace configuration found when loading json. Assuming no bandboxing",
                stacklevel=2,
            )

        if data.get("individual_sectors") is not None:
            airspace._individual_sectors = {
                name: Sector.from_json(json.dumps(sector)) for name, sector in data["individual_sectors"].items()
            }
        else:
            logger.debug(
                "No individual sectors found when loading json. Assuming all sectors are individual sectors",
                stacklevel=2,
            )

        if (airways := data.get("airways")) is not None:
            airspace.airways = {ident: Airway.from_json(json.dumps(data)) for ident, data in airways.items()}
        else:
            logger.debug(
                "No airway information found when loading json. Assuming there are no airways in this airspace.",
                stacklevel=2,
            )

        return airspace

    @staticmethod
    def load(filename: str) -> Airspace:
        """
        Construct a new instance from a file.

        Parameters
        ----------
        filename: str
            Path to a JSON file with an Airspace definition in dictionary format.

        Returns
        --------
        Airspace
        """

        with open(filename) as fd:
            return Airspace.from_json(fd.read())

    def data(self) -> dict[str, typing.Any]:
        """
        Create a dictionary with key/value pairs representing the Airspace data.

        Returns
        --------
        dict
        """

        return {
            "sectors": {pair[0]: pair[1].data() for pair in self.sectors.items()},
            "fixes": self.fixes.model_dump(),
            "airways": {pair.identifier: pair.data() for pair in self.airways.values()},
            "airspace_configuration": dict(self._airspace_configuration.items()),
            "individual_sectors": {pair[0]: pair[1].data() for pair in self._individual_sectors.items()},
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

    def boundary(self, sector_name: str | list[str] | None = None, precision: float = 0.00062) -> Area:
        """
        Return the combined Area of the Airspace i.e., the union of all its Volumes.
        If sector_name is given, will return Area of only the named Sector(s).

        Parameters
        ----------
        sector_name: str, optional
            Name of the Sector(s) to restrict the view to.
        precision: float, optional
            The level of precision used to create the union of the sectors.
            Default value of 0.00062 corresponds to up to 70 metres depending on latitude
            See grid_size in https://shapely.readthedocs.io/en/latest/reference/shapely.unary_union.htmlfor
            for more information.

        Returns
        --------
        Area
        """

        # extract each sector in the airspace (or the named sector(s)) and the associated volume(s)
        if sector_name is None:
            sectors = list(self.sectors.values())
        elif isinstance(sector_name, list):
            sectors = [self.sectors[name] for name in sector_name]
        else:
            sectors = [self.sectors[sector_name]]

        volumes = list(chain.from_iterable([sector.volumes for sector in sectors]))
        polygons = [volume.area.boundary for volume in volumes]

        poly = unary_union(polygons, grid_size=precision)

        # if union results in disjoint polygons (MultiPolygon) produce a warning
        if isinstance(poly, MultiPolygon):
            logger.warning(
                "The union operation has resulted in disjoint polygons, indicating potential issues with "
                "the input airspace geometry.",
                stacklevel=2,
            )

        # NOTE: Shapely returns lon,lat and it is Pos2D(lat,lon)
        coords = [Pos2D(lat, lon) for lon, lat in poly.exterior.coords]

        return Area(coords)

    def boundary_fixes(self) -> Fixes:
        """
        Return Fixes that are on the Airspace boundary.

        Returns
        --------
        Fixes
        """

        airspace_area = self.boundary()

        boundary_fixes = {fix: pos for fix, pos in self.fixes.places.items() if airspace_area.on_boundary(pos)}

        return Fixes(boundary_fixes)

    def route_boundary_fixes(self, route: Route) -> Fixes:
        """
        Return the Route Fixes that are on the Airspace boundary.

        Parameters
        ----------
        route: Route
            The route to be used.

        Returns
        --------
        Fixes
        """

        boundary_fixes = self.boundary_fixes()
        route_boundary_fixes = {fix: pos for fix, pos in boundary_fixes.places.items() if fix in route.filed}

        # ToDo:- Temporary fudge, route boundary fixes need to be reworked (fixes can lie on horizontal planes (
        #  floor, ceiling))
        if not route_boundary_fixes:
            logger.debug(
                f"Route fixes ({route.filed}) do not lie on airspace boundary... Increasing on boundary "
                f"tolerance to ~50NMI",
                stacklevel=2,
            )

            boundary_fixes = {}
            airspace_area = self.boundary()
            for fix, pos in self.fixes.places.items():
                if airspace_area.on_boundary(pos, epsilon=0.84):
                    boundary_fixes[fix] = pos

            boundary_fixes = Fixes(boundary_fixes)
            route_boundary_fixes = {fix: pos for fix, pos in boundary_fixes.places.items() if fix in route.filed}

        return Fixes(route_boundary_fixes)

    def route_entry_fix(self, route: Route) -> Pos2D | None:
        """
        Return Route entry Fix 2D position.

        Parameters
        ----------
        route: Route
            The route to be used.

        Returns
        --------
        Pos2D
        """

        boundary_fixes = self.route_boundary_fixes(route)
        # get the first boundary Fix on Route
        for fix in route.filed:
            if fix in boundary_fixes.places:
                return boundary_fixes.places[fix]
        return None

    def route_exit_fix(self, route: Route) -> Pos2D | None:
        """
        Return Route exit Fix 2D position.

        Parameters
        ----------
        route: Route
            The route to be used.

        Returns
        --------
        Pos2D
        """

        boundary_fixes = self.route_boundary_fixes(route)
        # get the last boundary Fix on Route
        for fix in route.filed[::-1]:
            if fix in boundary_fixes.places:
                return boundary_fixes.places[fix]
        return None

    def find_volume(self, position: Pos2D) -> Volume | None:
        """
        Return the Volume of the Airspace that contains the given position
        (out of possibly multiple Volumes within the Airspace).

        Parameters
        ----------
        position: Pos2D
            The 2D position to be used.

        Returns
        --------
        Volume
        """

        for sector in self.sectors.values():
            for volume in sector.volumes:
                if volume.area.contains(position):
                    return volume
        return None

    def find_fl_lim(self, position: Pos2D) -> tuple[int, int]:
        """
        Find the min/max flight level allowed in the Volume of the Airspace containing the given position.

        Parameters
        ----------
        position: Pos2D
            The 2D position to be used.

        Returns
        --------
        tuple[int, int]
            The [minimum, maximum] allowed flight level.
        """

        volume = self.find_volume(position)
        if volume is not None:
            return volume.min_fl, volume.max_fl
        raise ValueError("The given position is not contained in any of the Airspace volumes.")

    def _get_geometry_collection(self) -> GeometryCollection:
        """
        Return a shapely GeometryCollection of all Area boundaries and Fix coordinates.

        Returns
        --------
        GeometryCollection
        """

        polygons = []
        for sector_name in self.sectors:
            volumes = self.sectors[sector_name].volumes
            polygons.extend([volume.area.boundary for volume in volumes])
        # NOTE: Shapely uses (lon,lat)
        fixes = [Point(fix.lon, fix.lat) for fix in self.fixes.places.values()]
        return GeometryCollection(fixes + polygons)

    def _update_airspace(self, collection: GeometryCollection):
        """
        Update Airspace Fix locations and Area boundaries of all Volumes from shapely GeometryCollection.

        Parameters
        ----------
        collection: GeometryCollection
            The shapely GeometryCollection to be used.
        """

        polygons = []
        fixes = []
        for geom in collection.geoms:
            if isinstance(geom, Polygon):
                polygons.append(geom)
            elif isinstance(geom, Point):
                fixes.append(geom)

        # Update Fix positions
        for i, fix in enumerate(self.fixes.places):
            lon, lat = fixes[i].coords.xy
            self.fixes.places[fix] = Pos2D(lat[0], lon[0])

        # Update Area boundaries
        counter = 0
        for sector_name in self.sectors:
            volumes = self.sectors[sector_name].volumes
            for i in range(len(volumes)):
                self.sectors[sector_name].volumes[i].area.boundary = polygons[counter]
                counter += 1

    def rotate(self, angle: float):
        """
        Rotate Airspace by angle [deg].

        Parameters
        ----------
        angle: float
            The angle of rotation in degrees.
        """

        collection = self._get_geometry_collection()
        rotated_collection = rotate(collection, angle)
        self._update_airspace(rotated_collection)

    def scale(self, factor: float):
        """
        Scale Airspace by factor.

        Parameters
        ----------
        factor: float
            The factor for scaling.
        """

        collection = self._get_geometry_collection()
        scaled_collection = scale(collection, xfact=factor, yfact=factor)
        self._update_airspace(scaled_collection)

    def forward(self, distance: float = 0.0, heading: float = 0.0):
        """
        Move Airspace by distance [nmi] in direction of heading [deg].

        Parameters
        ----------
        distance: float = 0.0
            The distance to move in nautical miles.
        heading: float = 0.0
            The heading in deg.
        """
        collection = self._get_geometry_collection()
        translated_collection = transform(
            partial(self.geo_helper.forward, distance=distance, heading=heading),
            collection,
        )
        self._update_airspace(translated_collection)

    def contains(self, position: Pos3D | Pos4D, epsilon: float = 1e-10) -> bool:
        """
        Determine if a given position is contained within the Airspace.

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

        return any(sector.contains(position, epsilon=epsilon) for sector in self.sectors.values())

    def contains_laterally(self, position: Pos2D | Pos3D | Pos4D, epsilon: float = 1e-10) -> bool:
        """
        Returns True if the position is in the Airspace laterally, otherwise False.
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

        return any(sector.contains_laterally(position, epsilon=epsilon) for sector in self.sectors.values())

    def bandbox_sectors(self, sector_names: dict[str, list[str]]) -> None:
        """
        Combine 2 or more Sectors in the current Airspace into a single Sector.

        Parameters
        ----------
        sector_names: Dict[str, List[str]]
            <name for band-boxed sector>: <names of current sectors to bandbox>

        Returns
        ----------
        Airspace
            The restructured Airspace
        """

        # split any bandboxed sectors which contain individual sectors that need bandboxing
        sectors_to_split = {
            self.get_containing_bandboxed_sector(sector)
            for sector_list in sector_names.values()
            for sector in sector_list
        }

        # filter to remove individual sectors so only bandboxed sectors remain
        sectors_to_split = [
            sector for sector in sectors_to_split if sector not in self._individual_sectors and sector is not None
        ]

        # split the sectors needed to allow the bandboxing
        for sector in sectors_to_split:
            self.split_sector(sector)

        # update the airspace configuration
        self._airspace_configuration.update(sector_names)

        # bandbox sectors as requested
        for bandboxed_name, sectors_list in sector_names.items():
            # storage for the combined volumes and area of responsibilities
            volumes: list[Volume] = []
            area_of_responsibility: list[Volume] = []
            conditional_volume_dict: dict[str, Volume] = {}
            for sec_name in sectors_list:
                del self.airspace_configuration[sec_name]
                old_sector = self.sectors.pop(sec_name)

                # add the volumes and area of responsibility to the bandboxed sector
                volumes.extend(old_sector.volumes)
                if old_sector.area_of_responsibility is not None:
                    area_of_responsibility.extend(old_sector.area_of_responsibility)

                # add the conditional volumes to the bandboxed sector
                if old_sector.conditional_volume_dict is not None:
                    conditional_volume_dict.update(old_sector.conditional_volume_dict)

            self.sectors[bandboxed_name] = Sector(
                volumes=volumes,
                area_of_responsibility=area_of_responsibility if len(area_of_responsibility) > 0 else None,
                conditional_volume_dict=conditional_volume_dict if len(conditional_volume_dict) > 0 else None,
            )

    @property
    def airspace_configuration(self) -> dict[str, list[str]]:
        """The configuration of the Airspace.
        A key:value pair. The key is the bandboxed sector name and the value is a
        list of the individual sector names which together form the bandboxed sector.
        """
        return self._airspace_configuration

    def get_sector(self, sector_name: str) -> Sector:
        if sector_name in self.sectors:
            return self.sectors[sector_name]

        if sector_name in self._individual_sectors:
            return self._individual_sectors[sector_name]

        raise ValueError(f"{sector_name} not a Sector in this Airspace...")

    def contains_sector(self, sector_name: str) -> bool:
        """
        Returns True if the airspace contains this sector, either as a bandboxed sector, or a
        individual sector in its simplest form.
        """
        return sector_name in self.sectors or sector_name in self._individual_sectors

    def split_sector(self, sector_name: str):
        """
        The bandboxed sector to be split into it's individual sectors

        Parameters
        ----------
        sector_name: str
            The sector to be split
        """
        if sector_name not in self.sectors:
            # if sector_name not a sector in the airspace then give warning
            logger.warning(
                "Unable to split sector %s as it is not in the airspace. Airspace sectors are %s",
                sector_name,
                self.sectors.keys(),
                stacklevel=2,
            )
        elif len(self.airspace_configuration[sector_name]) > 1:  # no need to split if maps to a single sector
            # split the sector
            del self.sectors[sector_name]

            # add the individual sectors back to the sector list and airspace config
            individual_sectors = self._airspace_configuration[sector_name]
            for name in individual_sectors:
                self.sectors[name] = self._individual_sectors[name]
                self._airspace_configuration[name] = [name]

            # remove the previously bandboxed sector from airspace_configuration dict
            del self._airspace_configuration[sector_name]

    def get_bounds(self, sector_name: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the lower and upper [lon, lat, fl] bounds of the Airspace. More
        specifically, we calculate the smallest/largest possible of lon/lat/fl
        given every Volume of every Sector of the Airspace.

        Parameters
        ----------
        sector_name: str, optional = None
            Name of the Sector to restrict the view to. If None, the entire Airspace is considered.

        Returns
        ----------
        (np.ndarray, np.ndarray), shapes (3, ) and (3, )
            Lower and upper [lon, lat, fl] bounds of the Airspace (or Sector).
        """

        # initial the min/max values as the largest/smallest they could be
        min_fl = min_lat = min_lon = np.inf
        max_fl = max_lat = max_lon = -np.inf

        # extract each sector in the airspace and its volume(s)
        for name, sector in self.sectors.items():
            # if a sector name is given, only consider that sector
            if sector_name is not None and name != sector_name:
                continue

            (
                (sector_lon_min, sector_lat_min, sector_fl_min),
                (
                    sector_lon_max,
                    sector_lat_max,
                    sector_fl_max,
                ),
            ) = sector.get_bounds()

            # compare the min/max fls of the sector to those stored
            min_fl = np.minimum(min_fl, sector_fl_min)
            max_fl = np.maximum(max_fl, sector_fl_max)

            # compare the min/max lat and lons to those stored
            min_lat = np.minimum(min_lat, sector_lat_min)
            max_lat = np.maximum(max_lat, sector_lat_max)

            min_lon = np.minimum(min_lon, sector_lon_min)
            max_lon = np.maximum(max_lon, sector_lon_max)

        # combine them into lower and upper bounds
        min_vals = np.array([min_lon, min_lat, min_fl])
        max_vals = np.array([max_lon, max_lat, max_fl])

        return min_vals, max_vals

    def _find_boundary_intersection(
        self, start_pos: Pos2D, end_pos: Pos2D, airspace: Airspace
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """
        Given a path start_pos -> end_pos, find the Airspace boundary intersection point closest to the end_pos.

        Parameters
        ----------
        start_pos: Pos2D
            2D start position of path.
        end_pos: Pos2D
            2D end position of path.
        airspace: Airspace
            The Airspace.

        Returns
        ----------
        Tuple(np.ndarray, np.ndarray, float)
            tuple of (c, d, u), defining line c -> d:  c + u * (d - c)
            In this case, c -> d represents a sector edge (i.e., c and d are consecutive coordinates
            of the polygon border) and u is how far along the line the exact exit/entry location is.
        """
        # get the intersections with the sector boundary
        intersections = find_all_boundary_intersections(start_pos.location, end_pos.location, airspace)

        # if there are no intersections then the points must not have been
        # in the sector and the sector does not lie between them
        if len(intersections) == 0:
            return None

        # if there is exactly one intersection, we are done
        c, d, u = intersections[0]

        # # if there is more than one, find the one closest to the second point
        if len(intersections) > 1:
            min_dist = np.inf

            for _c, _d, _u in intersections:
                i_lat, i_lon = _c + _u * np.subtract(_d, _c)

                di_NMI = self.geo_helper.distance(lat=end_pos.lat, lon=end_pos.lon, lat_origin=i_lat, lon_origin=i_lon)

                if di_NMI < min_dist:
                    c, d, u = _c, _d, _u
                    min_dist = di_NMI

        return c, d, u

    def _get_route_windows(
        self,
        aircraft: Aircraft,
        sector_name: str | list[str] | None = None,
        entry: bool = False,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], list[tuple[str, str]]]:
        """
        Find the possible exit (or entry) window points for the Aircraft Route.
        These will either be on the Airspace boundary where the Route intersects
        it or within the Airspace, if the Route ends or starts there (e.g., Aircraft
        descends toward an Airport). If `sector_name` is specified, the considered
        boundary is restricted to the named Sector(s) rather than the entire Airspace.

        Parameters
        ----------
        aircraft: Aircraft
            Aircraft with a Route through the Airspace.
        sector_name: str, optional
            Name of the Sector(s) to restrict the method to.
        entry: bool, default = False
            Return the entry (rather than the exit) windows.

        Returns
        ----------
        Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[str, str]]]
            Window locations, Fix pairs
            - window location: tuple of (c, d, u), where the exit/entry location is
              along the line c -> d: c + u * (d - c)
            - fix pair: tuple of (fix1, fix2), indicating pair of fix names
              the window location corresponds to i.e., window_location[i] is between
              fix_pairs[i][0] and fix_pairs[i][1] (inclusive, e.g., it could be one of
              the fix locations if that is a boundary fix).
        """

        assert aircraft.flight_plan is not None, "Aircraft must have a flight plan to get route windows"
        route = aircraft.flight_plan.route.filed

        # either consider the entire airspace or some subset of sectors within it
        # the point here is to get the right boundary to get intersections with
        # we filter the fixes based on what is in the route - it's fine to leave all
        if sector_name is None:
            airspace = self
        else:
            if isinstance(sector_name, str):
                sector_name = [sector_name]
            sectors = {name: self.sectors[name] for name in sector_name}
            airspace = Airspace(sectors, self.fixes)

        # an Airspace contains a .fixes attribute (Fixes)
        # a Fixes contains a .places attribute (dict of str: Pos2D)
        places = self.fixes.places

        # remove fixes we do not have the location of
        route = [fix for fix in route if fix in places]

        # pos2D locations of the aircraft's position and each route fix
        # NOTE: this assumes that the aircraft is yet to fly the full route
        # if aircraft is already on route, the first window might be wrong
        # TODO: filter route fixes to next on route fix given aircraft position
        point_locations = [aircraft.pos2d(), *[places[fix] for fix in route]]

        # find out which fixes on the route are in or out of the sector laterally
        # do not use any buffer around boundary - it's fine if fixes aren't exactly on it
        point_in_sector = [airspace.contains_laterally(location, epsilon=0) for location in point_locations]

        # name the initial  position of the aircraft in the route
        route = ["START_LOC", *route]

        # we need to go through each consecutive pair of fixes on the route
        # and find which ones cross the sector boundary

        # we calculate the window locations, i.e. the start and end points of a
        # line that defines where the plane can fly through.
        # there are five cases:
        # 1. both points outside a sector and no intersections with it: no window
        # 2. both points outside a sector but intersecting with it (i.e., sector lies
        #    between the two points):
        #       - exit/entry is the route/border intersection centred on the intersection
        # 3. p0 -> p1 crosses INTO a sector:
        #       - exit window is a line perpendicular to p1, centred on it
        #       - entry window is the route/border intersection centred on the intersection
        # 4. p0 -> p1 crosses out of a sector:
        #       - exit window is the route/border intersection centred on the intersection
        #       - entry window is a line perpendicular to p0, centred on it
        # 5. both p0 and p1 are INSIDE the sector:
        #       - exit window is line perpendicular to p1, centred on it
        #       - entry window is line perpendicular to p0, centred on it

        # NOTE: Pos2D.location = np.array([lat, lon])

        # location of the window, tuples of (c, d, u), where the exit/entry location is
        # along the line c -> d: c + u * (d - c)
        exit_window_locations: list[tuple[np.ndarray, np.ndarray, float]] = []
        entry_window_locations: list[tuple[np.ndarray, np.ndarray, float]] = []

        # pairs of fixes that the window locations correspond to; i.e.
        # window_location[i] occurred between fixes fix_pairs[i][0] and fix_pairs[i][1]
        exit_fix_pairs: list[tuple[str, str]] = []
        entry_fix_pairs: list[tuple[str, str]] = []

        # NOTE: for entry we only need to find the first point so we return as soon as it's found
        # for exit we need the very last point, so we loop through all points before we return

        # if the aircraft is laterally inside the airspace, then this is the entry point
        if entry and point_in_sector[0]:
            # the entry point is the aircraft location
            _c, _d, _u = get_perpendicular_line(point_locations[1], point_locations[0], airspace)
            entry_window_locations.append((_c, _d, _u))
            entry_fix_pairs.append((route[0], route[1]))
            return entry_window_locations, entry_fix_pairs

        for i in range(len(point_locations) - 1):
            # get a consecutive pair of points and whether they lie in the sector
            p0, p0_in_sector = point_locations[i], point_in_sector[i]
            p1, p1_in_sector = point_locations[i + 1], point_in_sector[i + 1]

            # for entry, check if we are crossing into the sector
            # either first point is outside and second point inside OR both are outside
            if entry and ((not p0_in_sector and p1_in_sector) or (not p0_in_sector and not p1_in_sector)):
                # find intersection closest to p0
                intersection = self._find_boundary_intersection(p1, p0, airspace)
                if intersection is not None:
                    _c, _d, _u = intersection
                    entry_window_locations.append((_c, _d, _u))
                    entry_fix_pairs.append((route[i], route[i + 1]))
                    return entry_window_locations, entry_fix_pairs

            # if we're crossing out of the sector, the boundary intersection point is the exit
            if (p0_in_sector and not p1_in_sector) or (not p0_in_sector and not p1_in_sector):
                # find intersection closest to p1
                intersection = self._find_boundary_intersection(p0, p1, airspace)
                if intersection is None:
                    continue
                c, d, u = intersection

            # else, we are looking for exit && entering the sector or both points are in the sector
            # therefore p1 is the exit point
            else:
                c, d, u = get_perpendicular_line(p0, p1, airspace)

            exit_window_locations.append((c, d, u))
            exit_fix_pairs.append((route[i], route[i + 1]))

        return exit_window_locations, exit_fix_pairs

    def get_exit_point(self, aircraft: Aircraft, sector_name: str | list[str] | None = None) -> Pos2D:
        """
        Find Airspace exit point for Aircraft that is either:
        - the point on the Airspace boundary that the Aircraft exits at
        - the last point on Route (if Aircraft Route stops inside the Sector)
        If `sector_name` is specified, the considered boundary is restricted to the
        named Sector(s) rather than the entire Airspace.

        Parameters
        ----------
        aircraft: Aircraft
            Aircraft with a Route through the Airspace.
        sector_name: str, optional
            Name of the Sector(s) to restrict the method to.

        Returns
        ----------
        Pos2D
            The exit point.
        """

        # get the possible exit windows for the aircraft route - these are
        # points on the boundary that the route intersects or inside the
        # airspace if the route ends there
        exit_windows, _ = self._get_route_windows(aircraft, sector_name)

        if len(exit_windows) == 0:
            raise ValueError("No exit point found, the Route does not intersect with the Sector...")

        # since the method loops through the route fixes in order - use the last
        # window point -- the exit location is: c + alpha * (d - c)
        c, d, alpha = exit_windows[-1]
        exit_location_latlon = c + alpha * (d - c)

        return Pos2D(lat=exit_location_latlon[0], lon=exit_location_latlon[1])

    def get_exit_window(
        self,
        aircraft: Aircraft,
        exit_window_width: float,
        sector_name: str | list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find Airspace exit window for Aircraft (exiting within this window is
        part of the exit Coordination fulfilment conditions).

        The exit window has an exit point that is either:
        - the point on the Airspace boundary that the Aircraft exits at
        - the last point on Route (if Aircraft Route stops inside the Sector)

        The exit window is a line that goes through the exit point and
        starts/ends within at most +/- `exit_window_width` distance.

        If `sector_name` is specified, the considered boundary is restricted to the
        named Sector(s) rather than the entire Airspace.

        Note: each pair of the returned points in is the form [lat, lon].

        Parameters
        ----------
        aircraft: Aircraft
            Aircraft with a Route through the Airspace.
        exit_window_width : float (NMI)
            Magnitude of allowable lateral deviation from exit of each
            aircraft.
        sector_name: str, optional
            Name of the Sector(s) to restrict the method to.

        Returns
        ----------
        Tuple[np.ndarray, np.ndarray]
            The start and end points (defined as [lat, lon] each) of the exit
            window line.
        """

        # get the possible exit windows for the aircraft route - these are
        # points on the boundary that the route intersects or inside the
        # airspace if the route ends there
        exit_windows, _ = self._get_route_windows(aircraft, sector_name)

        # since the method loops through the route fixes in order - use the last
        # window point -- the exit location is: c + alpha * (d - c)
        c, d, alpha = exit_windows[-1]

        # calculate the distance between c and d -- we want to move along cd
        # in both directions by (at most) exit_window_width
        dcd_NMI = self.geo_helper.distance(lat=c[0], lon=c[1], lat_origin=d[0], lon_origin=d[1])

        # convert the desired window half-width to a fraction of dcd_NMI
        window_width_prop = exit_window_width / dcd_NMI

        # convert to proportions along the vector ab and clip so that
        # they always reside within the sector bounds, i.e. enforce that
        # alpha_1 and alpha_2 are always in the range [0, 1]
        alpha_1_unclipped = alpha - window_width_prop
        alpha_2_unclipped = alpha + window_width_prop

        # order them such that alpha_1_unclipped > alpha_2_unclipped
        if alpha_1_unclipped > alpha_2_unclipped:
            alpha_1_unclipped, alpha_2_unclipped = alpha_2_unclipped, alpha_1_unclipped

        alpha_1 = np.clip(alpha_1_unclipped, 0, 1)
        alpha_2 = np.clip(alpha_2_unclipped, 0, 1)

        # convert to two locations that define the ends of the window
        # in which the flight may pass through
        dc = np.subtract(d, c)
        window_loc_1 = c + alpha_1 * dc
        window_loc_2 = c + alpha_2 * dc

        dcd_NMI = self.geo_helper.distance(
            lat=window_loc_1[0],
            lon=window_loc_1[1],
            lat_origin=window_loc_2[0],
            lon_origin=window_loc_2[1],
        )

        # if the window is smaller than the desired width, then either
        # (or both) alpha_1 or (and) alpha_2 are touching the boundaries of the
        # sector, i.e. are equal to 0 or 1. if there was any amount of distance
        # left over when clipping them both to be in [0, 1], then add as much
        # as we can onto alpha_1 or alpha_2 to extend the window as close as
        # possible to the desired width while staying in the sector.
        if dcd_NMI < 2 * exit_window_width:
            if alpha_1 == 0:
                # alpha_1_unclipped must be negative (rounded up to 0)
                alpha_1_remaining = alpha_1_unclipped

                # subtract the negative amount (add) as much as we can onto alpha_2
                alpha_2 = np.clip(alpha_2 - alpha_1_remaining, 0, 1)

            if alpha_2 == 1:
                # amount above 1 we couldn't extend by
                alpha_2_remaining = alpha_2_unclipped - 1

                # subtract as much as we can from alpha_1
                alpha_1 = np.clip(alpha_1 - alpha_2_remaining, 0, 1)

            # convert to two locations that define the ends of the window
            # in which the flight may pass through
            dc = np.subtract(d, c)
            window_loc_1 = c + alpha_1 * dc
            window_loc_2 = c + alpha_2 * dc

        return window_loc_1, window_loc_2

    def get_entry_point(self, aircraft: Aircraft, sector_name: str | list[str] | None = None) -> Pos2D:
        """
        Find Airspace entry point for Aircraft that is either:
        - the point on the Airspace boundary that the Aircraft enters at
        - the first point on Route (if Aircraft Route starts inside the Sector)
        If `sector_name` is specified, the considered boundary is restricted to the
        named Sector(s) rather than the entire Airspace.

        Parameters
        ----------
        aircraft: Aircraft
            Aircraft with a Route through the Airspace.
        sector_name: str, optional
            Name of the Sector(s) to restrict the method to.

        Returns
        ----------
        Pos2D
            The entry point.
        """

        window_locations, _ = self._get_route_windows(aircraft, sector_name, entry=True)

        if len(window_locations) == 0:
            raise ValueError("No entry point found, the Route does not intersect with the Sector...")

        # use the first window point -- the entry location is: c + alpha * (d - c)
        c, d, alpha = window_locations[0]
        entry_location_latlon = c + alpha * (d - c)

        return Pos2D(lat=entry_location_latlon[0], lon=entry_location_latlon[1])

    def closest_forward_fix(
        self,
        aircraft: Aircraft,
        distance_threshold_NMI: float = 5.0,
        route_fixes: list[str] | None = None,
    ) -> str | None:
        """
        This function returns the next Fix on the Route that the Aircraft is flying in direction of travel. It does
        this by calculating the distance between the Aircraft and each leg, and returning the down-route Fix of the
        nearest leg. Returning None indicates the aircraft has passed the final fix.

        Additionally:
        - If two legs are equally close, the down-route leg is chosen.
        - If resulting Fix is closer than distance_threshold_NMI, the next route Fix outside this threshold is returned.
        - If the aircraft is approaching the first leg, the first Fix is used (not the down-route Fix of the first leg).

        Parameters
        ----------
        aircraft: Aircraft
            An instance of the Aircraft class, which contains the current position of the Aircraft.
        distance_threshold_NMI: float, default=5
            Consider Fix as having been passed if Aircraft is < distance_threshold from it (NMI).
        route_fixes: list[str] | None, default=None
            The fixes to be considered.  By default the aircraft's current flight plan will be used.

        Returns
        ----------
        str | None
            The name of the Fix on the Route closest to the given Aircraft's current position, moving in the direction
            of the Route's exit. Returns None if passed the final fix of the route.
        """

        if route_fixes is None:
            route_fixes = aircraft.flight_plan.route.current

        all_fixes = self.fixes.places

        # remove any fixes which aren't recognised and print a warning
        unknown_fixes = set(route_fixes) - self.fixes.as_set
        if len(unknown_fixes) > 0:
            logger.warning(
                f"Fix(es) {unknown_fixes} not in airspace."
                f"Ignoring when locating next fix forward for aircraft {aircraft.callsign}",
                stacklevel=2,
            )
            route_fixes = [fix for fix in route_fixes if fix not in unknown_fixes]

        # special case: the route has only one fix. Either the aircraft is near the end of its route
        # or this can occur in real-world data.
        # also use set, to account for real-world cases where the route is just the origin and destination, when
        # the origin and destination is the same airport, e.g. route_fixes = ["EGUN", "EGUN"]
        if len(set(route_fixes)) == 1:
            fix = route_fixes[0]
            distance_to_fix = aircraft.pos2d().distance(all_fixes[fix])

            if distance_to_fix > distance_threshold_NMI:
                # if outside threshold, then assume aircraft hasn't reached this single fix yet
                return fix

            # else aircraft is within threshold, and has therefore reached/passed this single (only) fix,
            # so we return None
            return None

        fix_idx_to_location = {i: all_fixes[fix].location for i, fix in enumerate(route_fixes)}
        n_route_fixes = len(route_fixes)
        aircraft_pos2d = aircraft.pos2d()
        aircraft_location = aircraft_pos2d.location

        # Compute distance to each leg
        distance_to_legs: dict[tuple[int, int], float] = {}
        for leg_idx_pair in pairwise(range(n_route_fixes)):
            leg_start_loc = fix_idx_to_location[leg_idx_pair[0]]
            leg_end_loc = fix_idx_to_location[leg_idx_pair[1]]

            leg_cpa = nearest_point_on_line_segment(
                aircraft_location, leg_start_loc, leg_end_loc, truncate_to_segment=True
            ).squeeze()
            distance_to_legs[leg_idx_pair] = aircraft_pos2d.distance(Pos2D(*leg_cpa))

        # Find the smallest distance
        min_distance_to_legs = min(distance_to_legs.values())
        nearest_legs = [leg for leg in distance_to_legs if distance_to_legs[leg] == min_distance_to_legs]

        # Prioritise the down-route fix if multiple legs are equally close
        nearest_leg_inds = nearest_legs[-1]

        # Next fix is the down-route fix of the nearest leg
        next_fix_idx = nearest_leg_inds[1]

        # Special consideration for aircraft being before the first leg (nearest_leg_inds = (0, 1))
        if next_fix_idx == 1:
            # Test if the aircraft is behind the first fix via the dot product
            fix0_location = fix_idx_to_location[0]
            fix1_location = fix_idx_to_location[1]
            aircraft_before_leg = Airspace._is_obtuse_angle(aircraft_location, fix0_location, fix1_location)

            if aircraft_before_leg:
                # Use fix0 instead of fix1
                next_fix_idx = 0

        # Skip fixes that are closer than distance_threshold_NMI
        distance_to_next_fix = -1
        for fix_idx in range(next_fix_idx, n_route_fixes):
            distance_to_next_fix = aircraft.pos2d().distance(all_fixes[route_fixes[fix_idx]])

            if distance_to_next_fix >= distance_threshold_NMI:
                next_fix_idx = fix_idx
                break

        # if all remaining fixes are closer, return None
        else:
            next_fix_idx = None

        # Special consideration for aircraft past end of last leg
        if next_fix_idx is not None and next_fix_idx == n_route_fixes - 1:
            # Test if the aircraft is past the last fix via the dot product
            fix_last_loc = fix_idx_to_location[next_fix_idx]
            fix_penultimate_loc = fix_idx_to_location[next_fix_idx - 1]

            aircraft_after_last_leg = Airspace._is_obtuse_angle(fix_penultimate_loc, fix_last_loc, aircraft_location)

            if aircraft_after_last_leg:
                next_fix_idx = None

        return route_fixes[next_fix_idx] if next_fix_idx is not None else None

    def list_individual_sectors(self) -> list[str]:
        """
        List all individual sector names

        Returns
        ----------
        list
            Name of ever individual sector in the airspace
        """
        return list(self._individual_sectors.keys())

    def list_all_sector_sets(self) -> list[set[str]]:
        """
        List all sets of individual sectors. Each set is the individual sectors making up a single bandboxed sector.

        Returns
        ----------
        list of sets of strings
            A list of sets. Each set are the constituent individual sectors of a bandboxed sectors. If the sector is
            not bandboxed, the set will contain only one element, which is that sector name
        """
        return [set(self._airspace_configuration[sector]) for sector in self.sectors]

    def sector_name_from_list_of_individual_sectors(self, sector_list: list[str]) -> str | None:
        """
        Return the name of the bandboxed sector defined by a list of individual sectors.

        Parameters
        ----------
        sector_list: list[str]
            List of individual sectors

        Returns
        ----------
        str
            Name of sector comprised from the list of individual sectors. None is no named bandboxed sector
            in airspace.
        """
        # create mapping from sector list to bandboxed name
        map_list_of_sectors_to_sector_name = {tuple(sorted(v)): k for k, v in self._airspace_configuration.items()}

        # find the name of the bandboxed sector from the list of individual sectors
        return map_list_of_sectors_to_sector_name.get(tuple(sorted(sector_list)))

    def get_containing_bandboxed_sector(self, individual_sector: str | None) -> str | None:
        """
        Return the name of the bandboxed sector containing an individual sector.

        Parameters
        ----------
        individual_sector: str
            Name of the individual sector for which the name of the containing bandboxed sector will be returned

        Returns
        ----------
        str
            Name of the bandboxed sector containing an individual_sector.
        """
        # default 'background' if sector isn't recognised e.g. outside simulated airspace
        containing_sector = "background"

        # if the sector is a bandboxed sector already, return it
        if individual_sector in self.airspace_configuration:
            containing_sector = individual_sector
        elif individual_sector in self.list_individual_sectors():
            # find bandboxed sector containing the sector
            for bandbox_name, sectors in self._airspace_configuration.items():
                if individual_sector in sectors:
                    containing_sector = bandbox_name

        return containing_sector

    def distance(self, point: Pos2D) -> float:
        """
        The lateral distance [nmi] between a point and the closest point on the Airspace boundary.

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

    @staticmethod
    def _is_obtuse_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        """Return True if three ordered points form an obtuse angle on the middle point

        Helper function, intended for testing whether the aircraft lies before or after a leg.
        """
        ba = np.subtract(b, a)
        bc = np.subtract(b, c)
        return np.dot(ba, bc) < 0

    def flight_plan_airways(self, flight_plan: FlightPlan) -> list[Airway]:
        """
        Retrieves Airway objects from the Airspace that match identifiers in a FlightPlan.

        This method identifies airway identifiers present in the `flight_plan.unexpanded_route` string and returns the
        corresponding `Airway` objects stored within this `Airspace` instance's `airways` dictionary.

        Parameters
        ----------
        flight_plan : FlightPlan
            The flight plan containing the route information.
            Its `unexpanded_route` attribute is used to find relevant airway identifiers.

        Returns
        -------
        list[Airway]
            A list of `Airway` objects from the airspace whose identifiers are found within the flight plan's
            unexpanded route. If no matching airways are found in the airspace, an empty list is returned.
        """
        # separate out the airways within the unexpanded route: "A B C" -> {"A", "B", "C"}
        aircraft_airways = set(flight_plan.unexpanded_route.split(" "))

        # find the airways that are relevant by checking if their identifier is in the set
        return [airway for airway in self.airways.values() if airway.identifier in aircraft_airways]

    def expand_bandbox_sector(self, sector: str) -> list[str]:
        if (bandboxed_config_individual_sector := self.airspace_configuration.get(sector)) is not None:
            return bandboxed_config_individual_sector

        return [sector]

    def get_conditional_volumes_for_aircraft(
        self, aircraft: Aircraft, sector_name: str | None = None
    ) -> dict[str, Volume]:
        """
        Retrieves conditional volumes applicable to an aircraft's route within a specific sector.

        This method identifies the relevant sector (either specified by `sector_name` or the aircraft's current sector)
        and then calls that sector's `get_conditional_volumes_for_aircraft` method to determine which conditional
        volumes apply to the aircraft's filed route.

        TODO: This method, and the corresponding one in the `Sector` class will eventually be changed to look for
        conditional route volumes based on airspace name rather than fix pairs. This is because some aircraft may not
        fly a specific route, e.g. A->C instead of the known pair B->C, even though they go through the same volume of
        airspace. They will, however, have the same Airway name. Once more conditional route structures are understood
        from Area Control, this will be updated.

        Parameters
        ----------
        aircraft : Aircraft
            The aircraft object containing the flight plan with the route to check.
        sector_name : str | None, optional
            The name of the sector for which to retrieve conditional volumes. If None, the aircraft's current_sector
            is used. Defaults to None.

        Returns
        -------
        dict[str, Volume]
            A dictionary mapping standardized fix-pair strings (sorted alphabetically, joined by underscore,
            e.g., "FIXA_FIXB") from the aircraft's route to their corresponding conditional `Volume` objects defined
            within the identified sector. Returns an empty dictionary if the target sector has no conditional volumes,
            the aircraft lacks a flight plan, or no route segments match conditional volume keys.

        Warns
        -----
        UserWarning
            If the aircraft has no flight plan.
        """
        # if there's no sector given, use the aircraft's current sector
        if sector_name is None:
            sector_name = aircraft.current_sector

        # get the sector and its area of responsibility volumes
        sector = self.sectors[sector_name]

        return sector.get_conditional_volumes_for_aircraft(aircraft)

    @staticmethod
    def combine_two_airspaces(
        airspace_1: Airspace, airspace_2: Airspace, new_sector_2_name: str, prefix: str
    ) -> Airspace:
        """
        Create new airspace from two existing airspaces which contain one sector only. This is used to create an
        airspace containing two artificial sectors, whilst utilising airspace creation methods that already exist.

        A few adjustments are required. We will rename the second sector, to ensure that there is no name clash. We will
        also add an alphanumeric prefix to the fix name.

        Parameters
        ----------
        airspace_1: Airspace
            An airspace containing a single sector.
        airspace_2: Airspace
            An airspace containing a single sector. This sector will be renamed.
        new_sector_2_name: string
            The new name of the sector from airspace_2.
        prefix: string
            A string prefix to prepend to each fix name.

        Returns
        ----------
        Airspace
            A new airspace containing the sector and fixes from airspace_1, the renamed sector from airspace_2, and the
            adjusted fixes from airspace_2, with their new prefix.
        """

        # Aggregate the sectors, and rename the second sector (from airspace 2)
        aggregate_sectors = {new_sector_2_name: next(iter(airspace_2.sectors.values()))}
        aggregate_sectors.update(airspace_1.sectors)

        # Adjust airspace 2 fixes and aggregate with airspace 1 fixes
        adjusted_fixes = {f"{prefix}{fix_name}": data for fix_name, data in airspace_2.fixes.data().items()}
        adjusted_fixes.update(airspace_1.fixes.data())
        adjusted_fixes_json = json.dumps(adjusted_fixes, indent=4)
        aggregate_fixes = Fixes.from_json(adjusted_fixes_json)

        return Airspace(sectors=aggregate_sectors, fixes=aggregate_fixes)
