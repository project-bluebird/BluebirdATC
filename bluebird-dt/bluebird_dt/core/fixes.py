from __future__ import annotations

import json

import geojson
from pydantic import BaseModel
from typing_extensions import Generator, override

from bluebird_dt.core.pos2d import Pos2D
from bluebird_dt.mixin import Comparison


class Fixes(BaseModel, Comparison):
    """
    Dictionary of named locations (2D positions) on the globe, and a dictionary of visibility flags for each location.
    """

    places: dict[str, Pos2D]

    def __init__(self, places: dict[str, Pos2D], visibility: dict[str, bool] | None = None):
        """
        Construct a new instance.

        Parameters
        ----------
        places: dict[str, Pos2D]
            The names and 2D positions of fixes.
        visibility: dict[str, bool], optional
            The names and visibility flag of fixes.
            Visibility is True by default.

        Examples
        --------
        >>> fixes = Fixes({"STHPN": Pos2D(50.9515, -1.3577)}, {"STHPN": True})
        """
        super().__init__(places=places)

        # If visibility dict not set, set to True for all places
        self._visibility: dict[str, bool] = {}
        if visibility is None:
            self.visibility = dict.fromkeys(self.places, True)
        else:
            self.visibility = visibility
        self._as_set = set(self.places.keys())

    @property
    def visibility(self) -> dict[str, bool]:
        """
        The visibility of the fixes.

        Returns
        -------
        dict[str, bool]
            The names and visibility flag of fixes.
        """
        return self._visibility

    @visibility.setter
    def visibility(self, visibility: dict[str, bool]):
        """
        Set the visibility of the fixes.
        Raises a key error if a fix name is not found in places.

        Parameters
        ----------
        visibility: dict[str, bool]
            The names and visibility flags of fixes.
        """
        for name in visibility:
            if name not in self.places:
                raise KeyError(f"Fix '{name}' not found in places")
        for name in self.places:
            self._visibility[name] = visibility.get(name, True)

    def get_visibility(self, key: str) -> bool:
        """
        Get the visibility flag for a specific fix.

        Parameters
        ----------
        key: str
            The name of the fix

        Returns
        -------
        bool
            The visibility flag for the fix. Returns True if the fix exists but has no visibility setting.
        """
        return self.visibility.get(key, True)

    @property
    def as_set(self) -> set[str]:
        """
        Return fixes places as set.
        Will be cached internally so hashes are only computed once.

        Returns
        -------
        set[str]
            The names of fixes.
        """
        if self._as_set is None:
            self._as_set = set(self.places.keys())
        return self._as_set

    @override
    def __setattr__(self, name: str, value) -> None:  # noqa: ANN001
        if name == "places":
            self._as_set = None
        super().__setattr__(name, value)

    @staticmethod
    def from_json(s: str) -> Fixes:
        """
        Construct a new instance from a string in JSON format.
        Does not include visibility flags.

        Parameters
        ----------
        s: str
            A string representation of Fixes in JSON/dictionary format.

        Returns
        --------
        Fixes

        Examples
        ----------
        >>> Fixes.from_json(
        >>>     '''
        >>>     {
        >>>     "CAMBG": [52.2069, 0.1713E],
        >>>     "EXTER": [50.7350, -3.4153],
        >>>     "HTHRW": [51.4700, -0.4543],
        >>>     "STHPN": [50.9515, -1.3577]"
        >>>     }
        >>>     '''
        >>> )
        """

        places = {}
        for name, coord in json.loads(s).items():
            places[name] = Pos2D(lat=coord[0], lon=coord[1])

        return Fixes(places)

    @staticmethod
    def from_geojson(features: geojson.FeatureCollection) -> Fixes:
        """
        Constructor loading all fixes and coordinates from a geojson feature collection, returning a Fixes object.

        Note the Pos2D constructor used does not perform coordinate transformations,
        therefore assuming they are already in WGS84.

        Parameters
        ----------
        features: geojson.FeatureCollection
            A collection of geojson Points.

        Returns
        -------
        Fixes
        """
        places: dict[str, Pos2D] = {}

        for feature in features["features"]:
            if isinstance(feature["geometry"], geojson.Point):
                if "name" in feature["properties"]:
                    waypoint_name: str = feature["properties"]["name"]
                else:
                    waypoint_name = feature["properties"]["fix"]
                places[waypoint_name] = Pos2D.from_geojson(feature["geometry"])

        return Fixes(places)

    @staticmethod
    def load(filename: str) -> Fixes:
        """
        Construct a new instance from a file.
        Does not include visibility flags.

        Parameters
        ----------
        filename: str
            Path to a JSON file with a Fixes definition.

        Returns
        --------
        Fixes
        """

        with open(filename) as fd:
            return Fixes.from_json(fd.read())

    def data(self) -> dict[str, list[float]]:
        """
        Create a dictionary with key/value pairs representing the Fixes places data.

        Returns
        --------
        dict
        """

        data: dict[str, str] = {}

        for name, coord in self.places.items():
            data[name] = [coord.lat, coord.lon]

        return data

    def to_json(self) -> str:
        """
        Serialise the instance to JSON string.
        Does not include visibility flags.

        Returns
        --------
        str
        """

        return json.dumps(self.data(), indent=4)

    def __or__(self, value: Fixes) -> Fixes:
        return Fixes(self.places | value.places, self.visibility | value.visibility)

    def save(self, filename: str):
        """
        Write the instance to a file.
        Does not include visibility flags.

        Parameters
        ----------
        filename: str
            Path to file.
        """

        with open(filename, "w") as fd:
            fd.write(self.to_json())

    def get(self, key: str) -> Pos2D | None:
        return self.places.get(key)

    def designators(self) -> Generator[str, None, None]:
        return (a for a in self.places)
