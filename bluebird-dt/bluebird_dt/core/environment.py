from __future__ import annotations

import copy
import itertools
import json
import typing
from typing import Generic

import pandas as pd
import typing_extensions

from bluebird_dt.core.aircraft import Aircraft
from bluebird_dt.core.airspace import Airspace
from bluebird_dt.core.coordination import Coordination, CoordinationsManager
from bluebird_dt.core.wind import WindField
from bluebird_dt.logger import logger
from bluebird_dt.mixin import Comparison
from bluebird_dt.utility import convert

TAircraft = typing_extensions.TypeVar("TAircraft", bound=Aircraft, default=Aircraft)
TWindField = typing_extensions.TypeVar("TWindField", bound=WindField, default=WindField)
TForecastWindField = typing_extensions.TypeVar("TForecastWindField", bound=WindField, default=WindField)


class Environment(
    Comparison,
    Generic[TAircraft, TWindField, TForecastWindField],
):
    """
    Environment holding all situational data.

    Attributes
    ----------
    aircraft: dict[str, TAircraft]
        Callsign indexed aircraft store, where the aircraft is of type TAircraft which implements bluebird_dt..Aircraft.
    airspace: TAirspace
        Airspace data store, of type TAirspace which implements bluebird_dt.core.airspace.Airspace.
    wind_field: TWindField
        Wind field object, of type TWindField which implements bluebird_dt.core.wind.WindField.
    forecast_wind_field: TForecastWindField
        Forecast wind field object, of type TForecastWindField which implements bluebird_dt.core.wind.WindField.
    coordinations: CoordinationsManager[TCoordination]
        Coordination data store, containing coordinations of type TCoordination which implement
        bluebird_dt.core.coordination.Coordination.
    airspace_bandboxing_dict: static dictionary of allowed bandbox configuration for the airspace in use.
        This is needed when initialising with real world data for managing coordinations which are not 'current'.
    start_time: float
    time: float
    """

    aircraft: dict[str, TAircraft]
    airspace: Airspace
    wind_field: TWindField | None
    forecast_wind_field: TForecastWindField | None
    coordinations: CoordinationsManager
    start_time: float
    time: float

    def __init__(
        self,
        time: float,
        airspace: Airspace,
        aircraft: dict[str, TAircraft],
        wind_field: TWindField | None = None,
        forecast_wind_field: TForecastWindField | None = None,
        coordinations: list[Coordination] | None = None,
        airspace_bandboxing_dict: dict[str, list[str]] | None = None,
    ):
        """
        Construct a new instance.

        Parameters
        ----------
        time: float
            Current time in the Environment in seconds (Posix/UNIX).
        airspace: Airspace
            An Airspace the Aircraft are flying through.
        aircraft: dict[str, Aircraft]
            A dictionary of {callsign, Aircraft} currently in the Environment.
        wind_field: WindField, optional
            A wind vector field
        forecast_wind_field: WindField, optional
            A forecasted wind vector field
        coordinations: list of Coordination or None
            A list Coordination indicating current active coordinations
        airspace_bandboxing_dict: dict, optional
            If provided, dict keyed by bandbox-sector-name, with
            value being the list of sectors within the bandboxed sector.


        Type Variables
        --------------
        Dependent packages can implement expanded representations of the following classes.

        TAircraft: bluebird_dt.core.Aircraft
        TWindField: bluebird_dt.core.WindField
        TForecastWindField: bluebird_dt.core.WindField
        """
        if time < 0.0:
            raise ValueError("Time must be non-negative.")

        # does not get updated so we can track elapsed time
        self.start_time = time

        # used for advancing time in sim
        self.time = time

        self.airspace = airspace
        self.aircraft = aircraft

        if wind_field is None:
            self.wind_field = None
        else:
            self.wind_field = copy.deepcopy(wind_field)

        if forecast_wind_field is None:
            self.forecast_wind_field = None
        else:
            self.forecast_wind_field = copy.deepcopy(forecast_wind_field)

        self.coordinations = CoordinationsManager(coordinations if coordinations is not None else [])

        if airspace_bandboxing_dict is None:
            self.airspace_bandboxing_dict = {}
        else:
            self.airspace_bandboxing_dict = airspace_bandboxing_dict

        # create dict to store agent plans
        self.agent_plans = {}

        # for caching the datetime update
        self._datetime = pd.to_datetime(self.time, unit="s")
        self._time_used_for_datetime = self.time

    def reset(self):
        """
        Reset the environment, e.g. when jumping to a new time.
        """
        # reset aircraft
        self.aircraft = {}
        # reset coordinations
        self.coordinations.coords = {}

    @classmethod
    def from_json(
        cls,
        s: str,
        typeof_aircraft: type[TAircraft] = Aircraft,
        typeof_windfield: type[TWindField] = WindField,
        typeof_forecastwindfield: type[TForecastWindField] = WindField,
    ) -> typing_extensions.Self:
        """
        Construct a new instance from JSON representation.

        Parameters
        ----------
        s: str
            A string representation of an Environment in a JSON/dictionary structure.
        typeof_airspace: type[TAirspace]
            Type of Airspace to deserialize, defaults to bluebird_dt.core.airspace.Airspace.
        typeof_aircraft: type[TAircraft]
            Type of Aircraft to deserialize, defaults to bluebird_dt.core.aircraft.Aircraft.
        typeof_coordination: type[TCoordination]
            Type of Coordination to deserialize, defaults to bluebird_dt.core.coordination.Coordination,
        typeof_windfield: type[TWindField]
            Type of WindField to deserialize, defaults to bluebird_dt.core.wind.WindField,
        typeof_forecastwindfield: type[TForecastWindField]
            Type of WindField to deserialize as the forecast wind field, default to bluebird_dt.core.wind.WindField,
        Returns
        --------
        Environment
        """

        data = json.loads(s)

        time = data["time"]
        start_time = data["start_time"]

        # Allow possibility that airspace was omitted from the environment
        airspace = Airspace.from_json(json.dumps(data["airspace"]))

        aircraft: dict[str, TAircraft] = {}
        for callsign, aircraft_data in data["aircraft"].items():
            aircraft[callsign] = typeof_aircraft.from_json(json.dumps(aircraft_data))

        wind_field = None
        if "wind_field" in data and data["wind_field"] is not None:
            wind_field = typeof_windfield.from_json(json.dumps(data["wind_field"]))

        forecast_wind_field = None
        if "forecast_wind_field" in data and data["forecast_wind_field"] is not None:
            forecast_wind_field = typeof_forecastwindfield.from_json(json.dumps(data["forecast_wind_field"]))

        coordinations: list[Coordination] | None = data.get("coordinations", None)

        if coordinations is not None:
            loaded_coordinations = [Coordination.from_json(json.dumps(coord)) for coord in coordinations]
            coordinations = [coord for coord in loaded_coordinations if coord is not None]

        env = cls(
            time=time,
            airspace=airspace,  # We sometimes support having no airspace for reading the json.
            aircraft=aircraft,
            wind_field=wind_field,
            forecast_wind_field=forecast_wind_field,
            coordinations=coordinations,
        )
        env.start_time = start_time

        return env

    @classmethod
    def load(
        cls,
        filename: str,
        typeof_aircraft: type[TAircraft] = Aircraft,
        typeof_windfield: type[TWindField] = WindField,
        typeof_forecastwindfield: type[TForecastWindField] = WindField,
    ) -> typing_extensions.Self:
        """
        Construct a new instance from a file.

        Parameters
        ----------
        filename: str
            Path to a JSON file with an Environment definition in dictionary format.
        typeof_airspace: type[TAirspace]
            Type of Airspace to deserialize, defaults to bluebird_dt.core.airspace.Airspace.
        typeof_aircraft: type[TAircraft]
            Type of Aircraft to deserialize, defaults to bluebird_dt.core.aircraft.Aircraft.
        typeof_coordination: type[TCoordination]
            Type of Coordination to deserialize, defaults to bluebird_dt.core.coordination.Coordination,
        typeof_windfield: type[TWindField]
            Type of WindField to deserialize, defaults to bluebird_dt.core.wind.WindField,
        typeof_forecastwindfield: type[TForecastWindField]
            Type of WindField to deserialize as the forecast wind field, default to bluebird_dt.core.wind.WindField,
        Returns
        --------
        Environment
        """

        with open(filename) as fd:
            return cls.from_json(
                fd.read(),
                typeof_aircraft,
                typeof_windfield,
                typeof_forecastwindfield,
            )

    @property
    def datetime(self) -> pd.Timestamp:
        """The environment time as a datetime.Timestamp"""
        # cache the datetime conversion
        if self._time_used_for_datetime != self.time:
            self._datetime = pd.to_datetime(self.time, unit="s")
            self._time_used_for_datetime = self.time

        return self._datetime

    def data(self) -> dict[str, typing.Any]:
        """
        Get the data as a serialisable dictionary.

        Returns
        --------
        dict
        """

        return {
            "start_time": self.start_time,
            "time": self.time,
            "time_str": convert.timestamp_to_string(self.time),
            "airspace": self.airspace.data(),
            "aircraft": {callsign: aircraft.data() for (callsign, aircraft) in self.aircraft.items()},
            "wind_field": None if self.wind_field is None else self.wind_field.data(),
            "forecast_wind_field": None if self.forecast_wind_field is None else self.forecast_wind_field.data(),
            "coordinations": self.coordinations.data(),
        }

    def to_json(self) -> str:
        """
        Serialise the instance to JSON.

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

    def extract_fixes(self) -> tuple[list[float], list[float], list[str]]:
        """
        Extract fixes as lists of longitudes, latitudes, names.

        Returns
        --------
        tuple[list, list, list]
            The fix longitudes, latitudes and names.
        """

        x_f: list[float] = []
        y_f: list[float] = []
        n_f: list[str] = []

        for name, location in self.airspace.fixes.places.items():
            x_f.append(location.lon)
            y_f.append(location.lat)
            n_f.append(name)

        return x_f, y_f, n_f

    def controllable_aircraft(self, sector_name: str | None = None) -> list[str]:
        """
        Return a list of callsigns for Aircraft that respond to Actions.
        Non-controllable Aircraft are, for example, military.
        Optionally restrict to a specific Sector.

        Parameters
        ----------
        sector_name: str, optional
            Name of the Sector to restrict view to.

        Returns
        --------
        list[str]
        """

        if sector_name is None:
            return [callsign for callsign, aircraft in self.aircraft.items() if aircraft.controllable]

        return [
            callsign
            for callsign, aircraft in self.aircraft.items()
            if (aircraft.controllable and aircraft.current_sector == sector_name)
        ]

    def not_controllable_aircraft(self, sector_name: str | None = None) -> list[str]:
        """
        Return a list of callsigns for Aircraft that do not respond to Agent-issued Actions.
        Non-controllable Aircraft are, for example, military. The Agent has to plan around them.
        Optionally restrict to a specific Sector.

        Parameters
        ----------
        sector_name: str, optional
            Name of the Sector to restrict the view to.

        Returns
        --------
        list[str]
        """

        if sector_name is None:
            return [callsign for callsign, aircraft in self.aircraft.items() if not aircraft.controllable]

        return [
            callsign
            for callsign, aircraft in self.aircraft.items()
            if (not aircraft.controllable and aircraft.current_sector == sector_name)
        ]

    def simulated_aircraft(self) -> list[str]:
        """
        Return a list of callsigns for Aircraft that can be acted on by a predictor.

        Returns
        --------
        list[str]
        """

        return [callsign for callsign, aircraft in self.aircraft.items() if aircraft.simulated]

    def replayed_aircraft(self) -> list[str]:
        """
        Return a list of callsigns for Aircraft that are being replayed.

        Returns
        --------
        list[str]
        """

        return [callsign for callsign, aircraft in self.aircraft.items() if not aircraft.simulated]

    def aircraft_in_sectors(self, sector_names: str | list[str]) -> list[str]:
        """
        The aircraft in a list of sectors

        Parameters
        ----------
        sector_names: str or list[str]
            A single sector name or a list of sector names

        Returns
        --------
        list[str]
            A list of callsign contained in the sectors named by sector_names
        """

        sector_names = [sector_names] if isinstance(sector_names, str) else sector_names

        return [callsign for callsign, aircraft in self.aircraft.items() if aircraft.current_sector in sector_names]

    def transform_coordination_to_bandboxed(self, coordination: Coordination) -> Coordination:
        """Update the to_sector and from_sector in a coordination to the bandboxed sector names
        rather than individual sector names

        Parameters
        ----------
        coordination: Coordination
            The coordination to be transformed

        Returns
        --------
        Coordination
            The input coordination but with the to_sector and from_sector updated to represent
            the bandboxed sector controlling the aircraft (rather than the individual sectors)
        """

        from_sector = self.airspace.get_containing_bandboxed_sector(coordination.from_sector)
        to_sector = self.airspace.get_containing_bandboxed_sector(coordination.to_sector)

        # transform None to "background"
        if from_sector is None:
            from_sector = "background"
        if to_sector is None:
            to_sector = "background"

        return Coordination(
            callsign=coordination.callsign,
            from_sector=from_sector,
            to_sector=to_sector,
            fl=coordination.fl,
            fix=coordination.fix,
            direction=coordination.direction,
            level_by=coordination.level_by,
            level_by_details=coordination.level_by_details,
            secondary_coord_conditions=coordination.secondary_coord_conditions,
            the_datetime=coordination.datetime,
        )

    def entry_coordination(self, sector: str, callsign: str, apply_bandboxing: bool = True) -> Coordination | None:
        """The entry coordination to a specific sector for a specific callsign

        Parameters
        ----------
        sector: str
            Entry coordination will be returned for this sector
        callsign: str
            Callsign of aircraft for which the entry coordination will be returned
        apply_bandboxing: bool, default is True
            If true, the bandboxed version of the coordination will be returned, even if the coordination
            is in terms of individual sectors

        Returns
        --------
        Coordination or None
            The entry coordination for a callsign, sector pair. None if the coordination does not exist.
        """
        # the coordinations may be in terms of individual sectors therefore
        # we iterate over the given sector and all its component sectors to get
        # all candidate entry coordinations

        if sector == "background":
            # any coordination to a sector that is unknown is an entry coordination
            # for "background"
            all_sector_names = set(self.airspace.list_individual_sectors())
            all_sector_names |= set(self.airspace.sectors.keys())

            # keep any coordination from a sector that is in the airspace
            # to a sector that isn't in the airspace
            candidate_entry_coords = [
                coord
                for coord in self.coordinations.coords.values()
                if coord.callsign == callsign
                and coord.from_sector in all_sector_names
                and coord.to_sector not in all_sector_names
            ]
        else:
            # get sector_to candidates, which is the named sector and all it's component sectors
            to_sector_candidates = set(self.airspace.airspace_configuration.get(sector, []))

            # if provided, include bandboxed sector
            to_sector_candidates |= set(self.airspace_bandboxing_dict.get(sector, sector.split("_")))

            # include the given (possibly bandboxed) sector
            to_sector_candidates.add(sector)

            # any coord with a (callsign, to_sec) where to_sec is a candidate to_sec is a candidate coordination
            candidate_entry_coords = [
                self.coordinations.get(callsign=callsign, to_sector=to_sec) for to_sec in to_sector_candidates
            ]
            # flatten
            candidate_entry_coords = [v for v_list in candidate_entry_coords for v in v_list]

            # only keep candidate entry_coords which are from a sector NOT in the to_sector candidate list
            candidate_entry_coords = [
                coord for coord in candidate_entry_coords if coord.from_sector not in to_sector_candidates
            ]

        # if no candidates, return None
        coordination = None

        # else if one candidate, that is the coordination
        if len(candidate_entry_coords) == 1:
            coordination = candidate_entry_coords[0]
        # if multiple candidates, warn, and keep one with latest datetime
        elif len(candidate_entry_coords) > 1:
            coordination = sorted(
                candidate_entry_coords,
                # sort by datetime unless datetime is None, in which case just return -1 (smaller than any datetime)
                key=lambda coord: coord.datetime if coord.datetime is not None else -1,
            )[-1]
            logger.warning(
                "WARNING: Multiple coordinations out of a sector, returning the one with the latest timestamp",
                stacklevel=2,
            )

        # Convert any unknown sector origin/destinations to background
        if coordination is not None:
            if apply_bandboxing:
                # transform coordination to bandboxed version if requested
                coordination = self.transform_coordination_to_bandboxed(coordination)
            else:
                if coordination.from_sector is None:
                    coordination.from_sector = "background"
                if coordination.to_sector is None:
                    coordination.to_sector = "background"

        return coordination

    def exit_coordination(
        self, sector: str, callsign: str, apply_bandboxing: bool = True
    ) -> Coordination | None | None:
        """The exit coordination to a specific sector for a specific callsign

        Parameters
        ----------
        sector: str
            Exit coordination will be returned for this sector
        callsign: str
            Callsign of aircraft for which the exit coordination will be returned
        apply_bandboxing: bool, default is True
            If true, the bandboxed version of the coordination will be returned, even if the coordination
            is in terms of individual sectors

        Returns
        --------
        Coordination or None
            The exit coordination for a callsign, sector pair. None if the coordination does not exist.
        """
        # the coordinations may be in terms of individual sectors therefore
        # we iterate over the given sector and all its component sectors to get
        # all candidate exit coordinations
        if sector == "background":
            # any coordination from a sector that is unknown is an exit coordination
            # for "background"
            all_sector_names = set(self.airspace.list_individual_sectors())
            all_sector_names |= set(self.airspace.sectors.keys())

            # keep any coordination from a sector that isn't in the airspace
            # to a sector that is in the airspace
            candidate_exit_coords = [
                coord
                for coord in self.coordinations.coords.values()
                if coord.callsign == callsign
                and coord.to_sector in all_sector_names
                and coord.from_sector not in all_sector_names
            ]

        else:
            # look for coordination from "sector"
            # get sector_from candidates, which is the named sector and all it's component sectors
            from_sector_candidates = set(self.airspace.airspace_configuration.get(sector, []))
            if self.airspace_bandboxing_dict:
                from_sector_candidates |= set(self.airspace_bandboxing_dict.get(sector, sector.split("_")))

            # include the given (possibly bandboxed) sector
            from_sector_candidates.add(sector)

            # any coord with a (callsign, from_sec) where from_sec is a candidate from sec
            # and the to_sector isn't in the candidate sector list, is a candidate coordination
            candidate_exit_coords = [
                self.coordinations.coords[(callsign, from_sec)]
                for from_sec in from_sector_candidates
                if (callsign, from_sec) in self.coordinations.coords
            ]

            # remove internal coordinations by only keeping candidate exit_coords which
            # are to a sector NOT in the from_sector candidate list
            candidate_exit_coords = [
                coord for coord in candidate_exit_coords if coord.to_sector not in from_sector_candidates
            ]

        # if no candidates, return None
        coordination = None

        # else if one candidate, that is the coordination
        if len(candidate_exit_coords) == 1:
            coordination = candidate_exit_coords[0]

        # if multiple candidates, warn, and keep one with latest datetime
        elif len(candidate_exit_coords) > 1:
            coordination = sorted(
                candidate_exit_coords,
                # sort by datetime unless datetime is None, in which case just return -1 (smaller than any datetime)
                key=lambda coord: coord.datetime if coord.datetime is not None else -1,
            )[-1]
            logger.warning(
                f"WARNING: Multiple coordinations out of a sector for {callsign},"
                " returning the one with the latest timestamp",
                stacklevel=2,
            )

        # Convert any unknown sector origin/destinations to background
        if coordination is not None:
            if apply_bandboxing:
                # transform coordination to bandboxed version if requested
                coordination = self.transform_coordination_to_bandboxed(coordination)
            else:
                if coordination.from_sector is None:
                    coordination.from_sector = "background"
                if coordination.to_sector is None:
                    coordination.to_sector = "background"

        return coordination

    def next_sector_of_aircraft_from_sector(
        self,
        callsign: str,
        from_sector: str,
        apply_bandboxing: bool = True,
        recursive_depth: int = 0,
        max_recursive_depth: int = 20,
    ) -> str | None:
        """
        The next coordinated sector for an aircraft from a specific sector

        Parameters
        ----------
        callsign: str
            Callsign of aircraft for which the next sector will be returned
        from_sector: str
            Sector before the "next sector" which will be returned
        apply_bandboxing: bool, default is True
            If true, the bandboxed version of the coordination will be returned, even if the coordination
            is in terms of individual sectors
        recursive_depth: int
            Recursive depth of function.
            Required to stop infinite loops which can occur with real-world data.
        max_recursive_depth: int
            Maximum recursive depth allowed before function emits a warning and returns None.

        Returns
        -------
        str or None
            Name of next sector for a callsign, from_sector pair.
            None if coordination out of from_sector does not exist.
        """
        # if recursive depth too large. Then there may be an infinite loop due to errors in real-world data.
        # in this case, return None
        if recursive_depth > max_recursive_depth:
            logger.warning(
                f"Maximum recursive depth of {max_recursive_depth} reached for next_sector_of_aircraft_from_sector."
                f"Exiting function and returning None. (callsign={callsign}, from_sector={from_sector})",
                stacklevel=2,
            )
            return None

        # need to use individual sector to step through possible bandboxed sector to find next
        # sector outside the bandboxed sector
        coordination = self.exit_coordination(from_sector, callsign, apply_bandboxing=False)

        # next sector is given by the coordination going FROM the aircraft's
        # current sector and TO the next sector
        if coordination is None:
            to_sector = None

        else:
            # next sector is where the aircraft is coordinated to
            to_sector = coordination.to_sector
            if apply_bandboxing:
                # if not a bandboxed sector, find the bandboxed sector containing from_sector
                bandboxed_sector = self.airspace.get_containing_bandboxed_sector(from_sector)

                # if bandboxed, the next coordination might be in the same bandboxed sector
                # in which case recursively repeat process until to_sector != aircraft.current_sector
                to_sector_bandboxed = self.airspace.get_containing_bandboxed_sector(to_sector)

                if bandboxed_sector == to_sector_bandboxed:
                    to_sector = self.next_sector_of_aircraft_from_sector(
                        callsign,
                        to_sector,
                        recursive_depth=recursive_depth + 1,
                        max_recursive_depth=max_recursive_depth,
                    )

                # as apply_bandboxing is True, return the bandboxed version of the next sector
                to_sector = self.airspace.get_containing_bandboxed_sector(to_sector)

        return to_sector

    def next_sector_of_aircraft(self, callsign: str, apply_bandboxing: bool = True) -> str | None:
        """The next coordinated sector for an aircraft, from its current sector

        Parameters
        ----------
        callsign: str
            The callsign of the aircraft of interest
        apply_bandboxing: bool, default is True
            If true, the bandboxed version of the coordination will be returned, even if the coordination
            is in terms of individual sectors

        Returns
        ----------
        Coordination or None if coordination from current sector does not exist.
        """

        aircraft = self.aircraft[callsign]
        return self.next_sector_of_aircraft_from_sector(
            callsign, aircraft.current_sector, apply_bandboxing=apply_bandboxing
        )

    def remove_coordinations_within_sector(self, sector_name: str, callsign: str) -> None:
        """Remove any coordinations between the individual sectors within the given bandboxed sector

        Parameters
        ----------
        sector_name: str
            The bandboxed sector for which internal coordination will be deleted
        """

        # Coordinations may be in terms of individual sectors even when the sectors are bandboxed
        # (this occurs in the real-world data). The incomm is in terms of the bandboxed sectors.
        # Remove any coordination between the individual sectors forming the bandboxed "sector_name" sector

        # find the individual sectors for each (possibly bandboxed) sector
        # assumes bandboxing naming convention is "sec1_sec2_sec3"
        # Use of name splitting is required as "sector_name" may not be a currently bandboxed sector
        individual_sectors = set(self.airspace.airspace_configuration.get(sector_name, []))
        if self.airspace_bandboxing_dict:
            individual_sectors |= set(self.airspace_bandboxing_dict.get(sector_name, sector_name.split("_")))

        # add the bandboxed sector names, as the coordinations may be in terms of the bandboxed names
        individual_sectors.add(sector_name)

        # find each possible combination of individual sectors to remove internal coordinations
        # within a bandboxed sector
        within_sector_combinations = set(itertools.product(individual_sectors, individual_sectors))

        coords_for_aircraft = self.coordinations.get(callsign)

        coords_to_remove = []
        for coord in coords_for_aircraft:
            # if to_sector or from_sectors don't match any bandboxed or individual sectors
            # they are 'background' sectors
            if (
                coord.from_sector in self.airspace.list_individual_sectors()
                or coord.from_sector in self.airspace.sectors
            ):
                coordinated_from_sector = coord.from_sector
            else:  # it's a background sector
                coordinated_from_sector = "background"

            if coord.to_sector in self.airspace.list_individual_sectors() or coord.to_sector in self.airspace.sectors:
                coordinated_to_sector = coord.to_sector
            else:  # it's a background sector
                coordinated_to_sector = "background"

            # if the coordination matches any possible combination of the individual sectors which
            # form the bandboxed sector then delete it
            if (
                coordinated_from_sector,
                coordinated_to_sector,
            ) in within_sector_combinations:
                # There is a maximum of one coordination from any sector per aircraft, therefore a
                # (callsign, from_sector) pair uniquely identifies a stored coordination to delete
                coords_to_remove.append((callsign, coord.from_sector))

        # remove the coord. Kept as list so coords aren't mutated in the loop
        for callsign, from_sector in coords_to_remove:
            self.coordinations.remove(callsign, from_sector)

    def remove_coordination_to_sector(self, to_sector: str, callsign: str, from_sector: str | None = None) -> None:
        """
        Remove coordinations to the named sector, or any of its component sectors (if bandboxed).
        - If the bandbox name is passed as the 'to sector', any internal coordinations
        (i.e. from one component sector to another) will not be removed.
        - However if a specific component sector name is passed as the 'to_sector' then any internal coordinations
        to that sector WILL be removed.
        - If from_sector is provided, only delete if this also matches.

        Parameters
        ----------
        to_sector: str
            Sector to which coordinations will be removed
        callsign: str
            Callsign for which coordinations will be removed
        from_sector: str, optional
            If not None, then only coordinations also matching from_sector will be removed
        """
        # Coordinations may be in terms of individual sectors even when the sectors are bandboxed
        # (this occurs in the real-world data). The incomm is in terms of the bandboxed sectors.
        # Remove coordinations representing all combinations of individual sectors which represent
        # a coordination to "to_sector"

        # find the individual sectors for each (possibly bandboxed) sector
        # assumes bandboxing naming convention is "sec1_sec2_sec3"
        # Use of name splitting is required as "sector_name" may not be a currently bandboxed sector
        individual_to_sectors = set(self.airspace.airspace_configuration.get(to_sector, []))
        # In the case of initialising from data we need to check airspace_bandboxing_dict which contains
        # more bandbox options than in the current configuration.
        if self.airspace_bandboxing_dict:
            individual_to_sectors |= set(self.airspace_bandboxing_dict.get(to_sector, to_sector.split("_")))

        # add the bandboxed sector names, as the coordinations may be in terms of the bandboxed names
        individual_to_sectors.add(to_sector)

        coords_for_aircraft = self.coordinations.get(callsign)

        # find each possible combination of individual sectors representing coordinations internal
        # to "to_sector"
        within_sector_combinations = set(itertools.product(individual_to_sectors, individual_to_sectors))

        coords_to_remove: list[tuple[str, str | None]] = []
        for coord in coords_for_aircraft:
            # if to_sector or from_sectors don't match any bandboxed or individual sectors
            # they are 'background' sectors
            if (
                coord.from_sector in self.airspace.list_individual_sectors()
                or coord.from_sector in self.airspace.sectors
            ):
                coordinated_from_sector = coord.from_sector
            else:  # it's a background sector
                coordinated_from_sector = "background"

            if coord.to_sector in self.airspace.list_individual_sectors() or coord.to_sector in self.airspace.sectors:
                coordinated_to_sector = coord.to_sector
            else:  # it's a background sector
                coordinated_to_sector = "background"

            # if from_sector is provided then only consider coordinations matching this
            if from_sector is not None:
                individual_from_sectors = set(self.airspace_bandboxing_dict.get(from_sector, from_sector.split("_")))
                individual_from_sectors.add(from_sector)

                # don't delete this coordination if it doesn't match from_sector
                if coordinated_from_sector not in individual_from_sectors:
                    continue

            # if the coordinate_to_sector matches any individual sectors representing
            # coordinations to "to_sector" we consider deleting it
            # We want to exclude internal coordinations so find and exclude these
            if (
                coordinated_to_sector in individual_to_sectors
                and (coordinated_from_sector, coordinated_to_sector) not in within_sector_combinations
            ):
                # There is a maximum of one coordination from any sector per aircraft, therefore a
                # (callsign, from_sector) pair uniquely identifies a stored coordination to delete
                coords_to_remove.append((callsign, coord.from_sector))

        # remove the coord. Kept as list so coords aren't mutated in the loop
        for callsign, from_sector in coords_to_remove:
            self.coordinations.remove(callsign, from_sector)

    def remove_coords_pre_outcomm(self, callsign: str):
        """
        When an aircraft is about to be outcommed from a sector, delete any coordinations that
        were either:
         - from the previous sector into that sector or
         - internal to that sector, (occurs for bandboxed sectors with real-world data)

        Parameters
        ----------
        callsign: str
            Callsign of aircraft that is about to be outcommed.
        """
        aircraft = self.aircraft[callsign]
        # delete coordinations which were from the previous sector to the current
        # sector that the aircraft is about to outcomm from
        # if outcomming from a non-background sector, then remove EVERY coordination into the sector the aircraft is
        # outcomming from, which is required to clean up real-world coordinations as coordinations often aren't
        # linked into a coherent path. However if outcomming from a background sector, then only delete coordination
        # matching the aircraft's previous sector by setting from_sector = aircraft.previous_sector,
        # as else the exit coordination will be deleted
        from_sector = aircraft.previous_sector if aircraft.current_sector == "background" else None

        self.remove_coordination_to_sector(
            to_sector=aircraft.current_sector,
            callsign=callsign,
            from_sector=from_sector,
        )
        # remove any coordinations internal to the sector the aircraft is about to outcommed from
        self.remove_coordinations_within_sector(aircraft.current_sector, callsign)
