from __future__ import annotations

import json
import typing
from datetime import datetime as datetime_type

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from bluebird_dt.core.aircraft import Aircraft
from bluebird_dt.core.airspace import Airspace
from bluebird_dt.core.flight_plan import FlightPlan
from bluebird_dt.core.pos3d import Pos3D
from bluebird_dt.core.route import Route
from bluebird_dt.mixin import Comparison
from bluebird_dt.utility.convert import nan_to_none

TAircraft = typing.TypeVar("TAircraft", bound=Aircraft)


class Coordination(BaseModel, Comparison):
    """
    Entry or exit agreement, i.e. a flight level and fix [str] when entry/exit should happen along with
    the vertical coordination indicator (how it enters/exits the sector)
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        strict=False,  # Non strict to allow string datetime input
        populate_by_name=True,  # Allow input by alias or fieldname
    )

    callsign: str
    from_sector: str | None = None
    to_sector: str | None = None
    fl: float = Field(..., ge=0.0, description="Flight level must be non-negative")
    fix: str
    # remove requirement for specific fix lengths for now.
    # should be enforced later down the line
    # if len(fix) not in [3, 5]:
    #     raise ValueError(f"Fix must be a 3 or 5 character string. Fix {fix} of length {len(fix)} was passed.")
    direction: typing.Literal["Horizontal", "Down", "Up"]
    level_by: bool | None = False
    level_by_details: dict[str, float] | typing.Literal["placeholder level by detail"] | None = None
    secondary_coord_conditions: str | None = None
    datetime: datetime_type | None = Field(default=None, alias="the_datetime")

    @field_validator("from_sector", "to_sector", "level_by_details", "secondary_coord_conditions", mode="before")
    @classmethod
    def _nan_to_none(cls, v: object) -> object:
        return nan_to_none(v)

    @field_validator("fl")
    @classmethod
    def _validate_fl(cls, v: float) -> float:
        if pd.isna(v):
            raise ValueError(f"{v} is not a valid flight level")
        return v

    @model_validator(mode="after")
    def _check_level_by_consistency(self) -> Self:
        if self.level_by and self.level_by_details is None:
            raise ValueError("Must have level_by_details if level_by is True")

        if not self.level_by and self.level_by_details is not None:
            raise ValueError("Cannot have level_by_details if level_by is False")

        return self

    def data(self) -> dict[str, typing.Any]:
        """
        Get the data as a serialisable dictionary.

        Returns
        --------
        dict
        """
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """
        Serialise the instance to JSON.

        Returns
        --------
        str
        """
        return self.model_dump_json(indent=4)

    @classmethod
    def from_json(cls, s: str) -> Self | None:
        """
        Construct a new instance from JSON representation.

        Parameters
        ----------
        s: str
            A string representation of an Environment in a JSON/dictionary structure.

        Returns
        --------
        Coordination or None
        """
        if not s or json.loads(s) is None or s.strip() in ("", "null", "None"):
            return None

        return cls.model_validate_json(s)

    @staticmethod
    def load(filename: str) -> Coordination | None:
        """
        Construct a new instance from a file.

        Parameters
        ----------
        filename: str
            Path to a JSON file with a Coordination definition in a dictionary format.

        Returns
        --------
        Coordination or None
        """

        with open(filename) as fd:
            return Coordination.from_json(fd.read())

    def same_excluding_datetime(self, coord: Coordination) -> bool:
        """
        Whether target coordination is identical to this instance excluding the datetime parameter

        Parameters
        ----------
        coord: Coordination
            Coordination to be compared to this instance

        Returns
        --------
        Bool
            True if the coordinations are identical other than the datetime attribute
        """
        return (
            self.callsign == coord.callsign
            and self.from_sector == coord.from_sector
            and self.to_sector == coord.to_sector
            and self.fl == coord.fl
            and self.fix == coord.fix
            and self.direction == coord.direction
            and self.level_by == coord.level_by
            and self.level_by_details == coord.level_by_details
            and self.secondary_coord_conditions == coord.secondary_coord_conditions
        )

    @staticmethod
    def from_dict(item: dict) -> Coordination:
        return Coordination.from_json(json.dumps(item))


class CoordinationsManager:
    """
    Class holding all coordinations in an environment and some convenient methods to manage them.

    Attributes
    ----------
    coords: dict[tuple[str, str | None], TCoordination]
        A coordination store indexed by (callsign, from_sector), where the coordination is of type
        bluebird_dt.Coordination, or bounded by it (implements bluebird_dt.Coordination).

    """

    coords: dict[tuple[str, str | None], Coordination]

    def __init__(self, coordinations: list[Coordination] | None = None):
        """
        Construct a new instance.

        Parameters
        ----------
        coordinations: list of Coordinations, optional
            The coordinations for the airspace and aircraft in the environment
        """

        # keep coordinations as a dict with tuple(callsign, from_sector) as the key
        # which enforces only 1 coordination from a sector for a specific aircraft
        self.coords = {}

        if coordinations is not None:
            for coord in coordinations:
                key = (coord.callsign, coord.from_sector)
                self.coords[key] = coord

    def add(self, coord: Coordination):
        """Add a single coordination

        Parameters
        ----------
        coord: TCoordination
            A Coordination to be added of type bluebird_dt.Coordination, or bounded by it (implements
            bluebird_dt.Coordination).
        """
        key = (coord.callsign, coord.from_sector)
        self.coords[key] = coord

    def values(self) -> list[Coordination]:
        """All coordinations as a list


        Returns
        ----------
        list[TCoordinations]
            List of all coordination of type bluebird_dt.Coordination, or bounded by it (implements
            bluebird_dt.Coordination).
        """
        return list(self.coords.values())

    def remove(self, callsign: str, from_sector: str) -> bool:
        """Remove a coordination from the set of coordinations if it matches
         both the given callsign and from_sector

        Parameters
        ----------
        callsign: str
            Callsign of the aircraft for which a coordination will be removed
        from_sector: str
            The from_sector value of the Coordination that will be removed if matched
            as well as the callsign

        Returns
        ----------
        bool
            True if coordination was removed, otherwise False
        """

        key = (callsign, from_sector)
        if key in self.coords:
            del self.coords[key]
            return True

        return False

    def get(
        self,
        callsign: str | None = None,
        from_sector: str | None = None,
        to_sector: str | None = None,
    ) -> list[Coordination]:
        """Get coordinations for any combination of callsigns, from_sector and to_sector
        with each being an optional parameter.

        Parameters
        ----------
        callsign: str, optional
            If not None, filter to coordinations matching this callsign
        from_sector: str, optional
            If not None, filter to coordinations matching this from_sector
        to_sector: str, optional
            If not None, filter to coordinations matching this to_sector

        Returns
        ----------
        list[TCoordination]
            List of coordinations with optional filtering to the parameters
        """
        # match all keys and then filter as appropriate
        matching_keys = self.coords.keys()
        if callsign is not None:
            matching_keys = [(call_sgn, from_sec) for (call_sgn, from_sec) in matching_keys if call_sgn == callsign]
        if from_sector is not None:
            matching_keys = [(call_sgn, from_sec) for (call_sgn, from_sec) in matching_keys if from_sec == from_sector]
        if to_sector is not None:
            matching_keys = [key for key in matching_keys if self.coords[key].to_sector == to_sector]

        return [self.coords[key] for key in matching_keys]

    def data(self) -> list[dict[str, typing.Any]]:
        """
        Get the data as a serialisable dictionary.

        Returns
        ----------
        dict
        """

        return [coord.data() for coord in self.coords.values()]

    @staticmethod
    def from_json(s: str) -> CoordinationsManager:
        """
        Construct a new instance from JSON representation.

        Parameters
        ----------
        s: str
            A string representation of a CoordinationsManager in a JSON/dictionary structure.

        Returns
        ----------
        CoordinationsManager[TCoordination]
        """

        data = json.loads(s)

        # Convert the coordination dictionaries to TCoordinations
        crds = [Coordination.from_dict(crd) for crd in data["coordinations"]]

        return CoordinationsManager(crds)

    @staticmethod
    def load(filename: str) -> CoordinationsManager:
        """
        Construct a new instance from a file.

        Parameters
        ----------
        filename: str
            Path to a JSON file with an CoordinationsManager definition in dictionary format.

        Returns
        ----------
        CoordinationsManager[TCoordination]
        """

        with open(filename) as fd:
            return CoordinationsManager.from_json(fd.read())

    def to_json(self) -> str:
        """
        Serialise the instance to JSON.

        Returns
        ----------
        str
        """
        return json.dumps(self.data(), indent=4)

    def contains_excluding_times(self, target_coord: Coordination) -> bool:
        """
        Whether target coordination is identical to this instance excluding the datetime parameter

        Parameters
        ----------
        target_coord: TCoordination
            Coordination to be compared to all coordinations

        Returns
        --------
        Bool
            True if the any of the coordinations are identical to the target
            coordination, excluding their datetime attribute
        """
        # get coordination by matching callsign and from_sector only
        potential_match = self.coords.get((target_coord.callsign, target_coord.from_sector), None)

        # if no potential match then coordination isn't present
        contains_target_coord = False

        if potential_match is not None:
            # if callsign and from_sector have matched then check if all other values
            # of is the same except for the datetime
            contains_target_coord = target_coord.same_excluding_datetime(potential_match)

        return contains_target_coord

    @staticmethod
    def find_entry_exit_fixes(airspace: Airspace, route: Route, sector_name: str) -> tuple[str, str]:
        """
        Given a route and a sector name, find the fixes to use
        for entry and exit coordinations.

        Parameters
        ==========
        airspace: Airspace
            the airspace under consideration
        route: Route
            the aircraft's route.
        sector_name: str
            name of the controlling sector
        Returns
        =======
        entry_fix, exit_fix: tuple[str, str]
            The fix names for entry and exit coordinations
        """
        sector = airspace.sectors[sector_name]
        entry_fix, exit_fix = None, None

        # If route starts inside the sector, take entry_fix to be first fix on route
        if sector.contains_laterally(airspace.fixes.places[route.filed[0]]):
            entry_fix = route.filed[0]
        # If route ends inside the sector, take exit_fix to be last fix on route
        if sector.contains_laterally(airspace.fixes.places[route.filed[-1]]):
            exit_fix = route.filed[-1]

        if entry_fix and exit_fix:
            return entry_fix, exit_fix
        # If we start or end outside the sector, find fixes on the boundary
        sector_boundary = sector.boundary()

        if not entry_fix:
            # start from the beginning of the route
            for fix in route.filed:
                fix_pos = airspace.fixes.places[fix]
                if sector_boundary.on_boundary(fix_pos):
                    entry_fix = fix
                    break
            # if no match, revert to first fix in route
            entry_fix = entry_fix if entry_fix is not None else route.filed[0]
        if not exit_fix:
            # start from the end of the route
            for fix in reversed(route.filed):
                fix_pos = airspace.fixes.places[fix]
                if sector_boundary.on_boundary(fix_pos) and fix is not entry_fix:
                    exit_fix = fix
                    break
            # if no match, use last fix in route
            exit_fix = exit_fix if exit_fix is not None else route.filed[-1]
        return entry_fix, exit_fix

    @classmethod
    def aircraft_with_coordinations(
        cls,
        callsign: str,
        pos: Pos3D,
        heading: float,
        speed: float,
        route: Route,
        sector_name: str,
        entry_fl: float,
        exit_fl: float,
        airspace: Airspace,
        on_route: bool = False,
        prev_sector: str = "background",
        next_sector: str = "background",
        coord_direction: typing.Literal["Horizontal", "Down", "Up"] = "Horizontal",
        typeof_aircraft: type[TAircraft] = Aircraft,
    ) -> tuple[TAircraft, Coordination, Coordination]:
        """
        Create an Aircraft instance, and entry and exit Coordinations,
        given the necessary input parameters.

        Parameters
        ----------
        callsign: str
            callsign of the aircraft being created.
        pos: Pos3D
            position (lat, lon, FL) at which aircraft will be created.
        heading: float
            initial heading for the aircraft
        speed: float
            initial speed for the aircraft
        route: Route
            sequence of fixes on aircraft's route.
        sector_name: str
            name of the sector for entry and exit coordinations
        entry_fl: float
            Flight Level for entry coordination
        exit_fl: float
            Flight Level for exit coordination
        airspace: Airspace
            specify the airspace containing the sector being controlled.
        on_route: bool
            Setting for  "on_route" flag on created aircraft. Default is False.
        prev_sector: str
            "from_sector" for entry coordination. Default is "background".
        next_sector: str
            "to_sector" for exit coordination. Default is "background".
        coord_direction: Literal["Horizontal", "Down", "Up"]
            whether the coordination is horizontal or vertical.  Default is "Horizontal".
        typeof_aircraft: type[TAircraft]
            if derived class of Aircraft is to be created, specify here.

        Returns
        =======
        tuple[TAircraft, Coordination, Coordination]
            Newly instantiated Aircraft, with Entry Coordination and Exit Coordination.
        """
        # find the entry and exit fixes for the coordinations.
        entry_fix, exit_fix = cls.find_entry_exit_fixes(airspace, route, sector_name)

        # create entry and exit coordinations
        coordination_entry = Coordination(
            callsign=callsign,
            from_sector=prev_sector,
            to_sector=sector_name,
            fl=entry_fl,
            fix=entry_fix,
            direction=coord_direction,
        )

        coordination_exit = Coordination(
            callsign=callsign,
            from_sector=sector_name,
            to_sector=next_sector,
            fl=exit_fl,
            fix=exit_fix,
            direction=coord_direction,
        )

        # generate the Aircraft instance
        flight_plan = FlightPlan(route)
        aircraft = typeof_aircraft(
            pos.lat,
            pos.lon,
            pos.fl,
            heading,
            flight_plan,
            callsign,
            selected_fl=int(pos.fl),
            current_sector=sector_name,
        )
        aircraft.speed_tas = speed
        aircraft.selected_instructions.cas = speed
        aircraft.on_route = on_route
        aircraft.simulated = True

        return aircraft, coordination_entry, coordination_exit
