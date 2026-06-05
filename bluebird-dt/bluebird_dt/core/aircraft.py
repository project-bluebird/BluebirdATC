from __future__ import annotations

import json
import math
import random
import typing
from enum import Enum

import typing_extensions

from bluebird_dt.core.action import Action
from bluebird_dt.core.flight_plan import FlightPlan
from bluebird_dt.core.pilot import Pilot
from bluebird_dt.core.pos2d import Pos2D
from bluebird_dt.core.pos3d import Pos3D
from bluebird_dt.core.sector import Sector
from bluebird_dt.logger import logger
from bluebird_dt.mixin import Comparison
from bluebird_dt.utility.convert import nan_to_none

if typing.TYPE_CHECKING:
    from bluebird_dt.core import Airspace


class FlightState(Enum):
    CLIMB = "climb"
    CRUISE = "cruise"
    DESCEND = "descend"


class Instructions:
    """
    A data structure to store the combined actions (lateral, vertical and speed) issued to an aircraft

    Each Aircraft has two instances of this class:
        `cleared_instructions` to store the actions given by the ATCO
        `selected_instructions` to store the actions enacted by the Pilot
    Normally these two instances are the same, but they can be made to differ in situations such as the pilot
    mis-hearing the ATCO
    """

    def __init__(
        self,
        fl: float | None = None,
        mach: float | None = None,
        cas: float | None = None,
        vertical_speed: float | None = None,
        heading: float | None = None,
        on_route: bool = True,
        speed_action: Action | None = None,
        vertical_speed_action: Action | None = None,
        vertical_action: Action | None = None,
        lateral_action: Action | None = None,
    ):
        """
        Construct a new instance.

        Parameters
        ----------
        fl: float, optional
            The instructed flight level
        mach: float, optional
            The instructed mach speed
        cas: float, optional
            The instructed calibrated air speed
        vertical_speed: float, optional
            The instructed vertical speed
        heading: float, optional
            The instructed heading
        on_route: bool , optional
            The instructed on_route
        speed_action: Action, optional
            The last instructed speed Action
        vertical_speed_action: Action, optional
            The last instructed vertical speed Action
        vertical_action: Action, optional
            The last instructed vertical Action
        lateral_action: Action, optional
            The last instructed lateral Action

        Returns
        --------
        Instructions
        """
        self.fl = nan_to_none(fl)
        self.mach = nan_to_none(mach)
        self.cas = nan_to_none(cas)
        self.vertical_speed = nan_to_none(vertical_speed)
        self.heading = nan_to_none(heading)
        self.on_route = on_route
        self.speed_action = speed_action
        self.vertical_speed_action = vertical_speed_action
        self.vertical_action = vertical_action
        self.lateral_action = lateral_action

    def data(self) -> dict[str, typing.Any]:
        """
        Create a dictionary containing the Instructions data.

        Returns
        --------
        data_dict: Dict[str, Any]
            Data dictionary of Instructions data
        """

        return {
            "fl": self.fl,
            "mach": self.mach,
            "cas": self.cas,
            "vertical_speed": self.vertical_speed,
            "heading": self.heading,
            "on_route": self.on_route,
            "speed_action": self.speed_action.data() if self.speed_action else None,
            "vertical_speed_action": self.vertical_speed_action.data() if self.vertical_speed_action else None,
            "vertical_action": self.vertical_action.data() if self.vertical_action else None,
            "lateral_action": self.lateral_action.data() if self.lateral_action else None,
        }

    def to_json(self) -> str:
        """
        Serialise the instance to JSON string.

        Returns
        --------
        str
        """

        return json.dumps(self.data(), indent=4)

    @staticmethod
    def from_json(s: str) -> Instructions:
        """
        Construct a new instance from a string in JSON format.

        Parameters
        ----------
        s: str
            A string representation of Instructions in a JSON/dictionary structure.

        Returns
        --------
        Instructions
        """

        data = json.loads(s)

        i = Instructions()

        i.fl = data.get("fl", None)
        i.mach = data.get("mach", None)
        i.cas = data.get("cas", None)
        i.vertical_speed = data.get("vertical_speed", None)
        i.heading = data.get("heading", None)
        i.on_route = data.get("on_route", None)

        i.speed_action = Action.from_json(json.dumps(data["speed_action"])) if data["speed_action"] else None
        i.vertical_speed_action = (
            Action.from_json(json.dumps(data["vertical_speed_action"])) if data["vertical_speed_action"] else None
        )
        i.vertical_action = Action.from_json(json.dumps(data["vertical_action"])) if data["vertical_action"] else None
        i.lateral_action = Action.from_json(json.dumps(data["lateral_action"])) if data["lateral_action"] else None

        return i


class Aircraft(Comparison):
    """
    An aeroplane, helicopter, or other machine capable of flight.

    In its most minimal form, an Aircraft has a position, heading, speed and a callsign.
    Optionally, it can have operational parameters and information associated with how
    it is viewed from an air traffic control point of view.
    """

    def __init__(
        self,
        lat: float,
        lon: float,
        fl: float,
        heading: float,
        flight_plan: FlightPlan | None,
        callsign: str,
        selected_fl: int | None = None,
        ufid: str | None = None,
        rate_of_turn: float | None = None,
        aircraft_type: str | None = None,
        operation_params: dict | None = None,
        controllable: bool = True,
        simulated: bool = True,
        current_sector: str | None = None,
        random_seed: int | None = None,
        pilot: Pilot | None = None,
        squawk: str | None = None,
        wake_vortex: str | None = None,
        last_passed_filed_idx: int | None = None,
        last_passed_current_idx: int | None = None,
        squawk_ident_until: float | None = None,
    ):
        """
        Construct a new instance.

        Parameters
        ----------
        lat: float
            The Aircraft latitude in degrees.
        lon: float
            The Aircraft longitude in degrees.
        fl: float
            The Aircraft flight level.
        heading: float
            The Aircraft heading in degrees.
        flight_plan: FlightPlan
            A dict of the Aircraft planned Route
        callsign: str
            The Aircraft callsign.
        selected_fl: int, optional
            The flight level selected on the aircraft computer
        ufid : str, optional
            The Aircraft's unique flight ID.
        rate_of_turn: float, optional
            The Aircraft rate of turn in degrees/sec.
        aircraft_type: str, optional
            The Aircraft type
        operation_params: dict, optional
            Dictionary of aircraft operational parameters.
        controllable: bool, optional
            Indicates whether the Aircraft responds to Agent issued Actions (default=True).
        simulated: bool, optional
            Indicates whether the Aircraft is subject to Predictor manipulation (default=True)
        current_sector: str, optional
            Indicates which Sector is currently controlling the Aircraft.
            If None, is set to "background"
        random_seed: int or None
            Random seed for sampling from speed distributions
        pilot: Aircraft Pilot, default is a Pilot()
            The Aircraft's Pilot, which processes actions issued to the aircraft
        squawk: str, optional
            Secondary surveillance radar (squawk) code
        wake_vortex: str, optional
            Wake vortex category of the aircraft. Will be set automatically using bada OPF files if None.
        last_passed_filed_idx: int or None
            Index of flight_plan.route.filed which was last passed.
            None indicates no filed fixes have been passed yet.
        last_passed_current_idx: int or None
            Index of flight_plan.route.current which was last passed.
            None indicates no current fixes have been passed yet.
        squawk_ident_until: The unix time that the aircraft squawk idents until.

        Attributes
        ----------
        cleared_instructions: Instructions
            The instructions given by the ATCO.
        selected_instructions: Instructions
            The instructions enacted by the Pilot.
        percentile_rank_dict: dict[str, [float, None]], optional
            The percentile rank dictionary of the aircraft. The allowed keys in the dict are: "cas_cr", "cas_cl",
            "cas_des", "mach_cr", "mach_cl", "mach_des", "rocd_cl" and "rocd_des". The values to these keys are used
            to assign the aircraft horizontal and vertical speed scores from a probability distribution.
        cleared_fl: float, multiple of 10.
            Last-cleared flight level.
            Upon instantiation this is set to the current flight level.
            Note that if a flight level is given that is not a multiple of 10, it will be rounded to the nearest
            multiple of 10.
        selected_fl: float, multiple of 10.
            Selected flight level (causes the Aircraft to climb/descend).
            Upon instantiation this is set to the current flight level.
            Note that if a flight level is given that is not a multiple of 10, it will be rounded to the nearest
            multiple of 10.
        speed_tas: float or None
            The true airspeed (TAS) of the Aircraft in knots.
            This is only set by the Predictors.
        vertical_speed: float
            The Aircraft vertical speed (climb or descend) in feet/min.
            Upon instantiation this is set to 0.0 (i.e. level flight).
        on_route: bool
            A flag indicating if the aircraft is route following or not
        heading_changing_to: float or None
            A flag indicating whether or not an aircraft needs its heading changing. If None, aircraft is not expected
            to change heading, otherwise it is expected to change to the given heading.
        next_fix_index: int or None
            Index of the next fix in the flight plan route, when aircraft is flying on_route.
            If None, aircraft has passed its last route fix.
        ground_speed: float or None
            The ground speed of the aircraft in knots.
        ground_track_angle: float or None
            The ground track angle of the aircraft in degrees.
        predictor_params: dict[str, Any]
            Dictionary of parameters for use by any of the predictors to store state.
        """

        # wrap the given heading
        heading = heading % 360.0

        self.lat = lat
        self.lon = lon
        self.fl = fl
        self.heading = heading
        self.flight_plan = nan_to_none(flight_plan)
        self.callsign = nan_to_none(callsign)
        self.ufid = nan_to_none(ufid)
        self.squawk = nan_to_none(squawk)
        self.squawk_ident_until = nan_to_none(squawk_ident_until)
        self.last_passed_filed_idx = nan_to_none(last_passed_filed_idx)
        self.last_passed_current_idx = nan_to_none(last_passed_current_idx)

        if rate_of_turn and rate_of_turn < 0.0:
            raise ValueError("Aircraft rate of turn must be non-negative.")
        self.rate_of_turn = rate_of_turn

        if nan_to_none(aircraft_type) is None:
            self.aircraft_type = "B753"
        else:
            self.aircraft_type = aircraft_type

        # if wake vortex manually set, then use it, else set to default value "M"
        if wake_vortex:
            self.wake_vortex = wake_vortex
        else:
            self.set_wake_vortex_category()

        self.operation_params = {} if operation_params is None else operation_params

        self.controllable = controllable
        self.simulated = simulated

        # start current and previous sector as 'background'
        self._previous_sector = "background"
        self._current_sector = "background"
        self.current_sector = current_sector

        # set the pilot and ensure pilot callsign is correct if Pilot instance provided
        if pilot is None:
            self.default_pilot()
        elif isinstance(pilot, Pilot):
            self.pilot = pilot
            if pilot.callsign != callsign:
                raise ValueError("Supplied pilot must have the same callsign as its intended Aircraft.")
        else:
            raise ValueError("Supplied pilot must be a (sub-)class of Pilot, or None.")

        # Store the clearances/actions in two separate instances:
        # - cleared_instructions: the instructions given by the ATCO
        # - selected_instructions: the instructions enacted by the Pilot
        # These can be made to differ in situations such as the pilot mis-hearing the ATCO
        selected_fl = nan_to_none(selected_fl)
        use_fl = selected_fl if selected_fl is not None else fl
        self.cleared_instructions = Instructions(fl=use_fl, on_route=True)
        self.selected_instructions = Instructions(fl=use_fl, on_route=True)

        self.vertical_speed: float = 0.0
        self.speed_tas: float | None = None
        self.ground_speed: float | None = None

        self.ground_track_angle: float | None = None

        # By default the aircraft will maintain current heading and flight level
        self.heading_changing_to: None | float = None

        # Set the percentile rank dictionary, based on random seed
        self.percentile_rank_dict: dict[str, float | None] = {}

        # set self.random_seed and randomise speeds accordingly
        self.random_seed = random_seed
        self.randomise_performance(random_seed)

        # Next fix index defaults to 1
        self.next_fix_index: int | None = 1

        # Predictor parameters
        self.predictor_params: dict[str, typing.Any] = {}

    def set_wake_vortex_category(self):
        """
        Derived classes may use data tables to set wake vortex category if none is provided.
        Here we just set a default value of "M"
        """
        self.wake_vortex = "M"

    def default_pilot(self):
        """
        Create a Pilot instance.
        """
        self.pilot = Pilot(self.callsign)

    def ident(self, time: float, duration_seconds: float = 12):
        """
        Makes the aircraft's squawk ident.

        The aircraft's ident is stored as seconds from epoch until when the aircraft is identing.

        Parameters
        ----------
        time: float
            Time from epoch the aircraft begins to ident
        duration_seconds: float (optional)
            How long to ident for. Defaults to 12 seconds.

        """
        self.squawk_ident_until = time + duration_seconds

    def is_in_or_aproximately_aproaching_sector(self, sector: Sector, max_distance: int = 30, ds: int = 15) -> bool:
        """
        Checks if an aircraft is in a sector, or approaching the sector by checking if the point a specified distance
        ahead of the aircraft is in the sector.

        Note this function does not consider turn radius.

        Parameters
        ----------
        sector: Sector
            The sector to check if the aircraft is in or approaching.

        max_distance: int
            The distance to consider ahead of the aircraft. Defaults to 30.

        ds: int
            The step distances to step through. Defaults to 15.

        Returns
        -------
        bool

        """

        # The following generator is completely lazy, returning true for the first value that is true
        return any(
            sector.contains_laterally(self.pos2d().forward(distance, self.heading))
            for distance in range(0, max_distance + 1, ds)
        )

    @property
    def cleared_fl(self) -> float:
        """Cleared Flight Level of the Aircraft"""
        return self.cleared_instructions.fl

    @cleared_fl.setter
    def cleared_fl(self, fl: float):
        """Set the Cleared Flight Level of the Aircraft"""
        self.cleared_instructions.fl = fl

    @property
    def selected_fl(self) -> float:
        """Selected Flight Level of the Aircraft"""
        return self.selected_instructions.fl

    @selected_fl.setter
    def selected_fl(self, fl: float):
        """Set the Selected Flight Level of the Aircraft"""
        self.selected_instructions.fl = fl

    @property
    def on_route(self) -> bool:
        """Whether the aircraft is route following"""
        return self.selected_instructions.on_route

    @on_route.setter
    def on_route(self, on_route: bool):
        """Setter for whether the aircraft is route following"""
        self.selected_instructions.on_route = on_route

    def set_squawk(self, transponder_code: int | str):
        self.squawk = str(transponder_code)

    def set_position(self, lat: float, lon: float):
        """
        Set the Aircraft position.

        Parameters
        ----------
        lat: float
            The Aircraft latitude in degrees.
        lon: float
            The Aircraft longitude in degrees.
        """
        self.lat = lat
        self.lon = lon

    def set_attributes(self, attributes: dict[str, typing.Any]):
        """
        Set the 'fl', 'heading', and 'controllable' attributes of the Aircraft.

        Parameters
        ----------
        attributes: dict
            A dictionary of attributes to set.
        """
        allowed_attributes = {"fl", "heading", "controllable"}
        for key, value in attributes.items():
            if key in allowed_attributes:
                setattr(self, key, value)
            else:
                logger.warning(
                    f"Attribute {key} is not allowed to be set via set_attributes.",
                    stacklevel=2,
                )

    @classmethod
    def from_json(cls, s: str) -> typing_extensions.Self:
        """
        Construct a new instance from a string in JSON format.

        Parameters
        ----------
        s: str
            A string representation of Aircraft in a JSON/dictionary structure.

        Returns
        --------
        Aircraft

        Examples
        --------
        >>> Aircraft.from_json(''' PROBABLY NEEDS UPDATING
        >>>  {
        >>>    "flight_plan": {
        >>>        "entry": "160.0FL 00:00:00",
        >>>        "exit": "200.0FL 00:15:00",
        >>>        "route": {"current": ["ALFA", "BRAVO", "CHAR"], "filed": ["ALFA", "BRAVO", "CHAR"]},
        >>>        },
        >>>    "lat": 51.4702,
        >>>    "lon": -0.4479,
        >>>    "fl": 120.0,
        >>>    "heading": 249.68,
        >>>    "aircraft_type": None,
        >>>    "callsign": "ABC123",=======
        >>>    "cleared_instructions": {
        >>>        "fl": 120.0,
        >>>        "mach": null,
        >>>        "cas": null,
        >>>        "vertical_speed": 0.0,
        >>>        "heading": 249.68,
        >>>        "on_route": true,
        >>>        "speed_action": null,
        >>>        "vertical_speed_action": null,
        >>>        "vertical_action": null,
        >>>        "lateral_action": null
        >>>    },
        >>>    "selected_instructions": {
        >>>        "fl": 120.0,
        >>>        "mach": null,
        >>>        "cas": null,
        >>>        "vertical_speed": 0.0,
        >>>        "heading": 249.68,
        >>>        "on_route": true,
        >>>        "speed_action": null,
        >>>        "vertical_speed_action": null,
        >>>        "vertical_action": null,
        >>>        "lateral_action": null,
        >>>        "last_passed_filed_idx": null,
        >>>        "last_passed_current_idx": null
        >>>    },
        >>>    "coordinations": {  # NO LONGER CORRECT
        >>>             "25": {
        >>>                 "entry": {
        >>>                     "fl": 160,
        >>>                     "fix": "ABC",
        >>>                     "direction": "Up",
        >>>                     "next_sector": null,
        >>>                     "time": null,
        >>>                     "level_by": false,
        >>>                     "level_by_details": null,
        >>>                     "secondary_coord_conditions": null
        >>>                 }
        >>>                 "exit": {
        >>>                     "fl": 320,
        >>>                     "fix": "DEF",
        >>>                     "direction": "Horizontal",
        >>>                     "next_sector": null,
        >>>                     "time": null,
        >>>                     "level_by": false,
        >>>                     "level_by_details": null,
        >>>                     "secondary_coord_conditions": null
        >>>                 }
        >>>             }
        >>>         },
        >>>  }''')
        """

        data = json.loads(s)

        heading = data["heading"] % 360.0
        fl = data["fl"]

        aircraft = Aircraft(
            lat=data["lat"],
            lon=data["lon"],
            fl=fl,
            heading=heading,
            flight_plan=FlightPlan.from_json(json.dumps(data["flight_plan"])),
            callsign=data["callsign"],
            ufid=data.get("ufid", None),
            rate_of_turn=data.get("rate_of_turn", None),
            aircraft_type=data.get("aircraft_type", None),
            operation_params=data.get("operation_params", {}),
            controllable=data.get("controllable", True),
            simulated=data.get("simulated", True),
            current_sector=data.get("current_sector", None),
            random_seed=data.get("random_seed", None),
            pilot=Pilot.from_json(json.dumps(data["pilot"])) if "pilot" in data else Pilot(data["callsign"]),
            squawk=data.get("squawk", None),
            wake_vortex=data.get("wake_vortex", None),
            last_passed_filed_idx=data.get("last_passed_filed_idx", None),
            last_passed_current_idx=data.get("last_passed_current_idx", None),
            squawk_ident_until=data.get("squawk_ident_until", None),
        )

        # attributes
        aircraft.speed_tas = data.get("speed_tas", None)
        aircraft.ground_speed = data.get("ground_speed", None)
        aircraft.ground_track_angle = data.get("ground_track_angle", None)
        aircraft.heading_changing_to = data.get("heading_changing_to", None)
        aircraft.next_fix_index = data.get("next_fix_index", 1)
        aircraft.vertical_speed = data.get("vertical_speed", 0.0)

        # set protected attributes
        aircraft._previous_sector = data.get("previous_sector", None)

        if "cleared_instructions" in data:
            cleared_instructions = Instructions.from_json(json.dumps(data["cleared_instructions"]))
        else:
            cleared_instructions = Instructions()

        if "selected_instructions" in data:
            selected_instructions = Instructions.from_json(json.dumps(data["selected_instructions"]))
        else:
            selected_instructions = Instructions()

        # Set cleared  and selected fl and heading to current values if not provided
        if cleared_instructions.heading is None:
            cleared_instructions.heading = heading
        if cleared_instructions.fl is None:
            cleared_instructions.fl = fl
        if selected_instructions.heading is None:
            selected_instructions.heading = heading
        if selected_instructions.fl is None:
            selected_instructions.fl = fl

        aircraft.cleared_instructions = cleared_instructions
        aircraft.selected_instructions = selected_instructions

        aircraft.predictor_params = {}
        if "predictor_params" in data:
            for param_name, param_value in data["predictor_params"].items():
                aircraft.predictor_params[param_name] = param_value

        return aircraft

    @staticmethod
    def load(filename: str) -> Aircraft:
        """
        Construct a new instance from a file.

        Parameters
        ----------
        filename: str
            Path to a JSON file with an Aircraft definition in a dictionary format.

        Returns
        ----------
        Aircraft
        """

        with open(filename) as fd:
            return Aircraft.from_json(fd.read())

    def data(self) -> dict[str, typing.Any]:
        """
        Create a dictionary with key/value pairs representing the Aircraft data.

        Returns
        ----------
        data_dict: Dict[str, Any]
            Data dictionary of Aircraft data
        """
        data_dict: dict[str, typing.Any] = {
            "lat": self.lat,
            "lon": self.lon,
            "fl": self.fl,
            "heading": self.heading,
            "flight_plan": None if self.flight_plan is None else self.flight_plan.data(),
            "callsign": self.callsign,
            "ufid": self.ufid,
            "rate_of_turn": self.rate_of_turn,
            "aircraft_type": self.aircraft_type,
            "operation_params": self.operation_params,
            "controllable": self.controllable,
            "simulated": self.simulated,
            "current_sector": self.current_sector,
            "previous_sector": self._previous_sector,
            "percentile_rank_dict": self.percentile_rank_dict,
            "pilot": self.pilot.data(),
            "squawk": self.squawk,
            "squawk_ident_until": self.squawk_ident_until,
            "wake_vortex": self.wake_vortex,
            "speed_tas": self.speed_tas,
            "vertical_speed": self.vertical_speed,
            "ground_speed": self.ground_speed,
            "ground_track_angle": self.ground_track_angle,
            "random_seed": int(self.random_seed) if self.random_seed is not None else None,
            "heading_changing_to": self.heading_changing_to,
            "next_fix_index": self.next_fix_index,
            "cleared_instructions": self.cleared_instructions.data(),
            "selected_instructions": self.selected_instructions.data(),
            "last_passed_filed_idx": self.last_passed_filed_idx,
            "last_passed_current_idx": self.last_passed_current_idx,
        }

        data_dict["predictor_params"] = {}

        for param_name, param_value in self.predictor_params.items():
            if param_name in ["track_history"]:
                data_dict["predictor_params"][param_name] = [trajectory_pt.data() for trajectory_pt in param_value]

            elif hasattr(param_value, "data") and callable(param_value.data):
                data_dict["predictor_params"][param_name] = param_value.data()

            else:
                data_dict["predictor_params"][param_name] = param_value

        return data_dict

    def to_json(self) -> str:
        """
        Serialise the instance to JSON string.

        Returns
        -------
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

    def pos2d(self) -> Pos2D:
        """
        Get the two-dimensional position of the Aircraft.

        Returns
        ----------
        Pos2D
            The latitude, longitude of the Aircraft.
        """

        return Pos2D(self.lat, self.lon)

    def pos3d(self) -> Pos3D:
        """
        Get the three-dimensional position of the Aircraft.

        Returns
        ----------
        Pos3D
            The latitude, longitude, altitude (flight level) of the Aircraft.
        """

        return Pos3D(self.lat, self.lon, self.fl)

    def distance(self, other: Pos2D | Pos3D) -> float:
        """
        Calculate lateral distance [nmi] between Aircraft and another Pos3D location
        (vertical distance is ignored).

        Parameters
        ----------
        other: Pos3D
            A latitude, longitude, altitude position.

        Returns
        ----------
        float
            Lateral distance in nautical miles between Aircraft and the given 3D position.
        """

        return self.pos3d().distance(other)

    def distance_to_abeam(self, position: Pos2D, radius: float = 20.0) -> float | None:
        """
        Calculate distance [nmi] between Aircraft and the point on the ground track angle which is abeam position.

        The supplied position is normally a fix location.
        The term 'abeam' means the fix is perpendicular to the aircraft's ground track. So we get a right-angled
        triangle with the hypotenuse being the line from the aircraft to the fix, and the other two sides being the
        path of the aircraft to the abeam point and the line from the abeam point to the fix.
        If the abeam point is behind the aircraft, the distance returned is negative.
        If the current path of the aircraft doesn't pass within the supplied radius of the fix, None is returned.

        Parameters
        ----------
        position: Pos2D
            A latitude, longitude position, normally of a fix.
        radius: float, optional
            The maximum distance from the ground track to consider the point abeam the aircraft. Default is 20.0 nmi.

        Returns
        ----------
        float | None
            Distance in nautical miles between Aircraft and the point on the current ground track which is abeam the
            supplied position. If the point is behind the aircraft the distance is negative. If the path of the aircraft
            doesn't pass within the radius of the point, None is returned.
        """

        bearing_to_pos = self.pos2d().bearing_to(position)
        bearing = self.ground_track_angle  # so account for wind

        if bearing is None:
            logger.warning(
                f"{self.callsign}: distance_to_abeam cannot be calculated as ground_track_angle is not set.",
                stacklevel=2,
            )
            return None

        angle = abs(bearing - bearing_to_pos)
        angle = min(angle, 360.0 - angle)  # ensure angle is the smallest of the two angles between the bearings

        # Distance from the aircraft to the fix
        distance = self.pos2d().distance(position)
        # Distance from the abeam point to the fix
        distance_abeam = distance * math.sin(math.radians(angle))
        # Distance from the aircraft to the abeam point - will be negative if the point is behind the aircraft
        distance_to_abeam = distance * math.cos(math.radians(angle))

        if distance_abeam > radius:
            return None
        return distance_to_abeam

    @property
    def flight_state(self) -> FlightState:
        """
        Returns the aircraft flight state from its flight level and its selected flight level.

        Returns
        ----------
        flight_state: FlightState
            The aircraft flight state, either descend, climb or cruise
        """

        flight_level = self.fl
        selected_fl = self.selected_fl

        if flight_level == selected_fl:
            flight_state = FlightState.CRUISE

        elif flight_level > selected_fl:
            flight_state = FlightState.DESCEND

        else:  # flight_level < selected_fl:
            flight_state = FlightState.CLIMB

        return flight_state

    def randomise_performance(self, random_seed: int | None):
        """
        If the aircraft random seed is not set this method sets it, and
        defines its performance characteristics.

        Parameters
        ----------
        random_seed: int
            The random seed to use. If None, the aircraft performance characteristics
            will effectively default to default values
        """
        self.random_seed = random_seed
        if self.random_seed is not None:
            # set random seed of random generator
            random.seed(self.random_seed)

            # Select a lateral speed percentile rank, shared by the CAS and Mach speeds
            cas_percentile_rank = random.uniform(0, 100.0)
            for key in ["cas_des", "cas_cr", "cas_cl", "mach_des", "mach_cr", "mach_cl"]:
                self.percentile_rank_dict[key] = cas_percentile_rank

            # Select a ROCD percentile rank
            rocd_percentile_rank = random.uniform(0, 100.0)
            for key in ["rocd_des", "rocd_cl"]:
                self.percentile_rank_dict[key] = rocd_percentile_rank

        else:
            for key in ["cas_des", "cas_cr", "cas_cl", "mach_des", "mach_cr", "mach_cl", "rocd_des", "rocd_cl"]:
                self.percentile_rank_dict[key] = None

    def set_performance(self, cas_pr: float | None = None, rocd_pr: float | None = None):
        """
        A method to set the percentile rank performance of the aircraft

        Parameters
        ----------
        cas_pr: float, optional
            The aircraft's calibrated airspeed percentile rank (0 to 100)
        rocd_pr: float, optional
            The aircraft's rate of climb and descent percentile rank (0 to 100)
        """
        if cas_pr:
            if cas_pr < 0.0 or cas_pr > 100.0:
                raise ValueError("Trying to set aircraft performance using a invalid CAS percentile rank")
            for key in ["cas_des", "cas_cr", "cas_cl", "mach_des", "mach_cr", "mach_cl"]:
                self.percentile_rank_dict[key] = cas_pr
        else:
            for key in ["cas_des", "cas_cr", "cas_cl", "mach_des", "mach_cr", "mach_cl"]:
                self.percentile_rank_dict[key] = None

        if rocd_pr:
            if rocd_pr < 0.0 or rocd_pr > 100.0:
                raise ValueError("Trying to set aircraft performance using a invalid ROCD percentile rank")
            for key in ["rocd_des", "rocd_cl"]:
                self.percentile_rank_dict[key] = rocd_pr
        else:
            for key in ["rocd_des", "rocd_cl"]:
                self.percentile_rank_dict[key] = None

    @property
    def current_sector(self) -> str:
        """Current sector that the aircraft is controlled by"""
        return self._current_sector

    @current_sector.setter
    def current_sector(self, sector: str | None):
        """Setter for the current sector that the aircraft is controlled by"""
        # default no sector to "background" sector
        if sector is None:
            sector = "background"
        self._previous_sector = self._current_sector
        self._current_sector = sector

    @property
    def previous_sector(self) -> str:
        """The previous sector that the aircraft is controlled by"""
        return self._previous_sector

    @property
    def route_status(self) -> dict[str, str]:
        """Report the progression status for each fix along the flight plan route.

        The status of each fix is reported as:

        - passed: these fixes are considered passed, whether via a close approach or from being skipped.
        - next: when on-route one fix is considered next, and indicated by the trajectory prediction.
        - skipping: fixes that were in the filed route, but are planned to be omitted due to a route-direct.
        - to-come (""): future fixes in the current flight plan route, typically following the next fix.

        Note that fixes marked "passed" are typically followed by "next", sometimes with intermediate "skipping".
        However, due to interactions between the trajectory prediction and the aircraft's route-progression status,
        there may be "to-come" fixes before the "next" fix.

        The underlying indices should be updated by update_route_status in Airspace.

        Returns
        -------
        dict[str,str]
            Key-value pairs of each fix along a route with the corresponding status.
        """
        PASSED = "passed"
        NEXT = "next"
        TO_SKIP = "skipping"
        TO_COME = ""

        # if not flight plan, return empty dict
        if self.flight_plan is None:
            status = {}
        else:
            # calculate and return route progression
            filed_fixes = self.flight_plan.route.filed
            n_filed_passed = 0 if self.last_passed_filed_idx is None else self.last_passed_filed_idx + 1

            current_fixes = self.flight_plan.route.current
            n_current_passed = 0 if self.last_passed_current_idx is None else self.last_passed_current_idx + 1

            full_route_fixes = Aircraft._get_full_route(filed_fixes, current_fixes, n_filed_passed)

            # Entry to mark as NEXT in the current flight plan
            next_idx = self.next_fix_index

            # Mark all fixes as TO_SKIP as default, then overwrite for PASSED, NEXT and TO_COME.
            status = dict.fromkeys(full_route_fixes, TO_SKIP)
            for fix in filed_fixes[:n_filed_passed]:
                status[fix] = PASSED
            for fix in current_fixes[:n_current_passed]:
                status[fix] = PASSED
            if self.on_route:
                if next_idx is not None:
                    status[current_fixes[next_idx]] = NEXT
                    for fix in current_fixes[next_idx + 1 :]:
                        status[fix] = TO_COME
            else:
                # Not on route, next_idx not being updated
                for fix in current_fixes[n_current_passed:]:
                    status[fix] = TO_COME

        return status

    @staticmethod
    def _get_full_route(filed_fixes: list[str], current_fixes: list[str], n_filed_passed: int = 0) -> list[str]:
        """Create a route of fixes composed from both filed and current fixes.

        Route returned includes all passed fixes from filed_fixes (based on the
        index n_filed_passed) and the complete set of current fixes, without
        repeated fixes.

        Parameters
        ----------
        filed_fixes: list[str]
            Fixes from a filed route - any fixes not in the current route will
            be included in the result, otherwise any passed fixes.
        current_fixes: list[str]
            Fixes from the current route - all these fixes will be included in
            the result.
        n_filed_passed: int, default=0
            The index of filed_fixes marking items that will be included if
            there is no cross-over between filed and current fixes.
        Returns
        -------
        list[str]
            The current route, preceded by all skipped or passed fixes.
        """
        # Include n entries from filed route up to the first entry also in current_fixes
        route_intersection = set(filed_fixes).intersection(current_fixes)
        if route_intersection:
            full_items_from_filed = min(filed_fixes.index(f) for f in route_intersection)
        else:
            full_items_from_filed = n_filed_passed

        return filed_fixes[:full_items_from_filed] + current_fixes

    def update_route_status(
        self,
        airspace: Airspace,
        distance_threshold_NMI: float = 5.0,
        set_next_fix_index: bool = False,
    ):
        """Update the status of the route progression (route_status)

        Check the position of an aircraft relative to fixes on its filed and current flight plan Route, and update the
        respective passed-fix indices.

        Parameters
        ----------
        airspace: Airspace
            Airspace in which the Aircraft is flying - route updated based on the closest forward fix method.
        distance_threshold_NMI: float, default=5
            Consider Fix as having been passed if Aircraft is less than distance_threshold from it (NMI).
        set_next_fix_index: bool, default False
            Set the next_fix_index to be aligned with the calculated passed-fix indices. This is False by default
            as setting next_fix_index is usually performed by the predictor. However, at initialisation it can be
            convenient to set the next_fix_index manually here.

        Side-Effects
        ------------
        aircraft.last_passed_filed_idx
            Updates the index if the aircraft is within a threshold distance of a fix later in the filed route.
        aircraft.last_passed_current_idx
            Matches the filed index if the filed and the current are the same, otherwise updates with the same logic
            but for the progression along the current flight plan (updated by a route-direct Action).
        aircraft.route_status
            The main intended change by this function is the route_status, which uses the above passed indices.
        """
        if self.flight_plan is None:
            logger.debug(
                f"Cannot update route status without a flight plan for aircraft {self.callsign}.",
                stacklevel=2,
            )
            return

        filed = self.flight_plan.route.filed
        current = self.flight_plan.route.current
        last_filed_idx = self.last_passed_filed_idx
        last_current_idx = self.last_passed_current_idx

        # Find the next fix - all other fixes are "passed"
        next_filed_fix = airspace.closest_forward_fix(
            aircraft=self,
            distance_threshold_NMI=distance_threshold_NMI,
            route_fixes=filed,
        )
        if next_filed_fix is None:
            # Past last fix
            self.last_passed_filed_idx = len(filed)
        else:
            next_filed_idx = filed.index(next_filed_fix)
            if last_filed_idx is None:
                if next_filed_idx > 0:
                    self.last_passed_filed_idx = next_filed_idx - 1
            else:
                self.last_passed_filed_idx = max(last_filed_idx, next_filed_idx - 1)

        # As above, with current
        if filed == current:
            next_current_fix = next_filed_fix
        else:
            next_current_fix = airspace.closest_forward_fix(
                aircraft=self,
                distance_threshold_NMI=distance_threshold_NMI,
                route_fixes=current,
            )

        if next_current_fix is None:
            # Past last fix
            self.last_passed_current_idx = len(current) - 1
        else:
            next_current_idx = current.index(next_current_fix)
            if last_current_idx is None:
                if next_current_idx > 0:
                    self.last_passed_current_idx = next_current_idx - 1
            else:
                self.last_passed_current_idx = max(last_current_idx, next_current_idx - 1)

        if set_next_fix_index:
            # For update next_fix_index and on_route using the route status
            if self.last_passed_current_idx is None:
                # no fixes passed yet
                self.next_fix_index = 0

            elif self.last_passed_current_idx == len(self.flight_plan.route.current) - 1:
                # last fix has been passed
                self.next_fix_index = None
                self.on_route = False

            else:
                # next_fix_index is one greater than the last passed index
                self.next_fix_index = self.last_passed_current_idx + 1
