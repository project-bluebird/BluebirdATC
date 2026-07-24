from __future__ import annotations

import json
import operator
import typing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import typing_extensions
from pydantic import BaseModel
from typing_extensions import Self

from bluebird_dt.core import (
    Action,
    Aircraft,
    ClearanceAndResponse,
    Coordination,
    Environment,
    FlightPlan,
    Instructions,
    Route,
    WindField,
)
from bluebird_dt.events.event_dtypes import EventDtypes
from bluebird_dt.logger import logger
from bluebird_dt.utility.convert import tas_from_ground_speed

if typing.TYPE_CHECKING:
    from bluebird_dt.events import EventLogger
    from bluebird_dt.manager import EnvironmentManager

pd.set_option("future.no_silent_downcasting", True)


def pd_concat_two_dfs(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Combine both dataframes
    """
    if df2.empty:
        return df1
    if df1.empty:
        return df2
    return pd.concat([df1, df2])


TAircraft = typing_extensions.TypeVar("TAircraft", bound=Aircraft, default=Aircraft)
TWindField = typing_extensions.TypeVar("TWindField", bound=WindField, default=WindField)
TForecastWindField = typing_extensions.TypeVar("TForecastWindField", bound=WindField, default=WindField)


class FlightPlanEvent(BaseModel):
    callsign: str
    datetime: datetime
    route_filed: list[str] | None = None
    start_datetime: datetime | None = None
    end_datetime: datetime | None = None
    squawk: int | None = None
    origin: str | None = None
    dest: str | None = None
    unexpanded_route: str | None = None
    sector_crossing_seq: str | None = None
    actype: str | None = None
    milcivil: str | None = None
    requested_flight_level: float | int | None = None
    filed_true_airspeed: float | int | None = None
    intention_code: str | None = None
    ufid: str | None = None
    assigned_squawk: int | None = None


class CoordinationEvent(BaseModel):
    datetime: datetime
    callsign: str
    from_sector: str
    to_sector: str
    fl: float
    fix: str
    direction: str
    level_by: bool | None = None
    level_by_details: str | None = None
    secondary_coord_conditions: str | None = None


class IncommEvent(BaseModel):
    datetime: datetime
    callsign: str
    sector_name: str


class ClearanceEvent(BaseModel):
    datetime: datetime
    callsign: str
    kind: str
    value: str
    agent: str | None = None
    sector: list[str] | None = None


class EventHandler(typing.Generic[TAircraft]):
    """
    Events to apply to the Environment.
    """

    typeof_aircraft: type[TAircraft]
    radar_df: pd.DataFrame
    flight_df: pd.DataFrame
    clearances_df: pd.DataFrame
    coordination_df: pd.DataFrame
    sectors_df: pd.DataFrame
    incomm_df: pd.DataFrame
    aircraft_internals_df: pd.DataFrame
    ac_attribute_update_df: pd.DataFrame

    @dataclass
    class IgnoreFlags:
        """
        radar_if_simmed: bool, Default True
            If True, radar events will be ignored for simmed aircraft
        flight_if_simmed: bool, Default True
            If True, flight plan events will be ignored for simmed aircraft
        clearance_if_simmed: bool, Default True
            If True, clearance events will be ignored for simmed aircraft
        coordination_if_simmed: bool, Default True
            If True, coordination events will be ignored for simmed aircraft
        sectors_if_simmed: bool, Default True
            If True, sector configuration events will be ignored for simmed aircraft
        incomm_if_simmed: bool, Default True
            If True, incomm events will be ignored for simmed aircraft
        aircraft_internals_if_simmed: bool, Default True
            If True, aircraft internal events will be ignored for simmed aircraft
        ac_attribute_if_simmed: bool, Default True
            If True aircraft attribute update events will be ignored for simmed aircraft
        airspace_config_updates: bool, Default False
            The airspace configuration will be ignored at each timestep. Updating only when 'jump_to_time' is used.
            Set this flag to True to speed up runtime when the airspace configuration never changes.
        """

        radar_if_simmed: bool = True
        flight_if_simmed: bool = True
        clearance_if_simmed: bool = True
        coordination_if_simmed: bool = True
        sectors_if_simmed: bool = True
        incomm_if_simmed: bool = True
        aircraft_internals_if_simmed: bool = True
        ac_attribute_if_simmed: bool = True
        airspace_config_updates: bool = False

    ignore: IgnoreFlags

    def __init__(
        self,
        radar_dataframe: pd.DataFrame | None = None,
        flight_df: pd.DataFrame | None = None,
        clearances_df: pd.DataFrame | None = None,
        coordination_df: pd.DataFrame | None = None,
        sectors_df: pd.DataFrame | None = None,
        incomm_df: pd.DataFrame | None = None,
        aircraft_internals_df: pd.DataFrame | None = None,
        ac_attribute_update_df: pd.DataFrame | None = None,
        ignore: IgnoreFlags | None = None,
        typeof_aircraft: type[TAircraft] = Aircraft,
    ):
        """
        Construct a new EventHandler.

        Parameters
        ----------
        radar_dataframe: pd.DataFrame, optional
            Each row is a radar event
        flight_df: pd.DataFrame, optional
            Each row is a flight plan event
        clearances_df: pd.DataFrame, optional
            Each row is a clearance event
        coordination_df: pd.DataFrame, optional
            Each row is a coordination event
        sectors_df: pd.DataFrame, optional
            Each row is a sector configuration event
        incomm_df: pd.DataFrame, optional
            Each row is an incomm event
        aircraft_internals_df: pd.DataFrame, optional
            Each row is a aircraft internals event
            These update the internal parameters of the aircraft
        ac_attribute_update_df: pd.DataFrame, optional
            Each row is an aircraft attribute update event
            These event update any named attribute of an aircraft
        ignore: IgnoreFlags object, optional
            Object containing all the ignore flags for the event handler.
            If None, default values will be used
        typeof_aircraft: type[TAircraft]
            Type of aircraft to initialise from event logs.
            Defaults to bluebird_dt.core.Aircraft

        Type Variables
        --------------
        Dependent packages can implement expanded representations of the following classes.

        TAircraft: bluebird_dt.core.Aircraft
        """

        self.typeof_aircraft = typeof_aircraft
        # Define the types that will be created and modified by the event handler.
        # This way we can have code in the base class create instances of types
        # from derived classes.
        # create empty dataframes if dataframe not present
        self.radar_df = (
            radar_dataframe if radar_dataframe is not None else pd.DataFrame(columns=EventDtypes.radar_dtypes)
        )

        self.flight_df = flight_df if flight_df is not None else pd.DataFrame(columns=EventDtypes.flight_dtypes)

        self.clearances_df = (
            clearances_df if clearances_df is not None else pd.DataFrame(columns=EventDtypes.clearance_dtypes)
        )

        self.coordination_df = (
            coordination_df if coordination_df is not None else pd.DataFrame(columns=EventDtypes.coord_dtypes)
        )

        self.sectors_df = sectors_df if sectors_df is not None else pd.DataFrame(columns=EventDtypes.sectors_dtypes)

        self.incomm_df = incomm_df if incomm_df is not None else pd.DataFrame(columns=EventDtypes.incomm_dtypes)

        self.aircraft_internals_df = (
            aircraft_internals_df
            if aircraft_internals_df is not None
            else pd.DataFrame(columns=EventDtypes.aircraft_internals_dtypes)
        )

        self.ac_attribute_update_df = (
            ac_attribute_update_df
            if ac_attribute_update_df is not None
            else pd.DataFrame(columns=EventDtypes.ac_attribute_update_dtypes)
        )

        # ensure dataframes are in correct format
        # avoids subtle bugs later on when new scenario generators are created
        self.add_optional_radar_columns_if_required()
        self.add_optional_clearance_columns_if_required()
        self.ensure_dataframe_data_types()
        self.set_dataframe_indices_to_datetime()
        self.ensure_dataframes_are_date_ordered()
        self.fillna_for_specific_columns()
        self.ensure_specific_format_for_specific_columns()

        self.ignore = EventHandler.IgnoreFlags() if ignore is None else ignore

    def reset_events(self) -> None:
        """Reset events by creating empty DataFrames for each event type"""
        self.radar_df = pd.DataFrame(columns=EventDtypes.radar_dtypes)
        self.flight_df = pd.DataFrame(columns=EventDtypes.flight_dtypes)
        self.clearances_df = pd.DataFrame(columns=EventDtypes.clearance_dtypes)
        self.coordination_df = pd.DataFrame(columns=EventDtypes.coord_dtypes)
        self.sectors_df = pd.DataFrame(columns=EventDtypes.sectors_dtypes)
        self.incomm_df = pd.DataFrame(columns=EventDtypes.incomm_dtypes)
        self.aircraft_internals_df = pd.DataFrame(columns=EventDtypes.aircraft_internals_dtypes)
        self.ac_attribute_update_df = pd.DataFrame(columns=EventDtypes.ac_attribute_update_dtypes)

    def add_optional_radar_columns_if_required(self) -> None:
        """Add optional radar columns if not already present"""
        columns = self.radar_df.columns

        optional_float_cols = [
            "speed_tas",
            "ground_speed",
            "ground_track_angle",
            "selected_fl",
        ]

        # add columns if not in radar dataframe
        for col in optional_float_cols:
            if col not in columns:
                self.radar_df[col] = np.nan

        # add a ufid column if not present
        if "ufid" not in columns:
            self.radar_df["ufid"] = ""

    def add_optional_clearance_columns_if_required(self) -> None:
        """Add optional clearance columns if not already present"""
        optional_columns = [
            "text_clearance",
            "text_pilot_response",
            "voice_clearance",
            "voice_pilot_response",
        ]

        # add columns if not in radar dataframe
        for col in optional_columns:
            if col not in self.clearances_df.columns:
                self.clearances_df[col] = None

    def set_dataframe_indices_to_datetime(self) -> None:
        """Set all event DataFrame indices to a datetime column"""
        self.radar_df = self.radar_df.set_index("datetime")
        self.flight_df = self.flight_df.set_index("datetime")
        self.clearances_df = self.clearances_df.set_index("datetime")
        self.coordination_df = self.coordination_df.set_index("datetime")
        self.sectors_df = self.sectors_df.set_index("datetime")
        self.incomm_df = self.incomm_df.set_index("datetime")
        self.aircraft_internals_df = self.aircraft_internals_df.set_index("datetime")
        self.ac_attribute_update_df = self.ac_attribute_update_df.set_index("datetime")

    def ensure_specific_format_for_specific_columns(self) -> None:
        """
        Ensure sector numbers are padded,
        e.g. sector "7" is "07"
        """
        self.coordination_df["to_sector"] = self.coordination_df["to_sector"].str.zfill(2)
        self.coordination_df["from_sector"] = self.coordination_df["from_sector"].str.zfill(2)

    def fillna_for_specific_columns(self) -> None:
        """Replace NA with empty strings for specific columns."""
        self.flight_df["sector_crossing_seq"] = self.flight_df["sector_crossing_seq"].fillna("")
        self.coordination_df["level_by_details"] = self.coordination_df["level_by_details"].fillna("")
        self.coordination_df["secondary_coord_conditions"] = self.coordination_df["secondary_coord_conditions"].fillna(
            ""
        )

    def ensure_dataframe_data_types(self) -> None:
        """Convert DataFrame columns to correct data types if required."""
        self.radar_df = self.radar_df.astype(EventDtypes.radar_dtypes)
        self.flight_df = self.flight_df.astype(EventDtypes.flight_dtypes)
        self.clearances_df = self.clearances_df.astype(EventDtypes.clearance_dtypes)
        self.coordination_df = self.coordination_df.astype(EventDtypes.coord_dtypes)
        self.sectors_df = self.sectors_df.astype(EventDtypes.sectors_dtypes)
        self.incomm_df = self.incomm_df.astype(EventDtypes.incomm_dtypes)
        self.aircraft_internals_df = self.aircraft_internals_df.astype(EventDtypes.aircraft_internals_dtypes)
        self.ac_attribute_update_df = self.ac_attribute_update_df.astype(EventDtypes.ac_attribute_update_dtypes)

    def ensure_dataframes_are_date_ordered(self) -> None:
        """Sort each dataframe by date order."""
        if not self.radar_df.index.is_monotonic_increasing:
            self.radar_df = self.radar_df.sort_index()
        if not self.flight_df.index.is_monotonic_increasing:
            self.flight_df = self.flight_df.sort_index()
        if not self.clearances_df.index.is_monotonic_increasing:
            self.clearances_df = self.clearances_df.sort_index()
        if not self.coordination_df.index.is_monotonic_increasing:
            self.coordination_df = self.coordination_df.sort_index()
        if not self.sectors_df.index.is_monotonic_increasing:
            self.sectors_df = self.sectors_df.sort_index()
        if not self.incomm_df.index.is_monotonic_increasing:
            self.incomm_df = self.incomm_df.sort_index()
        if not self.aircraft_internals_df.index.is_monotonic_increasing:
            self.aircraft_internals_df = self.aircraft_internals_df.sort_index()
        if not self.ac_attribute_update_df.index.is_monotonic_increasing:
            self.ac_attribute_update_df = self.ac_attribute_update_df.sort_index()

    def add_aircraft_attribute_update_event(
        self,
        the_datetime: datetime,
        callsign: str,
        attribute_name: str,
        value: None | float | int | str | list,
    ) -> None:
        """
        Add an aircraft_attribute_update event.

        Parameters
        ----------
        the_datetime: datetime
            The datetime at with the attribute will be updated
        callsign: str
            Callsign of aircraft that will have its attribute updated
        attribute_name: str
            Name of the attribute that will be updated
        value: None or float or int or str or list
            New value of the updated attribute
        """
        new_row = {
            "callsign": callsign,
            "attribute_name": attribute_name,
            "value": value,
        }

        new_row = pd.Series(new_row, name=the_datetime)
        self.ac_attribute_update_df = pd_concat_two_dfs(self.ac_attribute_update_df, new_row.to_frame().T)

        # ensure date ordering is preserved
        self.ac_attribute_update_df = self.ac_attribute_update_df.sort_index()

    def add_aircraft(self, the_datetime: datetime, aircraft: Aircraft):
        """
        Create appropriate events to add an aircraft object to the environment at a specific data and time.

        Parameters
        ----------
        the_datetime: datetime
            The datetime at with the aircraft will be added to the environment
        aircraft: Aircraft
            The aircraft that will be added to the environment
        """
        # split aircraft across radar, flight and aircraft internal events
        self.add_radar_event(
            the_datetime=the_datetime,
            callsign=aircraft.callsign,
            lat=aircraft.lat,
            lon=aircraft.lon,
            fl=aircraft.fl,
            heading=aircraft.heading,
            ufid=aircraft.ufid,
            speed_tas=aircraft.speed_tas,
            ground_speed=aircraft.ground_speed,
            ground_track_angle=aircraft.ground_track_angle,
            selected_fl=aircraft.selected_fl,
        )

        flight_plan = aircraft.flight_plan

        if flight_plan is not None:
            self.add_flight_plan_event(
                the_datetime=the_datetime,
                callsign=aircraft.callsign,
                route_filed=flight_plan.route.filed,
                start_datetime=flight_plan.start_datetime,
                end_datetime=flight_plan.end_datetime,
                squawk=aircraft.squawk,
                origin=flight_plan.origin,
                dest=flight_plan.dest,
                unexpanded_route=flight_plan.unexpanded_route,
                sector_crossing_seq=flight_plan.sector_crossing_seq,
                actype=aircraft.aircraft_type,
                milcivil=flight_plan.milcivil,
                requested_flight_level=flight_plan.requested_flight_level,
                filed_true_airspeed=flight_plan.filed_true_airspeed,
                intention_code=flight_plan.intention_code,
                ufid=aircraft.ufid,
                assigned_squawk=flight_plan.assigned_squawk,
            )

        if aircraft.cleared_instructions is None:
            cleared_speed_action = None
            cleared_vertical_speed_action = None
            cleared_vertical_action = None
            cleared_lateral_action = None
        else:
            cleared_speed_action = (
                aircraft.cleared_instructions.speed_action.to_json()
                if aircraft.cleared_instructions.speed_action is not None
                else None
            )
            cleared_vertical_speed_action = (
                aircraft.cleared_instructions.vertical_speed_action.to_json()
                if aircraft.cleared_instructions.vertical_speed_action is not None
                else None
            )
            cleared_vertical_action = (
                aircraft.cleared_instructions.vertical_action.to_json()
                if aircraft.cleared_instructions.vertical_action is not None
                else None
            )
            cleared_lateral_action = (
                aircraft.cleared_instructions.lateral_action.to_json()
                if aircraft.cleared_instructions.lateral_action is not None
                else None
            )

        if aircraft.selected_instructions is None:
            selected_speed_action = None
            selected_vertical_speed_action = None
            selected_vertical_action = None
            selected_lateral_action = None
        else:
            selected_speed_action = (
                aircraft.selected_instructions.speed_action.to_json()
                if aircraft.selected_instructions.speed_action is not None
                else None
            )
            selected_vertical_speed_action = (
                aircraft.selected_instructions.vertical_speed_action.to_json()
                if aircraft.selected_instructions.vertical_speed_action is not None
                else None
            )
            selected_vertical_action = (
                aircraft.selected_instructions.vertical_action.to_json()
                if aircraft.selected_instructions.vertical_action is not None
                else None
            )
            selected_lateral_action = (
                aircraft.selected_instructions.lateral_action.to_json()
                if aircraft.selected_instructions.lateral_action is not None
                else None
            )

        self.add_aircraft_internals_event(
            the_datetime,
            callsign=aircraft.callsign,
            rate_of_turn=aircraft.rate_of_turn,
            operation_params=aircraft.operation_params,
            controllable=aircraft.controllable,
            simulated=aircraft.simulated,
            current_sector=aircraft.current_sector,
            previous_sector=aircraft.previous_sector,
            percentile_rank_dict=aircraft.percentile_rank_dict,
            pilot_type=aircraft.pilot.__class__.__name__,
            pilot_action_queue=[json.dumps(queue_item) for queue_item in aircraft.pilot.action_queue],
            predictor_params=aircraft.predictor_params,
            wake_vortex=aircraft.wake_vortex,
            random_seed=aircraft.random_seed,
            heading_changing_to=aircraft.heading_changing_to,
            next_fix_index=aircraft.next_fix_index,
            cleared_fl=aircraft.cleared_instructions.fl,
            cleared_mach=aircraft.cleared_instructions.mach,
            cleared_cas=aircraft.cleared_instructions.cas,
            cleared_vertical_speed=aircraft.cleared_instructions.vertical_speed,
            cleared_heading=aircraft.cleared_instructions.heading,
            cleared_on_route=aircraft.cleared_instructions.on_route,
            cleared_speed_action=cleared_speed_action,
            cleared_vertical_speed_action=cleared_vertical_speed_action,
            cleared_vertical_action=cleared_vertical_action,
            cleared_lateral_action=cleared_lateral_action,
            vertical_speed=aircraft.vertical_speed,
            selected_fl=aircraft.selected_instructions.fl,
            selected_mach=aircraft.selected_instructions.mach,
            selected_cas=aircraft.selected_instructions.cas,
            selected_vertical_speed=aircraft.selected_instructions.vertical_speed,
            selected_heading=aircraft.selected_instructions.heading,
            selected_on_route=aircraft.selected_instructions.on_route,
            selected_speed_action=selected_speed_action,
            selected_vertical_speed_action=selected_vertical_speed_action,
            selected_vertical_action=selected_vertical_action,
            selected_lateral_action=selected_lateral_action,
            route_current=None if flight_plan is None else flight_plan.route.current,  # not part of real flight plan
            last_passed_filed_idx=aircraft.last_passed_filed_idx,
            last_passed_current_idx=aircraft.last_passed_current_idx,
            squawk_ident_until=aircraft.squawk_ident_until,
        )

    def add_radar_event(
        self,
        the_datetime: datetime,
        callsign: str,
        lat: float,
        lon: float,
        fl: float,
        heading: float,
        ufid: str | None = None,
        speed_tas: float | None = None,
        ground_speed: float | None = None,
        ground_track_angle: float | None = None,
        selected_fl: float | int | None = None,
    ):
        """
        Add a radar event to the EventHandler.

        Parameters
        ----------
        the_datetime: datetime
            The datetime of the event
        callsign: str
            Callsign of aircraft
        lat: float
            Latitude of aircraft
        lon: float
            Longitude of aircraft
        fl: float
            Flight Level of aircraft
        heading: float
            Heading of aircraft
        ufid: str, optional
            Unique Flight ID of aircraft
        speed_tas: float, optional
            True Airspeed of the aircraft
        ground_speed: float, optional
            Ground Speed of the aircraft
        ground_track_angle: float, optional
            Angle from North of ground path of aircraft
        selected_fl: float or int, optional
            Flight level selected on aircraft flight computer
        """
        new_row = {
            "callsign": callsign,
            "lat": lat,
            "lon": lon,
            "fl": fl,
            "heading": heading,
            "ufid": ufid,
            "speed_tas": speed_tas,
            "ground_speed": ground_speed,
            "ground_track_angle": ground_track_angle,
            "selected_fl": selected_fl,
        }

        new_row = pd.Series(new_row, name=the_datetime)
        dtypes = {key: val for key, val in EventDtypes.radar_dtypes.items() if key != "datetime"}
        self.radar_df = pd_concat_two_dfs(self.radar_df, new_row.to_frame().T.astype(dtypes))

        # ensure date ordering is preserved
        self.radar_df = self.radar_df.sort_index()

    def extend_flight_plan_events(self, events: list[FlightPlanEvent]):
        new_flights = pd.DataFrame([event.model_dump() for event in events])

        new_flights = new_flights.set_index("datetime")
        new_flights.index.name = "datetime"

        dtypes = {key: val for key, val in EventDtypes.flight_dtypes.items() if key != "datetime"}
        self.flight_df = pd_concat_two_dfs(self.flight_df, new_flights.astype(dtypes))

        # ensure date ordering is preserved
        self.flight_df = self.flight_df.sort_index()

    def add_flight_plan_event(
        self,
        the_datetime: datetime,
        callsign: str,
        route_filed: list[str],
        start_datetime: datetime | None = None,
        end_datetime: datetime | None = None,
        squawk: str | None = None,
        origin: str | None = None,
        dest: str | None = None,
        unexpanded_route: str | None = None,
        sector_crossing_seq: str | None = None,
        actype: str | None = None,
        milcivil: str | None = None,
        requested_flight_level: float | int | None = None,
        filed_true_airspeed: float | int | None = None,
        intention_code: str | None = None,
        ufid: str | None = None,
        assigned_squawk: str | None = None,
    ):
        """
        Add a flight plan event.

        Parameters
        ----------
        the_datetime: datetime
            The datetime of the flight plan event
        callsign: str
            Callsign of aircraft
        route_filed: list[str]
            Filed route of aircraft
        start_datetime: datetime, optional
            Datetime after which the flight plan is valid. If None then always valid.
        end_datetime: datetime, optional
            Datetime after which the flight plan is no longer valid. If None then always valid.
        squawk: str, optional
            Squawk/ssr code of aircraft
        origin: str, optional
            Origin airport
        dest: str, optional
            Destination airport
        unexpanded_route: str, optional
            Route as original string including airways rather than only individual fixes
        sector_crossing_seq: str, optional
            Sector crossing sequence. Planned sequence to cross sectors
        actype: str, optional
            Aircraft type
        milcivil: {'M' or 'C'}, optional
            Whether the aircraft is military ("M") or civilian ("C")
        requested_flight_level: float or int or None, default None
            Flight planned flight level of the aircraft
        filed_true_airspeed: float or int or None, default None
            Flight planned true airspeed of the aircraft
        intention_code: str, optional
            Intention code, denoting where the aircraft plans to leave the FIR
        ufid: str, optional
            Universal flight id of the aircraft
        """

        new_row = {
            "callsign": callsign,
            "datetime": the_datetime,
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "squawk": squawk,
            "origin": origin,
            "dest": dest,
            "unexpanded_route": unexpanded_route,
            "route_filed": route_filed,
            "sector_crossing_seq": "" if pd.isna(sector_crossing_seq) else sector_crossing_seq,
            "actype": actype,
            "milcivil": milcivil,
            "requested_flight_level": requested_flight_level,
            "filed_true_airspeed": filed_true_airspeed,
            "intention_code": intention_code,
            "ufid": ufid,
            "assigned_squawk": assigned_squawk,
        }

        new_row = pd.Series(new_row, name=the_datetime)
        self.flight_df = pd_concat_two_dfs(self.flight_df, new_row.to_frame().T.astype(EventDtypes.flight_dtypes))
        # ensure date ordering is preserved
        self.flight_df = self.flight_df.sort_index()

    def extend_clearance_events(self, clearance_events: list[ClearanceEvent]):
        new_clearances = pd.DataFrame([event.model_dump() for event in clearance_events])
        new_clearances["text_clearance"] = [None] * new_clearances.shape[0]
        new_clearances["text_pilot_response"] = [None] * new_clearances.shape[0]
        new_clearances["voice_clearance"] = [None] * new_clearances.shape[0]
        new_clearances["voice_pilot_response"] = [None] * new_clearances.shape[0]

        new_clearances = new_clearances.set_index("datetime")
        new_clearances.index.name = "datetime"

        dtypes = {key: val for key, val in EventDtypes.clearance_dtypes.items() if key != "datetime"}
        self.clearances_df = pd_concat_two_dfs(self.clearances_df, new_clearances.astype(dtypes))

        self.clearances_df = self.clearances_df.sort_index()

    def add_clearance_event(
        self,
        the_datetime: datetime,
        callsign: str,
        kind: str,
        value: str,
        agent: str | None = None,
        sector: list[str] | None = None,
    ):
        """
        Add a clearance event.

        Parameters
        ----------
        the_datetime: datetime
            The datetime of the event
        callsign: str
            Callsign of aircraft which will receive the clearance
        kind: str
            Instructional part of clearance, e.g. "change_heading_to"
        value: str
            Value to go with instruction, e.g. 240
            String may contain ints, float, lists or dicts, but the value should be a string
        agent: str, {"human", "agent"}, optional
            Optional string denoting origin of clearance (human or AI agent)
        """
        new_row = {
            "callsign": callsign,
            "kind": kind,
            "value": value,
            "agent": agent,
            "text_clearance": None,  # TODO: update with clearance_and_response function
            "text_pilot_response": None,  # TODO: update with clearance_and_response function
            "voice_clearance": None,
            "voice_pilot_response": None,
            "sector": sector,
        }

        new_row = pd.Series(new_row, name=the_datetime)
        dtypes = {key: val for key, val in EventDtypes.clearance_dtypes.items() if key != "datetime"}
        self.clearances_df = pd_concat_two_dfs(self.clearances_df, new_row.to_frame().T.astype(dtypes))
        # ensure date ordering is preserved
        self.clearances_df = self.clearances_df.sort_index()

    def add_sectors_event(
        self,
        the_datetime: datetime,
        sectors_configuration: list[tuple[str, list[str]]],
    ):
        """
        Add a sectors event.

        Parameters
        ----------
        the_datetime: datetime
            Datetime of event
        sectors_configuration: list[tuple[str, list[str]]]
            List of the bandboxed name followed by the individual sectors in that bandboxed sector
        """
        new_row = {
            "sectors_configuration": sectors_configuration,
        }

        new_row = pd.Series(new_row, name=the_datetime)
        dtypes = {key: val for key, val in EventDtypes.sectors_dtypes.items() if key != "datetime"}
        self.sectors_df = pd_concat_two_dfs(self.sectors_df, new_row.to_frame().T.astype(dtypes))
        # ensure date ordering is preserved
        self.sectors_df = self.sectors_df.sort_index()

    def extend_incomm_events(
        self,
        incomm_events: list[IncommEvent],
    ):
        new_incomm_events = pd.DataFrame([incomm_event.model_dump() for incomm_event in incomm_events])
        new_incomm_events = new_incomm_events.set_index("datetime")
        new_incomm_events.index.name = "datetime"

        dtypes = {key: val for key, val in EventDtypes.incomm_dtypes.items() if key != "datetime"}
        self.incomm_df = pd_concat_two_dfs(self.incomm_df, new_incomm_events.astype(dtypes))

        self.incomm_df = self.incomm_df.sort_index()

    def add_incomm_event(
        self,
        the_datetime: datetime,
        callsign: str,
        sector_name: str | None = None,
    ):
        """
        Add an incomm event.

        Parameters
        ----------
        the_datetime: datetime
            Datetime of the event
        callsign: str
            Callsign of the aircraft which is incommed
        sector_name: str, optional
            Optionally give the sector to which the aircraft should be incommed.
            If None, the aircraft will be incommed according to the coordinations.
        """
        new_row = {
            "callsign": callsign,
            "sector_name": sector_name,
        }

        new_row = pd.Series(new_row, name=the_datetime)
        dtypes = {key: val for key, val in EventDtypes.incomm_dtypes.items() if key != "datetime"}
        self.incomm_df = pd_concat_two_dfs(self.incomm_df, new_row.to_frame().T.astype(dtypes))
        # ensure date ordering is preserved
        self.incomm_df = self.incomm_df.sort_index()

    def extend_coordination_events(self, events: list[CoordinationEvent]):
        if len(events) == 0:
            return

        new_coordinations = pd.DataFrame([event.model_dump() for event in events])
        new_coordinations = new_coordinations.set_index("datetime")
        new_coordinations.index.name = "datetime"

        dtypes = {key: val for key, val in EventDtypes.coord_dtypes.items() if key != "datetime"}

        self.coordination_df = pd_concat_two_dfs(self.coordination_df, new_coordinations.astype(dtypes))

        self.coordination_df = self.coordination_df.sort_index()

    def add_coordination_event(
        self,
        the_datetime: datetime,
        callsign: str,
        from_sector: str,
        to_sector: str,
        fl: int | float,
        fix: str,
        direction: str,
        level_by: bool | None = None,
        level_by_details: str | None = None,
        secondary_coord_conditions: str | None = None,
    ):
        """
        Add a coordination event.

        Parameters
        ----------
        the_datetime: datetime
            Datetime of the event
        callsign: str
            Callsign of coordinated aircraft
        from_sector: str
            Sector coordinated from
        to_sector: str
            Sector coordinated to
        fl: int or float
            Flight level of coordination
        fix: str
            Fix denoting location of coordination
        direction: str, {"Vertical", "Horizontal"}
            Direction of coordination
        level_by: bool, optional
            True if the coordination is a level_by coordination
        level_by_details: str, optional
            Details of level by coordination. Should only be non-None if level_by is True
        secondary_coord_conditions: str, optional
            Secondary Coordination Conditions details
        """
        new_row = {
            "callsign": callsign,
            "from_sector": from_sector,
            "to_sector": to_sector,
            "fl": fl,
            "fix": fix,
            "direction": direction,
            "level_by": level_by,
            "level_by_details": "" if pd.isna(level_by_details) else level_by_details,
            "secondary_coord_conditions": "" if pd.isna(secondary_coord_conditions) else secondary_coord_conditions,
        }

        new_row = pd.Series(new_row, name=the_datetime)
        dtypes = {key: val for key, val in EventDtypes.coord_dtypes.items() if key != "datetime"}
        self.coordination_df = pd_concat_two_dfs(self.coordination_df, new_row.to_frame().T.astype(dtypes))
        # ensure date ordering is preserved
        self.coordination_df = self.coordination_df.sort_index()

    def add_coordination(self, the_datetime: datetime, coordination: Coordination):
        """
        Create a Coordination event from a coordination

        Parameters
        ----------
        the_datetime: datetime
            Datetime of the event
        coordination:
            Coordination to be added
        """
        self.add_coordination_event(
            the_datetime,
            coordination.callsign,
            coordination.from_sector,
            coordination.to_sector,
            coordination.fl,
            coordination.fix,
            coordination.direction,
            level_by=coordination.level_by,
            level_by_details=coordination.level_by_details,
            secondary_coord_conditions=coordination.secondary_coord_conditions,
        )

    def add_aircraft_internals_event(
        self,
        the_datetime: datetime,
        callsign: str | None = None,
        rate_of_turn: float | None = None,
        operation_params: dict | None = None,
        controllable: bool | None = None,
        simulated: bool | None = None,
        current_sector: str | None = None,
        previous_sector: str | None = None,
        percentile_rank_dict: dict | None = None,
        pilot_type: str | None = None,
        pilot_action_queue: list[str] | None = None,
        predictor_params: dict[str, float] | None = None,
        wake_vortex: str | None = None,
        random_seed: int | None = None,
        heading_changing_to: int | float | None = None,
        next_fix_index: int | None = None,
        vertical_speed: float | None = None,
        cleared_fl: int | float | None = None,
        cleared_mach: float | None = None,
        cleared_cas: int | float | None = None,
        cleared_vertical_speed: int | float | None = None,
        cleared_heading: int | float | None = None,
        cleared_on_route: bool | None = None,
        cleared_speed_action: str | None = None,
        cleared_vertical_speed_action: str | None = None,
        cleared_vertical_action: str | None = None,
        cleared_lateral_action: str | None = None,
        selected_fl: int | float | None = None,
        selected_mach: float | None = None,
        selected_cas: int | float | None = None,
        selected_vertical_speed: int | float | None = None,
        selected_heading: int | float | None = None,
        selected_on_route: bool | None = None,
        selected_speed_action: str | None = None,
        selected_vertical_speed_action: str | None = None,
        selected_vertical_action: str | None = None,
        selected_lateral_action: str | None = None,
        # not part of real flight plan, so considered an aircraft internal state
        route_current: list[str] | None = None,
        last_passed_filed_idx: int | None = None,
        last_passed_current_idx: int | None = None,
        squawk_ident_until: float | None = None,
    ):
        """
        Add an aircraft_internals event.

        Parameters
        ----------
        the_datetime: datetime
            Datetime of the event
        callsign: str, optional
            Aircraft callsign
        rate_of_turn: float, optional
            Aircraft Rate of Turn
        operation_params: dict, optional
            Aircraft operational parameters
        controllable: bool, optional
            Whether the aircraft is controllable
        simulated: bool, optional
            Whether the aircraft is simulated
        current_sector: str, optional
            Current sector of aircraft
        previous_sector: str, optional
            Previous sector of aircraft
        percentile_rank_dict: dict, optional
            Dictionary of cumulative distribution percentile rank. Used for speed randomisation.
        pilot_type: str, optional
            Pilot type
        pilot_action_queue: list[str], optional
            Action queue of pilot
        predictor_params: dict[str, float], optional
            Dictionary of predictor parameters
        wake_vortex: str, optional
            Aircraft wake vortex category
        random_seed: int, optional
            Base random seed to set any random attributes of aircraft
        heading_changing_to: int | float | None = None,
            Heading aircraft is changing to
        next_fix_index: int, optional
            Index of next fix in the list of fixes which constitute the filed route
        vertical_speed: float, optional
            The vertical speed of the aircraft
        cleared_fl: int | float | None = None,
            Aircraft cleared flight level
        cleared_mach: float, optional
            Aircraft cleared mach speed
        cleared_cas: int | float | None = None,
            Aircraft cleared Calibrated airspeed
        cleared_vertical_speed: int | float | None = None,
            Aircraft cleared vertical speed
        cleared_heading: int | float | None = None,
            Aircraft cleared heading
        cleared_on_route: bool, optional
            Whether the aircraft have been instructed to be route following
        cleared_speed_action: Action, optional
            The aircraft's last instructed speed Action
        cleared_vertical_speed_action: Action, optional
            The aircraft's last instructed vertical speed Action
        cleared_vertical_action: Action, optional
            The aircraft's last instructed vertical Action
        cleared_lateral_action: Action, optional
            The aircraft's last instructed lateral Action
        selected_fl: int | float | None = None,
            Aircraft's selected flight level
        selected_mach: float, optional
            Aircraft's selected mach speed
        selected_cas: int | float | None = None,
            Aircraft's selected calibrated air speed
        selected_vertical_speed: int | float | None = None,
            Aircraft's selected vertical speed
        selected_heading: int | float | None = None,
            Aircraft's selected heading
        selected_on_route: bool, optional
            Whether the aircraft is route_following
        selected_speed_action: str, optional
            The last instructed speed Action that the aircraft is following
        selected_vertical_speed_action: str, optional
            The last instructed vertical speed Action that the aircraft is following
        selected_vertical_action: str, optional
            The last instructed vertical Action that the aircraft is following
        selected_lateral_action: str, optional
            The last instructed lateral Action that the aircraft is following
        route_current: str, optional
            The aircraft's current route (not part of real flight plan, so considered an aircraft internal state)
        last_passed_filed_idx: int, optional
            The index of the last passed fix, relative to the filed route
        last_passed_current_idx: int, optional
            The index of the last passed fix, relative to the current route
        squawk_ident_until: float, optional
            Unix time in seconds that the aircraft squawk idents until
        """
        new_row = {
            "callsign": callsign,
            "rate_of_turn": rate_of_turn,
            "operation_params": operation_params,
            "controllable": controllable,
            "simulated": simulated,
            "current_sector": current_sector,
            "previous_sector": previous_sector,
            "percentile_rank_dict": percentile_rank_dict,
            "pilot_type": pilot_type,
            "pilot_action_queue": pilot_action_queue,
            "predictor_params": predictor_params,
            "wake_vortex": wake_vortex,
            "random_seed": random_seed,
            "heading_changing_to": heading_changing_to,
            "next_fix_index": next_fix_index,
            "vertical_speed": vertical_speed,
            "cleared_fl": cleared_fl,
            "cleared_mach": cleared_mach,
            "cleared_cas": cleared_cas,
            "cleared_vertical_speed": cleared_vertical_speed,
            "cleared_heading": cleared_heading,
            "cleared_on_route": cleared_on_route,
            "cleared_speed_action": cleared_speed_action,
            "cleared_vertical_speed_action": cleared_vertical_speed_action,
            "cleared_vertical_action": cleared_vertical_action,
            "cleared_lateral_action": cleared_lateral_action,
            "selected_fl": selected_fl,
            "selected_mach": selected_mach,
            "selected_cas": selected_cas,
            "selected_vertical_speed": selected_vertical_speed,
            "selected_heading": selected_heading,
            "selected_on_route": selected_on_route,
            "selected_speed_action": selected_speed_action,
            "selected_vertical_speed_action": selected_vertical_speed_action,
            "selected_vertical_action": selected_vertical_action,
            "selected_lateral_action": selected_lateral_action,
            "route_current": route_current,
            "last_passed_filed_idx": last_passed_filed_idx,
            "last_passed_current_idx": last_passed_current_idx,
            "squawk_ident_until": squawk_ident_until,
        }

        new_row = pd.Series(new_row, name=the_datetime)
        dtypes = {key: val for key, val in EventDtypes.aircraft_internals_dtypes.items() if key != "datetime"}
        self.aircraft_internals_df = pd_concat_two_dfs(self.aircraft_internals_df, new_row.to_frame().T.astype(dtypes))
        # ensure date ordering is preserved
        self.aircraft_internals_df = self.aircraft_internals_df.sort_index()

    def jump_to_time(
        self,
        env_manager: EnvironmentManager[TAircraft, TWindField, TForecastWindField],
        new_time: pd.Timestamp,
    ) -> Environment[TAircraft, TWindField, TForecastWindField]:
        """
        Set up the environment at a specific time according to events in the EventHandler

        Sets environment to a specific "new_time"

        Parameters
        ----------
        env_manager: EnvironmentManager
            Environment manager containing the environment to be updated
        new_time: pandas.Timestamp
            Time to update the environment to, according to the events in the EventHandler

        Returns
        ----------
        Environment
            Environment with time and state updated to new_time
        """
        environment = env_manager.environment

        # reset environment
        environment.reset()

        # set environment time to new_time
        environment.time = new_time.replace(tzinfo=timezone.utc).timestamp()

        # update from radar
        # only use last 6 seconds
        environment = update_from_radar(
            environment,
            self.radar_df,
            new_time - timedelta(seconds=6),
            new_time,
            ignore_simmed=False,
            aircraft_class=self.typeof_aircraft,
        )

        environment = update_from_flight_plans(
            environment,
            self.flight_df,
            episode_start=new_time - timedelta(days=1),
            episode_end=new_time.normalize() + timedelta(days=1),
            ignore_simmed=False,
        )

        # as the environment is jumping to a specific time, ignore any pending clearances issued before the jump
        env_manager._actions_to_issue = []

        environment = update_from_clearances(
            environment,
            env_manager,
            self.clearances_df,
            episode_start=new_time - timedelta(hours=1),
            episode_end=new_time,
            ignore_simmed=False,
        )

        # process the clearances created by the event handler and any other waiting
        # to be processed in the environment manager
        env_manager.process_actions()

        # don't log the new actions when jumping to time, so empty the action list
        env_manager._actions_to_issue = []

        environment = update_airspace_configuration(
            environment,
            env_manager,
            self.sectors_df,
            episode_start=new_time - timedelta(hours=1),
            episode_end=new_time,
        )

        environment = update_coordination(
            environment,
            self.coordination_df,
            episode_start=new_time - timedelta(days=1),
            episode_end=new_time,
            ignore_simmed=False,
        )
        environment = update_incomm(
            environment,
            self.incomm_df,
            episode_start=new_time - timedelta(hours=4),
            episode_end=new_time,
            ignore_simmed=False,
        )
        # reset selected_fl to match radar for non-simulated aircraft
        environment = update_selected_fl_from_radar(
            environment,
            self.radar_df,
            new_time - timedelta(seconds=6),
            new_time,
            ignore_simmed=False,
        )
        # when jumping to time, use selected_fl to set cleared_fl as it's more accurate than the clearances
        environment = set_cleared_fl_to_selected_fl(
            environment,
            self.radar_df,
            episode_start=new_time - timedelta(hours=1),
            episode_end=new_time,
            ignore_simmed=False,
        )
        environment = update_aircraft_attribute(
            environment,
            self.ac_attribute_update_df,
            new_time - timedelta(hours=1),
            new_time,
            ignore_simmed=False,
        )
        # update the route status of the aircraft.
        # will be overwritten by aircraft internal logs if available.
        environment = update_route_status(environment, ignore_simmed=False)
        # if previous internal aircraft state is available in logs, then this has the
        # final say on the state of non-simulated aircraft
        return update_aircraft_internals(
            environment,
            self.aircraft_internals_df,
            episode_start=new_time - timedelta(hours=1),
            episode_end=new_time,
            ignore_simmed=False,
        )

    def forward(
        self,
        env_manager: EnvironmentManager[TAircraft, TWindField, TForecastWindField],
        step_time: float,
    ) -> Environment[TAircraft, TWindField, TForecastWindField]:
        """
        Evolve Environment for the given time step [sec] by applying Events.

        Parameters
        ----------
        env_manager: EnvironmentManager
            Environment manager containing the environment to be updated
        step_time: float
            Amount of time in seconds the Environment is being evolved by events in the EventHandler.

        Returns
        ----------
        Environment
            Environment with time and state update by step_time seconds
        """
        environment = env_manager.environment

        episode_start = environment.datetime
        episode_end = episode_start + timedelta(seconds=step_time)

        environment = update_from_radar(
            environment,
            self.radar_df,
            episode_start,
            episode_end,
            ignore_simmed=self.ignore.radar_if_simmed,
            aircraft_class=self.typeof_aircraft,
        )

        all_aircraft_are_simmed = all(aircraft.simulated for aircraft in environment.aircraft.values())

        if not (all_aircraft_are_simmed and self.ignore.flight_if_simmed):
            # bypass if simmed aircraft are being ignored and all aircraft are simulated
            # for now use whole day's data to update flight plan as not time based
            environment = update_from_flight_plans(
                environment,
                self.flight_df,
                episode_start,
                episode_end,
                ignore_simmed=self.ignore.flight_if_simmed,
            )

        if not (all_aircraft_are_simmed and self.ignore.clearance_if_simmed):
            # bypass if simmed aircraft are being ignored and all aircraft are simulated
            environment = update_from_clearances(
                environment,
                env_manager,
                self.clearances_df,
                episode_start,
                episode_end,
                ignore_simmed=self.ignore.clearance_if_simmed,
            )

        # process the clearances created by the event handler and any other waiting
        # to be processed in the environment manager
        env_manager.process_actions()

        if not self.ignore.airspace_config_updates:
            environment = update_airspace_configuration(
                environment, env_manager, self.sectors_df, episode_start, episode_end
            )

        if not (all_aircraft_are_simmed and self.ignore.coordination_if_simmed):
            # bypass if simmed aircraft are being ignored and all aircraft are simulated
            environment = update_coordination(
                environment,
                self.coordination_df,
                episode_start,
                episode_end,
                ignore_simmed=self.ignore.coordination_if_simmed,
            )

        if not (all_aircraft_are_simmed and self.ignore.incomm_if_simmed):
            # bypass if simmed aircraft are being ignored and all aircraft are simulated
            environment = update_incomm(
                environment,
                self.incomm_df,
                episode_start,
                episode_end,
                ignore_simmed=self.ignore.incomm_if_simmed,
            )

        # reset selected_fl to match radar for non-simulated aircraft
        if not (all_aircraft_are_simmed and self.ignore.radar_if_simmed):
            # bypass if simmed aircraft are being ignored and all aircraft are simulated
            environment = update_selected_fl_from_radar(
                environment,
                self.radar_df,
                episode_start,
                episode_end,
                ignore_simmed=self.ignore.radar_if_simmed,
            )

        if not (all_aircraft_are_simmed and self.ignore.ac_attribute_if_simmed):
            # bypass if simmed aircraft are being ignored and all aircraft are simulated
            environment = update_aircraft_attribute(
                environment,
                self.ac_attribute_update_df,
                episode_start,
                episode_end,
                ignore_simmed=self.ignore.ac_attribute_if_simmed,
            )

        # update the route status of any aircraft which are not being simmed.
        # will be overwritten by aircraft internal logs if available.
        environment = update_route_status(environment, ignore_simmed=True)

        # if previous internal aircraft state is available in logs, then this has the
        # final say on the state of non-simulated aircraft
        if not (all_aircraft_are_simmed and self.ignore.aircraft_internals_if_simmed):
            # bypass if simmed aircraft are being ignored and all aircraft are simulated
            environment = update_aircraft_internals(
                environment,
                self.aircraft_internals_df,
                episode_start,
                episode_end,
                ignore_simmed=self.ignore.aircraft_internals_if_simmed,
            )

        return environment

    def add(self, other_event_handler: Self) -> Self:
        """
        Concatenate another EventHandler with this EventHandler.

        Parameters
        ----------
        other_event_handler: EventHandler
            Another event handler to be concatenated to the end of this one

        Returns
        ----------
        EventHandler
            The original instance concatenated with other_event_handler
        """
        self.radar_df = pd_concat_two_dfs(self.radar_df, other_event_handler.radar_df)
        self.flight_df = pd_concat_two_dfs(self.flight_df, other_event_handler.flight_df)
        self.clearances_df = pd_concat_two_dfs(self.clearances_df, other_event_handler.clearances_df)
        self.coordination_df = pd_concat_two_dfs(self.coordination_df, other_event_handler.coordination_df)
        self.sectors_df = pd_concat_two_dfs(self.sectors_df, other_event_handler.sectors_df)
        self.incomm_df = pd_concat_two_dfs(self.incomm_df, other_event_handler.incomm_df)
        self.aircraft_internals_df = pd_concat_two_dfs(
            self.aircraft_internals_df, other_event_handler.aircraft_internals_df
        )
        self.ac_attribute_update_df = pd_concat_two_dfs(
            self.ac_attribute_update_df, other_event_handler.ac_attribute_update_df
        )

        return self

    def trim(self, comparison_function: str, a_datetime: datetime) -> Self:
        """
        Remove event by comparing datetime to a reference datetime.
        Comparison function allows "<", "<=", ">", ">=".

        Any event which has a datetime satisfying the comparison function with the comparison datetime
        will be removed from the log.

        Parameters
        ----------
        comparison_function: str
            Function to use to compare the event datetime with a target datetime
        a_datetime: pandas.Timestamp
            Datetime to be compared to event in filter

        Returns
        -------
        EventHandler
            The EventHandler after removing any events which satisfied the comparison function
        """
        match comparison_function:
            # operator: "<", "<=", ">", ">="
            case "<":
                comp = operator.lt
            case "<=":
                comp = operator.le
            case ">":
                comp = operator.gt
            case ">=":
                comp = operator.ge
            case _:
                raise ValueError('Comparison operator unknown, must be one of "<", "<=", ">", ">="')

        if False:
            # for now, we don't trim the flight_plans as these can be at any time in the day
            self.flight_df = self.flight_df[~comp(self.flight_df.start_datetime, a_datetime)]

        self.radar_df = self.radar_df[~comp(self.radar_df.index, a_datetime)]
        self.clearances_df = self.clearances_df[~comp(self.clearances_df.index, a_datetime)]
        self.sectors_df = self.sectors_df[~comp(self.sectors_df.index, a_datetime)]
        self.incomm_df = self.incomm_df[~comp(self.incomm_df.index, a_datetime)]
        self.coordination_df = self.coordination_df[~comp(self.coordination_df.index, a_datetime)]
        self.aircraft_internals_df = self.aircraft_internals_df[~comp(self.aircraft_internals_df.index, a_datetime)]

        return self

    def remove_simmed(self, event_logger: EventLogger) -> Self:
        """
        Remove all events from this EventHandler for any aircraft that was simmed in an EventLogger.

        Parameters
        ----------
        event_logger: EventLogger
            All events for and aircraft which is simmed in this event logger will be removed from the EventHandler

        Returns
        ----------
        EventHandler
            Original EventHandler with all events for any aircraft which are simmed at any time in the event logger
            removed.
        """
        callsigns_to_remove = {ac["callsign"] for ac in event_logger.aircraft_internals_log if ac["simulated"]}

        self.radar_df = self.radar_df[~self.radar_df.callsign.isin(callsigns_to_remove)]
        self.clearances_df = self.clearances_df[~self.clearances_df.callsign.isin(callsigns_to_remove)]
        self.incomm_df = self.incomm_df[~self.incomm_df.callsign.isin(callsigns_to_remove)]
        self.coordination_df = self.coordination_df[~self.coordination_df.callsign.isin(callsigns_to_remove)]
        self.aircraft_internals_df = self.aircraft_internals_df[
            ~self.aircraft_internals_df.callsign.isin(callsigns_to_remove)
        ]

        return self


def update_route_status(
    environment: Environment[TAircraft, TWindField, TForecastWindField], ignore_simmed: bool
) -> Environment[TAircraft, TWindField, TForecastWindField]:
    """
    Update the route status of the aircraft.

    Parameters
    ----------
    environment: Environment
        Environment containing aircraft to be updated
    ignore_simmed: bool
        If True, do not update the next_fix_index and on_route for simmed aircraft

    Returns
    ----------
    Environment
        Environment with aircraft route status updated
    """
    for aircraft in environment.aircraft.values():
        if aircraft.flight_plan is None:
            logger.warning(
                f"Cannot update route status without a flight plan for aircraft {aircraft.callsign}.",
                stacklevel=2,
            )
            continue

        # next_fix_index is updated by the predictor, and we only update it here when jumping to a time
        # which is specified by setting the ignore_simmed flag
        set_next_fix_index = not ignore_simmed or not aircraft.simulated

        aircraft.update_route_status(environment.airspace, set_next_fix_index=set_next_fix_index)

    return environment


def update_aircraft_attribute(
    environment: Environment[TAircraft, TWindField, TForecastWindField],
    df_ac_attribute: pd.DataFrame,
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
    ignore_simmed: bool,
) -> Environment[TAircraft, TWindField, TForecastWindField]:
    """
    Update single aircraft attributes using the Aircraft attribute Events DataFrame.

    Parameters
    ----------
    environment: Environment
        Environment containing aircraft to be updated
    df_ac_attribute: pd.DataFrame
        DataFrame containing aircraft internal attribute events
    episode_start: pd.Timestamp
        Time of the start of the episode for which events are to be used to update the environment
    episode_end: pd.Timestamp
        Time of the end of the episode for which events are to be used to update the environment
    ignore_simmed: bool
        If True, ignore simmed aircraft when updating using aircraft internal attribute events

    Returns
    ----------
    Environment
        Environment with aircraft updated using aircraft internal attribute events
    """

    # filter to relevant time window
    df = df_ac_attribute[(df_ac_attribute.index > episode_start) & (df_ac_attribute.index <= episode_end)]

    if ignore_simmed:
        # don't update simulated aircraft using data
        df = df[~df.callsign.isin(environment.simulated_aircraft())]

    # replace NANs with None as this is what the starling classes require
    df = df.fillna(np.nan).replace([np.nan], [None])
    # TODO: speed up other itertuples. named=None or use row.col or use zip
    for row in df.itertuples():
        if row.callsign in environment.aircraft:
            aircraft = environment.aircraft[row.callsign]

            # set the aircraft attribute
            setattr(aircraft, row.attribute_name, row.value)

    return environment


def update_from_radar(
    environment: Environment[TAircraft, TWindField, TForecastWindField],
    df_rad: pd.DataFrame,
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
    ignore_simmed: bool,
    aircraft_class: type[TAircraft],
) -> Environment[TAircraft, TWindField, TForecastWindField]:
    """
    Update Aircraft positions or create new aircraft using the Radar Events DataFrame.

    Parameters
    ----------
    environment: Environment
        Environment containing aircraft to be updated
    df_rad: pd.DataFrame
        DataFrame containing radar events
    episode_start: pd.Timestamp
        Time of the start of the episode for which events are to be used to update the environment
    episode_end: pd.Timestamp
        Time of the end of the episode for which events are to be used to update the environment
    ignore_simmed: bool
        If True, ignore simmed aircraft when updating using radar events
    aircraft_class: type
        The class of Aircraft to instantiate - either bluebird_dt.Aircraft or a derived class.
    Returns
    ----------
    Environment
        Environment with aircraft updated using radar events
    """

    # if all radar events is before episode_start then bypass this function
    # this is a speedup for fully artificial scenarios once all aircraft have been created
    if df_rad.index.max() < episode_start:
        return environment

    # filter to relevant time window
    df = df_rad[(df_rad.index > episode_start) & (df_rad.index <= episode_end)]

    # bypass function if dataframe is empty
    if len(df) == 0:
        return environment

    # take the last value of each callsign
    df = df.groupby("callsign", sort=False, observed=True).tail(1)

    if ignore_simmed:
        # don't update simulated aircraft using data
        df = df[~df.callsign.isin(environment.simulated_aircraft())]

    # replace NANs with None as this is what the starling classes require
    df = df.fillna(np.nan).replace([np.nan], [None])

    # if aircraft already in environment, update its position,
    # else create a new aircraft
    all_aircraft: list[TAircraft] = []

    # TODO: use itertuples
    for _, row in df.iterrows():
        # add aircraft if it's not in the environment
        if row.callsign not in environment.aircraft:
            # when aircraft is first created, it must have a lat, lon, fl, heading
            # after this it may be updated with partial information
            if None in (row.lat, row.lon, row.fl, row.heading):
                continue

            aircraft = aircraft_class(
                lat=row.lat,
                lon=row.lon,
                fl=row.fl,
                heading=row.heading,
                flight_plan=None,
                callsign=row.callsign,
                selected_fl=row.selected_fl,
                ufid=row.ufid if row.ufid != "" else None,
                simulated=False,
            )
        else:  # callsign is in environment, so update aircraft
            aircraft = environment.aircraft[row.callsign]
            aircraft.set_position(row.lat, row.lon)

            # don't update values if new values are zero (originally nan, transformed to zero)
            if row.fl is not None:
                aircraft.fl = row.fl

            if row.selected_fl is not None:
                aircraft.selected_fl = row.selected_fl

        # don't update values if new values are zero (originally nan, transformed to zero)
        if row.ground_speed is not None:
            aircraft.ground_speed = row.ground_speed

        if row.ground_track_angle is not None:
            aircraft.ground_track_angle = row.ground_track_angle

        if row.speed_tas is not None:
            aircraft.speed_tas = row.speed_tas
        elif aircraft.ground_speed is not None and aircraft.ground_track_angle is not None:
            # find the wind vector at the location of this aircraft
            if environment.wind_field is not None:
                wind_vector = environment.wind_field.get_wind_vector(aircraft.fl, aircraft.lat, aircraft.lon)
            else:
                wind_vector = None

            # calculate the tas
            aircraft.speed_tas = tas_from_ground_speed(aircraft.ground_speed, aircraft.ground_track_angle, wind_vector)

        if row.heading is not None:
            aircraft.heading = row.heading

        all_aircraft.append(aircraft)

    if ignore_simmed:
        # filter to keep only simulated aircraft
        environment.aircraft = {
            callsign: aircraft for callsign, aircraft in environment.aircraft.items() if aircraft.simulated
        }
    else:
        # we are NOT ignoring simmed aircraft, so filter out all aircraft
        environment.aircraft = {}

    # add back in the aircraft replaying from data
    for aircraft in all_aircraft:
        environment.aircraft[aircraft.callsign] = aircraft

    return environment


def update_from_flight_plans(
    environment: Environment[TAircraft, TWindField, TForecastWindField],
    df_fl: pd.DataFrame,
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
    ignore_simmed: bool,
) -> Environment[TAircraft, TWindField, TForecastWindField]:
    """
    Update flight plans from FlightPlan Events DataFrame.

    Parameters
    ----------
    environment: Environment
        Environment containing aircraft to be updated
    df_fl: pd.DataFrame
        DataFrame containing flight plan events
    episode_start: pd.Timestamp
        Time of the start of the episode for which events are to be used to update the environment
    episode_end: pd.Timestamp
        Time of the end of the episode for which events are to be used to update the environment
    ignore_simmed: bool
        If True, ignore simmed aircraft when updating using flight plan events
    Returns
    ----------
    Environment
        Environment with aircraft updated using flight plan events
    """
    # flight_plans may be available many hours before the flight, or from data have a time stamp after the flight
    # has started. For aircraft that have just appeared in the environment, we allow
    # the flight plan to have been on the system for up to 24 hours
    df = df_fl[(df_fl.index >= episode_start - timedelta(days=1)) & (df_fl.index <= episode_end + timedelta(hours=12))]

    # only update callsigns that are in environment
    df = df[df.callsign.isin(environment.aircraft.keys())]

    if ignore_simmed:
        # don't use logs/data to update simulated aircraft
        df = df[~df.callsign.isin(environment.simulated_aircraft())]

    # keep if time is for all time (None), or after the start_datetime or before the end_datetime
    df = df[
        (pd.isna(df.start_datetime) | (df.start_datetime < episode_end))
        & (pd.isna(df.end_datetime) | (episode_start < df.end_datetime))
    ]

    # exit early if dataframe is empty
    if len(df) == 0:
        return environment

    # sort by end_datetime (None values will be at the end)
    df = df.sort_values("end_datetime")

    # only keep ufids if they are None, an empty string or are in the environment
    all_ufids = {a.ufid for a in environment.aircraft.values()}
    all_ufids.add("")
    df = df[(df["ufid"].isna()) | (df["ufid"].isin(all_ufids))]

    # take the first value of each callsign, corresponding to the earliest end_datetime
    df = df.groupby("callsign", sort=False, observed=True).head(1)

    # replace NANs with None as this is what the starling classes require
    df = df.fillna(np.nan).replace([np.nan], [None])

    # update the flight plans
    # NOTE: We iterate over all flight plans at each timestep. This allows flight plan updates (takes 0.02 seconds)
    for _, row in df.iterrows():
        aircraft = environment.aircraft[row.callsign]

        # create flight plan
        if aircraft.flight_plan is None:
            # set current route to filed if this is the first flight plan of the aircraft
            new_route_current = row.route_filed

            # when flight plan is first added, set to route following or not, depending on previous instructions
            aircraft.on_route = aircraft.cleared_instructions.on_route

        else:
            route = aircraft.flight_plan.route
            # if filed route hasn't changed, keep current route
            new_route_current = route.current if list(row.route_filed) == list(route.filed) else row.route_filed

        route = Route(filed=list(row.route_filed), current=list(new_route_current))

        aircraft.flight_plan = FlightPlan(
            route=route,
            unexpanded_route=row.unexpanded_route,
            origin=row.origin,
            dest=row.dest,
            milcivil=row.milcivil,
            sector_crossing_seq=row.sector_crossing_seq,
            requested_flight_level=row.requested_flight_level,
            filed_true_airspeed=row.filed_true_airspeed,
            intention_code=row.intention_code,
            assigned_squawk=row.assigned_squawk,
            start_datetime=row.start_datetime,
            end_datetime=row.end_datetime,
        )

        # Once the squawk is initialised, it should only be updated by the pilot from instructions from ATC,
        # or when leaving controlled airspace, and not from flight plan updates.
        if aircraft.squawk is None:
            aircraft.squawk = row.squawk

        aircraft.aircraft_type = row.actype

    return environment


def last_notna(group: pd.Series) -> float:
    """
    Return the last value that isn't NA in a series.

    Intended for use in groupby of a single float or int column.

    Parameters
    ----------
    group: pd.Series
        Group created from a single float or int column

    Returns
    ----------
    float
        The last non-NA value
    """

    non_null_vals = group.dropna()
    if len(non_null_vals) > 0:
        return non_null_vals.iloc[-1]
    return np.nan


def update_selected_fl_from_radar(
    environment: Environment[TAircraft, TWindField, TForecastWindField],
    df_radar: pd.DataFrame,
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
    ignore_simmed: bool,
) -> Environment[TAircraft, TWindField, TForecastWindField]:
    """
    Clearances will update the CFL and SFL (if no delay) when actions are issued.
    For replayed aircraft we want this to be overruled by the radar data.

    Parameters
    ----------
    environment: Environment
        Environment containing aircraft to be updated
    df_radar: pd.DataFrame
        DataFrame containing radar event which contain the selected flight level
    episode_start: pd.Timestamp
        Time of the start of the episode for which events are to be used to update the environment
    episode_end: pd.Timestamp
        Time of the end of the episode for which events are to be used to update the environment
    ignore_simmed: bool
        If True, ignore simmed aircraft

    Returns
    ----------
    Environment
        Environment with aircraft selected flight level set to the values in the radar events dataframe
    """

    # filter to relevant time window
    df = df_radar[(df_radar.index > episode_start) & (df_radar.index <= episode_end)]

    # take the last non_null selected_fl for each callsign
    df = (
        df.groupby("callsign", observed=True)["selected_fl"].apply(last_notna).dropna()
    )  # if there are no values, then np.nan is returned from apply which needs to be dropped

    if ignore_simmed:
        # don't update simulated aircraft using data
        df = df[~df.index.isin(environment.simulated_aircraft())]

    for callsign, aircraft in environment.aircraft.items():
        if callsign in df.index:
            aircraft.selected_fl = df.loc[callsign]

    return environment


def set_cleared_fl_to_selected_fl(
    environment: Environment[TAircraft, TWindField, TForecastWindField],
    df_radar: pd.DataFrame,
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
    ignore_simmed: bool,
) -> Environment[TAircraft, TWindField, TForecastWindField]:
    """
    Set the cleared_fl to the selected_fl.
    To be used with the jump_to_time function.

    Parameters
    ----------
    environment: Environment
        Environment containing aircraft to be updated
    df_radar: pd.DataFrame
        DataFrame containing radar events
    episode_start: pd.Timestamp
        Time of the start of the episode for which events are to be used to update the environment
    episode_end: pd.Timestamp
        Time of the end of the episode for which events are to be used to update the environment
    ignore_simmed: bool
        If True, ignore simmed aircraft when updating using radar events

    Returns
    ----------
    Environment
        Environment with aircraft updated using radar events
    """

    # filter to relevant time window
    df = df_radar[(df_radar.index > episode_start) & (df_radar.index <= episode_end)]

    # take the last non_null selected_fl for each callsign
    df = (
        df.groupby("callsign", observed=True)["selected_fl"].apply(last_notna).dropna()
    )  # if there are no values, then np.nan is returned from apply which needs to be dropped

    if ignore_simmed:
        # don't update simulated aircraft using data
        df = df[~df.index.isin(environment.simulated_aircraft())]

    for callsign, aircraft in environment.aircraft.items():
        if callsign in df.index:
            selected_fl = df.loc[callsign]

            # update the aircraft using the clearance
            aircraft.cleared_fl = selected_fl

    return environment


def update_from_clearances(
    environment: Environment[TAircraft, TWindField, TForecastWindField],
    env_manager: EnvironmentManager[TAircraft, TWindField, TForecastWindField],
    df_clr: pd.DataFrame,
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
    ignore_simmed: bool,
) -> Environment[TAircraft, TWindField, TForecastWindField]:
    """
    Send clearances to environment using clearance events for a specific time period.

    Parameters
    ----------
    environment: Environment
        Environment containing aircraft to be updated
    env_manager: EnvironmentManager
        Environment Manager. Required to receive the new clearances.
    df_clr: pd.DataFrame
        DataFrame containing clearance events
    episode_start: pd.Timestamp
        Time of the start of the episode for which events are to be used to update the environment
    episode_end: pd.Timestamp
        Time of the end of the episode for which events are to be used to update the environment
    ignore_simmed: bool
        If True, ignore simmed aircraft when updating using clearance events

    Returns
    ----------
    Environment
        Environment with new clearances received from clearance events
    """
    # filter to relevant time window
    df = df_clr[(df_clr.index > episode_start) & (df_clr.index <= episode_end)]

    # no update if no clearance events in relevant timeframe
    if len(df) == 0:
        return environment

    # take the last value of each kind of action for each callsign
    df = (
        df.reset_index(names="datetime")
        .sort_values("datetime")
        .groupby(["callsign", "kind"], sort=False)
        .tail(1)
        .reset_index(drop=True)
        .set_index(["datetime"])
    )

    # only update callsigns that are in environment
    df = df[df.callsign.isin(environment.aircraft.keys())]

    if ignore_simmed:
        # don't update simulated aircraft using data
        df = df[~df.callsign.isin(environment.simulated_aircraft())]

    # replace NANs with None as this is what the bluebird_dt classes require
    df = df.fillna(np.nan).replace([np.nan], [None])

    # if aircraft already in environment update its state using the clearance
    # TODO: use itertuples
    for _the_datetime, row in df.iterrows():
        callsign = row.callsign
        aircraft = environment.aircraft[callsign]

        # ignore route_direct_to if no flight plan on aircraft
        if row.kind == "route_direct_to" and aircraft.flight_plan is None:
            continue

        # update the aircraft using the clearance
        action = Action(
            callsign,
            row.kind,
            row.value,
            agent=row.agent,
            text_representation=ClearanceAndResponse(
                clearance=row.text_clearance, pilot_response=row.text_pilot_response
            ),
            voice_representation=ClearanceAndResponse(
                clearance=row.voice_clearance, pilot_response=row.voice_pilot_response
            ),
            sector=(
                row["sector"]
                if "sector" in row.keys()  # noqa: SIM118,
                else environment.airspace.expand_bandbox_sector(aircraft.current_sector)
            ),
        )

        env_manager.receive_actions([action])

    return environment


def update_airspace_configuration(
    environment: Environment[TAircraft, TWindField, TForecastWindField],
    env_manager: EnvironmentManager[TAircraft, TWindField, TForecastWindField],
    df_sectors: pd.DataFrame,
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
) -> Environment[TAircraft, TWindField, TForecastWindField]:
    """
    Update the configuration/bandboxing of the airspace according to the sectorisation events DataFrame.

    Parameters
    ----------
    environment: Environment
        Environment containing aircraft to be updated
    env_manager: EnvironmentManager
        Environment Manager. Required to split and bandbox sectors.
    df_sectors: pd.DataFrame
        DataFrame containing sector configuration events
    episode_start: pd.Timestamp
        Time of the start of the episode for which events are to be used to update the environment
    episode_end: pd.Timestamp
        Time of the end of the episode for which events are to be used to update the environment

    Returns
    ----------
    Environment
        Environment with bandboxing updated according to events occurring in the the desired time period
    """
    if df_sectors is None:
        return environment
    # get relevant time period
    df = df_sectors[(df_sectors.index > episode_start) & (df_sectors.index <= episode_end)].tail(1)

    # only update if there is relevant update data in the time period
    if len(df) == 0:
        return environment

    # keep last config only
    # sector_config = {sec_name: sectors for (sec_name, sectors) in df.iloc[-1].to_numpy()[0]}
    sector_config = dict(df.iloc[-1]["sectors_configuration"])

    # split any bandboxed sectors that are not in the current configuration
    # create copy as split_sector mutates the iterable
    for sector_name in list(environment.airspace.airspace_configuration):
        if sector_name not in sector_config:
            env_manager.split_sector(sector_name)

    # bandbox sectors that aren't already bandboxed
    new_bandboxing = {}
    for sec_name, sec_numbers in sector_config.items():
        # don't try and bandbox one individual sector of the same name, or sectors which are already bandboxed
        if (
            len(sec_numbers) == 0
            or (len(sec_numbers) == 1 and sec_numbers[0] == sec_name)
            or sec_name in environment.airspace.airspace_configuration
        ):
            continue

        # only bandbox sectors if all individual sectors are in the airspace
        all_individual_sectors_in_airspace = all(
            sector in environment.airspace.list_individual_sectors() for sector in sec_numbers
        )

        if all_individual_sectors_in_airspace:
            new_bandboxing[sec_name] = sec_numbers
        else:
            logger.error(
                "Failed to bandbox sectors because not all individual sectors exist. Bandox configuration ="
                f"{sec_numbers}"
            )

    # bandbox if required
    if len(new_bandboxing) > 0:
        env_manager.bandbox_sectors(new_bandboxing)

    return environment


def update_coordination(
    environment: Environment[TAircraft, TWindField, TForecastWindField],
    df_coord: pd.DataFrame,
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
    ignore_simmed: bool,
) -> Environment[TAircraft, TWindField, TForecastWindField]:
    """
    Update coordinations in environment using the coordinations Events.

    Parameters
    ----------
    environment: Environment
        Environment containing aircraft to be updated
    df_coord: pd.DataFrame
        DataFrame containing coordination events
    episode_start: pd.Timestamp
        Time of the start of the episode for which events are to be used to update the environment
    episode_end: pd.Timestamp
        Time of the end of the episode for which events are to be used to update the environment
    ignore_simmed: bool
        If True, ignore simmed aircraft when updating using coordination events

    Returns
    ----------
    Environment
        Environment with coordinations updated according to events occurring in the the desired time period
    """
    # keep coordinations between start and end time
    df = df_coord[(df_coord.index > episode_start) & (df_coord.index <= episode_end)]
    if ignore_simmed:
        # don't update simulated aircraft using data
        df = df[~df.callsign.isin(environment.simulated_aircraft())]
    # return early if possible for speedup
    if len(df) == 0:
        return environment

    # only keep the last coordination from each sector for each callsign
    df = (
        df.reset_index(names="datetime")
        .sort_values("datetime")
        .groupby(["callsign", "from_sector"], dropna=False)
        .tail(1)
        .reset_index(drop=True)
        .set_index(["datetime"])
    )
    # replace "" with None as this is what the coordination class requires
    df["level_by_details"] = df["level_by_details"].fillna(np.nan).replace([np.nan], [None])
    df["level_by_details"] = df["level_by_details"].replace([""], [None])
    df["secondary_coord_conditions"] = df["secondary_coord_conditions"].fillna(np.nan).replace([np.nan], [None])
    df["secondary_coord_conditions"] = df["secondary_coord_conditions"].replace([""], [None])

    # update the aircraft's coordinations
    for (
        the_datetime,
        callsign,
        from_sector,
        to_sector,
        fl,
        fix,
        direction,
        level_by,
        level_by_details,
        secondary_coord_conditions,
    ) in df.itertuples():
        new_coordination = Coordination(
            callsign=callsign,
            from_sector=from_sector,
            to_sector=to_sector,
            fl=fl,
            fix=fix,
            direction=direction,
            level_by=level_by,
            level_by_details=level_by_details,
            secondary_coord_conditions=secondary_coord_conditions,
            the_datetime=the_datetime,
        )

        # don't add coordination if already present (possibly with a different timestamp)
        coord_already_present = environment.coordinations.contains_excluding_times(new_coordination)

        if not coord_already_present:
            environment.coordinations.add(new_coordination)

    return environment


def update_incomm(
    environment: Environment[TAircraft, TWindField, TForecastWindField],
    df_incomm: pd.DataFrame,
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
    ignore_simmed: bool,
) -> Environment[TAircraft, TWindField, TForecastWindField]:
    """
    Update incomm status from Incomm Events DataFrame.

    Parameters
    ----------
    environment: Environment
        Environment containing aircraft to be updated
    df_incomm: pd.DataFrame
        DataFrame containing incomm events
    episode_start: pd.Timestamp
        Time of the start of the episode for which events are to be used to update the environment
    episode_end: pd.Timestamp
        Time of the end of the episode for which events are to be used to update the environment
    ignore_simmed: bool
        If True, ignore simmed aircraft when updating using incomm events

    Returns
    ----------
    Environment
        Environment with aircraft updated using incomm events
    """
    df = df_incomm[(df_incomm.index > episode_start) & (df_incomm.index <= episode_end)]

    # no update if no events in timeframe
    if len(df) == 0:
        return environment

    # only update callsigns that are in environment
    df = df[df.callsign.isin(environment.aircraft.keys())]

    if ignore_simmed:
        # don't update simulated aircraft using data
        df = df[~df.callsign.isin(environment.simulated_aircraft())]

    for _the_datetime, callsign, new_sector in df.itertuples():
        leaving_sector = environment.aircraft[callsign].current_sector
        # don't allow incomm to the sector it's already in (happens in real-world data)
        if new_sector != leaving_sector:
            # any coordinations complete due to incomm needs to be removed. We keep the coordination
            # corresponding to this new incomm, which is needed so that the NFL of the current sector is still
            # available, but remove the coordination into the sector the aircraft is now leaving
            environment.remove_coords_pre_outcomm(callsign)

            # update sector on aircraft
            # (aircraft.previous sector is automatically updated from aircraft.current_sector)
            # if new sector is in the airspace (or is an individual sector in the airspace)
            # then allow the incomm to that sector, else treat the new_sector as "background"
            if new_sector in environment.airspace.sectors:
                environment.aircraft[callsign].current_sector = new_sector
            elif new_sector in environment.airspace.list_individual_sectors():
                environment.aircraft[callsign].current_sector = environment.airspace.get_containing_bandboxed_sector(
                    new_sector
                )
            else:
                environment.aircraft[callsign].current_sector = "background"

    return environment


def update_aircraft_internals(
    environment: Environment[TAircraft, TWindField, TForecastWindField],
    df_aircraft_internals: pd.DataFrame,
    episode_start: pd.Timestamp,
    episode_end: pd.Timestamp,
    ignore_simmed: bool,
) -> Environment[TAircraft, TWindField, TForecastWindField]:
    """
    Update Aircraft internal attributes using the aircraft_internals Events DataFrame.

    Parameters
    ----------
    environment: Environment
        Environment containing aircraft to be updated
    df_aircraft_internals: pd.DataFrame
        DataFrame containing aircraft internal attribute events
    episode_start: pd.Timestamp
        Time of the start of the episode for which events are to be used to update the environment
    episode_end: pd.Timestamp
        Time of the end of the episode for which events are to be used to update the environment
    ignore_simmed: bool
        If True, ignore simmed aircraft when updating using aircraft internal attribute events

    Returns
    ----------
    Environment
        Environment with aircraft updated using aircraft internal attribute events
    """
    df = df_aircraft_internals[
        (df_aircraft_internals.index > episode_start) & (df_aircraft_internals.index <= episode_end)
    ]
    # no update if no events in timeframe
    if len(df) == 0:
        return environment

    # take the last value of each callsign
    df = df.groupby("callsign", sort=False, observed=True).tail(1)

    if ignore_simmed:
        # don't update simulated aircraft using data
        df = df[~df.callsign.isin(environment.simulated_aircraft())]

    # only update aircraft that are in the environment
    df = df[df.callsign.isin(environment.aircraft)]

    # replace NANs with None as this is what the starling classes require
    df = df.fillna(np.nan).replace([np.nan], [None])

    # TODO: speed up other itertuples. named=None or use row.col or use zip
    for row in df.itertuples():
        aircraft = environment.aircraft[row.callsign]  # index is callsign
        aircraft.rate_of_turn = row.rate_of_turn
        aircraft.operation_params = row.operation_params
        aircraft.controllable = row.controllable
        aircraft.simulated = row.simulated
        aircraft.current_sector = row.current_sector
        aircraft._previous_sector = row.previous_sector
        aircraft.predictor_params = row.predictor_params
        aircraft.percentile_rank_dict = row.percentile_rank_dict

        # TODO: need to handle pilot type in a better way
        assert aircraft.pilot.__class__.__name__ == row.pilot_type

        aircraft.pilot.action_queue = [json.dumps(queue_item) for queue_item in row.pilot_action_queue]

        aircraft.wake_vortex = row.wake_vortex
        aircraft.random_seed = row.random_seed
        aircraft.heading_changing_to = row.heading_changing_to
        aircraft.vertical_speed = row.vertical_speed
        aircraft.squawk_ident_until = row.squawk_ident_until

        if pd.isna(row.next_fix_index):
            aircraft.next_fix_index = None
        else:
            aircraft.next_fix_index = int(row.next_fix_index)

        if aircraft.cleared_instructions is None:
            cleared_instructions = Instructions()
        else:
            cleared_instructions = aircraft.cleared_instructions

        if aircraft.selected_instructions is None:
            selected_instructions = Instructions()
        else:
            selected_instructions = aircraft.selected_instructions

        cleared_speed_action = None if row.cleared_speed_action is None else Action.from_json(row.cleared_speed_action)
        cleared_vertical_speed_action = (
            None if row.cleared_vertical_speed_action is None else Action.from_json(row.cleared_vertical_speed_action)
        )
        cleared_vertical_action = (
            None if row.cleared_vertical_action is None else Action.from_json(row.cleared_vertical_action)
        )
        cleared_lateral_action = (
            None if row.cleared_lateral_action is None else Action.from_json(row.cleared_lateral_action)
        )

        selected_speed_action = (
            None if row.selected_speed_action is None else Action.from_json(row.selected_speed_action)
        )
        selected_vertical_speed_action = (
            None if row.selected_vertical_speed_action is None else Action.from_json(row.selected_vertical_speed_action)
        )
        selected_vertical_action = (
            None if row.selected_vertical_action is None else Action.from_json(row.selected_vertical_action)
        )
        selected_lateral_action = (
            None if row.selected_lateral_action is None else Action.from_json(row.selected_lateral_action)
        )

        cleared_instructions.fl = row.cleared_fl
        cleared_instructions.mach = row.cleared_mach
        cleared_instructions.cas = row.cleared_cas
        cleared_instructions.vertical_speed = row.cleared_vertical_speed
        cleared_instructions.heading = row.cleared_heading
        cleared_instructions.on_route = row.cleared_on_route
        cleared_instructions.speed_action = cleared_speed_action
        cleared_instructions.vertical_speed_action = cleared_vertical_speed_action
        cleared_instructions.vertical_action = cleared_vertical_action
        cleared_instructions.lateral_action = cleared_lateral_action

        selected_instructions.fl = row.selected_fl
        selected_instructions.mach = row.selected_mach
        selected_instructions.cas = row.selected_cas
        selected_instructions.vertical_speed = row.selected_vertical_speed
        selected_instructions.heading = row.selected_heading
        selected_instructions.on_route = row.selected_on_route
        selected_instructions.speed_action = selected_speed_action
        selected_instructions.vertical_speed_action = selected_vertical_speed_action
        selected_instructions.vertical_action = selected_vertical_action
        selected_instructions.lateral_action = selected_lateral_action

        aircraft.cleared_instructions = cleared_instructions
        aircraft.selected_instructions = selected_instructions

        aircraft.last_passed_filed_idx = None if pd.isna(row.last_passed_filed_idx) else int(row.last_passed_filed_idx)
        aircraft.last_passed_current_idx = (
            None if pd.isna(row.last_passed_current_idx) else int(row.last_passed_current_idx)
        )

        if aircraft.flight_plan is not None:
            if row.route_current is None:
                aircraft.flight_plan.route.current = None
            else:
                aircraft.flight_plan.route.current = list(row.route_current)

    return environment
