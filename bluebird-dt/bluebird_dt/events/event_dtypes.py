from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict

from bluebird_dt.core import Route


class Record(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )


class RadarEvent(Record):
    datetime: datetime
    callsign: str
    lat: float
    lon: float
    fl: float
    heading: float

    # Optional columns
    ufid: str | None = None
    speed_tas: float | None = None
    ground_speed: float | None = None
    ground_track_angle: float | None = None
    selected_fl: float | None = None


class FlightPlanEvent(Record):
    datetime: datetime
    callsign: str
    route_filed: Route
    start_datetime: datetime
    end_datetime: datetime
    squawk: str
    origin: str
    dest: str
    unexpanded_route: str
    sector_crossing_seq: str
    actype: str
    milcivil: str
    requested_flight_level: int | None
    filed_true_airspeed: int | None
    intention_code: str
    ufid: str
    assigned_squawk: str


class ClearanceEvent(Record):
    datetime: datetime
    callsign: str
    kind: str
    value: str
    agent: str
    text_clearance: str
    text_pilot_response: str
    voice_clearance: str
    voice_pilot_response: str
    sector: list[str]


class SectorsEvent(Record):
    datetime: datetime
    sectors_configuration: list[list[str]]


class IncommEvent(Record):
    datetime: datetime
    callsign: str
    sector_name: str


class CoordinationEvent(Record):
    callsign: str
    from_sector: str
    to_sector: str
    fl: float
    fix: str
    direction: str
    level_by: bool
    level_by_details: str
    secondary_coord_conditions: str
    datetime: datetime


class AircraftInternalsEvent(Record):
    callsign: str
    datetime: datetime
    rate_of_turn: float
    operation_params: Any
    controllable: bool
    simulated: bool
    current_sector: str
    previous_sector: str
    percentile_rank_dict: dict[str, Any]
    pilot_type: str
    pilot_action_queue: Any
    predictor_params: Any
    wake_vortex: str
    random_seed: int | None
    heading_changing_to: float
    next_fix_index: int | None
    vertical_speed: float
    cleared_fl: float
    cleared_mach: float
    cleared_cas: float
    cleared_vertical_speed: float
    cleared_heading: float
    cleared_on_route: bool
    cleared_speed_action: str
    cleared_vertical_speed_action: str
    cleared_vertical_action: str
    cleared_lateral_action: str
    selected_fl: float
    selected_mach: float
    selected_cas: float
    selected_vertical_speed: float
    selected_heading: float
    selected_on_route: bool
    selected_speed_action: str
    selected_vertical_speed_action: str
    selected_vertical_action: str
    selected_lateral_action: str
    route_current: Any
    last_passed_filed_idx: int | None
    last_passed_current_idx: int | None
    squawk_ident_until: float


class AircraftAttributeUpdateEvent(Record):
    callsign: str
    datetime: datetime
    attribute_name: str
    value: Any
