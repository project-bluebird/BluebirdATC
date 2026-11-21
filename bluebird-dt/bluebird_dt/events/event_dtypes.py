from typing import ClassVar


class EventDtypes:
    """Fields and types for every dataframe of events in the EventHandler"""

    # specify expected datatypes for each EventHandler dataframe
    radar_dtypes: ClassVar[dict[str, str]] = {
        # ufid, speed_tas, ground_speed, ground_track angle and selected_fl
        # are all optional columns and values and will be added automatically
        # if required
        "datetime": "datetime64[us]",  # datetime in utc
        "callsign": "string",
        "lat": "float64",
        "lon": "float64",
        "fl": "float64",
        "heading": "float64",
        "ufid": "category",  # optional
        "speed_tas": "float64",  # optional
        "ground_speed": "float64",  # optional
        "ground_track_angle": "float64",  # optional
        "selected_fl": "float64",  # optional
    }

    flight_plan_dtypes: ClassVar[dict[str, str]] = {
        "datetime": "datetime64[us]",
        "callsign": "category",
        "route_filed": "object",  # numpy array
        "start_datetime": "datetime64[us]",
        "end_datetime": "datetime64[us]",
        "squawk": "string",  # string as may start with a zero
        "origin": "string",
        "dest": "string",
        "unexpanded_route": "string",
        "sector_crossing_seq": "string",
        "actype": "string",
        "milcivil": "string",
        "requested_flight_level": "Int64",
        "filed_true_airspeed": "Int64",
        "intention_code": "string",
        "ufid": "category",
        "assigned_squawk": "string",
    }

    clearance_dtypes: ClassVar[dict[str, str]] = {
        "datetime": "datetime64[us]",
        "callsign": "string",
        "kind": "string",
        "value": "string",  # may be float or list of strings. Represent all as a string.
        "agent": "string",
        "text_clearance": "string",
        "text_pilot_response": "string",
        "voice_clearance": "string",
        "voice_pilot_response": "string",
        "sector": "object",  # list[str]
    }

    sectors_dtypes: ClassVar[dict[str, str]] = {
        "datetime": "datetime64[us]",
        "sectors_configuration": "object",  # list of lists of (possibly bandboxes) sectors
    }

    incomm_dtypes: ClassVar[dict[str, str]] = {
        "datetime": "datetime64[us]",
        "callsign": "string",
        "sector_name": "string",
    }

    coord_dtypes: ClassVar[dict[str, str]] = {
        "callsign": "string",
        "from_sector": "string",  # e.g. "02". Also sectors outside AC may not be ints
        "to_sector": "string",  # e.g. "02". Also sectors outside AC may not be ints
        "fl": "float64",
        "fix": "string",
        "direction": "string",
        "level_by": "bool",
        "level_by_details": "string",
        "secondary_coord_conditions": "string",
        "datetime": "datetime64[us]",
    }

    ac_internals_dtypes: ClassVar[dict[str, str]] = {
        "callsign": "string",
        "datetime": "datetime64[us]",
        "rate_of_turn": "float64",
        "operation_params": "object",
        "controllable": "bool",
        "simulated": "bool",
        "current_sector": "string",
        "previous_sector": "string",
        "percentile_rank_dict": "object",
        "pilot_type": "string",
        "pilot_action_queue": "object",
        "predictor_params": "object",
        "wake_vortex": "string",
        "random_seed": "Int64",
        "heading_changing_to": "float64",
        "next_fix_index": "Int64",
        "vertical_speed": "float64",
        "cleared_fl": "float64",
        "cleared_mach": "float64",
        "cleared_cas": "float64",
        "cleared_vertical_speed": "float64",
        "cleared_heading": "float64",
        "cleared_on_route": "bool",
        "cleared_speed_action": "string",
        "cleared_vertical_speed_action": "string",
        "cleared_vertical_action": "string",
        "cleared_lateral_action": "string",
        "selected_fl": "float64",
        "selected_mach": "float64",
        "selected_cas": "float64",
        "selected_vertical_speed": "float64",
        "selected_heading": "float64",
        "selected_on_route": "bool",
        "selected_speed_action": "string",
        "selected_vertical_speed_action": "string",
        "selected_vertical_action": "string",
        "selected_lateral_action": "string",
        "route_current": "object",  # stored on flight_plan but not part of flight plan data
        "last_passed_filed_idx": "Int64",
        "last_passed_current_idx": "Int64",
        "squawk_ident_until": "float64",
    }

    ac_attribute_update_dtypes: ClassVar[dict[str, str]] = {
        "callsign": "string",
        "datetime": "datetime64[us]",
        "attribute_name": "string",
        "value": "object",
    }
