# Dictionary containing the currently supported actions.
SUPPORTED_ACTIONS = {
    "vertical": [
        "change_flight_level_to",
        "change_flight_level_by",
        "descend_when_ready,level_by_fix",
        "descend_now,level_by_fix",
    ],
    "vertical_speed": ["change_vertical_speed_to"],
    "lateral": [
        "route_direct_to",
        "change_heading_to",
        "change_heading_to_by_direction",
        "change_heading_by",
        "maintain_current_heading",
        "intercept_radial",
    ],
    "speed": [
        "change_cas_to",
        "change_mach_to",
        "using_speed_limit",
    ],
    "outcomm": ["outcomm"],
    "transponder": ["set_squawk", "squawk_ident"],
    "system_lateral": [
        "route_segment",
        "route_turn_segment",
        "heading_segment",
        "heading_turn_segment",
        "track_segment",
        "fixed_radius_turn",
        "fixed_rate_turn",
    ],
    "message": ["message"],
}
