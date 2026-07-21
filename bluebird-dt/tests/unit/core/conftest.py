import copy
import datetime
import random
import string
import typing

import pytest

from bluebird_dt.core import (
    Action,
    Aircraft,
    Airspace,
    Airway,
    AirwayLeg,
    Area,
    ClearanceAndResponse,
    Coordination,
    Environment,
    Fixes,
    FlightPlan,
    HoldAtFixParameters,
    Pilot,
    Pos2D,
    Pos3D,
    Pos4D,
    Route,
    Sector,
    Volume,
    WindField,
)
from bluebird_dt.utility.supported_actions import SUPPORTED_ACTIONS
from bluebird_dt.utility.geo_helper import GeoHelper

def make_random_lat() -> float:
    """Return a latitude between +/- 90 rounded to 2 decimal places"""
    return round(random.uniform(-90.000, +90.000), 2)


def make_random_lon() -> float:
    """Return a longitude between +/- 180 rounded to 2 decimal places"""
    return round(random.uniform(-180.000, +180.000), 2)


def make_random_lat_lon() -> tuple[float, float]:
    """Return a lat lon pair rounded to 2 decimal places"""
    lat = make_random_lat()
    lon = make_random_lon()
    return lat, lon


def make_random_fl() -> int:
    """Return a flight level between 50 and 300"""
    return random.choice(range(50, 310, 10))


def make_random_min_max_fl() -> tuple[int, int]:
    """Returns two unique flight levels between 50 and 300"""
    fls = random.sample(range(50, 310, 10), 2)
    return min(fls), max(fls)


def make_random_min_max_lat_lon() -> tuple[float, float, float, float]:
    """Return two unique latitude and longitudes"""
    lats = [make_random_lat() for _ in range(10)]
    lons = [make_random_lon() for _ in range(10)]
    return min(lats), max(lats), min(lons), max(lons)


def make_random_time() -> float:
    """Return a random time since epoch between 1st Jan 2000 and 1st Jan 2020 to 6 decimal places"""
    start = 946684800
    end = 1577836800
    return round(random.uniform(start, end), 6)

def make_random_datetime() -> datetime.datetime:
    """
    Return a random datetime.
    """
    time = make_random_time()
    return datetime.datetime.fromtimestamp(time)

def make_random_string() -> str:
    """Return a sector name 6 random characters long with an uppercase first letter"""
    return str.capitalize("".join(random.choices(string.ascii_lowercase, k=6)))

def make_random_dict() -> dict[str, str]:
    """Return an dictionary with meaningless key, value pair"""
    return {make_random_string(): make_random_string()}

def make_random_heading() -> float:
    """Return a random heading between -180 and +180 to 2 decimal places"""
    return round(random.uniform(-180.00, 180.00), 2)

def make_random_callsign() -> str:
    """Return a random callsign AIRXYZ where XYZ are digits"""
    return "AIR" + "".join(random.choice(string.digits) for _ in range(3))

def make_random_fix_name() -> str:
    """Return a fix name made of three or 5 random uppercase letters"""
    return "".join(random.choices(string.ascii_uppercase, k=random.choice([3, 5])))

def make_random_sector_name() -> str:
    """Return a sector name 6 random characters long with an uppercase first letter"""
    return str.capitalize("".join(random.choices(string.ascii_lowercase, k=6)))

def make_random_date_string() -> str:
    """
    Return a random date as a string
    """
    time = make_random_time()
    return datetime.datetime.fromtimestamp(time).strftime("%Y%m%d")

def make_random_ufid(callsign: str | None = None) -> str:
    """Return a random UFID string for a given callsign"""
    if callsign is None:
        callsign = make_random_callsign()
    the_date = make_random_date_string()
    return "-".join([the_date, str(random.randint(10000, 99999)), callsign])

def make_random_squawk() -> str:
    """Return a random secondary surveillance radar (squawk) code"""
    return str(random.randint(0, 7777))

def make_random_fixes() -> Fixes:
    """Return a 2 to 9 random fixes"""
    names = [make_random_fix_name() for _ in range(random.choice(range(2, 10)))]
    return Fixes({name: make_posnd(2) for name in names}, {name: random.choice([True, False]) for name in names})


def make_random_route() -> Route:
    """Return a Route with different 'current' and 'filed' values"""
    return Route(
        current=[make_random_fix_name() for _ in range(random.choice(range(3, 8)))],
        filed=[make_random_fix_name() for _ in range(random.choice(range(3, 8)))],
    )

def make_random_sector() -> Sector:
    """Return a random Sector made of two adjacent volumes"""
    volume_1, volume_2 = make_random_adjacent_volumes()
    area_of_responsibility = make_random_volume()
    cva, cvb = make_random_adjacent_volumes()
    conditional_volume_dict = {"A_B": cva, "E_F": cvb}
    return Sector([volume_1, volume_2], [area_of_responsibility], conditional_volume_dict)

def make_random_airspace() -> Airspace:
    """Return an Airspace object. Note there is no correlation between the fixes and sector positions"""
    sector = make_random_sector()
    sector_name = make_random_sector_name()
    fixes = make_random_fixes()
    airway = make_random_airway(fixes)
    airways = {airway.identifier: airway}
    return Airspace(sectors={sector_name: sector}, fixes=fixes, airways=airways)

def make_random_aircraft_type() -> str:
    """Return a random aircraft type"""
    return "".join([random.choice(["A", "B", "C"]), str(random.randint(100, 999))])


def make_random_wake_vortex() -> str:
    """Return 'H' or 'M'"""
    return random.choice(["H", "M"])


def make_random_cas() -> float:
    """Return a random value for the cleared airspeed"""
    return round(random.uniform(300, 500), 2)

def make_random_mach() -> float:
    """Return a random value for the mach"""
    return round(random.uniform(0.8, 1.2), 4)


def make_random_vertical_speed() -> float:
    """Return a random float for the vertical speed"""
    return round(random.uniform(100, 200), 2)

def make_random_ground_speed() -> int:
    """Return a float for the ground speed"""
    return int(random.uniform(300, 500))

def make_random_ground_track_angle() -> float:
    """Return an angle (east of north) between 0 and 360 degrees"""
    return round(random.uniform(0.000, 360.00), 2)

def make_random_milcivil() -> typing.Literal["C", "M"]:
    """Randomly return 'M' or 'C', the allowed milcivil strings"""
    return random.choice(["C", "M"])

def make_random_intention_code() -> str:
    """Return a non-null intention code"""
    return random.choice(["D1", "C2", "V", "JJ", "D3", "E", "FF", "H", "NX", "AS", "BB", "GD", "EB"])

def make_random_airway(fixes: Fixes | None = None) -> Airway:
    """Returns a random airway"""

    if fixes is None:
        fixes = make_random_fixes()

    n_fixes = random.randint(2, len(fixes.places))

    return Airway.from_list_of_fixes(
        identifier="L1",
        fix_names=random.sample(list(fixes.places.keys()), n_fixes),
        lower_limit_fl=0,
        upper_limit_fl=660,
        fixes=fixes,
    )

def make_random_airwayleg(fixes: Fixes | None = None) -> AirwayLeg:
    """Returns a airway leg"""

    if fixes is None:
        fixes = make_random_fixes()

    # generate two random fixes
    p0_identifier, p1_identifier = random.sample(list(fixes.places.keys()), 2)
    p0, p1 = fixes.places[p0_identifier], fixes.places[p1_identifier]

    # generate two random flight levels and order them
    lower_limit_fl, upper_limit_fl = (make_random_fl(), make_random_fl())
    if lower_limit_fl > upper_limit_fl:
        lower_limit_fl, upper_limit_fl = upper_limit_fl, lower_limit_fl
    elif lower_limit_fl == upper_limit_fl:
        upper_limit_fl = upper_limit_fl + 50
    return AirwayLeg(
        upper_limit_fl=upper_limit_fl,
        lower_limit_fl=lower_limit_fl,
        p0=p0,
        p0_identifier=p0_identifier,
        p1=p1,
        p1_identifier=p1_identifier,
    )

def make_random_message() -> str:
    """Return a random heading between -180 and +180 to 2 decimal places"""
    charset = string.ascii_letters + string.digits + ' ' # A–Z, a–z, 0–9
    return ''.join(random.choices(charset, k=random.randint(1, 30)))

def make_random_flight_plan() -> FlightPlan:
    """Return a random flight plan consisting of random coordinations, routes, and times"""
    route = make_random_route()
    sector_crossing_seq = str([make_random_sector_name() for _ in range(3)])
    unexpanded_route = " ".join(route.filed[1:-1]).strip()

    return FlightPlan(
        route=route,
        unexpanded_route=unexpanded_route,
        origin=make_random_sector_name(),
        dest=make_random_sector_name(),
        milcivil=make_random_milcivil(),
        sector_crossing_seq=sector_crossing_seq,
        requested_flight_level=make_random_fl(),
        filed_true_airspeed=make_random_ground_speed(),
        intention_code=make_random_intention_code(),
        assigned_squawk=make_random_squawk(),
        start_datetime=make_random_datetime(),
        end_datetime=make_random_datetime(),
    )

def make_random_area() -> Area:
    """Return an n-sided irregular polygon centred on a random location"""
    n = random.choice([5, 6, 7, 8, 9])
    initial_headings = [round(random.uniform(10.0, 350.0), 4) for _ in range(n)]
    headings = sorted(initial_headings)
    origin = make_random_lat_lon()
    gh = GeoHelper(origin)
    boundary = []
    for heading in headings:
        lon, lat, _ = gh.forward(x=origin[1], y=origin[0], z=0, distance=random.uniform(15.0, 20.0), heading=heading)
        boundary.append(Pos2D(lat, lon))
    return Area(boundary)

def make_random_volume() -> Volume:
    """Return a random volume"""
    area = make_random_area()
    min_fl, max_fl = make_random_min_max_fl()
    sector_name = make_random_sector_name()
    description = make_random_sector_name()
    airspace_id = "A_B"
    return Volume(
        area=area,
        min_fl=min_fl,
        max_fl=max_fl,
        sector_name=sector_name,
        description=description,
        airspace_id=airspace_id,
    )

def make_random_uniform_windfield() -> WindField:
    """Return a random uniform wind field"""
    min_lat, max_lat, min_lon, max_lon = make_random_min_max_lat_lon()
    n_grid_points = random.choice(range(10, 60, 10))
    wind_speed = random.uniform(20, 70)
    wind_direction = make_random_ground_track_angle()
    return WindField.uniform(
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        no_grid_points=n_grid_points,
    )

def make_random_adjacent_areas() -> tuple[Area, Area]:
    """Return two irregular Areas with adjacent edges"""
    origin = make_random_lat_lon()
    gh = GeoHelper(origin)

    areas = []
    for min_head, max_head in [(0, 180), (180, 360)]:
        n = random.choice([2, 3, 4])
        initial_headings = [round(random.uniform(min_head + 10, max_head - 10), 4) for _ in range(n)] + [
            min_head,
            max_head,
        ]
        headings = sorted(initial_headings)
        boundary = []
        for heading in headings:
            lon, lat, _ = gh.forward(x=origin[1], y=origin[0], z=0, distance=10.0, heading=heading)
            boundary.append(Pos2D(lat, lon))
        boundary.append(Pos2D(origin[0], origin[1]))

        areas.append(Area(boundary))

    return areas[0], areas[1]


def make_random_adjacent_volumes() -> tuple[Volume, Volume]:
    """Return two irregular Volumes with adjacent edges"""

    area_1, area_2 = make_random_adjacent_areas()
    min_fl, max_fl = make_random_min_max_fl()

    volume_1 = Volume(area=area_1, min_fl=min_fl, max_fl=max_fl)
    volume_2 = Volume(area=area_2, min_fl=min_fl, max_fl=max_fl)

    return volume_1, volume_2

def make_random_rate_of_turn() -> float:
    """Return a random float - range is pure guesswork"""
    return round(random.uniform(10.00, 20.00), 2)

def make_random_coordination(callsign=None, the_datetime=None) -> Coordination:
    """Return a coordination object with random parameters"""

    if callsign is None:
        callsign = make_random_callsign()

    if the_datetime is None:
        the_datetime = make_random_datetime()

    return Coordination(
        callsign=callsign,
        from_sector=make_random_sector_name(),
        to_sector=make_random_sector_name(),
        fl=make_random_fl(),
        fix=make_random_fix_name(),
        direction=random.choice(["Horizontal", "Up", "Down"]),
        level_by=True,
        level_by_details={make_random_fix_name(): make_random_fl()},
        secondary_coord_conditions=make_random_string(),
        the_datetime=the_datetime,
    )

def make_random_pilot(callsign: str | None = None) -> Pilot:
    """Return a random pilot with either a specified or random callsign"""
    if callsign is None:
        callsign = make_random_callsign()
    return Pilot(callsign=callsign)


def make_random_aircraft(callsign: str | None = None) -> Aircraft:
    """Return an aircraft where all parameters and attributes are not null"""
    lat, lon = make_random_lat_lon()
    fl = make_random_fl()
    heading = make_random_heading()
    flight_plan = make_random_flight_plan()
    if callsign is None:
        callsign = make_random_callsign()
    selected_fl = make_random_fl()
    ufid = make_random_ufid(callsign)
    rate_of_turn = make_random_rate_of_turn()
    aircraft_type = make_random_aircraft_type()
    operation_params = make_random_dict()
    current_sector = make_random_sector_name()
    previous_sector = make_random_sector_name()
    random_seed = random.randint(1, 100)
    pilot = make_random_pilot(callsign)
    squawk = make_random_squawk()
    wake_vortex = make_random_wake_vortex()
    heading_changing_to = make_random_heading()
    squawk_ident_until = make_random_time()

    cleared_cas = make_random_cas()
    cleared_mach = make_random_mach()
    cleared_vertical_speed = make_random_vertical_speed()
    vertical_speed = 0.0  # level flight
    ground_speed = make_random_ground_speed()
    ground_track_angle = make_random_ground_track_angle()

    aircraft = Aircraft(
        lat=lat,
        lon=lon,
        fl=fl,
        heading=heading,
        flight_plan=flight_plan,
        callsign=callsign,
        selected_fl=selected_fl,
        ufid=ufid,
        rate_of_turn=rate_of_turn,
        aircraft_type=aircraft_type,
        operation_params=operation_params,
        controllable=True,
        simulated=True,
        current_sector=current_sector,
        random_seed=random_seed,
        pilot=pilot,
        squawk=squawk,
        wake_vortex=wake_vortex,
        last_passed_filed_idx=0,
        last_passed_current_idx=0,
        squawk_ident_until=squawk_ident_until,
    )

    aircraft.speed_tas = cleared_cas  # based on what happens in linear predictor
    aircraft.vertical_speed = vertical_speed
    aircraft.heading_changing_to = heading_changing_to
    aircraft.next_fix_index = 1
    aircraft.ground_speed = ground_speed
    aircraft.ground_track_angle = ground_track_angle
    aircraft.predictor_params = make_random_dict()

    aircraft.cleared_instructions.fl = fl
    aircraft.cleared_instructions.heading = heading
    aircraft.cleared_instructions.cas = cleared_cas
    aircraft.cleared_instructions.mach = cleared_mach
    aircraft.cleared_instructions.vertical_speed = cleared_vertical_speed
    aircraft.cleared_instructions.on_route = True
    aircraft.cleared_instructions.speed_action = make_action("change_cas_to")
    aircraft.cleared_instructions.vertical_speed_action = make_action("change_vertical_speed_to")
    aircraft.cleared_instructions.vertical_action = make_action("change_flight_level_to")
    aircraft.cleared_instructions.lateral_action = make_action("change_heading_to")

    aircraft.selected_instructions = copy.deepcopy(aircraft.cleared_instructions)

    # previous sector is an attribute that's updated from current_sector
    # when outcommed. It usually shouldn't be set explicitly,
    # but can be via the protected attribute _previous_sector
    aircraft._previous_sector = previous_sector

    return aircraft

def make_random_environment() -> Environment:
    """Return a random environment object. Sectors and aircraft position are all random"""
    time = make_random_time()
    airspace = make_random_airspace()
    aircraft = {callsign: make_random_aircraft(callsign) for callsign in [make_random_callsign() for _ in range(5)]}
    wind_field = make_random_uniform_windfield()

    # make 10 random coordinations, half for the same aircraft
    callsign = make_random_callsign()
    coord_for_aircraft = [make_random_coordination(callsign) for i in range(5)]
    coords_random_aircraft = [make_random_coordination() for i in range(5)]

    coordinations = coord_for_aircraft + coords_random_aircraft

    return Environment(
        time=time,
        airspace=airspace,
        aircraft=aircraft,
        wind_field=wind_field,
        forecast_wind_field=wind_field,
        coordinations=coordinations,
    )

@typing.overload
def make_posnd(dimension: typing.Literal[2]) -> Pos2D: ...


@typing.overload
def make_posnd(dimension: typing.Literal[3]) -> Pos3D: ...


@typing.overload
def make_posnd(dimension: typing.Literal[4]) -> Pos4D: ...


def make_posnd(dimension: int) -> Pos2D | Pos3D | Pos4D:
    """Return a PosND instance N = 2, 3, or 4"""
    match dimension:
        case 2:
            pos = Pos2D(*make_random_lat_lon())

        case 3:
            pos = Pos3D(*make_random_lat_lon(), make_random_fl())

        case 4:
            pos = Pos4D(*make_random_lat_lon(), make_random_fl(), make_random_time())

        case _:
            raise ValueError("Dimension must be 2, 3, or 4")

    return pos

def make_full_range_of_actions() -> list[Action]:
    """Return a list of one of each kind of supported actions"""
    supported_actions = [action_kind for action_list in SUPPORTED_ACTIONS.values() for action_kind in action_list]
    return [make_action(kind) for kind in supported_actions]


def make_action(kind: str) -> Action:
    """Return an Action with correct type matching the kind"""
    callsign = make_random_callsign()
    value = None
    if kind == "route_direct_to":
        value = random.choice([make_random_fix_name(), [make_random_fix_name()]])
    if kind == "route_direct_to,hold_at_location":
        value = HoldAtFixParameters(fix=make_random_fix_name(), hold_orientation_deg=make_random_heading() % 360.0)
    if kind == "change_heading_to":
        value = make_random_heading()
    if kind == "change_heading_to_by_direction":
        value = (make_random_heading(), random.choice(['left', 'right', 'shortest']))
    if kind == "change_heading_by":
        value = make_random_heading()
    if kind == "maintain_current_heading":
        value = 0
    if kind == "change_flight_level_to":
        value = make_random_fl()
    if kind == "change_flight_level_by":
        value = random.randint(-50, 50)
    if kind == "descend_when_ready,level_by_fix":
        value = (make_random_fl(), make_random_fix_name())
    if kind == "descend_now,level_by_fix":
        value = (make_random_fl(), make_random_fix_name())
    if kind == "change_cas_to":
        value = round(random.uniform(300, 500), 2)
    if kind == "change_mach_to":
        value = round(random.uniform(0.8, 1.2), 4)
    if kind == "change_vertical_speed_to":
        value = round(random.uniform(200, 400), 2)
    if kind == "outcomm" in kind:
        value = make_random_sector_name()
    if kind == "squawk_ident":
        value = ""
    if kind == "set_squawk":
        value = make_random_squawk()
    if kind == "using_speed_limit":
        value = "True"
    if kind == "route_segment":
        value = make_random_fix_name() + "," + make_random_fix_name() + "," + make_random_fix_name()
    if kind == "route_turn_segment":
        value = make_random_fix_name() + "," + make_random_fix_name() + "," + make_random_fix_name()
    if kind == "heading_segment":
        value = make_random_heading()
    if kind == "heading_turn_segment":
        value = make_random_heading()
    if kind == "message":
        value = make_random_message()
    agent = random.choice(["Smith", "Bond", "Powers", "J"])
    clearance = " ".join((callsign, kind, str(value), agent))
    pilot_response = " ".join((kind, str(value), agent, callsign))
    sector = random.choice(
        ["Andromeda Galaxy", "Antennae Galaxies", "Backward Galaxy", "Bear Paw Galaxy", "Black Eye Galaxy"]
    )

    return Action(
        callsign=callsign,
        kind=kind,
        value=value,
        agent=agent,
        text_representation=ClearanceAndResponse(clearance=clearance,pilot_response=pilot_response),
        voice_representation=ClearanceAndResponse(clearance=clearance,pilot_response=pilot_response),
        sector=[sector],
    )

@pytest.fixture
def random_area():
    return make_random_area


@pytest.fixture
def build_random_full_instance_or_json_list(
    request,
) -> tuple[
    list[
        Action
        | Aircraft
        | Airspace
        | Airway
        | AirwayLeg
        | Area
        | Coordination
        | Environment
        | Fixes
        | FlightPlan
        | Pos2D
        | Pos3D
        | Pos4D
        | Route
        | Sector
        | Volume
        | WindField
    ],
    str,
    type,
]:
    """Build a list of random full (no 'None') instances or json of the specified class"""
    class_str, class_type, inst_or_json = request.param
    n_instances = 3

    rnd_instances = {
        "Aircraft": make_random_aircraft,
        "Area": make_random_area,
        "Airspace": make_random_airspace,
        "Airway": make_random_airway,
        "AirwayLeg": make_random_airwayleg,
        "Coordination": make_random_coordination,
        "Environment": make_random_environment,
        "Fixes": make_random_fixes,
        "FlightPlan": make_random_flight_plan,
        "Pos2D": lambda: make_posnd(dimension=2),
        "Pos3D": lambda: make_posnd(dimension=3),
        "Pos4D": lambda: make_posnd(dimension=4),
        "Route": make_random_route,
        "Sector": make_random_sector,
        "Volume": make_random_volume,
        "WindField": make_random_uniform_windfield,
    }

    # Build the instances
    if class_str in rnd_instances:
        op_list = [rnd_instances[class_str]() for _ in range(n_instances)]
    elif class_str == "Action":
        op_list = make_full_range_of_actions()
    else:
        raise ValueError(f"Unknown class type {class_str}")

    # Return appropriate list
    if inst_or_json == "inst":
        return op_list, (class_str, class_type)  # type: ignore
    if inst_or_json == "json":
        return [inst.to_json() for inst in op_list], (class_str, class_type)  # type: ignore
    raise ValueError("Must specify either 'inst' or 'json'")

@pytest.fixture
def example_coordination_json() -> str:
    return """
                     {
                         "callsign": "AIR0",
                         "from_sector": null,
                         "to_sector": "sector_i",
                         "fl": 200,
                         "fix": "SPIRIT",
                         "direction": "Horizontal",
                         "level_by": false,
                         "level_by_details": null,
                         "secondary_coord_conditions": null,
                         "datetime": "1970-01-01 00:02:00"
                     }
    """


@pytest.fixture
def example_coordination_dict_json() -> str:
    return """
        {
            "coordinations": [
                {
                    "callsign": "AIR0",
                    "from_sector": null,
                    "to_sector": "sector_i",
                    "fl": 200,
                    "fix": "SPIRIT",
                    "direction": "Horizontal",
                    "level_by": false,
                    "level_by_details": null,
                    "secondary_coord_conditions": null,
                    "datetime": "1970-01-01 00:02:00"
                }
            ]
        }
    """

@pytest.fixture
def example_flightplan() -> FlightPlan:
    flightplan = FlightPlan.from_json(
        """
        {
            "route": {
                "current": ["SPIRIT", "AIR", "EARTH", "FIRE"],
                "filed": ["SPIRIT", "AIR", "EARTH", "FIRE"]
            },
            "unexpanded_route": "SPIRIT-AIR-EARTH-FIR",
            "origin": "EGPF",
            "dest": "EGSS",
            "milcivil": "C",
            "sector_crossing_seq": "sector2",
            "requested_flight_level": 340,
            "filed_true_airspeed": 400,
            "example_squawk": "1234"
        }
        """
    )
    assert flightplan is not None
    return flightplan

@pytest.fixture
def random_sector() -> Sector:
    return make_random_sector

@pytest.fixture
def example_airwayleg() -> AirwayLeg:
    return make_random_airwayleg()

@pytest.fixture
def example_airway() -> Airway:
    return make_random_airway()

@pytest.fixture
def example_aircraft_pilot() -> str:
    return """
    {
    "flight_plan": {
        "route": {"current": ["ABC", "DEF", "GHI"], "filed": ["ABC", "DEF", "GHI"]},
        "sectors": [
            "first_sector_name",
            "second_sector_name"
        ]
    },
    "callsign": "AIR0",
    "lat": 51.4702,
    "lon": -0.4479,
    "fl": 120,
    "heading": 249.68,
    "speed": 200,
    "vertical_speed": 0,
    "aircraft_type": "DEFAULT",
    "pilot": {
        "pilot_type": "Pilot",
        "callsign": "AIR0"
        }
    }
    """


@pytest.fixture
def example_aircraft_json_no_pilot() -> str:
    return """
    {
    "flight_plan": {
        "route": {"current": ["ABC", "DEF", "GHI"], "filed": ["ABC", "DEF", "GHI"]},
        "sectors": [
            "first_sector_name",
            "second_sector_name"
        ]
    },
    "callsign": "AIR0",
    "lat": 51.4702,
    "lon": -0.4479,
    "fl": 120,
    "heading": 249.68,
    "speed": 200,
    "vertical_speed": 0,
    "aircraft_type": "DEFAULT"
    }
    """

@pytest.fixture
def random_pos2d() -> Pos2D:
    return lambda: make_posnd(2)

@pytest.fixture
def random_pos3d() -> Pos3D:
    return lambda: make_posnd(3)

@pytest.fixture
def random_pos4d() -> Pos4D:
    return lambda: make_posnd(4)

@pytest.fixture
def random_adjacent_volumes() -> list[Volume]:
    return make_random_adjacent_volumes

@pytest.fixture(scope="function")
def simple_airspace():
    # Return a fresh Airspace per test. Some tests mutate sectors and bandbox
    # configuration in place, so this fixture should not be shared or widened.
    coords = [
        Pos2D(lat=0.0, lon=0.0),
        Pos2D(lat=0.0, lon=1.0),
        Pos2D(lat=1.0, lon=1.0),
        Pos2D(lat=1.0, lon=0.0),
        Pos2D(lat=0.0, lon=0.0),
    ]
    sector = Sector([Volume(Area(coords), 0, 100)])
    fixes = Fixes({"FIX1": Pos2D(lat=0.5, lon=0.5)})
    return Airspace({"S1": sector}, fixes)
