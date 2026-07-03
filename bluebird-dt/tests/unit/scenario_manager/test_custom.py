from collections import defaultdict
from bluebird_dt.core import Pos3D
from bluebird_dt.manager import EnvironmentManager
from bluebird_dt.simulator import Simulator

import pytest
import numpy as np
import pandas as pd
from datetime import timedelta

from bluebird_dt.scenario_manager import Custom
from bluebird_dt.predictor import SimplePredictor

@pytest.mark.parametrize(
    "lateral_offset",
    [
        None,
        (0, 10),
    ],
)
def test_init_exceptions(lateral_offset, generate_i):
    """
    Test ValueError is raised when class is initiated with invalid inputs.
    """

    airspace, routes = generate_i

    with pytest.raises(ValueError):
        Custom(-1, lateral_offset=lateral_offset, airspace=airspace, routes=routes)

    with pytest.raises(ValueError):
        Custom(1, balance=[1], lateral_offset=lateral_offset, airspace=airspace, routes=routes)

    with pytest.raises(ValueError):
        Custom(1, balance=[0.25, 0.25, 0.25, 0.25], lateral_offset=lateral_offset, airspace=airspace, routes=routes)

    with pytest.raises(ValueError):
        Custom(1, speed_range=[300], lateral_offset=lateral_offset, airspace=airspace, routes=routes)

    with pytest.raises(ValueError):
        Custom(1, speed_range=[300, 310, 320], lateral_offset=lateral_offset, airspace=airspace, routes=routes)

    with pytest.raises(ValueError):
        Custom(1, time_entry_gap=-1, lateral_offset=lateral_offset, airspace=airspace, routes=routes)


@pytest.mark.parametrize(
    "airspace_routes, lateral_offset",
    [
        ("generate_i", None),
        ("generate_x", None),
        ("generate_y", None),
        ("generate_thunderdome", None),
        ("generate_i", (0, 10)),
        ("generate_x", (0, 10)),
        ("generate_y", (0, 10)),
    ],
)
def test_all_airspaces(airspace_routes, lateral_offset, request):
    """
    Test generator works for all available Airspaces.

    For each num_aircraft and Airspace, check:
       - the correct number of Aircraft is generated
       - the entry/exit Coordination FLs are within the Airspace limits at entry/exit
    """

    airspace, routes = request.getfixturevalue(airspace_routes)
    sector_name = list(airspace.sectors.keys())[0]
    volume = airspace.sectors[sector_name].volumes[0]

    for num_aircraft in [1, 2, 5, 10, 100]:
        env_manager = Custom(num_aircraft, lateral_offset=lateral_offset, airspace=airspace, routes=routes).create_env_manager()

        radar_events_df = env_manager.event_handler.radar_df
        ac_internal_events_df = env_manager.event_handler.aircraft_internals_df
        coord_df = env_manager.event_handler.coordination_df

        # test number of events matches number of aircraft
        assert len(radar_events_df) == num_aircraft
        assert len(ac_internal_events_df) == num_aircraft

        # test number of coordination is double the number of aircraft (exit/entry per aircraft)
        assert len(coord_df) == num_aircraft * 2

        # check coordinations are in the environment and within the sector upper/lower fl
        assert len(env_manager.environment.coordinations.values()) > 0
        for coord in env_manager.environment.coordinations.values():
            assert volume.min_fl <= coord.fl <= volume.max_fl
            assert volume.min_fl <= coord.fl <= volume.max_fl


@pytest.mark.parametrize(
    "lateral_offset",
    [
        None,
        (0, 10),
    ],
)
def test_repeat(generate_i, lateral_offset):
    """
    Check generator works when called multiple times.
    """

    airspace, routes = generate_i
    gen = Custom(5, lateral_offset=lateral_offset, airspace=airspace, routes=routes)

    for _ in range(5):
        env_manager = gen.create_env_manager()

        radar_events_df = env_manager.event_handler.radar_df
        ac_internal_events_df = env_manager.event_handler.aircraft_internals_df
        coord_df = env_manager.event_handler.coordination_df

        # test number of events matches number of aircraft
        assert len(radar_events_df) == 5
        assert len(ac_internal_events_df) == 5

        # test number of coordination is double the number of aircraft (exit/entry per aircraft)
        assert len(coord_df) == 5 * 2


@pytest.mark.parametrize(
    "lateral_offset",
    [
        None,
        (0, 10),
    ],
)
def test_speed_range(generate_i, lateral_offset):
    """
    Test the speed-range parameter.
    """

    airspace, routes = generate_i

    # uses default speed when not provided
    env_manager = Custom(10, lateral_offset=lateral_offset, airspace=airspace, routes=routes).create_env_manager()

    radar_events_df = env_manager.event_handler.radar_df

    for _, row in radar_events_df.iterrows():
        assert row.speed_tas == 400.0

    # choose Tactically from speed range, test 10 times
    for _ in range(10):
        env_manager = Custom(10, speed_range=[100, 300], airspace=airspace, routes=routes).create_env_manager()

        radar_events_df = env_manager.event_handler.radar_df
        for _, row in radar_events_df.iterrows():
            assert row.speed_tas is not None
            assert row.speed_tas != 400.0
            assert row.speed_tas >= 100 and row.speed_tas <= 300


@pytest.mark.parametrize(
    "lateral_offset",
    [
        None,
        (0, 10),
    ],
)
def test_balance(generate_i, lateral_offset):
    """
    Test the balance parameter gives the right proportions.
    """

    airspace, routes = generate_i
    sector_name = list(airspace.sectors.keys())[0]

    # only overfliers
    environment = (
        Custom(10, balance=[0, 0, 1], lateral_offset=lateral_offset, airspace=airspace, routes=routes).create_env_manager().environment
    )
    for callsign in environment.aircraft:
        entry_coord = environment.entry_coordination(sector_name, callsign)
        exit_coord = environment.exit_coordination(sector_name, callsign)
        assert entry_coord is not None
        assert exit_coord is not None
        assert entry_coord.fl == exit_coord.fl

    # only descenders
    environment = (
        Custom(10, balance=[0, 1, 0], lateral_offset=lateral_offset, airspace=airspace, routes=routes).create_env_manager().environment
    )
    for callsign in environment.aircraft:
        entry_coord = environment.entry_coordination(sector_name, callsign)
        exit_coord = environment.exit_coordination(sector_name, callsign)
        assert entry_coord is not None
        assert exit_coord is not None
        assert entry_coord.fl > exit_coord.fl

    # only climbers
    environment = (
        Custom(10, balance=[1, 0, 0], lateral_offset=lateral_offset, airspace=airspace, routes=routes).create_env_manager().environment
    )
    for callsign in environment.aircraft:
        entry_coord = environment.entry_coordination(sector_name, callsign)
        exit_coord = environment.exit_coordination(sector_name, callsign)
        assert entry_coord is not None
        assert exit_coord is not None
        assert entry_coord.fl < exit_coord.fl


@pytest.mark.parametrize(
    "lateral_offset",
    [
        None,
        (0, 10),
    ],
)
def test_aircraft_entry_coordinations(generate_i, lateral_offset):
    """
    Test no two Aircraft have the same entry coordinations.
    """

    airspace, routes = generate_i
    sector_name = list(airspace.sectors.keys())[0]
    env_manager = Custom(100, lateral_offset=lateral_offset, airspace=airspace, routes=routes).create_env_manager()

    radar_events_df = env_manager.event_handler.radar_df
    coord_events_df = env_manager.event_handler.coordination_df

    # assert aircraft only appear once each in radar event dataframe
    assert len(radar_events_df) == len(radar_events_df.callsign.unique())

    # as only one event per callsign in radar_df, use this to get the start times
    rad_df = radar_events_df.reset_index(names="datetime").set_index("callsign")
    all_start_time_fix_fl_pairs = [
        (rad_df.loc[coord.callsign].iloc[0], coord.fix, coord.fl)
        for _, coord in coord_events_df.iterrows()
        if coord.to_sector == sector_name
    ]

    # check there is only ever one entry Coordination for any Fix,FL and time combination.
    assert len(all_start_time_fix_fl_pairs) == len(set(all_start_time_fix_fl_pairs))


def test_aircraft_entry_spawn_point(generate_i):
    """
    Test no two Aircraft with the same entry fix have the lateral
    same spawn point.
    """

    airspace, routes = generate_i
    em = Custom(100, lateral_offset=(3, 5), airspace=airspace, routes=routes).create_env_manager()

    entries = defaultdict(list)

    flight_plans = em.event_handler.flight_plan_df.groupby("callsign").last()

    # check there is a unique callsign per flight plan entry
    assert len(flight_plans) == len(em.event_handler.flight_plan_df)

    for _, radar_row in em.event_handler.radar_df.iterrows():
        start_fix = flight_plans.loc[radar_row.callsign].route_filed[0]
        entries[start_fix].append(radar_row)

    # check no two aircraft with the same entry fix have the same
    # lateral position (latitude and longitude)
    # note, flight level information is ignored.
    for fix in entries:
        rows = entries[fix]
        n_aircraft = len(rows)
        for i in range(0, n_aircraft - 1):
            for j in range(i + 1, n_aircraft):
                assert rows[i].lat != rows[j].lat and rows[i].lon != rows[j].lon


@pytest.mark.parametrize(
    "lateral_offset",
    [
        (1, 2),
        (4, 6),
        (0, 10),
    ],
)
def test_laterally_offset_start_pos(lateral_offset):
    """
    Test stochastic_start_pos generates a laterally offset start position by that
    for each aircraft, the starting Pos2D is within the correct offset range from 
    the starting fix.
    """
    
    sim = Custom.setup(num_aircraft=2, lateral_offset=lateral_offset, scenario_name="X-Sector")
    for aircraft in sim.manager.environment.aircraft.values():
        start_fix = aircraft.flight_plan.route.filed[0]
        start_fix_pos = sim.manager.environment.airspace.fixes.places[start_fix]
        aircraft_pos = aircraft.pos2d()
        assert lateral_offset[0] < start_fix_pos.distance(aircraft_pos) < lateral_offset[1]

def _convert_datetime_index_to_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper: event handler dataframes tend to be indexed by datetime in this codebase.
    Standardise to a df with a 'datetime' column.
    """
    if df.index.name == "datetime":
        return df.reset_index()
    # fallback: some implementations may not name the index
    if isinstance(df.index, pd.DatetimeIndex):
        return df.reset_index(names="datetime")
    return df

@pytest.mark.parametrize("lateral_offset", [None, (0, 10)])
def test_create_event_handler(generate_i, lateral_offset):
    """
    Check that create_event_handler:
    - returns an EventHandler with N aircraft radar events
    - returns 2N coordination events
    - each callsign has 2 coords exactly 1 second before its radar event time
    """

    airspace, routes = generate_i
    n = 20
    gen = Custom(n, lateral_offset=lateral_offset, airspace=airspace, routes=routes, start_time=0, random_seed=12345)

    eh = gen.create_event_handler()

    radar_df = _convert_datetime_index_to_column(eh.radar_df)
    coord_df = _convert_datetime_index_to_column(eh.coordination_df)

    assert len(radar_df) == n
    assert len(coord_df) == n * 2

    # Map callsign -> aircraft event datetime
    aircraft_time_by_callsign = {row.callsign: row.datetime for _, row in radar_df.iterrows()}
    assert len(aircraft_time_by_callsign) == n

    # Group coordinations by callsign and check timing
    coords_by_callsign = coord_df.groupby("callsign")

    for callsign, group in coords_by_callsign:
        assert callsign in aircraft_time_by_callsign
        assert len(group) == 2  # entry + exit
        aircraft_dt = aircraft_time_by_callsign[callsign]

        # Both coords should be exactly 1 second before aircraft time
        for _, coord_row in group.iterrows():
            assert coord_row.datetime == aircraft_dt - timedelta(seconds=1)


def test_to_simulator(generate_i):
    """
    Test the Custom.to_simulator() method creates the required components and sets the appropriate variables correctly.
    """

    airspace, routes = generate_i
    num_aircraft = 10

    simulator = Custom(
        num_aircraft=num_aircraft,
        airspace=airspace,
        routes=routes,
        speed_range=[100, 300]
    ).to_simulator(scenario_name="test-scenario",)

    assert isinstance(simulator.manager, EnvironmentManager)
    assert isinstance(simulator.scenario_manager, Custom)
    assert simulator.projection_centre is None
    assert isinstance(simulator, Simulator)

    assert simulator.scenario_name == "test-scenario"
    assert simulator.category == "Custom"

    radar_events_df = simulator.manager.event_handler.radar_df
    ac_internal_events_df = simulator.manager.event_handler.aircraft_internals_df
    coord_df = simulator.manager.event_handler.coordination_df

    # test number of events matches number of aircraft
    assert len(radar_events_df) == num_aircraft
    assert len(ac_internal_events_df) == num_aircraft

    # test number of coordination is double the number of aircraft (exit/entry per aircraft)
    assert len(coord_df) == num_aircraft * 2

    # Check speed range
    for _, row in radar_events_df.iterrows():
        assert row.speed_tas is not None
        assert row.speed_tas != 400.0
        assert row.speed_tas >= 100 and row.speed_tas <= 300

    # Check returned simulator works as expected
    simulator.evolve(6)
    assert len(simulator.manager.environment.aircraft) is not None

def test_create_env_manager(generate_i):
    """
    Test create_env_manager default behaviour and initialisation control by checking:
    - a SimplePredictor is created when no predictor is provided
    """
    airspace, routes = generate_i

    # When predictor is None, should create SimplePredictor(1.0, 2.0)
    em = Custom(3, airspace=airspace, routes=routes).create_env_manager(predictor=None)
    assert isinstance(em.predictor, SimplePredictor)

    assert len(em.environment.aircraft) > 0
    assert len(em.environment.coordinations.values()) > 0


def test_add_custom_aircraft_only(generate_i):
    """
    Test that we can create a scenario with fully user-specified aircraft only
    """
    airspace, routes = generate_i
    sm = Custom(0, airspace=airspace, routes=routes)
    sm.add_aircraft_with_coordinations(
        aircraft_start_time=10.0, 
        callsign="TEST-1", 
        pos=Pos3D(1,0,200),
        heading=180, 
        speed=350., 
        route=routes[0], 
        entry_fl=200, 
        exit_fl=400
    )
    sm.add_aircraft_with_coordinations(
        aircraft_start_time=20.0, 
        callsign="TEST-2", 
        pos=Pos3D(0,0,210),
        heading=0, 
        speed=350., 
        route=routes[-1], 
        entry_fl=200, 
        exit_fl=400
    )
    sim = sm.to_simulator()
    sim.evolve(6)
    assert len(sim.manager.environment.aircraft) == 1
    assert "TEST-1" in sim.manager.environment.aircraft
    sim.evolve(6)
    assert len(sim.manager.environment.aircraft) == 2
    assert "TEST-2" in sim.manager.environment.aircraft


def test_add_custom_aircraft_with_random_ones(generate_i):
    """
    Test that we can create a scenario with fully user-specified aircraft as
    well as randomly-generated aircraft
    """
    airspace, routes = generate_i
    sm = Custom(3, airspace=airspace, routes=routes)
    sm.add_aircraft_with_coordinations(
        aircraft_start_time=10.0, 
        callsign="TEST-1", 
        pos=Pos3D(1,0,200),
        heading=180, 
        speed=350., 
        route=routes[0], 
        entry_fl=200, 
        exit_fl=400
    )
    sm.add_aircraft_with_coordinations(
        aircraft_start_time=20.0, 
        callsign="TEST-2", 
        pos=Pos3D(0,0,210),
        heading=0, 
        speed=350., 
        route=routes[-1], 
        entry_fl=200, 
        exit_fl=400
    )
    sim = sm.to_simulator()
    # wait 1 minute for all aircraft to spawn
    sim.evolve(60)
    # should be 5 aircraft - 3 randomly generated and 2 user-specified.
    assert len(sim.manager.environment.aircraft) == 5
    assert "TEST-1" in sim.manager.environment.aircraft
    assert "TEST-2" in sim.manager.environment.aircraft

@pytest.mark.parametrize(
        "scenario_name", 
        ("I-Sector","Y-Sector", "X-Sector", "Xplus-Sector", "Springfield")
)
def test_sim_from_category(scenario_name):
    """
    Test that we can instantiate the simulator using "from_category"
    """
    s = Simulator.from_category("Custom", scenario_name)
    assert isinstance(s, Simulator)