import copy
import os
import warnings
from datetime import datetime, timedelta
import tarfile
from typing import Any

import pandas as pd
import pytest

from bluebird_dt.airspace_generator.sector_i import SectorI
from bluebird_dt.core import Action
from bluebird_dt.events.event_dtypes import EventDtypes
from bluebird_dt.events.event_logger import EventLogger
from bluebird_dt.events.event_handler import EventHandler
from bluebird_dt.manager import EnvironmentManager
from bluebird_dt.predictor import RouteFollowPredictor, SimplePredictor, LinearPredictor
from bluebird_dt.scenario_manager import TwoAircraft
from bluebird_dt.simulator import Simulator

from bluebird_dt.simulator.simconfig import SaveConfig, SimulatorConfig
from bluebird_dt.utility.logging_utils import read_tar_csv_to_df, read_tar_json_to_dict, read_tar_parquet_to_df
from bluebird_dt.utility.paths import LOG_DIR

from tests.unit.logging.conftest import extract_first_row_from_df, query_row_from_df, pick_random_next_sector

@pytest.mark.parametrize(
    ("scenario_category", "scenario_name", "predictor_type", "fix_proximity", "projection_lon", "projection_lat"),
    [
        ("Springfield", "example-scenario", "LinearPredictor", 3, 0.2, 51),
        ("Springfield", "llm-scenario", "LinearPredictor", 5, -3, 23),
        ("Two Aircraft", "I-Sector", "SimplePredictor", 3, 0.2, 51),
        ("Two Aircraft", "Y-Sector", "SimplePredictor", 5, -3, 23),
        ("Springfield", "example-scenario", "RouteFollowPredictor", 3, 0.2, 51),
        ("Springfield", "llm-scenario", "RouteFollowPredictor", 5, -3, 23),
        ("Two Aircraft", "I-Sector", "RouteFollowPredictor", 3, 0.2, 51),
        ("Two Aircraft", "Y-Sector", "RouteFollowPredictor", 5, -3, 23),
    ],
)
def test_config_logged(
    scenario_category: str,
    scenario_name: str,
    predictor_type: str,
    fix_proximity: float,
    projection_lon: float,
    projection_lat: float,
    unique_log_name: str,
):
    """
    Test predictor type and parameters as well as simulator projection_centre are logged and used in replay
    """
    log_name = f"{unique_log_name}_test_predictor_logged"

    # create predictor
    if predictor_type == "SimplePredictor":
        predictor = SimplePredictor(dt=1, fix_proximity_threshold=fix_proximity)
    elif predictor_type == "RouteFollowPredictor":
        predictor = RouteFollowPredictor(dt=1, fix_proximity_threshold=fix_proximity, fixes=None)
    elif predictor_type == "LinearPredictor":
        predictor = LinearPredictor(dt=1, fix_proximity_threshold=fix_proximity)
    else:
        raise ValueError("Unknown predictor type")

    sim = Simulator.from_category(category=scenario_category, scenario_name=scenario_name, predictor=predictor, log_filename=log_name)

    # set the projection centre
    projection_centre = (projection_lon, projection_lat)
    sim.projection_centre = projection_centre

    sim.evolve(6)
    sim.save()
    del sim

    # check predictor is logged correctly
    tar_file_name = os.path.join(LOG_DIR, f"{log_name}.tar.gz")
    try:
        with tarfile.open(tar_file_name, "r:gz") as tar:
            data = read_tar_json_to_dict(tar, "config")
            config = SaveConfig.model_validate(data)

            assert config.environment_manager.predictor.predictor_type == predictor_type
            assert config.environment_manager.predictor.fix_proximity == fix_proximity

            # check simulator is logged correctly. Tuples are deserialised as lists
            assert config.simulator.projection_centre == projection_centre

            # check scenario category and name is logged correctly.
            # scenario_category is deprecated but we still test it for backwards compatibility
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                assert config.scenario_category == scenario_category
            assert config.scenario_name == scenario_name
    finally:
        os.remove(tar_file_name)

def test_log_environment(unique_log_name: str):
    """
    Test that the environment gets logged when a simulation is saved by checking the relevant generated files.
    """
    scenario_category = "Springfield"
    scenario_name = "example-scenario"
    sim = Simulator.from_category(category=scenario_category, scenario_name=scenario_name, log_filename=unique_log_name)
    sim.save()
    del sim

    # Open log files for each category that should be logged.
    # Note that wind, forecast and clearances are not checked as they need to be generated
    tar_file_name = os.path.join(LOG_DIR, f"{unique_log_name}.tar.gz")
    try:
        with tarfile.open(tar_file_name, "r:gz") as logfile:
            config = read_tar_json_to_dict(logfile, "config")
            assert config['scenario_category'] == scenario_category
            assert config['scenario_name'] == scenario_name
            log_file_names = ["radar", "flight_plan", "sectors", "coordination", "incomm", "ac_internals", "fixes"]

            for name in log_file_names:
                csv = read_tar_csv_to_df(logfile, name)
                assert len(csv) > 0
                df = read_tar_parquet_to_df(logfile, name)
                assert df.empty is False
    finally:
        os.remove(tar_file_name)

def test_is_same_flight_logs(flight_plan: dict[str, Any]):
    """
    Test that is_same_flight_logs correctly identifies identical vs non identical logs
    """
    later_flight_plan = copy.deepcopy(flight_plan)
    # Test identical logs
    event_logger = EventLogger()
    assert event_logger.is_same_flight_logs(flight_plan, later_flight_plan) == True

    # Test change to a mutable property
    later_flight_plan['callsign'] = 'ABC123'
    assert event_logger.is_same_flight_logs(flight_plan, later_flight_plan) == True

    # Test a change to a non mutable property
    later_flight_plan['milcivil'] = 'M'
    assert event_logger.is_same_flight_logs(flight_plan, later_flight_plan) == False

def test_log_clearances(unique_log_name: str):
    """
    Test that clearances can be logged and retrieved as dataframes, from csv file and from parquet file
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", log_filename=unique_log_name)
    em = sim.manager

    callsign = sorted(em.environment.aircraft.keys())[0]
    kind = "change_flight_level_to"
    value = 250
    new_action = Action(callsign, kind, value)
    em.event_logger.log_clearances(em.environment.datetime, [new_action])
    clearance_df = em.event_logger.clearance_log_as_df()
    sim.save()
    assert clearance_df.empty is False
    logged_action = clearance_df.iloc[0]
    assert logged_action["callsign"] == callsign
    assert logged_action["kind"] == kind
    assert logged_action["value"] == value

    del sim

    tar_file_name = os.path.join(LOG_DIR, f"{unique_log_name}.tar.gz")
    try:
        with tarfile.open(tar_file_name, "r:gz") as logfile:
            clearances = read_tar_csv_to_df(logfile,"clearances")
            assert len(clearances) > 0
            clearances_df = read_tar_parquet_to_df(logfile, "clearances")
            assert clearances_df.empty is False

    finally:
        os.remove(tar_file_name)

def test_coordination_log_as_df(unique_log_name: str):
    """
    Test that coordination events are logged correctly
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", log_filename=unique_log_name)
    env = sim.manager.environment

    fl = 320
    fix = "TABSY"
    direction = "Up"
    callsign = sorted(env.aircraft.keys())[0]
    coord_time = env.datetime + timedelta(seconds=1)

    # Set current sector and to_sector (must not be the aircraft's current sector)
    sectors = [sector for sector in env.airspace.sectors]
    current_sector = env.aircraft[callsign].current_sector
    to_sector = pick_random_next_sector(sectors, current_sector)

    # Force a coordination, force log and get the coord log as a DataFrame
    sim.manager.event_handler.add_coordination_event(coord_time, callsign, current_sector, to_sector, fl, fix, direction)
    sim.evolve(2)
    coordination_df = sim.manager.event_logger.coordination_log_as_df()
    sim.save()
    del sim

    # Test that the Dataframe contains the input coord data
    [foundRow, row] = query_row_from_df(callsign, coord_time, coordination_df)
    assert foundRow == True
    assert row["from_sector"] == current_sector
    assert row["to_sector"] == to_sector
    assert row["fl"] == fl
    assert row["fix"] == fix
    assert row["direction"] == direction

def test_radar_log_as_df(unique_log_name: str):
    """
    Test that the radar dataframe log gets created and with the correct properties
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", log_filename=unique_log_name)
    columns = [col for col in EventDtypes.radar_dtypes if col != "datetime"]
    current_radar = [{col: getattr(a, col) for col in columns} for a in sim.manager.environment.aircraft.values()]
    assert len(current_radar) > 0
    first_aircraft_radar = current_radar[0]

    # Call event logger method which creates dataframe from internal radar state
    radar_df = sim.manager.event_logger.radar_log_as_df()
    sim_time = sim.manager.environment.datetime
    del sim

    # Take the first aircraft and store the properties we're interested in
    callsign = first_aircraft_radar["callsign"]
    heading = first_aircraft_radar["heading"]
    lat = first_aircraft_radar["lat"]
    lon = first_aircraft_radar["lon"]

    # Now retrieve the relevant entry from the dataframe and test that it holds the expected properties
    [found, log_entry ]= query_row_from_df(callsign, sim_time, radar_df)
    assert found is True
    assert callsign == log_entry["callsign"]
    assert lat == log_entry["lat"]
    assert lon == log_entry["lon"]
    assert heading == log_entry["heading"]

def test_flight_log_as_df(unique_log_name: str):
    """
    Test that the flight plan dataframe log gets created and with the correct properties
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", log_filename=unique_log_name)
    first_aircraft = next(iter(sim.manager.environment.aircraft.values()))
    flight_plan = first_aircraft.flight_plan
    assert flight_plan is not None
    sim_time = sim.manager.environment.datetime
    callsign = first_aircraft.callsign

    # Call event logger method which creates dataframe from internal radar state
    flight_df = sim.manager.event_logger.flight_log_as_df()
    del sim

    # Now retrieve the flight from the dataframe and test that its properties match the original fp (for those props that match)
    [found, log_entry] = query_row_from_df(callsign, sim_time, flight_df)
    assert found is True
    expected_missing_properties = ["start_datetime", "end_datetime", "route", "filed_true_airspeed", "intention_code"]
    props_match = all(
        log_entry[k] == v
        for k, v in vars(flight_plan).items()
        if k not in expected_missing_properties
    )
    assert props_match == True

def test_incomm_log_as_df(unique_log_name: str):
    """
    Test that the test_logger's incomm_log list gets correctly converted to a dataframe
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", log_filename=unique_log_name)
    env = sim.manager.environment
    event_logger = sim.manager.event_logger
    callsign = list(env.aircraft.keys())[0]
    sectors = [sector for sector in env.airspace.sectors]
    current_sector = env.aircraft[callsign].current_sector
    to_sector = pick_random_next_sector(sectors, current_sector)
    incomm_time = env.datetime + timedelta(seconds=1)
    del sim

    # Set incomm_log as an empty dict and check that an empty dataframe is created
    event_logger.incomm_log = []
    df_logged = event_logger.incomm_log_as_df()
    assert df_logged.empty == True

    # Create a mock incomm event, append to incomm_log and check a corresponding dataframe is created
    mock_incomm = {
        "datetime": incomm_time,
        "callsign": callsign,
        "sector_name": to_sector
    }
    event_logger.incomm_log.append(mock_incomm)
    df_logged = event_logger.incomm_log_as_df()
    reference_df = pd.DataFrame([mock_incomm])
    assert df_logged.equals(reference_df) == True

def test_sectors_log_as_df(generate_two_sector):
    """
    Test that sector changes are logged correctly
    """
    airspace, _ = generate_two_sector
    event_handler = EventHandler()
    # Create an environment manager with our dummy sectors and record the sectors
    em = EnvironmentManager(
        airspace=airspace,
        event_handler=event_handler,
        predictor=SimplePredictor(dt=1, fix_proximity_threshold=2)
    )
    assert len(em.environment.airspace.sectors) == 2
    sectors = em.environment.airspace.sectors

    # Log the sectors and check sectors_df has the same pair as above
    em.event_logger.log_environment(em.environment)
    sectors_df = em.event_logger.sectors_log_as_df()
    assert sectors_df.empty == False
    log_entry = sectors_df.iloc[0]
    assert log_entry.empty == False
    logged_sectors = [key for key, _ in log_entry["sectors_configuration"]]
    assert all(sector in logged_sectors for sector in sectors)



@pytest.mark.parametrize(
    ("time_period", "ticks_to_evolve", "ticks_to_trim"),
    [
        (6, 20, 9),
        (3, 20, 10),
        (3, 11, 10),
        (6, 20, 12)
    ],
)
def test_trim_and_clip(time_period: int, ticks_to_evolve: int, ticks_to_trim: int):
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", autosave=False)

    for _ in range(0, ticks_to_evolve):
        sim.evolve(time_period)

    sim.manager.event_logger.trim_and_clip("<=", pd.Timestamp(year=1970, month=1, day=1, hour=9, minute=0, second=0) + pd.Timedelta(seconds=ticks_to_trim * time_period))

    # Sectorisation in Springfield is constant, so we need to ensure that after trimming the event is still there
    assert len(sim.manager.event_logger.sectors_log) == 1
    assert sim.manager.event_logger.sectors_log[0]["datetime"] == pd.Timestamp(year=1970, month=1, day=1, hour=9, minute=0, second=0)


    log = sim.manager.event_logger
    
    # Aircraft will appear with time, so this depends on where we trim. The following should always be true because the radar log should get trimmed as it gets updated every tick
    assert len(log.radar_log) <= len(sim.manager.environment.aircraft) * (ticks_to_evolve - ticks_to_trim)

    # The tests don't have any aircraft disappear, so we expect at least an equal set of initialisers of aircraft internals as aircraft we have in the airspace
    assert set(log["callsign"] for log in log.aircraft_internals_log) == set(sim.manager.environment.aircraft.keys())


    # The tests don't have any aircraft disappear, so we expect at least an equal set of initialisers of aircraft incomms as aircraft we have in the airspace
    assert len(log.incomm_log) == len(sim.manager.environment.aircraft)

    # We inject zero clearances so expect zero of these
    assert len(log.clearances_log) == 0

    # Expect two coordinations per aircraft in the airspace
    assert len(log.coordination_log) == 2 * len(sim.manager.environment.aircraft)


def test_trim():
    """
    Test that trimming a simulation run removes events
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", autosave=False)
    em = sim.manager

    # Prepare inputs for incomm and coordination
    callsign = sorted(em.environment.aircraft.keys())[0]
    fl = 320
    fix = "TABSY"
    direction = "Up"
    event_time = em.environment.datetime + timedelta(seconds=1)

    # Set current sector and to_sector (must not be the aircraft's current sector)
    sectors = [sector for sector in em.environment.airspace.sectors]
    current_sector = em.environment.aircraft[callsign].current_sector
    to_sector = pick_random_next_sector(sectors, current_sector)

    # Perform an incomm and coord, force log generation and get the dataframes
    em.event_handler.add_coordination_event(event_time, callsign, current_sector, to_sector, fl, fix, direction)
    em.event_handler.add_incomm_event(event_time, callsign, to_sector)
    sim.evolve(2)
    del sim

    assert em.environment.aircraft[callsign].current_sector == to_sector
    incomm_df = em.event_logger.incomm_log_as_df()
    incomm_time = event_time + timedelta(seconds=1)
    [foundRow, row ]= query_row_from_df(callsign, incomm_time, incomm_df)
    assert foundRow == True
    assert row["sector_name"] == to_sector

    coordination_df = em.event_logger.coordination_log_as_df()
    [foundRow, row ]= query_row_from_df(callsign, event_time, coordination_df)
    assert foundRow == True
    assert row["from_sector"] == current_sector
    assert row["to_sector"] == to_sector
    assert row["fl"] == fl
    assert row["fix"] == fix
    assert row["direction"] == direction

    # Call trim() which should remove everything in the log after the new (earlier) time
    earlier_time = em.environment.datetime - timedelta(seconds=5)
    em.event_logger = em.event_logger.trim(">", earlier_time)

    # Now retrieve the dataframes and check the above events have been removed
    coordination_df = em.event_logger.coordination_log_as_df()
    [foundRow, _ ]= query_row_from_df(callsign, event_time, coordination_df)
    assert foundRow == False

    incomm_df = em.event_logger.incomm_log_as_df()
    [foundRow, _ ]= query_row_from_df(callsign, incomm_time, incomm_df)
    assert foundRow == False

def test_to_event_handler(generate_two_sector):
    """
    Test that an event handler can be created from a log handler
    """
    # Create an environment manager with dummy sectors and record the sectors present
    airspace, _ = generate_two_sector
    event_handler = EventHandler()
    event_handler.reset_events()
    em = EnvironmentManager(
        airspace=airspace,
        event_handler=event_handler,
        predictor=SimplePredictor(dt=1, fix_proximity_threshold=2)
    )
    assert len(em.environment.airspace.sectors) == 2

    em.event_logger.log_environment(em.environment)
    sectors_df = em.event_logger.sectors_log_as_df()
    [found, log_entry ] = extract_first_row_from_df(sectors_df)
    assert found == True

    # Create a new event handler from the previous event logger
    new_event_handler = em.event_logger.to_event_handler()
    new_sectors_df = new_event_handler.sectors_df
    [found, new_log_entry ] = extract_first_row_from_df(new_sectors_df)
    assert found == True
    assert log_entry["sectors_configuration"] == new_log_entry["sectors_configuration"]
