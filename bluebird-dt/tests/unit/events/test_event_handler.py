import copy
from datetime import datetime, timedelta
from typing import Any, TypeVar
import typing
from bluebird_dt.scenario_manager import Tactical
from bluebird_dt.simulator import Simulator
from bluebird_dt.events.event_handler import (
    EventHandler, update_incomm, update_coordination, set_cleared_fl_to_selected_fl, update_from_clearances, 
    update_airspace_configuration, update_aircraft_internals, update_from_radar, update_from_flight_plans, 
    update_selected_fl_from_radar, update_aircraft_attribute
)
from bluebird_dt.events.event_logger import EventLogger
from bluebird_dt.events.event_dtypes import EventDtypes
from bluebird_dt.core import (
    Aircraft,
    Coordination,
    Environment,
    HoldAtLocationParameters,
    WindField,
)
from conftest import (
    pick_random_next_sector, build_default_test_df, is_deeply_equal, setup_test_sim, get_to_and_from_sectors, 
    build_test_df_for_multiple_ac, build_coordination_test_df
)
import pandas as pd
import pandas.api.types as ptypes
import pytest
from pydantic import ValidationError

T = TypeVar("T")

LOWER_EPISODE_BOUNDARY = 1
UPPER_EPISODE_BOUNDARY = 3

def test_init():
    """
    Test object construction.  Instantiate EventHandler and check proper construction
    """
    event_handler = EventHandler()
    expected_radar_keys = [k for k in EventDtypes.radar_dtypes if k != "datetime"]
    assert list(event_handler.radar_df.columns) == expected_radar_keys
    expected_flight_keys = [k for k in EventDtypes.flight_dtypes if k != "datetime"]
    assert list(event_handler.flight_df.columns) == expected_flight_keys
    expected_clearance_keys = [k for k in EventDtypes.clearance_dtypes if k != "datetime"]
    assert list(event_handler.clearances_df.columns) == expected_clearance_keys
    expected_coord_keys = [k for k in EventDtypes.coord_dtypes if k != "datetime"]
    assert list(event_handler.coordination_df.columns) == expected_coord_keys
    expected_sectors_keys = [k for k in EventDtypes.sectors_dtypes if k != "datetime"]
    assert list(event_handler.sectors_df.columns) == expected_sectors_keys
    expected_incomm_keys = [k for k in EventDtypes.incomm_dtypes if k != "datetime"]
    assert list(event_handler.incomm_df.columns) == expected_incomm_keys
    expected_aircraft_internals_keys = [k for k in EventDtypes.aircraft_internals_dtypes if k != "datetime"]
    assert list(event_handler.aircraft_internals_df.columns) == expected_aircraft_internals_keys
    expected_ac_attribute_update_keys = [k for k in EventDtypes.ac_attribute_update_dtypes if k != "datetime"]
    assert list(event_handler.ac_attribute_update_df.columns) == expected_ac_attribute_update_keys

def test_reset_events():
    """
    Test that reset_event correctly emptys member dfs.
    """
    env, manager, _, _, _, _ = setup_test_sim()
    event_handler = manager.event_handler

    radar_df_before = build_default_test_df(EventDtypes.radar_dtypes, env)
    flight_df_before = build_default_test_df(EventDtypes.flight_dtypes, env)
    clearances_df_before = build_default_test_df(EventDtypes.clearance_dtypes, env)
    coordination_df_before = build_default_test_df(EventDtypes.coord_dtypes, env)
    sectors_df_before = build_default_test_df(EventDtypes.sectors_dtypes, env)
    incomm_df_before = build_default_test_df(EventDtypes.incomm_dtypes, env)
    aircraft_internals_df_before = build_default_test_df(EventDtypes.aircraft_internals_dtypes, env)
    ac_attribute_update_df_before = build_default_test_df(EventDtypes.ac_attribute_update_dtypes, env)

    event_handler.radar_df = copy.deepcopy(radar_df_before)
    event_handler.flight_df = copy.deepcopy(flight_df_before)
    event_handler.clearances_df = copy.deepcopy(clearances_df_before)
    event_handler.coordination_df = copy.deepcopy(coordination_df_before)
    event_handler.sectors_df = copy.deepcopy(sectors_df_before)
    event_handler.incomm_df = copy.deepcopy(incomm_df_before)
    event_handler.aircraft_internals_df = copy.deepcopy(aircraft_internals_df_before)
    event_handler.ac_attribute_update_df = copy.deepcopy(ac_attribute_update_df_before)

    event_handler.reset_events()

    assert event_handler.radar_df.empty
    assert event_handler.flight_df.empty
    assert event_handler.clearances_df.empty
    assert event_handler.coordination_df.empty
    assert event_handler.sectors_df.empty
    assert event_handler.incomm_df.empty
    assert event_handler.aircraft_internals_df.empty
    assert event_handler.ac_attribute_update_df.empty

def test_add_optional_radar_columns_if_required():
    """
    Test adding optional radar columns
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario")
    assert "SPRINGFIELD" in sim.manager.environment.airspace.sectors
    event_handler = sim.manager.event_handler

    # Create a df with the optional columns missing and set as the event handler's radar dataframe
    required_cols = ["callsign", "lat", "lon", "fl", "heading"]
    cutdown_df = pd.DataFrame(columns=required_cols)
    event_handler.radar_df = cutdown_df

    # Now call the method and then test the optional columns have been added
    event_handler.add_optional_radar_columns_if_required()
    radar_df_after = event_handler.radar_df
    expected_cols_added = [
        "speed_tas",
        "ground_speed",
        "ground_track_angle",
        "selected_fl",
    ]
    assert all(col in radar_df_after.columns for col in expected_cols_added)

def test_add_optional_clearance_columns_if_required():
    """
    Test adding optional clearance columns if required
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario")
    assert "SPRINGFIELD" in sim.manager.environment.airspace.sectors
    event_handler = sim.manager.event_handler

    # Create a df with the optional columns missing and set as the event handler's dataframe
    required_cols = ["callsign", "lat", "lon", "fl", "heading"]
    cutdown_df = pd.DataFrame(columns=required_cols)
    event_handler.clearances_df = cutdown_df

    # Now call the method and then test the optional columns have been added
    event_handler.add_optional_clearance_columns_if_required()
    clearances_df_after = event_handler.clearances_df
    
    expected_cols_added = [
            "text_clearance",
            "text_pilot_response",
            "voice_clearance",
            "voice_pilot_response",
        ]
    assert all(col in clearances_df_after.columns for col in expected_cols_added)

def test_set_dataframe_indices_to_datetime():
    """
    Test setting dataframe indices
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario")
    assert "SPRINGFIELD" in sim.manager.environment.airspace.sectors
    event_handler = sim.manager.event_handler
    event_handler.radar_df = pd.DataFrame(columns=EventDtypes.radar_dtypes)
    event_handler.clearances_df = pd.DataFrame(columns=EventDtypes.clearance_dtypes)
    event_handler.flight_df = pd.DataFrame(columns=EventDtypes.flight_dtypes)
    event_handler.clearances_df = pd.DataFrame(columns=EventDtypes.clearance_dtypes)
    event_handler.coordination_df = pd.DataFrame(columns=EventDtypes.coord_dtypes)
    event_handler.sectors_df = pd.DataFrame(columns=EventDtypes.sectors_dtypes)
    event_handler.incomm_df = pd.DataFrame(columns=EventDtypes.incomm_dtypes)
    event_handler.aircraft_internals_df = pd.DataFrame(columns=EventDtypes.aircraft_internals_dtypes)
    event_handler.ac_attribute_update_df = pd.DataFrame(columns=EventDtypes.ac_attribute_update_dtypes)

    event_handler.set_dataframe_indices_to_datetime()

    assert event_handler.radar_df.index.name == 'datetime'
    assert event_handler.clearances_df.index.name == 'datetime'
    assert event_handler.flight_df.index.name == 'datetime'
    assert event_handler.clearances_df.index.name == 'datetime'
    assert event_handler.coordination_df.index.name == 'datetime'
    assert event_handler.sectors_df.index.name == 'datetime'
    assert event_handler.incomm_df.index.name == 'datetime'
    assert event_handler.aircraft_internals_df.index.name == 'datetime'
    assert event_handler.ac_attribute_update_df.index.name == 'datetime'

def test_ensure_specific_format_for_specific_columns():
    """
    Test that coord dataframe sectors with less than 2 chars are left padded with 0s else unchanged
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario")
    assert "SPRINGFIELD" in sim.manager.environment.airspace.sectors
    event_handler = sim.manager.event_handler

    test_df = pd.DataFrame([
        { "callsign": "test1", "from_sector": "7", "to_sector": "9" },  
        { "callsign": "test2", "from_sector": "S1", "to_sector": "TYNE" }
    ])

    # Push the test df into event handler. 
    event_handler.coordination_df = test_df
    event_handler.ensure_specific_format_for_specific_columns()
    coord_df_after = event_handler.coordination_df
    coord_df_after = coord_df_after[coord_df_after["callsign"] == "test1"].iloc[0]
    # Test that the 1 char sectors are padded
    assert coord_df_after["from_sector"] == "07"
    assert coord_df_after["to_sector"] == "09"

    # Now check that the 2 char sectors are not padded
    coord_df_after = event_handler.coordination_df
    coord_df_after = coord_df_after[coord_df_after["callsign"] == "test2"].iloc[0]
    assert coord_df_after["from_sector"] == "S1"
    assert coord_df_after["to_sector"] == "TYNE"

def test_fillna_for_specific_columns():
    """
    Test fillna column data cleaning method
    """

    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario")
    assert "SPRINGFIELD" in sim.manager.environment.airspace.sectors
    event_handler = sim.manager.event_handler

    # Set up test dfs with None's for the relevant fields
    test_coord_df = pd.DataFrame([{ "callsign": "test1", "level_by_details": None, "secondary_coord_conditions": None }])
    event_handler.coordination_df = test_coord_df
    test_flight_df = pd.DataFrame([{ "callsign": "test1", "sector_crossing_seq": None }])
    event_handler.flight_df = test_flight_df

    event_handler.fillna_for_specific_columns()

    # Test all Nones have been changed to "" for coord then flight
    coord_df_after = event_handler.coordination_df
    coord_df_after = coord_df_after[coord_df_after["callsign"] == "test1"].iloc[0]
    assert coord_df_after["level_by_details"] == ""
    assert coord_df_after["secondary_coord_conditions"] == ""

    flight_df_after = event_handler.flight_df
    flight_df_after = flight_df_after[flight_df_after["callsign"] == "test1"].iloc[0]
    assert flight_df_after["sector_crossing_seq"] == ""

def test_ensure_dataframe_data_types():
    """
    Test ensure dataframe data types
    """

    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario")
    assert "SPRINGFIELD" in sim.manager.environment.airspace.sectors
    event_handler = sim.manager.event_handler

    # Reset the indexes back to datetime field for all event handler dataframes
    event_handler.radar_df = event_handler.radar_df.reset_index().rename(columns={"index": "datetime"})
    event_handler.flight_df = event_handler.flight_df.reset_index().rename(columns={"index": "datetime"})
    event_handler.clearances_df = event_handler.clearances_df.reset_index().rename(columns={"index": "datetime"})
    event_handler.coordination_df = event_handler.coordination_df.reset_index().rename(columns={"index": "datetime"})
    event_handler.sectors_df = event_handler.sectors_df.reset_index().rename(columns={"index": "datetime"})
    event_handler.incomm_df = event_handler.incomm_df.reset_index().rename(columns={"index": "datetime"})
    event_handler.aircraft_internals_df = event_handler.aircraft_internals_df.reset_index().rename(columns={"index": "datetime"})
    event_handler.ac_attribute_update_df = event_handler.ac_attribute_update_df.reset_index().rename(columns={"index": "datetime"})

    # Test with an incorrect but castable type
    event_handler.radar_df["lat"] = "180.0"

    assert ptypes.is_float_dtype(event_handler.radar_df["lat"]) == False
    event_handler.ensure_dataframe_data_types()

    assert ptypes.is_float_dtype(event_handler.radar_df["lat"]) == True

    # Test with an incorrect and uncastable type
    event_handler.radar_df["lat"] = "uncastable"
    with pytest.raises(ValueError):
        event_handler.ensure_dataframe_data_types()
    assert ptypes.is_float_dtype(event_handler.radar_df["lat"]) == False

def test_ensure_dataframes_are_date_ordered():
    """
    Test ensure dataframes are date ordered
    """

    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario")
    assert "SPRINGFIELD" in sim.manager.environment.airspace.sectors
    event_handler = sim.manager.event_handler

    unordered_dates = ["2025-03-01", "2025-01-15", "2025-02-10", "2025-01-01" ]
    df = pd.DataFrame({"value": [1,2,3,4]}, index=pd.to_datetime(unordered_dates))
    event_handler.flight_df = df
    assert df.index.is_monotonic_increasing == False
    event_handler.ensure_dataframes_are_date_ordered()
    assert event_handler.flight_df.index.is_monotonic_increasing == True

def test_add_aircraft(generate_simple_environment: Environment):
    """
    Test adding an aircraft.
    In these tests we are only testing that the relevant events are added to the events list (not that it is are processed / enacted)
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario")
    assert "SPRINGFIELD" in sim.manager.environment.airspace.sectors
    environment = generate_simple_environment
    callsign = "AIR0"
    selected_aircraft = environment.aircraft[callsign]
    event_handler = sim.manager.event_handler

    radar_df_before = event_handler.radar_df
    radar_df_before = radar_df_before[radar_df_before["callsign"]==callsign]
    assert radar_df_before.empty is True

    flight_df_before = event_handler.flight_df
    flight_df_before = flight_df_before[flight_df_before["callsign"]==callsign]
    assert flight_df_before.empty is True

    before_ac_internals_df = event_handler.aircraft_internals_df
    before_ac_internals_df = before_ac_internals_df[before_ac_internals_df["callsign"]==callsign]
    assert before_ac_internals_df.empty is True

    rate_of_turn = 1.5
    selected_aircraft.rate_of_turn = rate_of_turn
    the_datetime = environment.datetime# + timedelta(seconds=1)

    event_handler.add_aircraft(the_datetime, selected_aircraft)

    # Test radar event created
    radar_df_after = event_handler.radar_df
    radar_df_after = radar_df_after[radar_df_after["callsign"]==callsign]
    assert len(radar_df_after) > len(radar_df_before)

    # Test flight event created
    flight_df_after = event_handler.flight_df
    flight_df_after = flight_df_after[flight_df_after["callsign"]==callsign]
    assert len(flight_df_after) > len(flight_df_before)

    # Test a/c internals updated
    after_ac_internals_df = event_handler.aircraft_internals_df
    assert len(after_ac_internals_df) > len(before_ac_internals_df)
    after_ac_internals_df = after_ac_internals_df[after_ac_internals_df["callsign"]==callsign]
    newly_added_row = after_ac_internals_df.iloc[-1]
    assert newly_added_row["rate_of_turn"] == rate_of_turn

def test_add_aircraft_attribute_update_event():
    """
    Test adding an aircraft attribute update event.
    """
    env, manager, _, callsign, _, _ = setup_test_sim()
    event_handler = manager.event_handler
    the_datetime = env.datetime
    attribute_name = "selected_fl"
    value = 275

    before_df = event_handler.ac_attribute_update_df
    before_df = before_df[before_df["callsign"] == callsign]

    event_handler.add_aircraft_attribute_update_event(the_datetime, callsign, attribute_name, value)

    after_df = event_handler.ac_attribute_update_df
    after_df = after_df[after_df["callsign"] == callsign]
    assert len(after_df) > len(before_df)
    newly_added_row = after_df.iloc[-1]
    assert newly_added_row["attribute_name"] == attribute_name
    assert newly_added_row["value"] == value

def test_add_radar_event():
    """
    Test adding a radar event.
    In these tests we are only testing that events are added to the events list (not that it is are processed / enacted)
    """
    env, manager, _, callsign, _, _ = setup_test_sim()
    event_handler = manager.event_handler
    the_datetime = env.datetime

    radar_df_before = event_handler.radar_df
    radar_df_before = radar_df_before[radar_df_before["callsign"]==callsign]
    lat = 51.0
    lon = 3.1
    fl = 310
    heading = 180.0
    event_handler.add_radar_event(the_datetime, callsign, lat, lon, fl, heading)
    radar_df_after = event_handler.radar_df
    radar_df_after = radar_df_after[radar_df_after["callsign"]==callsign]
    assert len(radar_df_after) > len(radar_df_before)
    last_row_after = radar_df_after.iloc[-1]
    assert last_row_after["lat"] == lat
    assert last_row_after["lon"] == lon
    assert last_row_after["fl"] == fl
    assert last_row_after["heading"] == heading

def test_add_flight_plan_event(flight_plan: dict[str, Any]):
    """
    Test adding a flight plan event.
    In these tests we are only testing that events are added to the events list (not that it is are processed / enacted)
    """
    env, manager, _, callsign, _, _ = setup_test_sim()
    event_handler = manager.event_handler
    flight_df_before = event_handler.flight_df
    flight_df_before = flight_df_before[flight_df_before["callsign"]==callsign]
    event_handler.add_flight_plan_event(env.datetime, callsign, flight_plan["route_filed"])
    flight_df_after = event_handler.flight_df
    flight_df_after = flight_df_after[flight_df_after["callsign"]==callsign]
    assert len(flight_df_after) > len(flight_df_before)

def test_add_clearance_event():
    """
    Test adding a clearance event.
    In these tests we are only testing that events are added to the events list (not that it is are processed / enacted)
    """
    env, manager, _, callsign, _, _ = setup_test_sim()
    event_handler = manager.event_handler
    kind = "change_flight_level_to"
    value = "250"
    clearances_df_before = event_handler.clearances_df
    clearances_df_before = clearances_df_before[clearances_df_before["callsign"]==callsign]
    event_handler.add_clearance_event(env.datetime, callsign, kind, value)

    clearances_df_after = event_handler.clearances_df
    clearances_df_after = clearances_df_after[clearances_df_after["callsign"]==callsign]
    assert len(clearances_df_after) > len(clearances_df_before)

    newly_added_row = clearances_df_after.iloc[-1]
    assert newly_added_row["kind"] == kind
    assert newly_added_row["value"] == value

def test_add_sectors_event():
    """
    Test adding a sectors update event.
    """
    env, manager, _, _, _, _ = setup_test_sim()
    event_handler = manager.event_handler
    the_datetime = env.datetime
    sectors_configuration = [("SPRINGFIELD", ["JIMMI", "NATOR"])]

    before_df = event_handler.sectors_df
    event_handler.add_sectors_event(the_datetime, sectors_configuration)
    after_df = event_handler.sectors_df

    assert len(after_df) > len(before_df)
    newly_added_value = after_df.loc[the_datetime, "sectors_configuration"]
    if isinstance(newly_added_value, pd.Series):
        newly_added_value = newly_added_value.iloc[-1]
    assert newly_added_value == sectors_configuration

def test_add_coordination_event():
    """
    Test adding a coordination event.
    In these tests we are only testing that events are added to the events list (not that the coords are processed / enacted)
    """
    env, manager, _, callsign, _, _ = setup_test_sim()
    event_handler = manager.event_handler

    # Set current sector and to_sector (must not be the aircraft's current sector
    sectors = [sector for sector in env.airspace.sectors]
    current_sector = env.aircraft[callsign].current_sector
    to_sector = pick_random_next_sector(sectors, current_sector)
    fl = 290.0
    fix = "SMUDJ"
    direction = "Horizontal"

    def test_coord_with_minimal_props():
        coord_time = env.datetime + timedelta(seconds=1)
        event_handler.add_coordination_event(coord_time, callsign, current_sector, to_sector, fl, fix, direction)
        coordination_df = event_handler.coordination_df
        results = coordination_df[coordination_df["callsign"] == callsign]
        assert results.empty is False
        row = results.loc[coord_time]
        assert row.empty is False
        assert row["from_sector"] == current_sector
        assert row["to_sector"] == to_sector
        assert row["fl"] == fl
        assert row["fix"] == fix
        assert row["direction"] == direction

    def test_coord_with_full_props():
        # Let's give the coord a time different to previous tests so we can extract it uniquely
        coord_time = env.datetime + timedelta(seconds=10)
        level_by = True
        level_by_details = "{ fix : 300 }"
        coord_condition = "CONDITION"
        secondary_coord_conditions = coord_condition
        event_handler.add_coordination_event(
            coord_time, callsign, current_sector, to_sector, fl, fix, 
            direction, level_by, level_by_details, secondary_coord_conditions
        )
        coordination_df = event_handler.coordination_df
        results = coordination_df[coordination_df["callsign"] == callsign]
        assert results.empty is False
        row = results.loc[coord_time]
        assert row.empty == False
        assert row["from_sector"] == current_sector
        assert row["to_sector"] == to_sector
        assert row["fl"] == fl
        assert row["fix"] == fix
        assert row["level_by"] == level_by
        assert row["level_by_details"] == level_by_details
        assert row["secondary_coord_conditions"] == secondary_coord_conditions
    
    test_coord_with_minimal_props()
    test_coord_with_full_props()

def test_add_coordination():
    """
    Test creating a coordination event by adding a coordination object.
    In these tests we are only testing that events are added to the events list (not that the coords are processed / enacted)
    """
    env, manager, _, callsign, _, _ = setup_test_sim()

    event_handler = manager.event_handler

    # Set current sector and to_sector (must not be the aircraft's current sector
    sectors = [sector for sector in env.airspace.sectors]
    from_sector = env.aircraft[callsign].current_sector
    to_sector = pick_random_next_sector(sectors, from_sector)
    fl = 290.0
    fix = "SMUDJ"
    direction = "Horizontal"
    level_by = False
      
    def test_add_coordination_minimal_props():
        coord = Coordination(
            callsign=callsign,
            from_sector=from_sector,
            to_sector=to_sector,
            fl=fl,
            fix=fix,
            direction=direction,
            level_by=level_by,
        )
        coord_time = env.datetime + timedelta(seconds=1)
        event_handler.add_coordination(coord_time, coord)
        coordination_df = event_handler.coordination_df
        results = coordination_df[coordination_df["callsign"] == callsign]
        assert results.empty is False
        row = results.loc[coord_time]
        assert row.empty == False
        assert row["from_sector"] == from_sector
        assert row["to_sector"] == to_sector
        assert row["fl"] == fl
        assert row["fix"] == fix
        assert row["level_by"] == level_by

    def test_add_coordination_full_props():
        level_by = True
        level_by_details = {fix: 300.0}
        coord_condition = "CONDITION"
        secondary_coord_conditions = coord_condition
        coord = Coordination(
            callsign=callsign,
            from_sector=from_sector,
            to_sector=to_sector,
            fl=fl,
            fix=fix,
            direction=direction,
            level_by=level_by,
            level_by_details=level_by_details,
            secondary_coord_conditions=secondary_coord_conditions
        )
        coord_time = env.datetime + timedelta(seconds=2)
        event_handler.add_coordination(coord_time, coord)
        coordination_df = event_handler.coordination_df
        results = coordination_df[coordination_df["callsign"] == callsign]
        assert results.empty is False
        row = results.loc[coord_time]
        assert row.empty == False
        assert row["from_sector"] == from_sector
        assert row["to_sector"] == to_sector
        assert row["fl"] == fl
        assert row["fix"] == fix
        assert row["level_by"] == level_by
        assert row["level_by_details"] == str(level_by_details)
        assert row["secondary_coord_conditions"] == secondary_coord_conditions

    test_add_coordination_minimal_props()
    test_add_coordination_full_props()

def test_add_aircraft_internals_event():
    """
    Test creating an aircraft internals event.
    """
    env, manager, _, callsign, _, _ = setup_test_sim()
    event_handler = manager.event_handler

    # Set a date later than previous events for aircraft
    the_datetime = env.datetime + timedelta(seconds=1)

    before_ac_internals_df = event_handler.aircraft_internals_df
    before_ac_internals_df = before_ac_internals_df[before_ac_internals_df["callsign"]==callsign]
    rate_of_turn = 1.5
    event_handler.add_aircraft_internals_event(the_datetime, callsign, rate_of_turn)

    after_ac_internals_df = event_handler.aircraft_internals_df
    assert len(after_ac_internals_df) > len(before_ac_internals_df)
    after_ac_internals_df = after_ac_internals_df[after_ac_internals_df["callsign"]==callsign]
    newly_added_row = after_ac_internals_df.iloc[-1]
    assert newly_added_row["rate_of_turn"] == rate_of_turn

def test_jump_to_time(flight_plan):
    """
    Test jumping to different sim times
    """
    selected_fl_to_test = 135
    cleared_fl_to_test = 310
    attribute_fl = 333
    time_jump_back_delta = timedelta(minutes=1)
    radar_props = [ 
        ("selected_fl",selected_fl_to_test), 
        ("fl", selected_fl_to_test), 
        ("lat", 53.0), 
        ("lon", 0.5), 
        ("heading", 180)
    ]
    clearance_props = [ ('kind', 'change_flight_level_to'), ('value', cleared_fl_to_test) ]
    flight_plan_props = list(flight_plan.items())
    fix = "SMUDJ"
    direction = "Horizontal"
    sector_name_to_test =  "SPRINGFIELD"
    coord_props = [
        ("to_sector", sector_name_to_test),
        ("fl", selected_fl_to_test),
        ("fix", fix),
        ("direction", direction),
        ("level_by", False),
        ("level_by_details", None),
        ("secondary_coord_conditions", ""),
    ]
    attribute_props = [("attribute_name", "selected_fl"), ("value", attribute_fl)]
    aircraft_internals_props = [
        ("current_sector", sector_name_to_test), 
        ("pilot_type", "Pilot"), 
        ("selected_fl", selected_fl_to_test),
        ("cleared_fl", cleared_fl_to_test)
    ]
    sectors_configuration = [[("SPRINGFIELD", ["SPRINGFIELD"])]]

    def get_latest_coord_for_callsign(env: Environment, callsign: str) -> list:
        [current_sector, _] = get_to_and_from_sectors(env, callsign)
        coords = env.coordinations.get(callsign)
        assert coords is not None and len(coords) > 0
        return coords[0]

    def build_jump_to_df(env: Environment[Aircraft, WindField, WindField], props_to_set: list[tuple[str, typing.Any]], timedelta_to_test: int, dataframe_schema: dict[str, str]) ->pd.DataFrame:
        df = build_test_df_for_multiple_ac(dataframe_schema, env, props_to_set)
        df.index = df.index - time_jump_back_delta - timedelta(seconds=timedelta_to_test)
        return df

    def execute_jump_test(timedelta_to_test: int):
        env, manager, _, _, _, _ = setup_test_sim()
        event_handler = manager.event_handler

        event_handler.radar_df = build_jump_to_df(env, radar_props, timedelta_to_test, EventDtypes.radar_dtypes)
        event_handler.flight_df = build_jump_to_df(env, flight_plan_props, timedelta_to_test, EventDtypes.flight_dtypes)
        event_handler.clearances_df = build_jump_to_df(env, clearance_props, timedelta_to_test, EventDtypes.clearance_dtypes)

        sectors_df = build_default_test_df(EventDtypes.sectors_dtypes, env)
        sectors_df.index = sectors_df.index + timedelta(seconds=timedelta_to_test)
        sectors_df["sectors_configuration"] = sectors_configuration
        event_handler.sectors_df = sectors_df

        crd_time_delta = -(time_jump_back_delta.total_seconds() + timedelta_to_test)
        event_handler.coordination_df = build_coordination_test_df(env, coord_props, crd_time_delta, EventDtypes.coord_dtypes)
        event_handler.incomm_df = build_jump_to_df(env, [("sector_name", sector_name_to_test)], timedelta_to_test, EventDtypes.incomm_dtypes)
        event_handler.ac_attribute_update_df = build_jump_to_df(env, attribute_props, timedelta_to_test, EventDtypes.ac_attribute_update_dtypes)
        event_handler.aircraft_internals_df = build_jump_to_df(env, aircraft_internals_props, timedelta_to_test, EventDtypes.aircraft_internals_dtypes)

        # Now perform the jump and then verify that the env time has changed correctly
        env_jump_to_time = env.datetime - time_jump_back_delta
        env_after = event_handler.jump_to_time(manager, env_jump_to_time)
        assert env_after.datetime == env_jump_to_time

        return env_after
    
    def test_jump_before_episode_start():
        env_after = execute_jump_test(7)
        assert len(env_after.aircraft) == 0

    def test_jump_to_episode_start():
        env_after = execute_jump_test(6)
        assert len(env_after.aircraft) == 0

    def test_jump_within_time_range():
        env = execute_jump_test(5)
        assert len(env.aircraft) != 0
        assert all(ac.selected_fl == selected_fl_to_test for ac in env.aircraft.values())
        assert all(ac.cleared_fl == cleared_fl_to_test for ac in env.aircraft.values())
        flight_plan_props_to_compare = [ "origin", "dest", "unexpanded_route", "milcivil", "requested_flight_level", "filed_true_airspeed", "intention_code", "assigned_squawk" ]
        assert all(
            all(getattr(ac.flight_plan, prop) == flight_plan[prop] for prop in flight_plan_props_to_compare)
            for ac in env.aircraft.values()
        )
        for callsign, ac in env.aircraft.items():
            # Test coordination
            latest_coord = get_latest_coord_for_callsign(env, callsign)
            assert latest_coord.fl == selected_fl_to_test
            assert latest_coord.fix == fix
            assert latest_coord.direction == direction
            # Test incomm        
            assert ac.current_sector == sector_name_to_test

    def test_jump_to_episode_end():
        env = execute_jump_test(0)
        assert len(env.aircraft) != 0
        assert all(ac.selected_fl == selected_fl_to_test for ac in env.aircraft.values())
        assert all(ac.cleared_fl == cleared_fl_to_test for ac in env.aircraft.values())

        flight_plan_props_to_compare = [ "origin", "dest", "unexpanded_route", "milcivil", "requested_flight_level", "filed_true_airspeed", "intention_code", "assigned_squawk" ]
        assert all(
            all(getattr(ac.flight_plan, prop) == flight_plan[prop] for prop in flight_plan_props_to_compare)
            for ac in env.aircraft.values()
        )

        for callsign, ac in env.aircraft.items():
            # Test coordination
            latest_coord = get_latest_coord_for_callsign(env, callsign)
            assert latest_coord.fl == selected_fl_to_test
            assert latest_coord.fix == fix
            assert latest_coord.direction == direction
            # Test incomm        
            assert ac.current_sector == sector_name_to_test

    def test_jump_after_episode_end():
        env_after = execute_jump_test(-1)
        assert len(env_after.aircraft) == 0

    test_jump_before_episode_start()
    test_jump_to_episode_start()
    test_jump_within_time_range()
    test_jump_to_episode_end()
    test_jump_after_episode_end()

def test_forward(flight_plan):
    """
    Test forwarding with all event dataframes populated.
    """
    selected_fl_to_test = 333
    test_turn_radius = 45.5
    test_coord_fl = 290.0
    cleared_fl_to_test = 280
    sectors_configuration = [[("SPRINGFIELD", ["SPRINGFIELD"])]]
    radar_props = [("lat", 51.0), ("lon", 3.3), ("fl", 280), ("heading", 200), ("selected_fl", 280)]
    clearance_props = [("kind", "change_flight_level_to"), ("value", cleared_fl_to_test)]
    attribute_props = [("attribute_name", "selected_fl"), ("value", selected_fl_to_test)]
    flight_plan_props = list(flight_plan.items())
    step_time = 2
    to_sector = "background"
    coord_props = [
        ("to_sector", to_sector),
        ("fl", test_coord_fl),
        ("fix", "SMUDJ"),
        ("direction", "Horizontal"),
        ("level_by", False),
        ("level_by_details", None),
        ("secondary_coord_conditions", ""),
    ]

    def get_latest_coord_for_callsign(env: Environment[Aircraft, WindField, WindField], callsign: str) -> Coordination:
        coords = env.coordinations.get(callsign)
        assert coords is not None and len(coords) > 0
        return coords[0]

    def build_test_df(env: Environment[Aircraft, WindField, WindField], props_to_set: list[tuple[str, typing.Any]], timedelta_to_test: int, dataframe_schema: dict[str, str]) -> pd.DataFrame:
        test_df = build_test_df_for_multiple_ac(dataframe_schema, env, props_to_set)
        test_df.index = test_df.index + timedelta(seconds=timedelta_to_test)
        return test_df

    def execute_forward_test(timedelta_to_test: int, include_internals: bool):
        env, manager, _, _, _, _ = setup_test_sim()
        event_handler = manager.event_handler
        event_handler.reset_events()

        # Ensure no event path is bypassed for simulated aircraft.
        event_handler.ignore.radar_if_simmed = False
        event_handler.ignore.flight_if_simmed = False
        event_handler.ignore.clearance_if_simmed = False
        event_handler.ignore.airspace_config_updates = False
        event_handler.ignore.coordination_if_simmed = False
        event_handler.ignore.incomm_if_simmed = False
        event_handler.ignore.ac_attribute_if_simmed = False
        event_handler.ignore.aircraft_internals_if_simmed = not include_internals

        event_handler.radar_df = build_test_df(env, radar_props, timedelta_to_test, EventDtypes.radar_dtypes)
        event_handler.flight_df = build_test_df(env, flight_plan_props, timedelta_to_test, EventDtypes.flight_dtypes)
        event_handler.clearances_df = build_test_df(env, clearance_props, timedelta_to_test, EventDtypes.clearance_dtypes)

        sectors_df = build_default_test_df(EventDtypes.sectors_dtypes, env)
        sectors_df.index = sectors_df.index + timedelta(seconds=timedelta_to_test)
        sectors_df["sectors_configuration"] = sectors_configuration
        event_handler.sectors_df = sectors_df
        event_handler.coordination_df = build_coordination_test_df(env, coord_props, timedelta_to_test, EventDtypes.coord_dtypes)
        event_handler.incomm_df = build_test_df(env, [("sector_name", to_sector)], timedelta_to_test, EventDtypes.incomm_dtypes)
        event_handler.ac_attribute_update_df = build_test_df(env, attribute_props, timedelta_to_test, EventDtypes.ac_attribute_update_dtypes)
        if include_internals:
            internals_props = [
                ("current_sector", to_sector),
                ("pilot_type", "Pilot"),
                ("predictor_params", {"turn_radius": test_turn_radius}),
                ("selected_fl", selected_fl_to_test),
                ("cleared_fl", cleared_fl_to_test)
            ]
            event_handler.aircraft_internals_df = build_test_df(env, internals_props, timedelta_to_test, EventDtypes.aircraft_internals_dtypes)

        env_before = copy.deepcopy(env)
        env_after = event_handler.forward(manager, step_time=step_time)
        return env_before, env_after, to_sector

    def test_jump_before_episode_start():
        env_before, env_after, _ = execute_forward_test(-1, include_internals=False)
        for callsign in env_after.aircraft.keys():
            assert env_after.aircraft[callsign].current_sector == env_before.aircraft[callsign].current_sector
            assert env_after.aircraft[callsign].selected_fl == env_before.aircraft[callsign].selected_fl
            assert env_after.aircraft[callsign].lat == env_before.aircraft[callsign].lat

    def test_forward_to_episode_start():
        env_before, env_after, _ = execute_forward_test(0, include_internals=False)
        for callsign in env_after.aircraft.keys():
            assert env_after.aircraft[callsign].current_sector == env_before.aircraft[callsign].current_sector
            assert env_after.aircraft[callsign].selected_fl == env_before.aircraft[callsign].selected_fl
            assert env_after.aircraft[callsign].lat == env_before.aircraft[callsign].lat

    def test_forward_within_time_range():
        env_before, env_after, to_sector = execute_forward_test(1, include_internals=False)
        assert len(env_after.aircraft) != 0
        assert "SPRINGFIELD" in env_after.airspace.airspace_configuration
        flight_plan_props_to_compare = [ "origin", "dest", "unexpanded_route", "milcivil", "requested_flight_level", "filed_true_airspeed", "intention_code", "assigned_squawk" ]
        assert all(
            all(getattr(ac.flight_plan, prop) == flight_plan[prop] for prop in flight_plan_props_to_compare)
            for ac in env_after.aircraft.values()
        )
        for callsign, ac in env_after.aircraft.items():
            assert ac.cleared_instructions.fl == cleared_fl_to_test
            assert ac.cleared_fl == cleared_fl_to_test
            assert ac.selected_fl == selected_fl_to_test
            assert ac.lat != env_before.aircraft[callsign].lat
            assert ac.lon != env_before.aircraft[callsign].lon
            # Test coordination
            latest_coord = get_latest_coord_for_callsign(env_after, callsign)
            assert latest_coord.fl == test_coord_fl
            assert latest_coord.to_sector == to_sector
            # Test incomm        
            assert ac.current_sector == to_sector

    def test_forward_within_time_range_with_internals():
        env_before, env_after, to_sector = execute_forward_test(1, include_internals=True)
        for callsign, ac in env_after.aircraft.items():
            assert ac.current_sector == to_sector
            assert ac.predictor_params["turn_radius"] == test_turn_radius
            assert ac.cleared_instructions.fl == cleared_fl_to_test
            assert ac.cleared_fl == cleared_fl_to_test
            assert ac.selected_fl == selected_fl_to_test
            assert ac.lat != env_before.aircraft[callsign].lat
            assert ac.lon != env_before.aircraft[callsign].lon
            # Test coordination
            latest_coord = get_latest_coord_for_callsign(env_after, callsign)
            assert latest_coord.fl == test_coord_fl
            assert latest_coord.to_sector == to_sector
            # Test incomm        
            assert ac.current_sector == to_sector

    def test_forward_to_episode_end():
        env_before, env_after, to_sector = execute_forward_test(2, include_internals=False)
        assert len(env_after.aircraft) != 0
        assert "SPRINGFIELD" in env_after.airspace.airspace_configuration
        flight_plan_props_to_compare = [ "origin", "dest", "unexpanded_route", "milcivil", "requested_flight_level", "filed_true_airspeed", "intention_code", "assigned_squawk" ]
        assert all(
            all(getattr(ac.flight_plan, prop) == flight_plan[prop] for prop in flight_plan_props_to_compare)
            for ac in env_after.aircraft.values()
        )
        for callsign, ac in env_after.aircraft.items():
            assert ac.cleared_instructions.fl == cleared_fl_to_test
            assert ac.cleared_fl == cleared_fl_to_test
            assert ac.selected_fl == selected_fl_to_test
            assert ac.lat != env_before.aircraft[callsign].lat
            assert ac.lon != env_before.aircraft[callsign].lon
            # Test coordination
            latest_coord = get_latest_coord_for_callsign(env_after, callsign)
            assert latest_coord.fl == test_coord_fl
            assert latest_coord.to_sector == to_sector
            # Test incomm        
            assert ac.current_sector == to_sector

    def test_forward_after_episode_end():
        env_before, env_after, _ = execute_forward_test(3, include_internals=False)
        for callsign in env_after.aircraft.keys():
            assert env_after.aircraft[callsign].current_sector == env_before.aircraft[callsign].current_sector
            assert env_after.aircraft[callsign].selected_fl == env_before.aircraft[callsign].selected_fl
            assert env_after.aircraft[callsign].lat == env_before.aircraft[callsign].lat

    test_jump_before_episode_start()
    test_forward_to_episode_start()
    test_forward_within_time_range()
    test_forward_within_time_range_with_internals()
    test_forward_to_episode_end()
    test_forward_after_episode_end()

def test_add():
    """
    Test concatenating event handlers.
    """
    env, _, _, callsign, _, _ = setup_test_sim()

    event_handler_a = EventHandler()
    event_handler_b = EventHandler()
    datetime_a = env.datetime
    datetime_b = env.datetime + timedelta(seconds=1)
    sectors_configuration_a = [("SECTOR_A", ["A1", "A2"])]
    sectors_configuration_b = [("SECTOR_B", ["B1", "B2"])]

    event_handler_a.add_radar_event(datetime_a, callsign, 51.0, 3.1, 310, 180.0)
    event_handler_a.add_sectors_event(datetime_a, sectors_configuration_a)
    event_handler_b.add_radar_event(datetime_b, callsign, 52.0, 2.9, 300, 200.0)
    event_handler_b.add_sectors_event(datetime_b, sectors_configuration_b)

    result = event_handler_a.add(event_handler_b)

    assert result is event_handler_a
    assert len(event_handler_a.radar_df) == 2
    assert len(event_handler_a.sectors_df) == 2
    assert set(event_handler_a.radar_df["fl"].tolist()) == {310, 300}
    assert sectors_configuration_a in event_handler_a.sectors_df["sectors_configuration"].tolist()
    assert sectors_configuration_b in event_handler_a.sectors_df["sectors_configuration"].tolist()

def test_trim():
    """
    Test trimming events by datetime comparison.
    """
    env, _, _, callsign, _, _ = setup_test_sim()
    datetime_a = env.datetime
    datetime_b = env.datetime + timedelta(seconds=1)
    sectors_configuration = [("SPRINGFIELD", ["JIMMI", "NATOR"])]
    expected_by_operator = {
        "<": [datetime_a, datetime_b],
        "<=": [datetime_b],
        ">": [datetime_a],
        ">=": [],
    }

    def build_handler_with_events() -> EventHandler[Aircraft]:
        event_handler = EventHandler()
        event_handler.add_radar_event(datetime_a, callsign, 51.0, 3.1, 310, 180.0)
        event_handler.add_radar_event(datetime_b, callsign, 52.0, 3.2, 320, 190.0)
        event_handler.add_clearance_event(datetime_a, callsign, "change_flight_level_to", "280")
        event_handler.add_clearance_event(datetime_b, callsign, "change_flight_level_to", "300")
        event_handler.add_sectors_event(datetime_a, sectors_configuration)
        event_handler.add_sectors_event(datetime_b, sectors_configuration)
        event_handler.add_incomm_event(datetime_a, callsign, "SPRINGFIELD")
        event_handler.add_incomm_event(datetime_b, callsign, "SPRINGFIELD")
        event_handler.add_coordination_event(
            datetime_a,
            callsign,
            "JIMMI",
            "NATOR",
            290.0,
            "SMUDJ",
            "Horizontal",
            False,
            None,
            "",
        )
        event_handler.add_coordination_event(
            datetime_b,
            callsign,
            "JIMMI",
            "NATOR",
            300.0,
            "SMUDJ",
            "Horizontal",
            False,
            None,
            "",
        )
        event_handler.add_aircraft_internals_event(datetime_a, callsign, 1.0)
        event_handler.add_aircraft_internals_event(datetime_b, callsign, 2.0)
        event_handler.add_flight_plan_event(datetime_a, callsign, ["SMUDJ", "NATEB"])
        return event_handler

    for operator, expected_index in expected_by_operator.items():
        event_handler = build_handler_with_events()
        event_handler.trim(operator, datetime_a)

        assert list(event_handler.radar_df.index) == expected_index
        assert list(event_handler.clearances_df.index) == expected_index
        assert list(event_handler.sectors_df.index) == expected_index
        assert list(event_handler.incomm_df.index) == expected_index
        assert list(event_handler.coordination_df.index) == expected_index
        assert list(event_handler.aircraft_internals_df.index) == expected_index
        # trim currently does not remove flight plan events
        assert list(event_handler.flight_df.index) == [datetime_a]

    with pytest.raises(ValueError):
        build_handler_with_events().trim("==", datetime_b)

def test_remove_simmed():
    """
    Test removing events associated with simulated aircraft callsigns.
    """
    event_handler = EventHandler()
    event_logger = EventLogger()
    datetime_test = datetime(2020, 1, 1, 10, 30, 0)
    simmed_callsign = "SIMMED1"
    non_simmed_callsign = "REAL1"

    for callsign in (simmed_callsign, non_simmed_callsign):
        event_handler.add_radar_event(datetime_test, callsign, 51.0, 3.1, 310, 180.0)
        event_handler.add_clearance_event(datetime_test, callsign, "change_flight_level_to", "290")
        event_handler.add_incomm_event(datetime_test, callsign, "SPRINGFIELD")
        event_handler.add_coordination_event(
            datetime_test,
            callsign,
            "JIMMI",
            "NATOR",
            290.0,
            "SMUDJ",
            "Horizontal",
            False,
            None,
            "",
        )
        event_handler.add_aircraft_internals_event(datetime_test, callsign, 1.0)

    event_logger.aircraft_internals_log = [
        {"callsign": simmed_callsign, "simulated": True},
        {"callsign": non_simmed_callsign, "simulated": False},
    ]

    result = event_handler.remove_simmed(event_logger)

    assert result is event_handler
    assert simmed_callsign not in event_handler.radar_df.callsign.values
    assert simmed_callsign not in event_handler.clearances_df.callsign.values
    assert simmed_callsign not in event_handler.incomm_df.callsign.values
    assert simmed_callsign not in event_handler.coordination_df.callsign.values
    assert simmed_callsign not in event_handler.aircraft_internals_df.callsign.values

    assert non_simmed_callsign in event_handler.radar_df.callsign.values
    assert non_simmed_callsign in event_handler.clearances_df.callsign.values
    assert non_simmed_callsign in event_handler.incomm_df.callsign.values
    assert non_simmed_callsign in event_handler.coordination_df.callsign.values
    assert non_simmed_callsign in event_handler.aircraft_internals_df.callsign.values
    # remove_simmed only filters callsign-indexed dataframes; these remain unchanged.
    assert event_handler.flight_df.empty
    assert event_handler.sectors_df.empty

class TestUpdateSelectedFlFromRadar:
    """
    Test setting a selected fl from the cleared fl
    """
    test_fl: float = 275.0
    ignore_simmed: bool = False
    props_to_set: list[tuple[str, typing.Any]] = [ ('selected_fl', test_fl) ]

    def test_null_input_df(self):
        env, _, _, _, episode_start, episode_end = setup_test_sim()
        df = build_test_df_for_multiple_ac(EventDtypes.radar_dtypes, env, self.props_to_set)
        df_empty = df.drop(df.index)
        update_selected_fl_from_radar(env, df_empty, episode_start, episode_end, self.ignore_simmed)
        assert all(ac.selected_fl != self.test_fl for ac in env.aircraft.values())

    def test_across_time_window(self):
        # Test, before, on start boundary, during, on end boundary and after
        # Set start time one second earlier than episode start then add 1 second each time
        # On times during the interval and on final boundary should pass (we check properties have changed)
        for i in range(0, 5):
            env, _, _, _, episode_start, episode_end = setup_test_sim()
            ac_before = { callsign: copy.deepcopy(ac) for callsign, ac in env.aircraft.items() }
            start_time = episode_start - timedelta(seconds=1)
            df_datetime = start_time + timedelta(seconds=1) * i
            df = build_test_df_for_multiple_ac(EventDtypes.radar_dtypes, env, self.props_to_set, df_datetime)
            update_selected_fl_from_radar(env, df, episode_start, episode_end, self.ignore_simmed)

            if i > LOWER_EPISODE_BOUNDARY and i <= UPPER_EPISODE_BOUNDARY:
                assert all(ac.selected_fl == self.test_fl for ac in env.aircraft.values())
            else:
                assert all(is_deeply_equal(ac, ac_before[callsign]) for callsign, ac in env.aircraft.items())

    def test_ignore_simmed(self):
        # We picked a simmed aircraft so set ignore simmed and check no cfl change
        env, _, _, _, episode_start, episode_end = setup_test_sim()
        df = build_test_df_for_multiple_ac(EventDtypes.radar_dtypes, env, self.props_to_set)
        set_cleared_fl_to_selected_fl(env, df, episode_start, episode_end, ignore_simmed=True)
        assert all(ac.selected_fl != self.test_fl for ac in env.aircraft.values())

class TestSetCleardeFlToSelectedFl:
    """
    Test setting a cleared fl from the selected fl of a radar event
    """

    test_fl: float = 275.0
    ignore_simmed: bool = False
    props_to_set: list[tuple[str, typing.Any]] = [ ("selected_fl", test_fl) ]

    def test_null_input_df(self):
        env, _, _, _, episode_start, episode_end = setup_test_sim()
        df = build_test_df_for_multiple_ac(EventDtypes.radar_dtypes, env, self.props_to_set)
        df_empty = df.drop(df.index)
        set_cleared_fl_to_selected_fl(env, df_empty, episode_start, episode_end, self.ignore_simmed)
        assert all(ac.cleared_fl != self.test_fl for ac in env.aircraft.values())

    def test_across_time_window(self):
        # Test, before, on start boundary, during, on end boundary and after
        # Set start time one second earlier than episode start then add 1 second each time
        # On times during the interval and on final boundary should pass (we check properties have changed)
        for i in range(0, 5):
            env, _, _, _, episode_start, episode_end = setup_test_sim()
            ac_before = { callsign: copy.deepcopy(ac) for callsign, ac in env.aircraft.items() }
            start_time = episode_start - timedelta(seconds=1)
            df_datetime = start_time + timedelta(seconds=1) * i
            df = build_test_df_for_multiple_ac(EventDtypes.radar_dtypes, env, self.props_to_set, df_datetime)
            set_cleared_fl_to_selected_fl(env, df, episode_start, episode_end, self.ignore_simmed)

            if i > LOWER_EPISODE_BOUNDARY and i <= UPPER_EPISODE_BOUNDARY:
                assert all(ac.cleared_fl == self.test_fl for ac in env.aircraft.values())
            else:
                assert all(is_deeply_equal(ac, ac_before[callsign]) for callsign, ac in env.aircraft.items())
    
    def test_ignore_simmed(self):
        # We picked a simmed aircraft so set ignore simmed and check no cfl change
        env, _, _, _, episode_start, episode_end = setup_test_sim()
        df = build_test_df_for_multiple_ac(EventDtypes.radar_dtypes, env, self.props_to_set)
        set_cleared_fl_to_selected_fl(env, df, episode_start, episode_end, ignore_simmed=True)
        assert all(ac.cleared_fl != self.test_fl for ac in env.aircraft.values())

class TestUpdateFromClearances():
    """
    Test passing a clearance dataframe with the various options on and off
    """

    mock_fl: int = 310
    ignore_simmed: bool = False
    props_to_set: list[tuple[str, typing.Any]]= [ ('kind', 'change_flight_level_to'), ('value', mock_fl) ]

    def test_null_input_df(self):
        env, manager, _, _, episode_start, episode_end = setup_test_sim()
        df = build_test_df_for_multiple_ac(EventDtypes.clearance_dtypes, env, self.props_to_set)
        empty_df = df.drop(df.index)
        update_from_clearances(env, manager, empty_df, episode_start, episode_end, self.ignore_simmed)
        manager.process_actions()
        assert all(ac.cleared_instructions.fl != self.mock_fl for ac in env.aircraft.values())
    
    def test_across_time_window(self):
        # Test, before, on start boundary, during, on end boundary and after
        # Set start time one second earlier than episode start then add 1 second each time
        # On times during the interval and on final boundary should pass (we check properties have changed)
        for i in range(0, 5):
            env, manager, _, _, episode_start, episode_end = setup_test_sim()
            ac_before = { callsign: copy.deepcopy(ac) for callsign, ac in env.aircraft.items() }
            start_time = episode_start - timedelta(seconds=1)
            df_datetime = start_time + timedelta(seconds=1) * i
            df = build_test_df_for_multiple_ac(EventDtypes.clearance_dtypes, env, self.props_to_set, df_datetime)
            update_from_clearances(env, manager, df, episode_start, episode_end, self.ignore_simmed)
            manager.process_actions()

            if i > LOWER_EPISODE_BOUNDARY and i <= UPPER_EPISODE_BOUNDARY:
                assert all(ac.cleared_instructions.fl == self.mock_fl for ac in env.aircraft.values())
            else:
                assert all(is_deeply_equal(ac, ac_before[callsign]) for callsign, ac in env.aircraft.items())

    def test_ignore_simmed(self):
        env, manager, _, _, episode_start, episode_end = setup_test_sim()
        df = build_test_df_for_multiple_ac(EventDtypes.clearance_dtypes, env, self.props_to_set)
        update_from_clearances(env, manager, df, episode_start, episode_end, ignore_simmed=True)
        manager.process_actions()
        assert all(ac.cleared_instructions.fl != self.mock_fl for ac in env.aircraft.values())

    def test_parses_hold_parameters(self):
        env, manager, _, _, episode_start, episode_end = setup_test_sim()
        props_to_set = [
            ("kind", "route_direct_to,hold_at_location"),
            (
                "value",
                '{"location":[51.0,-1.0],"hold_orientation_deg":90.0,'
                '"outbound_time_s":60.0,"turn_direction":"left"}',
            ),
        ]
        df = build_test_df_for_multiple_ac(EventDtypes.clearance_dtypes, env, props_to_set)

        update_from_clearances(env, manager, df, episode_start, episode_end, ignore_simmed=False)

        assert manager._actions_to_issue
        assert all(isinstance(action.value, HoldAtLocationParameters) for action in manager._actions_to_issue)


class TestUpdateAirspaceConfiguration:
    """
    Test passing an airspace configuration update with the various options on and off
    """

    def test_null_input_df(self):
        env, manager, _, _, episode_start, episode_end = setup_test_sim()
        df = build_default_test_df(EventDtypes.sectors_dtypes, env)
        empty_df = df.drop(df.index)
        expected_airspace = copy.deepcopy(env.airspace.airspace_configuration)
        update_airspace_configuration(env, manager, empty_df, episode_start, episode_end)
        resulting_airspace = env.airspace.airspace_configuration
        assert resulting_airspace == expected_airspace

    def test_bandbox_removal(self):
        # Single-sector layout: re-applying the existing SPRINGFIELD config is a no-op
        # (split_sector skips when the bandbox maps to only one individual sector)
        env, manager, _,  _, episode_start, episode_end = setup_test_sim()
        df = build_default_test_df(EventDtypes.sectors_dtypes, env)
        df["sectors_configuration"] = [[("SPRINGFIELD", ["SPRINGFIELD"])]]
        expected_airspace = copy.deepcopy(env.airspace.airspace_configuration)
        update_airspace_configuration(env, manager, df, episode_start, episode_end)
        resulting_airspace = env.airspace.airspace_configuration
        assert resulting_airspace == expected_airspace

    def test_bandbox_creation(self):
        # Attempt to create a bandbox from non-existent individual sectors.
        # split_sector("SPRINGFIELD") is a no-op (maps to single sector),
        # so the bandbox creation is rejected and the config is unchanged.
        env, manager, _,  _, episode_start, episode_end = setup_test_sim()
        df = build_default_test_df(EventDtypes.sectors_dtypes, env)
        bandbox_name = "BANDBOX1"
        bandboxed_sectors = ["NONEXISTENT_A", "NONEXISTENT_B"]
        new_bandboxed_sectors = {bandbox_name: bandboxed_sectors}
        df["sectors_configuration"] = [ new_bandboxed_sectors ]
        expected_airspace = copy.deepcopy(env.airspace.airspace_configuration)
        update_airspace_configuration(env, manager, df, episode_start, episode_end)
        resulting_airspace = env.airspace.airspace_configuration
        assert resulting_airspace == expected_airspace

    def test_updates_outside_time_window(self):
        # Call function under test with (1) late episode start and (2) early episode end and check rejection (i.e. no state change)
        env, manager, _,  _, episode_start, episode_end = setup_test_sim()
        df = build_default_test_df(EventDtypes.sectors_dtypes, env)
        df["sectors_configuration"] = [[("JIMMI", ["JIMMI"])]]
        expected_airspace = copy.deepcopy(env.airspace.airspace_configuration)
        later_start_time = episode_start + timedelta(seconds=1)
        update_airspace_configuration(env, manager, df, later_start_time, episode_end)
        earlier_end_time = episode_end - timedelta(seconds=2)
        update_airspace_configuration(env, manager, df, episode_start, earlier_end_time)
        resulting_airspace = env.airspace.airspace_configuration
        assert resulting_airspace == expected_airspace

class TestUpdateCoordination:
    """
    Test passing a coordination dataframe with the various options on and off
    """

    fl: float = 290.0
    fix: str = "SMUDJ"
    direction: str = "Horizontal"
    ignore_simmed: bool = False

    @staticmethod
    def get_latest_coord_for_callsign(env: Environment[Aircraft, WindField, WindField], callsign: str) -> Coordination:
        [current_sector, _] = get_to_and_from_sectors(env, callsign)        
        coords = env.coordinations.get(callsign)
        assert coords is not None and len(coords) > 0
        return coords[0]
    
    def test_null_coord_df(self):
        env, _, _, callsign, episode_start, episode_end = setup_test_sim()
        df = build_default_test_df(EventDtypes.coord_dtypes, env)
        initial_coord = self.get_latest_coord_for_callsign(env, callsign)
        df_empty = df.drop(df.index)
        update_coordination(env, df_empty, episode_start, episode_end, self.ignore_simmed)
        latest_coord = self.get_latest_coord_for_callsign(env, callsign)
        assert initial_coord == latest_coord

    def test_coord_outside_time_window(self):
        # Call update coordination with (1) late episode start and (2) early episode end and check rejection (i.e. no coord change)
        env, _, _, callsign, episode_start, episode_end = setup_test_sim()
        df = build_default_test_df(EventDtypes.coord_dtypes, env)
        initial_coord = self.get_latest_coord_for_callsign(env, callsign)

        later_start_time = episode_start + timedelta(seconds=1)
        update_coordination(env, df, later_start_time, episode_end, self.ignore_simmed)
        latest_coord = self.get_latest_coord_for_callsign(env, callsign)
        assert initial_coord == latest_coord

        earlier_end_time = episode_end - timedelta(seconds=2)
        update_coordination(env, df, episode_start, earlier_end_time, self.ignore_simmed)
        latest_coord = self.get_latest_coord_for_callsign(env, callsign)
        assert initial_coord == latest_coord

    def test_valid_coordination(self):
        env, _, _, callsign, episode_start, episode_end = setup_test_sim()
        df = build_default_test_df(EventDtypes.coord_dtypes, env)
        existing_coords = env.coordinations.get(callsign)
        from_sector = existing_coords[0].from_sector if existing_coords else env.aircraft[callsign].current_sector
        [_, to_sector] = get_to_and_from_sectors(env, callsign)
        df["callsign"] = callsign
        df["from_sector"] = from_sector
        df["to_sector"] = to_sector
        df["fl"] = self.fl
        df["fix"] = self.fix
        df["direction"] = self.direction
        df["level_by"] = False
        df["level_by_details"] = None
        df["secondary_coord_conditions"] = ""
        update_coordination(env, df, episode_start, episode_end, self.ignore_simmed)
        latest_coord = self.get_latest_coord_for_callsign(env, callsign)
        assert latest_coord.fl == self.fl
        assert latest_coord.fix == self.fix
        assert latest_coord.to_sector == to_sector
        assert latest_coord.direction == self.direction

    def test_earlier_coordinations_removed(self):
        env, _, _, callsign, episode_start, episode_end = setup_test_sim()
        [current_sector, to_sector] = get_to_and_from_sectors(env, callsign)
        df = build_default_test_df(EventDtypes.coord_dtypes, env)
        df["callsign"] = callsign
        df["from_sector"] = current_sector
        df["to_sector"] = to_sector
        df["fl"] = self.fl
        df["fix"] = self.fix
        df["direction"] = self.direction
        df["level_by"] = False
        df["level_by_details"] = None
        df["secondary_coord_conditions"] = ""
  
        # Create a second coordination with the datetime later than first but within the window
        the_datetime = episode_end
        second_coord_fl = 340
        second_coord = pd.DataFrame({
            "callsign": callsign,
            "from_sector": current_sector, 
            "to_sector": to_sector, 
            "fl": second_coord_fl, 
            "fix": self.fix, 
            "direction": self.direction,  
            "level_by": False,
            "level_by_details": None, 
            "secondary_coord_conditions": ""
            },
            index=[the_datetime]
        )
        df = pd.concat([df, second_coord])
        update_coordination(env, df, episode_start, episode_end, self.ignore_simmed)
        # Test that the coordinations for flight leaving the current sector have been filtered to latest only
        latest_coord = self.get_latest_coord_for_callsign(env, callsign)
        assert latest_coord.fl == second_coord_fl

    def test_ignore_simmed(self):
        env, _, _, callsign, episode_start, episode_end = setup_test_sim()
        [current_sector, to_sector] = get_to_and_from_sectors(env, callsign)
        df = build_default_test_df(EventDtypes.coord_dtypes, env)
        df["callsign"] = callsign
        df["from_sector"] = current_sector
        df["to_sector"] = to_sector
        df["fl"] = self.fl
        df["fix"] = self.fix
        df["direction"] = self.direction
        df["level_by"] = False
        df["level_by_details"] = None
        df["secondary_coord_conditions"] = ""

        coords = env.coordinations.get(callsign)
        assert coords is not None
        initial_coord = coords[0]
        env.aircraft[callsign].simulated = True
        update_coordination(env, df, episode_start, episode_end, ignore_simmed=True)
        coords = env.coordinations.get(callsign)
        assert coords is not None
        latest_coord = coords[0]
        assert initial_coord == latest_coord

    def test_coord_with_level_by(self):
        env, _, _, callsign, episode_start, episode_end = setup_test_sim()
        [current_sector, to_sector] = get_to_and_from_sectors(env, callsign)
        df = build_default_test_df(EventDtypes.coord_dtypes, env)
        df["callsign"] = callsign
        df["from_sector"] = current_sector
        df["to_sector"] = to_sector
        df["fl"] = self.fl
        df["fix"] = self.fix
        df["direction"] = self.direction
        df["level_by_details"] = None
        df["secondary_coord_conditions"] = ""
        df["level_by"] = True
        with pytest.raises(ValidationError) as exception:
            update_coordination(env, df, episode_start, episode_end, self.ignore_simmed)
        assert "Must have level_by_details if level_by is True" in str(exception.value)

        level_by_details = {self.fix: 290.0}
        df.at[df.index[0], "level_by_details"] = level_by_details
        update_coordination(env, df, episode_start, episode_end, self.ignore_simmed)
        coords = env.coordinations.get(callsign, current_sector)
        assert coords is not None
        latest_coord = coords[0]
        assert latest_coord.level_by_details == level_by_details

    def test_secondary_coord_conditions(self):
        env, _, _, callsign, episode_start, episode_end = setup_test_sim()
        [current_sector, to_sector] = get_to_and_from_sectors(env, callsign)
        df = build_default_test_df(EventDtypes.coord_dtypes, env)
        df["callsign"] = callsign
        df["from_sector"] = current_sector
        df["to_sector"] = to_sector
        df["fl"] = self.fl
        df["fix"] = self.fix
        df["direction"] = self.direction
        df["level_by"] = False
        df["level_by_details"] = None
        df["secondary_coord_conditions"] = ""
        coord_condition = "CONDITION"
        df["secondary_coord_conditions"] = coord_condition
        update_coordination(env, df, episode_start, episode_end, self.ignore_simmed)
        coords = env.coordinations.get(callsign, current_sector)
        assert coords is not None
        latest_coord = coords[0]
        assert latest_coord.secondary_coord_conditions == coord_condition

class TestUpdateFromIncomm:
    """
    Test passing an incomm dataframe with the various options on and off
    """
    ignore_simmed = False


    def test_null_incomm_df(self):
        env, _, _, callsign, episode_start, episode_end = setup_test_sim()
        [current_sector, _] = get_to_and_from_sectors(env, callsign)
        df = build_default_test_df(EventDtypes.incomm_dtypes, env)

        df_empty = df.drop(df.index)
        update_incomm(env, df_empty, episode_start, episode_end, self.ignore_simmed)
        assert env.aircraft[callsign].current_sector == current_sector

    def test_incomm_outside_time_window(self):
        env, _, _, callsign, episode_start, episode_end = setup_test_sim()
        df = build_default_test_df(EventDtypes.incomm_dtypes, env)
        [current_sector, to_sector] = get_to_and_from_sectors(env, callsign)
        df["callsign"] = callsign
        df["sector_name"] = to_sector

        update_incomm(env, df, episode_start + timedelta(seconds=1), episode_end, self.ignore_simmed)
        assert env.aircraft[callsign].current_sector == current_sector

        episode_end = env.datetime + timedelta(seconds=-1)
        update_incomm(env, df, episode_start, episode_end, self.ignore_simmed)
        assert env.aircraft[callsign].current_sector == current_sector

    def test_valid_incomm(self):
        env, _, _, callsign, episode_start, episode_end = setup_test_sim()
        df = build_default_test_df(EventDtypes.incomm_dtypes, env)
        [_, to_sector] = get_to_and_from_sectors(env, callsign)
        df["callsign"] = callsign
        df["sector_name"] = to_sector
        df.index = df.index + timedelta(seconds=1)
        update_incomm(env, df, episode_start, episode_end, self.ignore_simmed)
        assert env.aircraft[callsign].current_sector == to_sector

    def test_incomm_to_current_sector(self):
        env, _, _, callsign, episode_start, episode_end = setup_test_sim()
        df = build_default_test_df(EventDtypes.incomm_dtypes, env)
        [current_sector, _] = get_to_and_from_sectors(env, callsign)
        df["callsign"] = callsign
        df["sector_name"] = current_sector
        df.index = df.index + timedelta(seconds=1)
        update_incomm(env, df, episode_start, episode_end, self.ignore_simmed)
        assert env.aircraft[callsign].current_sector == current_sector

    def test_incomm_to_non_existent_sector(self):
        env, _, _, callsign, episode_start, episode_end = setup_test_sim()
        df = build_default_test_df(EventDtypes.incomm_dtypes, env)
        df["callsign"] = callsign
        df.index = df.index + timedelta(seconds=1)
        non_existent_sector = "NON_EXISTENT"
        df["sector_name"] = non_existent_sector
        update_incomm(env, df, episode_start, episode_end, self.ignore_simmed)
        assert env.aircraft[callsign].current_sector == "background"

    def test_ignore_simmed(self):
        env, _, _, callsign, episode_start, episode_end = setup_test_sim()
        [current_sector, to_sector] = get_to_and_from_sectors(env, callsign)
        df = build_default_test_df(EventDtypes.incomm_dtypes, env)
        df["callsign"] = callsign
        df["sector_name"] = to_sector
        df.index = df.index + timedelta(seconds=1)
        update_incomm(env, df, episode_start, episode_end, True)
        assert env.aircraft[callsign].current_sector == current_sector

class TestUpdateAircraftInternals:
    """
    Test setting aircraft internals
    """
    ignore_simmed = False
    props_to_set = [("current_sector", "IMAGINARY_SECTOR"), ("pilot_type", "Pilot"), ("predictor_params", { "turn_radius": 45.5 })]

    def test_null_input_df(self):
        env, _, _, _, episode_start, episode_end = setup_test_sim()
        df = build_test_df_for_multiple_ac(EventDtypes.aircraft_internals_dtypes, env, self.props_to_set)
        empty_df = df.drop(df.index)
        ac_before = { callsign: copy.deepcopy(ac) for callsign, ac in env.aircraft.items() }
        update_aircraft_internals(env, empty_df, episode_start, episode_end, self.ignore_simmed)
        assert all(is_deeply_equal(ac, ac_before[callsign]) for callsign, ac in env.aircraft.items())

    def test_outside_allowed_time_window(self ):
        # Test, before, on start boundary, during, on end boundary and after
        # Set start time one second earlier than episode start then add 1 second each time
        # On times during the interval and on final boundary should pass (we check properties have changed)
        for i in range(0, 5):
            env, _, _, _, episode_start, episode_end = setup_test_sim()
            ac_before = { callsign: copy.deepcopy(ac) for callsign, ac in env.aircraft.items() }
            start_time = episode_start - timedelta(seconds=1)
            df_datetime = start_time + timedelta(seconds=1) * i
            df = build_test_df_for_multiple_ac(EventDtypes.aircraft_internals_dtypes, env, self.props_to_set, df_datetime)
            env_after = update_aircraft_internals(env, df, episode_start, episode_end, self.ignore_simmed)
            if i > LOWER_EPISODE_BOUNDARY and i <= UPPER_EPISODE_BOUNDARY:
                assert all(
                    all(getattr(ac, prop) == value for prop, value in self.props_to_set if prop != "pilot_type")
                    for ac in env.aircraft.values()
                )
            else:
                assert all(is_deeply_equal(ac, ac_before[callsign]) for callsign, ac in env_after.aircraft.items())


def test_add_incomm_event():
    simulator = Simulator.from_category(category="Springfield", scenario_name="example-scenario")
    
    # Ensure the airspace is as we expect it, with Springfield.
    assert "SPRINGFIELD" in simulator.manager.environment.airspace.sectors

    # Force an incomm of AIR01 into the bandboxed sector of SPRINGFIELD in the next tick.
    simulator.manager.event_handler.add_incomm_event(
            simulator.manager.environment.datetime + timedelta(seconds=1),
            "AIR01",
            "SPRINGFIELD"
            )

    simulator.evolve(6)

    # It should have SPRINGFIELD as the current sector
    assert simulator.manager.environment.aircraft["AIR01"].current_sector == "SPRINGFIELD"

def test_update_aircraft_attribute():
    """
    Test passing a update to aircraft attributes with the various options on and off
    """
    ignore_simmed = False
    mock_selected_fl = 275
    props_to_set = [("attribute_name", "selected_fl"), ("value", mock_selected_fl)]

    def test_null_input_df():
        env, _, ac, _, episode_start, episode_end = setup_test_sim()
        df = build_default_test_df(EventDtypes.ac_attribute_update_dtypes, env)
        empty_df = df.drop(df.index)
        ac_before = copy.deepcopy(ac)
        update_aircraft_attribute(env, empty_df, episode_start, episode_end, ignore_simmed)
        assert is_deeply_equal(ac, ac_before)

    def test_across_time_window():
        # Test, before, on start boundary, during, on end boundary and after
        # Set start time one second earlier than episode start then add 1 second each time
        # On times during the interval and on final boundary should pass (we check properties have changed)
        for i in range(0, 5):
            [env, _, _, _, episode_start, episode_end] = setup_test_sim()
            ac_before = { callsign: copy.deepcopy(ac) for callsign, ac in env.aircraft.items() }
            start_time = episode_start - timedelta(seconds=1)
            df_datetime = start_time + timedelta(seconds=1) * i
            df = build_test_df_for_multiple_ac(EventDtypes.ac_attribute_update_dtypes, env, props_to_set, df_datetime)
            update_aircraft_attribute(env, df, episode_start, episode_end, ignore_simmed)

            if i > LOWER_EPISODE_BOUNDARY and i <= UPPER_EPISODE_BOUNDARY:
                assert all(
                    ac.selected_fl == mock_selected_fl for ac in env.aircraft.values()
                )
            else:
                assert all(is_deeply_equal(ac, ac_before[callsign]) for callsign, ac in env.aircraft.items())

    test_null_input_df()
    test_across_time_window()

def test_update_from_radar():
    """
    Test passing a radar update with the various options on and off
    """
    ignore_simmed = False
    props_to_set = [ ('lat', 51.0), ('lon', 3.3) ]

    def test_null_input_df():
        env, manager, _, _, episode_start, episode_end = setup_test_sim()
        df = build_test_df_for_multiple_ac(EventDtypes.radar_dtypes, env, props_to_set)
        empty_df = df.drop(df.index)
        ac_before = { callsign: copy.deepcopy(ac) for callsign, ac in env.aircraft.items() }
        type = manager.event_handler.typeof_aircraft
        update_from_radar(env, empty_df, episode_start, episode_end, ignore_simmed, type)
        assert all(is_deeply_equal(ac, ac_before[callsign]) for callsign, ac in env.aircraft.items())

    def test_across_time_window():
        # Test, before, on start boundary, during, on end boundary and after
        # Set start time one second earlier than episode start then add 1 second each time
        # On times during the interval and on final boundary should pass (we check properties have changed)
        for i in range(0, 5):
            env, manager, _, _, episode_start, episode_end = setup_test_sim()
            ac_before = { callsign: copy.deepcopy(ac) for callsign, ac in env.aircraft.items() }
            start_time = episode_start - timedelta(seconds=1)
            df_datetime = start_time + timedelta(seconds=1) * i
            df = build_test_df_for_multiple_ac(EventDtypes.radar_dtypes, env, props_to_set, df_datetime)
            type = manager.event_handler.typeof_aircraft
            update_from_radar(env, df, episode_start, episode_end, ignore_simmed, type)

            if i > LOWER_EPISODE_BOUNDARY and i <= UPPER_EPISODE_BOUNDARY:
                assert all(
                    all(getattr(ac, prop) == value for prop, value in props_to_set)
                    for ac in env.aircraft.values()
                )
            else:
                assert all(is_deeply_equal(ac, ac_before[callsign]) for callsign, ac in env.aircraft.items())

    test_null_input_df()
    test_across_time_window()

class TestUpdateFromFlightPlans:
    ignore_simmed: bool = False

    def test_null_input_df(self, flight_plan: dict[str, Any]):
        env, manager, _, _, episode_start, episode_end = setup_test_sim()
        props_to_set = list(flight_plan.items())
        df = build_test_df_for_multiple_ac(EventDtypes.flight_dtypes, env, props_to_set)
        empty_df = df.drop(df.index)
        ac_before = { callsign: copy.deepcopy(ac) for callsign, ac in env.aircraft.items() }
        update_from_flight_plans(env, empty_df, episode_start, episode_end, self.ignore_simmed)
        assert all(is_deeply_equal(ac, ac_before[callsign]) for callsign, ac in env.aircraft.items())

    def test_across_time_window(self, flight_plan: dict[str, Any]):
        # This test is slightly different to the equivalent tests for other kinds of update.
        # Specify the time window and boundaries exactly.
        # The FP check is >= episode start - 1 day and <= episode end + 12h
        env, _, _, _, episode_start, episode_end = setup_test_sim()
        time_samples = [
            episode_start - timedelta(days=1) - timedelta(seconds=1),
            episode_start - timedelta(days=1),
            episode_start - timedelta(days=1) + timedelta(hours=6),
            episode_end + timedelta(hours=12),
            episode_end + timedelta(hours=12) + timedelta(seconds=1) 
        ]
        LOWER_EPISODE_BOUNDARY = 1
        props_to_set = list(flight_plan.items())
        for i in range(0, len(time_samples)):
            env, _, _, _, episode_start, episode_end = setup_test_sim()
            df_datetime = time_samples[i]
            fp_before = { callsign: copy.deepcopy(ac.flight_plan) for callsign, ac in env.aircraft.items() }
            df = build_test_df_for_multiple_ac(EventDtypes.flight_dtypes, env, props_to_set, df_datetime)
            update_from_flight_plans(env, df, episode_start, episode_end, self.ignore_simmed)
            props_to_test = [ "origin", "dest", "unexpanded_route", "milcivil", "requested_flight_level", "filed_true_airspeed", "intention_code", "assigned_squawk" ]
            if i >= LOWER_EPISODE_BOUNDARY and i <= UPPER_EPISODE_BOUNDARY:
                assert all(
                    all(getattr(ac.flight_plan, prop) == flight_plan[prop] for prop in props_to_test)
                    for ac in env.aircraft.values()
                )
            else:
                assert all(is_deeply_equal(ac.flight_plan, fp_before[callsign]) for callsign, ac in env.aircraft.items())

def test_update_flight_plan(generate_i):
    # Airspace, AddAircraft Events and Predictor
    airspace, routes = generate_i

    # Single simulated aircraft
    em = Tactical(1, airspace=airspace, routes=routes, initialise_with_event_handler=False).create_env_manager()

    # duplicate the flight plan event, give the first an earlier end time and the second a different route
    df = em.event_handler.flight_df
    em.event_handler.flight_df = pd.concat([df, pd.DataFrame([df.iloc[0]])], ignore_index=False)
    first_plan_route = ["FIRE", "EARTH", "WATER"]
    second_plan_route = ["AIR", "SPIRIT"]
    em.event_handler.flight_df["route_filed"] = [first_plan_route, second_plan_route]

    # set end datetime of first and second flight plan to be 1 minute and 1 hour after scenario start
    first_plan_end_time = em.event_handler.radar_df.index.min() + timedelta(minutes=1)
    second_plan_end_time = em.event_handler.radar_df.index.min() + timedelta(hours=1)
    em.event_handler.flight_df["end_datetime"] = [first_plan_end_time, second_plan_end_time]

    # initialise the environment
    em.initialise_env_with_event_handler()

    # ensure that the flight plans are being updated from the event handler
    em.event_handler.ignore.flight_if_simmed = False

    # loop for 2 mins and check flight plan changes
    end_time = 120
    while em.environment.time < end_time:
        em.evolve(6)

        aircraft = em.environment.aircraft["AIR0"]

        assert aircraft.flight_plan is not None

        if em.environment.time <= first_plan_end_time.timestamp():
            assert aircraft.flight_plan.route.filed == first_plan_route

        else:
            assert aircraft.flight_plan.route.filed == second_plan_route
