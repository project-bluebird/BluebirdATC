import os
import numpy as np
import pytest

from bluebird_dt.core.wind import WindField
from bluebird_dt.scenario_manager.replayer_from_logs import ReplayerFromLogs
from bluebird_dt.simulator import Simulator

@pytest.mark.parametrize(
    ("category", "scenario_name"), 
    [
        ("Two Aircraft", "I-Sector"),
        ("Regular", "X-Sector"),
        ("Springfield", "Test_Scenario_1_3Aircraft_Easy")
    ],
)
def test_replay_scenario(tmp_path, monkeypatch, category, scenario_name):
    log_path = os.path.join(tmp_path, scenario_name)

    monkeypatch.setenv("BLUEBIRD_LOG_DIR", str(log_path))
    s = Simulator.from_category(category, scenario_name)
    # roll the simulation forward a bit
    for _ in range(50):
        s.evolve(6)
    s.save()
    orig_time = s.manager.environment.time

    # filename of logfile can be obtained from event_logger
    log_name = s.manager.event_logger.log_name

    s_replay = Simulator.from_category("Replay", log_name)
    assert isinstance(s, Simulator)
    # anecdotally, may need to do different number of evolve()
    # in order to match up.
    while s_replay.manager.environment.time < orig_time:
        s_replay.evolve(6)
    for callsign in s.manager.environment.aircraft:
        assert callsign in s_replay.manager.environment.aircraft
        aircraft_orig = s.manager.environment.aircraft[callsign]
        aircraft_replay = s_replay.manager.environment.aircraft[callsign]
        assert aircraft_orig.lat == aircraft_replay.lat
        assert aircraft_orig.lon == aircraft_replay.lon
        assert aircraft_orig.fl == aircraft_replay.fl
        assert aircraft_orig.heading == aircraft_replay.heading
        assert aircraft_orig.flight_plan == aircraft_replay.flight_plan
        assert aircraft_orig.ufid == aircraft_replay.ufid
        assert aircraft_orig.squawk == aircraft_replay.squawk
        assert aircraft_orig.last_passed_filed_idx == aircraft_replay.last_passed_filed_idx
        assert aircraft_orig.last_passed_current_idx == aircraft_replay.last_passed_current_idx
        assert aircraft_orig.simulated == True
        assert aircraft_replay.simulated == False
    # test the event handler dataframes
    eh_orig = s.manager.event_handler
    eh_replay = s_replay.manager.event_handler
    df_list = ["radar_df", "flight_plan_df", "clearances_df",
               "coordination_df", "sectors_df", "incomm_df", 
               "ac_internals_df", "ac_attribute_update_df"]
    for df_name in df_list:
        df_orig = eh_orig.__getattribute__(df_name)
        df_replay = eh_replay.__getattribute__(df_name)
        # not guaranteed to be same length, but replay
        # dataframe should be non-zero if original was.
        if len(df_orig) > 0:
            assert len(df_replay) > 0

def test_replay_wind(tmp_path, monkeypatch):
    s = Simulator.from_category("Two Aircraft", "I-Sector")
    wf = WindField.uniform(wind_speed=20, wind_direction=90)
    fwf = WindField.uniform(wind_speed=25, wind_direction=85)
    s.manager.environment.wind_field = wf
    s.manager.environment.forecast_wind_field = fwf
    for _ in range(5):
        s.evolve(6)
    s.save()
    
    # filename of logfile can be obtained from event_logger
    log_name = s.manager.event_logger.log_name

    s_replay = Simulator.from_category("Replay", log_name)
    assert s_replay.manager.environment.wind_field == wf
    assert s_replay.manager.environment.forecast_wind_field == fwf