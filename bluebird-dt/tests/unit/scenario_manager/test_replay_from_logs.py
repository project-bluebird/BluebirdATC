import os
import pytest

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
