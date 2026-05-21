from bluebird_gymnasium.rewards.conflict_resolution import (
    conflict_resolution_exp,
    conflict_resolution_tanh,
)


def test_conflict_resolution_exp(gym_env):
    """Test `conflict_resolution_exp` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = list(tracked_aircraft.keys())[0]

    action = 0
    assert -1.0 <= conflict_resolution_exp(gym_env, callsign, action) <= 0.0


def test_conflict_resolution_tanh(gym_env):
    """Test `conflict_resolution_tanh` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = list(tracked_aircraft.keys())[0]

    action = 0
    assert -1.0 <= conflict_resolution_tanh(gym_env, callsign, action) <= 0.0
