from bluebird_gymnasium.rewards.expeditious import (
    expeditious_const,
    expeditious_linear,
    expeditious_quad,
    expeditious_exp,
)


def test_expeditious_const(gym_env):
    """Test `expeditious_const` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = list(tracked_aircraft.keys())[0]

    action = 0
    assert -1.0 <= expeditious_const(gym_env, callsign, action) <= 1.0


def test_expeditious_linear(gym_env):
    """Test `expeditious_linear` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = list(tracked_aircraft.keys())[0]

    action = 0
    assert -1.0 <= expeditious_linear(gym_env, callsign, action) <= 1.0


def test_expeditious_quad(gym_env):
    """Test `expeditious_quad` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = list(tracked_aircraft.keys())[0]

    action = 0
    assert -1.5 <= expeditious_quad(gym_env, callsign, action) <= 1.5


def test_expeditious_exp(gym_env):
    """Test `expeditious_exp` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = list(tracked_aircraft.keys())[0]

    action = 0
    assert 0.0 <= expeditious_exp(gym_env, callsign, action) <= 1.0
