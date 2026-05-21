from bluebird_gymnasium.rewards.lateral_centreline_distance import (
    lateral_centreline_distance_exp,
    lateral_centreline_distance_linear,
    lateral_centreline_distance_quad,
    lateral_centreline_distance_shaped,
)


def test_lateral_centreline_distance_linear(gym_env):
    """Test `lateral_centreline_distance_linear` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = list(tracked_aircraft.keys())[0]

    action = 0
    assert lateral_centreline_distance_linear(gym_env, callsign, action) <= 0.0


def test_lateral_centreline_distance_quad(gym_env):
    """Test `lateral_centreline_distance_quad` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = list(tracked_aircraft.keys())[0]

    action = 0
    assert lateral_centreline_distance_quad(gym_env, callsign, action) <= 2.5


def test_lateral_centreline_distance_exp(gym_env):
    """Test `lateral_centreline_distance_exp` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = list(tracked_aircraft.keys())[0]

    action = 0
    assert (
        0.0 <= lateral_centreline_distance_exp(gym_env, callsign, action) <= 1.0
    )


def test_lateral_centreline_distance_shaped(gym_env):
    """Test `lateral_centreline_distance_shaped` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = list(tracked_aircraft.keys())[0]

    action = 0
    assert lateral_centreline_distance_shaped(gym_env, callsign, action) == 1.0

    action = 1
    assert (
        -1.0
        <= lateral_centreline_distance_shaped(gym_env, callsign, action)
        <= 1.0
    )
