from bluebird_gymnasium.rewards.lateral_next_fix_proximity import (
    lateral_next_fix_proximity_dist_exp,
    lateral_next_fix_proximity_bacnf,
    lateral_next_fix_proximity_bpfnf,
)


def test_lateral_next_fix_proximity_dist_exp(gym_env):
    """Test `lateral_next_fix_proximity_dist_exp` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = list(tracked_aircraft.keys())[0]

    action = 0
    assert (
        0.0
        <= lateral_next_fix_proximity_dist_exp(gym_env, callsign, action)
        <= 1.0
    )


def test_lateral_next_fix_proximity_bpfnf(gym_env):
    """Test `lateral_next_fix_proximity_bpfnf` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = list(tracked_aircraft.keys())[0]

    action = 0
    next_fix_idx = 1  # assume the next fix index is 1
    assert (
        lateral_next_fix_proximity_bpfnf(
            gym_env, callsign, action, next_fix_idx
        )
        <= 0.0
    )


def test_lateral_next_fix_proximity_bacnf(gym_env):
    """Test `lateral_next_fix_proximity_bacnf` reward function."""

    tracked_aircraft = gym_env.get_tracked_aircraft_data()
    callsign = list(tracked_aircraft.keys())[0]

    action = 0
    assert lateral_next_fix_proximity_bacnf(gym_env, callsign, action) <= 0.0
