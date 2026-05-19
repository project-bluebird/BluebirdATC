import numpy as np
import pytest

from datetime import timezone

from bluebird_gymnasium.envs import ViewType
from bluebird_gymnasium.envs.springfield import SpringfieldEnv
from bluebird_gymnasium.utils.types import PositionStatus, ACPositionInfo

VIEW_TYPES = list(ViewType)

def _get_env_instance(view_type: ViewType=ViewType.CENTRALIZED):
    config = SpringfieldEnv.get_default_env_config(view_type)
    gym_env = SpringfieldEnv(config=config)
    return gym_env


@pytest.mark.parametrize("view_type", VIEW_TYPES)
def test_init_exceptions(view_type: ViewType):
    """Test initialisation of the gym environment.

    Args:
        view_type: set to either CENTRALIZED (single agent)
            or DECENTRALIZED (multi agent) representations.
    """

    gym_env = _get_env_instance(view_type)


@pytest.mark.parametrize("view_type", VIEW_TYPES)
def test_reset(view_type: ViewType):
    """Test reset of the gym environment.

    Args:
        view_type: set to either CENTRALIZED (single agent)
            or DECENTRALIZED (multi agent) representations.
    """

    gym_env = _get_env_instance(view_type)
    obs, info = gym_env.reset()

    if view_type == ViewType.CENTRALIZED:
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert obs.shape == gym_env.observation_space.shape
    else:  # decentralized
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        if len(obs) > 0:
            callsign = list(obs.keys())[0]
            assert obs[callsign].shape == gym_env.observation_space.shape


@pytest.mark.parametrize("view_type", VIEW_TYPES)
def test_step(view_type: ViewType):
    """Test a forward step (given an action) in the gym environment.

    Args:
        view_type: set to either CENTRALIZED (single agent)
            or DECENTRALIZED (multi agent) representations.
    """

    gym_env = _get_env_instance(view_type)
    obs, info = gym_env.reset()
    timestep_before = gym_env.timestep

    if view_type == ViewType.CENTRALIZED:
        action = 0  # NOOP action.
        obs, reward, done, truncated, info = gym_env.step(action)
        timestep_after = gym_env.timestep

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert (timestep_before + 1) == timestep_after
        assert obs.shape == gym_env.observation_space.shape

    else:  # decentralized
        action = {}  # no action on any aircraft
        obs, reward, done, truncated, info = gym_env.step(action)
        timestep_after = gym_env.timestep

        assert isinstance(obs, dict)
        assert isinstance(reward, dict)
        assert isinstance(done, dict)
        assert isinstance(truncated, dict)
        assert isinstance(info, dict)
        assert (timestep_before + 1) == timestep_after
        if len(obs) > 0:
            callsign = list(obs.keys())[0]
            assert obs[callsign].shape == gym_env.observation_space.shape


def test_decentralized_step_does_not_stop_before_controllable_aircraft():
    """Test the initial Springfield decentralized empty-agent warm-up."""

    gym_env = _get_env_instance(ViewType.DECENTRALIZED)
    obs, info = gym_env.reset(seed=100)

    assert obs == {}

    obs, reward, done, truncated, info = gym_env.step({})

    assert done == {"__all__": False}
    assert truncated == {"__all__": False}
    assert not all(done.values())

    for _ in range(25):
        if len(obs) > 0:
            break
        obs, reward, done, truncated, info = gym_env.step({})

    assert len(obs) > 0
    assert "__all__" not in done
    assert not all(done.values())


@pytest.mark.parametrize("view_type", VIEW_TYPES)
def test_pos_information(view_type: ViewType):
    """Test `check_pos_information` method of the gym environment.

    Note, after every reset of the environment (and also the underlying
    simulator), aircraft are always spawned just outside the sector.
    When a new aircraft is spawned, the position status (pos_status)
    would be `BEFORE_ENTRY` and `transfer` flag of the aircraft
    would be `False` as the aircraft is yet to fly through the airspace to
    reach the exit fix on its filed route. If we perform some forward
    simulation steps before any test, the aircraft should be in the sector,
    thus, it should have the status `IN_SECTOR`.

    Args:
        view_type: set to either CENTRALIZED (single agent)
            or DECENTRALIZED (multi agent) representations.
    """

    if view_type == ViewType.CENTRALIZED:
        action = 0
    else:  # decentralized
        action = {}  # no action on any aircraft

    gym_env = _get_env_instance(view_type)
    obs, info = gym_env.reset()
    simulator_env = gym_env.get_simulator_env()

    # forward the simulation to the time when at least one aircraft is being
    # tracked
    tracked_data = {}
    max_steps = 100
    for _ in range(max_steps):
        tracked_data = gym_env.get_tracked_aircraft_data()
        if tracked_data:
            break

        gym_env.step(action)

    else:
        pytest.fail(f"No aircraft were tracked within the step limit of {max_steps}.")

    callsign = next(iter(tracked_data))
    # assume that aircraft is currently position before the sector entry.
    # this is not always the case as aircraft are sometimes spawned in
    # within the sector in lms/lus. the method call below will check
    # and return the correct information (status and distances).
    ret: ACPositionInfo = gym_env.check_pos_information(
        callsign, PositionStatus.BEFORE_ENTRY, False, False, None
    )

    if ret.position_status == PositionStatus.BEFORE_ENTRY:
        assert ret.incomm_status is False
        assert ret.outcomm_status is False
        assert ret.dist_to_sector_entry > 0.0
        assert ret.dist_away_from_sector_exit == 0.0
        assert ret.dist_away_from_incorrect_sector_exit == 0.0
        assert ret.incorrect_exit_position is None
    elif ret.position_status == PositionStatus.IN_SECTOR:
        assert ret.incomm_status is True
        assert ret.outcomm_status is False
        assert ret.dist_to_sector_entry == 0.0
        assert ret.dist_away_from_sector_exit == 0.0
        assert ret.dist_away_from_incorrect_sector_exit == 0.0
        assert ret.incorrect_exit_position is None
    else:
        assert False
