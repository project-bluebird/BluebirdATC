import numpy as np
import pytest
from gymnasium.error import ResetNeeded

from bluebird_gymnasium.envs import ViewType
from bluebird_gymnasium.envs.infinite import InfiniteEnv
from bluebird_gymnasium.envs.sector_x import SectorXEnv
from bluebird_gymnasium.utils.types import ACStateTracker, PositionStatus

VIEW_TYPES = list(ViewType)


def _get_env_instance(view_type: ViewType = ViewType.CENTRALIZED) -> SectorXEnv:
    config = SectorXEnv.get_default_env_config(view_type)
    return SectorXEnv(config=config)


@pytest.mark.parametrize("view_type", VIEW_TYPES)
def test_global_terminated_truncated(view_type: ViewType):
    """Test the global terminated and truncated flags.

    Args:
        view_type: set to either CENTRALIZED (single agent)
            or DECENTRALIZED (multi agent) representations.
    """

    gym_env = _get_env_instance(view_type)
    obs, info = gym_env.reset()
    timestep_before = gym_env.timestep

    if view_type == ViewType.CENTRALIZED:
        action = 0  # NOOP action.
        obs, reward, terminated, truncated, info = gym_env.step(action)
        timestep_after = gym_env.timestep

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert (timestep_before + 1) == timestep_after
        assert obs.shape == gym_env.observation_space.shape
        assert "__global_terminated__" in info
        assert "__global_truncated__" in info
        assert terminated == info["__global_terminated__"]
        assert truncated == info["__global_truncated__"]

        if gym_env.timestep < gym_env.maxstep:
            assert terminated is False
            assert truncated is False
            assert info["__global_terminated__"] is False
            assert info["__global_truncated__"] is False

            num_remaining_steps = gym_env.maxstep - gym_env.timestep
            for _ in range(num_remaining_steps):
                obs, reward, terminated, truncated, info = gym_env.step(action)

        # which of the two flags is set depends on the aircraft in the
        # scenario at the end timestep (see the `end_criteria` tests below).
        # here, check that exactly one of them is set.
        assert terminated is not truncated
        assert terminated == info["__global_terminated__"]
        assert truncated == info["__global_truncated__"]

        # try to step in a scenario that has ended. this should raise a
        # ResetNeeded error which is caught by pytest. failure to raise the
        # error leads to a failed test.
        with pytest.raises(ResetNeeded):
            _ = gym_env.step(action)

    else:  # decentralised
        action = {}  # no action on any aircraft
        obs, reward, terminated, truncated, info = gym_env.step(action)
        timestep_after = gym_env.timestep

        assert isinstance(obs, dict)
        assert isinstance(reward, dict)
        assert isinstance(terminated, dict)
        assert isinstance(truncated, dict)
        assert isinstance(info, dict)
        assert (timestep_before + 1) == timestep_after
        if len(obs) > 0:
            callsign = next(iter(obs.keys()))
            assert obs[callsign].shape == gym_env.observation_space.shape
        assert "__global_terminated__" in info
        assert "__global_truncated__" in info

        if gym_env.timestep < gym_env.maxstep:
            assert info["__global_terminated__"] is False
            assert info["__global_truncated__"] is False

            num_remaining_steps = gym_env.maxstep - gym_env.timestep
            for _ in range(num_remaining_steps):
                obs, reward, terminated, truncated, info = gym_env.step(action)

        # which of the two flags is set depends on the aircraft in the
        # scenario at the end timestep (see the `end_criteria` tests below).
        # here, check that exactly one of them is set.
        assert info["__global_terminated__"] is not info["__global_truncated__"]

        # try to step in a scenario that has ended. this should raise a
        # ResetNeeded error which is caught by pytest. failure to raise the
        # error leads to a failed test.
        with pytest.raises(ResetNeeded):
            _ = gym_env.step(action)


def _set_end_of_episode(gym_env: SectorXEnv, tracker: dict[str, ACStateTracker]) -> None:
    """Put the environment at its end timestep with the given tracked aircraft."""

    gym_env.timestep = gym_env.maxstep
    gym_env.ac_tracker = tracker


@pytest.mark.parametrize("view_type", VIEW_TYPES)
def test_end_criteria_before_maxstep(view_type: ViewType):
    """Both flags are False before the end timestep is reached."""

    gym_env = _get_env_instance(view_type)
    gym_env.reset()
    gym_env.timestep = gym_env.maxstep - 1

    assert gym_env.end_criteria() == (False, False)


@pytest.mark.parametrize("view_type", VIEW_TYPES)
@pytest.mark.parametrize(
    "tracker",
    [
        {},
        {"AC1": ACStateTracker(pos_status=PositionStatus.EXIT_REACHED, outcomm_status=False)},
        {"AC1": ACStateTracker(pos_status=PositionStatus.OUT_SECTOR, outcomm_status=False)},
        # outcommed aircraft still inside the sector are no longer active
        {"AC1": ACStateTracker(pos_status=PositionStatus.IN_SECTOR, outcomm_status=True)},
        {
            "AC1": ACStateTracker(pos_status=PositionStatus.IN_SECTOR, outcomm_status=True),
            "AC2": ACStateTracker(pos_status=PositionStatus.EXIT_REACHED, outcomm_status=True),
        },
    ],
)
def test_end_criteria_terminated(view_type: ViewType, tracker: dict):
    """terminated is set when no aircraft is active at the end timestep."""

    gym_env = _get_env_instance(view_type)
    gym_env.reset()
    _set_end_of_episode(gym_env, tracker)

    assert gym_env.end_criteria() == (True, False)


@pytest.mark.parametrize("view_type", VIEW_TYPES)
@pytest.mark.parametrize(
    "tracker",
    [
        {"AC1": ACStateTracker(pos_status=PositionStatus.IN_SECTOR, outcomm_status=False)},
        {"AC1": ACStateTracker(pos_status=PositionStatus.BEFORE_ENTRY, outcomm_status=False)},
        {
            "AC1": ACStateTracker(pos_status=PositionStatus.EXIT_REACHED, outcomm_status=False),
            "AC2": ACStateTracker(pos_status=PositionStatus.IN_SECTOR, outcomm_status=False),
        },
    ],
)
def test_end_criteria_truncated(view_type: ViewType, tracker: dict):
    """truncated is set when at least one aircraft is active at the end timestep."""

    gym_env = _get_env_instance(view_type)
    gym_env.reset()
    _set_end_of_episode(gym_env, tracker)

    assert gym_env.end_criteria() == (False, True)


@pytest.mark.parametrize("view_type", VIEW_TYPES)
def test_end_criteria_out_sector_control(view_type: ViewType):
    """Out of sector aircraft are active only when out of sector control is enabled."""

    gym_env = _get_env_instance(view_type)
    gym_env.reset()
    tracker = {"AC1": ACStateTracker(pos_status=PositionStatus.OUT_SECTOR, outcomm_status=False)}
    _set_end_of_episode(gym_env, tracker)

    gym_env.out_sector_control = True
    assert gym_env.end_criteria() == (False, True)

    gym_env.out_sector_control = False
    assert gym_env.end_criteria() == (True, False)

    # an outcommed out of sector aircraft is not active, even with out of
    # sector control enabled
    gym_env.out_sector_control = True
    gym_env.ac_tracker["AC1"].outcomm_status = True
    assert gym_env.end_criteria() == (True, False)


@pytest.mark.parametrize("view_type", VIEW_TYPES)
def test_end_criteria_infinite_env(view_type: ViewType):
    """InfiniteEnv always truncates (never terminates) at the end timestep."""

    config = InfiniteEnv.get_default_env_config(view_type)
    gym_env = InfiniteEnv(config=config)
    gym_env.reset()

    gym_env.timestep = gym_env.maxstep - 1
    assert gym_env.end_criteria() == (False, False)

    gym_env.timestep = gym_env.maxstep
    gym_env.ac_tracker = {}
    assert gym_env.end_criteria() == (False, True)
