import numpy as np
import pytest

from gymnasium.error import ResetNeeded

from bluebird_gymnasium.envs import ViewType
from bluebird_gymnasium.envs.sector_x import SectorXEnv

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
            
        if terminated:
            assert truncated is False
            assert info["__global_terminated__"] is True
            assert info["__global_truncated__"] is False

        else:
            assert terminated is False
            assert truncated is True
            assert info["__global_terminated__"] is False
            assert info["__global_truncated__"] is True

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
            for i in range(num_remaining_steps):
                obs, reward, terminated, truncated, info = gym_env.step(action)
            
        assert info["__global_terminated__"] is not info["__global_truncated__"]

        # try to step in a scenario that has ended. this should raise a
        # value error which is caught by pytest. failure to raise the
        # error leads to a failed test.
        with pytest.raises(ResetNeeded):
            _ = gym_env.step(action)