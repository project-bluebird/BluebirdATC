import gymnasium as gym
import numpy as np

import bluebird_gymnasium  # noqa: F401
from bluebird_dt.scenario_manager import Infinite
from bluebird_gymnasium.envs import SCENARIO_CLS, ViewType
from bluebird_gymnasium.envs.flight_school import FlightSchoolEnv


def test_flight_school_uses_infinite_scenario_manager():
    assert SCENARIO_CLS["infinite"] is Infinite


def test_flight_school_reset_step():
    config = FlightSchoolEnv.get_default_env_config(ViewType.CENTRALIZED)
    config.scenario_config["args"]["random_seed"] = 7
    config.scenario_duration = 60
    gym_env = FlightSchoolEnv(config=config)

    obs, info = gym_env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    assert obs.shape == gym_env.observation_space.shape
    assert isinstance(gym_env.scenario_manager, Infinite)

    obs, reward, done, truncated, info = gym_env.step(0)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_flight_school_gym_make():
    gym_env = gym.make("FlightSchoolEnv-v0")
    obs, info = gym_env.reset()

    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
