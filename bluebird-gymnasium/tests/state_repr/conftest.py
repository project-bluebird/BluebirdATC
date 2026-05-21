import pytest

from bluebird_gymnasium.envs import SectorXEnv


@pytest.fixture
def gym_env():
    """Sample environment instantiator for testing."""

    # default env config
    config = SectorXEnv.get_default_env_config()

    # set number of aircraft in the scenario to 4
    config.scenario_config["args"]["num_aircraft"] = 4

    env = SectorXEnv(config=config)
    _, _ = env.reset()

    # simulate a few steps forward before return
    # action 0 denotes no action taken to alter aircraft trajectory
    action = 0
    num_steps = 20
    for _ in range(num_steps):
        env.step(action)

    return env
