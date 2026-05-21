import pytest

from bluebird_gymnasium.envs import SectorXEnv


@pytest.fixture
def gym_env():
    """Sample environment instantiator for testing."""

    # default env config
    config = SectorXEnv.get_default_env_config()

    # set number of aircraft in the scenario to 1
    config.scenario_config["args"]["num_aircraft"] = 1

    # set actions
    config.action_config = {
        "simple_heading_left": [
            10,
        ],
        "simple_heading_right": [
            10,
        ],
        "simple_heading_route_parallel": True,
        "simple_fl_descent": True,
        "simple_fl_climb": True,
        "simple_fl_intermediate": True,
        "simple_fl_exit": True,
        "simple_speed_increase": True,
        "simple_speed_decrease": True,
        "simple_route_direct": True,
        "simple_outcomm": True,
    }

    env = SectorXEnv(config=config)
    _, _ = env.reset()

    # simulate a few steps forward before return
    # 0 => means NOOP action, denote no action taken
    # to alter the aircraft trajectory
    action = 0
    num_steps = 50
    for _ in range(num_steps):
        env.step(action)

    return env
