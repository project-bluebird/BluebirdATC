import pytest

from bluebird_gymnasium.envs import SectorXEnv
from bluebird_gymnasium.utils.types import PositionStatus


@pytest.fixture
def gym_env():
    """Sample environment instantiator for testing."""

    # default env config
    config = SectorXEnv.get_default_env_config()
    config.action_config = {
        "simple_heading_left": [
            10,
        ],
        "simple_heading_right": [
            10,
        ],
        "simple_heading_route_parallel": True,
        "simple_fl_descent": [
            10,
        ],
        "simple_fl_climb": [
            10,
        ],
        "simple_fl_intermediate": True,
        "simple_fl_exit": True,
        "simple_speed_increase": False,
        "simple_speed_decrease": False,
        "simple_route_direct": [1, 2],
        "simple_outcomm": False,
    }
    config.forward_fixes_config = {
        "num_fixes": 2,
        "use_filed_route": True,
    }

    # set number of aircraft in the scenario to 1
    config.scenario_config["args"]["num_aircraft"] = 1

    env = SectorXEnv(config=config)
    _, _ = env.reset()

    # simulate a few steps forward before return
    # action 0 denotes no action taken to alter aircraft trajectory
    action = 0
    num_steps = 100
    at_least_one_ac_in_sector = False
    for _ in range(num_steps):
        env.step(action)

        tracked_aircraft = env.get_tracked_aircraft_data()
        for callsign, ac_data in tracked_aircraft.items():
            if ac_data.pos_status == PositionStatus.IN_SECTOR:
                at_least_one_ac_in_sector = True
                break

        if at_least_one_ac_in_sector:
            break

    return env
