import pytest

from bluebird_gymnasium.envs import SectorIEnv, SectorXEnv, SectorXPlusEnv, SectorYEnv


@pytest.mark.parametrize("rotation", [-45, 0, 30])
def test_configured_xplus_predictor_geometry(rotation: float):
    config = SectorXPlusEnv.get_default_env_config()
    config.airspace_config["rotation_deg"] = rotation
    env = SectorXPlusEnv(config=config)
    try:
        assert env.rollout_predictor.fixes == env.get_simulator_env().airspace.fixes
        env.config.airspace_config["rotation_deg"] = rotation + 15
        env.reset(seed=12)
        assert env.rollout_predictor.fixes == env.get_simulator_env().airspace.fixes
    finally:
        env.close()


@pytest.mark.parametrize("env_type", [SectorIEnv, SectorXEnv, SectorXPlusEnv, SectorYEnv])
@pytest.mark.parametrize("reset_seed", [0, 456])
def test_scenario_seed_and_reset_seed(
    env_type: type[SectorIEnv | SectorXEnv | SectorXPlusEnv | SectorYEnv], reset_seed: int
):
    config = env_type.get_default_env_config()
    config.scenario_config["args"]["random_seed"] = 123
    env = env_type(config=config)
    try:
        initial = env.scenario_manager.create_event_handler().radar_df.copy()
        env.reset()
        assert initial.equals(env.scenario_manager.create_event_handler().radar_df)
        env.reset(seed=reset_seed)
        overridden = env.scenario_manager.create_event_handler().radar_df.copy()
        env.reset(seed=reset_seed)
        assert overridden.equals(env.scenario_manager.create_event_handler().radar_df)
        assert not initial.equals(overridden)
        assert config.scenario_config["args"]["random_seed"] == 123
    finally:
        env.close()


@pytest.mark.parametrize("env_type", [SectorIEnv, SectorXEnv, SectorXPlusEnv, SectorYEnv])
def test_missing_origin_uses_loader_default(env_type: type[SectorIEnv | SectorXEnv | SectorXPlusEnv | SectorYEnv]):
    config = env_type.get_default_env_config()
    del config.airspace_config["origin"]
    env = env_type(config=config)
    try:
        assert tuple(env.config.airspace_config["origin"]) == (50.716667, -3.533333)
        assert env.rollout_predictor.fixes == env.get_simulator_env().airspace.fixes
    finally:
        env.close()
