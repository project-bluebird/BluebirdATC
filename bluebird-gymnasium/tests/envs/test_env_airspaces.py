"""
Test that the environments' airspaces are created correctly
"""
import pytest
from bluebird_gymnasium.envs import SectorIEnv, SectorXEnv, SectorXPlusEnv, SectorYEnv, SpringfieldEnv


@pytest.mark.parametrize('sector_env', [SectorIEnv, SectorXEnv, SectorXPlusEnv, SectorYEnv, SpringfieldEnv])
def test_env_vs_predictor_fixes(sector_env):
    config = sector_env.get_default_env_config()
    env = sector_env(config=config)
    env_fixes = env.get_simulator_env().airspace.fixes
    predictor_fixes = env.rollout_predictor.fixes
    assert env_fixes == predictor_fixes


