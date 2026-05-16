from gymnasium.envs.registration import register

__all__ = []

register(
    id="CustomInfiniteEnv-v0",
    entry_point="bluebird_gymnasium.envs:CustomInfiniteEnv",
)

register(id="InfiniteEnv-v0", entry_point="bluebird_gymnasium.envs:InfiniteEnv")

register(id="SectorIEnv-v0", entry_point="bluebird_gymnasium.envs:SectorIEnv")

register(id="SectorXEnv-v0", entry_point="bluebird_gymnasium.envs:SectorXEnv")

register(
    id="SectorXPlusEnv-v0", entry_point="bluebird_gymnasium.envs:SectorXPlusEnv"
)

register(id="SectorYEnv-v0", entry_point="bluebird_gymnasium.envs:SectorYEnv")

register(
    id="SpringfieldEnv-v0", entry_point="bluebird_gymnasium.envs:SpringfieldEnv"
)
