# types and enums
# added first before import envs module to prevent circular import

## types
import numpy
import typing
from dataclasses import dataclass
from numpy.typing import NDArray
from typing import TypeAlias

ActionConfig: TypeAlias = dict[str, bool | list[int]]
AirspaceConfig: TypeAlias = dict[str, None | bool | str | list]
ForwardFixesConfig: TypeAlias = dict[str, bool | int]
RadarConfig: TypeAlias = dict[str, None | bool | str]
RewardConfig: TypeAlias = dict[str, list[str | int | float]]
ScenarioConfig: TypeAlias = dict[str, str | dict | int | float]
SimulationLogConfig: TypeAlias = dict[str, bool | str]
StateReprConfig: TypeAlias = dict[str, bool | str]
ViewConfig: TypeAlias = dict[str, str | dict]
Config: TypeAlias = typing.Union[
    ActionConfig,
    AirspaceConfig,
    ForwardFixesConfig,
    RadarConfig,
    RewardConfig,
    ScenarioConfig,
    SimulationLogConfig,
    StateReprConfig,
    ViewConfig,
]

ObsType: TypeAlias = typing.Union[
    NDArray[numpy.float32], dict[str, NDArray[numpy.float32]]
]
RewardType: TypeAlias = typing.Union[float, dict[str, float]]
DoneType: TypeAlias = typing.Union[bool, dict[str, bool]]
TruncatedType: TypeAlias = typing.Union[bool, dict[str, bool]]
InfoType: TypeAlias = typing.Union[
    dict[str, typing.Any], dict[str, dict[str, typing.Any]]
]
ActionType: TypeAlias = typing.Union[int, dict[str, int]]


## enums
from enum import IntEnum
from bluebird_gymnasium.utils.types import MetaEnum, StrEnum


class SuccessMetric(IntEnum, metaclass=MetaEnum):
    FAIL = -1
    PASS = 1
    PENDING = 0


class Diagnostics(StrEnum, metaclass=MetaEnum):
    MINIMAL = "minimal"
    FULL = "full"


class ViewType(StrEnum, metaclass=MetaEnum):
    CENTRALIZED = "centralized"  # single agent
    DECENTRALIZED = "decentralized"  # multi agent


class CentralizedSampler(StrEnum, metaclass=MetaEnum):
    """Aircraft sampler for centralized setup."""

    EARLIEST = "earliest_entries"
    LATEST = "latest_entries"
    RANDOM_STEP = "random_step"
    RANDOM_EPISODAL = "random_episodal"
    COMBINED = "combined"


## dataclasses
@dataclass
class EnvConfig:
    """
    The configuration for the gymnasium environments.

    Args:
        action_config: defines the action types that is available in a
            simulator instance (lateral, vertical and so on).
            Defaults to `None` which leads to the use of default configuration.
        airspace_config: defines the parameters for the airspace in the
            simulator.
            Defaults to `None` which leads to the use of default configuration.
        radar_config: defines the parameters for the radar instance to
            visualize the airspace.
            Defaults to `None` which leads to the use of default configuration.
        reward_config: defines the reward function(s) employed and their
            respective weight/coefficient to the total reward per step.
            Defaults to `None` which leads to the use of default configuration.
        scenario_config: defines the parameters used for generating aircraft
            scenarios in the defined airspace.
            Defaults to `None` which leads to the use of default configuration.
        simulation_log_config: defines the information used for setting up
            the logging of simulation to disk at the end of an episode.
            Includes a flag to switch on/off simulation logging.
            Defaults to `None` which leads to the use of default configuration.
        state_repr_config: defines the parameters used for genrating the
            state representation per aircraft.
            Defaults to `None` which leads to the use of default configuration.
        view_config: defines the parameters used for generating final state
            emitted from the environment to the external agent. It's based on
            the strategy used for combining the state representation from all
            aircraft currently in the airspace. The environment can be defined
            as either:
            "centralized" (single agent) or "decentralized" (multi agent)
            Defaults to `None` which leads to the use of default configuration.
        forward_fixes_config: defines the parameters used to configure the
            aircraft route information used in actions and state representation
            Defaults to `None` which leads to the use of default configuration.
        scenario_duration: defines the total time (in seconds) to run a
            scenario within an episode.
            Defaults to 1800.
        scenario_sec_per_step: defines the time duration (in seconds) to
            elapse per simulation step.
            Defaults to 6.
            Note, the number of RL steps within an episode is computed as:
            `scenario_duration / scenario_sec_per_step`.
        diagnostics_level: defines the level of environment information to
            log in the `info` dict return from a call to `.step(...)`. The
            information is useful for analysis and debugging purpose. It
            should be set to one of three values which indicates the level
            information to log: None, "minimal", or "full".
            Defaults to None.
        use_default_outcomm_policy: defines whether or not a default outcomm
            policy (within the env) should be employed. When set to `True` an
            in-built policy is employed within the gymnasium environment to
            outcomm aircraft (without the need for an external agent to
            select an outcomm action to outcomm aircraft).
            Defaults to False.
            Note, when it can only be set to `True` if the outcomm action is
            disabled (i.e., 'outcomm' in the `action_config` dict argument is
            set to False). Otherwise, an exception will be raised if it set
            to `True` while the outcomm action is enabled in the env.
    """

    action_config: ActionConfig | None = None
    airspace_config: AirspaceConfig | None = None
    forward_fixes_config: ForwardFixesConfig | None = None
    radar_config: RadarConfig | None = None
    reward_config: RewardConfig | None = None
    scenario_config: ScenarioConfig | None = None
    simulation_log_config: SimulationLogConfig | None = None
    state_repr_config: StateReprConfig | None = None
    view_config: ViewConfig | None = None
    scenario_duration: int = 1800
    scenario_sec_per_step: int = 6
    diagnostics_level: None | Diagnostics = (None,)
    use_default_outcomm_policy: bool = (False,)


# aircraft scenario generator class(es)
from bluebird_dt.scenario_manager.scenario_manager import ScenarioManager
from bluebird_dt.scenario_manager import (
    Regular,
    Tactical,
    TwoAircraft,
)

SCENARIO_CLS: dict[str, ScenarioManager] = {
    "regular": Regular,
    "tactical": Tactical,
    "twoaircraft": TwoAircraft,
}


# now envs module imports
from bluebird_gymnasium.envs.base import BaseEnv
from bluebird_gymnasium.envs.infinite import CustomInfiniteEnv, InfiniteEnv
from bluebird_gymnasium.envs.sector_i import SectorIEnv
from bluebird_gymnasium.envs.sector_x import SectorXEnv
from bluebird_gymnasium.envs.sector_xplus import SectorXPlusEnv
from bluebird_gymnasium.envs.sector_y import SectorYEnv
from bluebird_gymnasium.envs.springfield import SpringfieldEnv

registry_env: dict[str, BaseEnv] = {
    "base": BaseEnv,
    "CustomInfiniteEnv-v0": CustomInfiniteEnv,
    "InfiniteEnv-v0": InfiniteEnv,
    "SectorIEnv-v0": SectorIEnv,
    "SectorXEnv-v0": SectorXEnv,
    "SectorXPlusEnv-v0": SectorXPlusEnv,
    "SectorYEnv-v0": SectorYEnv,
    "SpringfieldEnv-v0": SpringfieldEnv,
}

name_to_gym_key: dict[str, str] = {
    "CustomInfiniteEnv-v0": "custom_infinite",
    "InfiniteEnv-v0": "infinite",
    "SectorIEnv-v0": "sector_i",
    "SectorXEnv-v0": "sector_x",
    "SectorXPlusEnv-v0": "sector_xplus",
    "SectorYEnv-v0": "sector_y",
    "SpringfieldEnv-v0": "springfield",
}

available_names = ", ".join(name_to_gym_key.values())
available_gym_keys = ", ".join(name_to_gym_key.keys())


def get_gym_key(env_name):
    for gym_key, name in name_to_gym_key.items():
        if env_name == name:
            return gym_key

        elif env_name == gym_key:
            return gym_key

    raise ValueError(
        f"Environment '{env_name}' not found. Environment name"
        f" can be specified using a name in: {available_names}"
        f" or a gym key in: {available_gym_keys}"
    )


def get_default_config(env_name: str) -> Config:
    if env_name in name_to_gym_key.keys():
        env_cls = registry_env.get(env_name)
        return env_cls.get_default_env_config()

    elif env_name in name_to_gym_key.values():
        env_name = get_gym_key(env_name)
        env_cls = registry_env.get(env_name)
        return env_cls.get_default_env_config()

    else:
        raise ValueError(
            f"Environment '{env_name}' not found. Environment name"
            f" can be specified using a name in: {available_names}"
            f" or a gym key in: {available_gym_keys}"
        )


def get_env_cls_and_config(env_name):
    env_name = get_gym_key(env_name)
    config = get_default_config(env_name)

    return registry_env.get(env_name), config


__all__ = [
    "BaseEnv",
    "CustomInfiniteEnv",
    "InfiniteEnv",
    "SectorIEnv",
    "SectorXEnv",
    "SectorXPlusEnv",
    "SectorYEnv",
    "SpringfieldEnv",
    "registry_env",
    "name_to_gym_key",
    "available_names",
    "available_gym_keys",
    "get_gym_key",
    "get_default_config",
    "get_env_cls_and_config",
]
