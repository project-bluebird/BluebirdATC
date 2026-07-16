from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

from bluebird_dt.scenario_manager.custom import CustomScenarioManagerConfig
from bluebird_dt.scenario_manager.infinite import InfiniteScenarioManagerConfig
from bluebird_dt.scenario_manager.regular import RegularScenarioManagerConfig
from bluebird_dt.scenario_manager.springfield import SpringfieldScenarioManagerConfig
from bluebird_dt.scenario_manager.two_aircraft import TwoAircraftScenarioManagerConfig
from bluebird_dt.utility.config_models import SaveConfig


class FallbackScenarioManagerConfig(BaseModel):
    """Fallback for scenario manager configs from derived simulators we don't
    have a concrete model for. Keeps whatever fields were present in the JSON."""

    model_config = ConfigDict(extra="allow")

    scenario_manager: str


_KNOWN_TAGS = {"regular", "custom", "infinite", "springfield", "two_aircraft"}


def _scenario_manager_tag(v: dict[str, object] | BaseModel) -> str:
    tag = v.get("scenario_manager") if isinstance(v, dict) else getattr(v, "scenario_manager", None)
    return tag if tag in _KNOWN_TAGS else "unknown"


ScenarioManagerConfigs = Annotated[
    Annotated[RegularScenarioManagerConfig, Tag("regular")]
    | Annotated[CustomScenarioManagerConfig, Tag("custom")]
    | Annotated[CustomScenarioManagerConfig, Tag("infinite")]
    | Annotated[SpringfieldScenarioManagerConfig, Tag("springfield")]
    | Annotated[TwoAircraftScenarioManagerConfig, Tag("two_aircraft")],
    Discriminator(_scenario_manager_tag),
]


class ReplayScenarioConfig(BaseModel):
    original_scenario_manager: ScenarioManagerConfigs | None
    scenario_manager: Literal["replay"] = Field(default="replay")


BluebirdSaveConfig: TypeAlias = SaveConfig[
    ReplayScenarioConfig
    | RegularScenarioManagerConfig
    | CustomScenarioManagerConfig
    | InfiniteScenarioManagerConfig
    | SpringfieldScenarioManagerConfig
    | TwoAircraftScenarioManagerConfig
    | FallbackScenarioManagerConfig
]
