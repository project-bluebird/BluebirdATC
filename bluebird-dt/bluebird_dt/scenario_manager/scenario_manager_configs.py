from typing import Literal, TypeAlias

from pydantic import BaseModel, Field

from bluebird_dt.scenario_manager.custom import CustomScenarioManagerConfig
from bluebird_dt.scenario_manager.regular import RegularScenarioManagerConfig
from bluebird_dt.scenario_manager.springfield import SpringfieldScenarioManagerConfig
from bluebird_dt.scenario_manager.two_aircraft import TwoAircraftScenarioManagerConfig
from bluebird_dt.utility.config_models import SaveConfig

ScenarioManagerConfigs = (
    RegularScenarioManagerConfig
    | CustomScenarioManagerConfig
    | SpringfieldScenarioManagerConfig
    | TwoAircraftScenarioManagerConfig
)


class ReplayScenarioConfig(BaseModel):
    original_scenario_manager: ScenarioManagerConfigs | None
    scenario_manager: Literal["replay"] = Field(default="replay")


BluebirdSaveConfig: TypeAlias = SaveConfig[
    ReplayScenarioConfig
    | RegularScenarioManagerConfig
    | CustomScenarioManagerConfig
    | SpringfieldScenarioManagerConfig
    | TwoAircraftScenarioManagerConfig
]
