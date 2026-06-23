from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from bluebird_dt.manager.environment_manager import EnvironmentConfig


class SimulatorConfig(BaseModel):
    projection_centre: tuple[float, float] | None


# Config objects used by the Simulator when saving to logfiles.

TScenarios = TypeVar("TScenarios", bound=BaseModel)


class SaveConfig(BaseModel, Generic[TScenarios]):
    scenario_name: str | None = Field(description="Scenario name loaded by the simulator.")
    scenario_category: str | None = Field(
        description="Scenario category loaded by the simulator.", deprecated="Use scenario.scenario_manager."
    )
    save_real_datetime: datetime = Field(
        description=("The real datetime from the host system clock when the save was created.")
    )
    load_real_datetime: datetime = Field(
        description="The real datetime from the host system clock when the scenario was loaded."
    )
    # start_simulator_datetime: datetime
    save_simulator_datetime: datetime = Field(description="The simulator datetime from the environment.")
    simulator: SimulatorConfig
    environment_manager: EnvironmentConfig
    scenario: TScenarios | None
