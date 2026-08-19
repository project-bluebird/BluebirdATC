from datetime import datetime, timedelta
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from bluebird_dt.manager.environment_manager import EnvironmentConfig


class SimulatorConfig(BaseModel):
    projection_centre: tuple[float, float] | None


class SaveConfig(BaseModel):
    save_csv: bool = Field(description=("The log will be saved with csv files in addition to parquet if True."))
    autosave_interval: timedelta | None = Field(
        description=("The autosave simtime interval which enables autosave if not None")
    )
    save_chunk_interval: timedelta | None = Field(
        description=("The chunking simtime interval which enables chunking if not None")
    )
    load_simtime: datetime = Field(
        description=("The simulator datetime from the environment when the scenario was created.")
    )
    load_realtime: datetime = Field(
        description=("The real datetime from the host system clock in UTC when the scenario was created.")
    )
    save_simtime: datetime | None = Field(
        default=None, description=("The simulator datetime from the environment when the save was created.")
    )
    save_realtime: datetime | None = Field(
        default=None, description=("The real datetime from the host system clock in UTC when the save was created.")
    )
    chunk_start_simtime: datetime | None = Field(
        default=None,
        description=(
            "The simulator datetime from the environment when the chunk was created. "
            "If None, chunking is not enabled for this save."
        ),
    )
    chunk_start_realtime: datetime | None = Field(
        default=None,
        description=(
            "The real datetime from the host system clock in UTC when the chunk was created. "
            "If None, chunking is not enabled for this save."
        ),
    )
    save_chunk_id: int | None = Field(
        default=None,
        description=("The chunk id when the chunk was created. If None, chunking is not enabled for this save."),
    )
    # Internal logic, excluded for serialization
    last_save_task_success: bool | None = Field(
        default=None,
        exclude=True,
        description=("A boolean of if the last save task was successful."),
    )
    last_save_task_save_simtime: datetime | None = Field(
        default=None,
        exclude=True,
        description=("The simulator datetime from the environment of last save task savedata's save_simtime."),
    )


TScenarios = TypeVar("TScenarios", bound=BaseModel)


class SimConfig(BaseModel, Generic[TScenarios]):
    scenario_name: str | None = Field(description="Scenario name loaded by the simulator.")
    scenario_category: str | None = Field(
        description="Scenario category loaded by the simulator.", deprecated="Use scenario.scenario_manager."
    )
    simulator: SimulatorConfig
    environment_manager: EnvironmentConfig
    scenario: TScenarios | None
    save_config: SaveConfig
