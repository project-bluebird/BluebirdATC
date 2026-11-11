import typing

from bluebird_dt.core.action import ActionValueType
from bluebird_dt.utility.models import BandboxSectorName, IndividualSectorName
from bluebird_dt.utility.supported_actions import SUPPORTED_ACTIONS
from pydantic import BaseModel, Field

SUPPORTED_ACTIONS_LIST = [action for actions in SUPPORTED_ACTIONS.values() for action in actions]


class ActionInput(BaseModel):
    """An action to be sent to the simulator."""

    agent: str = Field(
        description="The agent performing the action.",
        json_schema_extra={"example": "atc_1"},
    )
    callsign: str = Field(
        description="The callsign of the flight being acted on.",
        json_schema_extra={"example": "AIR123"},
    )
    kind: typing.Literal[tuple(SUPPORTED_ACTIONS_LIST)] = Field(
        description="The kind of action to perform. Must be a string that is one of the supported actions.",
        json_schema_extra={"example": "change_heading_to"},
    )
    value: ActionValueType = Field(
        description=(
            "The value associated with the action. The type of this value depends on the action being performed. "
            'An action that takes a tuple takes it as a JSON list, e.g. [200, "FIX1"] for descend_now,level_by_fix.'
        ),
        json_schema_extra={"example": 90},
    )
    sector: list[IndividualSectorName] | BandboxSectorName = Field(
        description="The sector that the action is being performed in.",
        json_schema_extra={"example": "sector_1"},
    )


class HmiRunnerInformation(BaseModel):
    selected_aircraft: str | None
