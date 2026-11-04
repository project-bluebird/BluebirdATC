import typing

from bluebird_dt.core import Action
from bluebird_dt.utility.supported_actions import SUPPORTED_ACTIONS
from pydantic import BaseModel, Field, model_validator

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
    value: typing.Any = Field(
        description=(
            "The value associated with the action. The type of this value depends on the action being performed."
        ),
        json_schema_extra={"example": 90},
    )
    sector: str = Field(
        description="The sector that the action is being performed in.",
        json_schema_extra={"example": "sector_1"},
    )

    @model_validator(mode="after")
    def check_simulator_accepts_action(self) -> typing.Self:
        """Build the simulator's `Action` so that a value it rejects fails validation instead of being dropped."""
        try:
            Action(self.callsign, self.kind, self.value, self.agent)
        except Exception as err:
            # Action can also raise a bare Exception or a TypeError, which pydantic would let through instead of
            # reporting it as a validation error.
            raise ValueError(str(err)) from err

        return self


class HmiRunnerInformation(BaseModel):
    selected_aircraft: str | None
