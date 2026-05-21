from __future__ import annotations

import ast
import json
import re
import typing

import numpy as np
from pydantic import BaseModel
from typing_extensions import override

from bluebird_dt.mixin import Comparison
from bluebird_dt.utility.supported_actions import SUPPORTED_ACTIONS


class ClearanceAndResponse(BaseModel):
    clearance: str | None
    pilot_response: str | None


class Action(Comparison):
    """
    Fundamental unit of communication between Agent and Predictor.
    """

    callsign: str
    _kind: str
    _value: int | float | str | list[str] | tuple[int, str] | None = None
    agent: str | None
    voice_representation: ClearanceAndResponse | None
    text_representation: ClearanceAndResponse | None
    sector: list[str] | None

    def __init__(
        self,
        callsign: str,
        kind: str,
        value: int | float | str | list[str] | tuple[int, str] | None,
        agent: str | None = None,
        text_representation: ClearanceAndResponse | None = None,
        voice_representation: ClearanceAndResponse | None = None,
        sector: list[str] | None = None,
    ):
        """
        Construct a new instance.

        Parameters
        ----------
        callsign: str
            The Aircraft callsign.
        kind: str
            Action type. All supported actions are found in the SUPPORTED_ACTIONS dictionary in the utility folder.

            - `route_direct_to`: go directly to named Fix(es) on Route (and set to route following)
            - `change_heading_to`: change heading to the specified degrees
            - `change_heading_to_by_direction`: change heading to the specified degrees by turning in a specific
            direction
            - `change_heading_by`: change heading by given degrees
            - `maintain_current_heading`: follow the current heading (useful if was following route)
            - `change_flight_level_to`: go to the specified flight level
            - `change_flight_level_by`: descend/ascend by specified flight levels
            - `descend_when_ready,level_by_fix`: descend when ready to the specified flight level by the named Fix
            - `descend_now,level_by_fix`: descend to the specified flight level by the named Fix, starting to
                descend immediately
            - `change_cas_to`: change horizontal fly speed to the specified knots
            - `change_mach_to`: change horizontal fly speed to the specified mach
            - `change_vertical_speed_to`: ascend/descend with the specified vertical speed (in feet/min)
            - `outcomm`: hand control over to the named sector or next coordinated sector if no name given
            - `using_speed_limit`: set whether aircraft needs to obey TMA speed limit (for basic training only)
            - `message`: pass a text string (primarily to display on the HMI)

        value: Union[int, float, str]
            Allowed values for each Action kind differs:

            - `route_direct_to`: str or list[str]
            - `change_heading_to`: int
            - `change_heading_to_by_direction`: tuple[int, Literal['left', 'right', 'shortest']]
            - `change_heading_by`: int
            - `maintain_current_heading`: value will be automatically set to 0
            - `change_flight_level_to`: int
            - `change_flight_level_by`: int
            - `change_cas_to`: int or None (None indicates "fly at own speed")
            - `descend_when_ready,level_by_fix`: tuple[int, str]
            - `descend_now,level_by_fix`: tuple[int, str]
            - `change_cas_to`: float or None (None indicates "fly at own speed")
            - `change_mach_to`: float or None (None indicates "fly at own speed")
            - `change_vertical_speed_to`: float
            - `outcomm`: str
            - `using_speed_limit`: bool
            - `message`: str

        agent: str, optional
            Optional string indicating who sent the Action. In the case of having multiple
            agents interacting with the simulation, this can be used to filter who was
            responsible for what Action.
        text_representation: ClearanceAndResponse, optional
            Optional text representation of the clearance, including clearance and pilot response
            - the clearance depends on the environment at the time that its
            generated, so creation is handled elsewhere.
        voice_representation: ClearanceAndResponse, optional
            Optional voice representation of the clearance, including clearance and pilot response
        sector: list of str, optional
            List of the individual sector frequency this action was sent from/to. Its value is optional for
            backwards compatibility, with `None` values either being replaced by the name of the sector frequency the
            aircraft is currently on if sent to the API, or assumed to be for the entire airspace ("all").

        Examples
        --------
        >>> action = Action("AIR123", "change_flight_level_to", 300, sector="sector_xplus")
        >>> action = Action("AIR123", "route_direct_to", ["PORT1"], sector="sector_xplus")
        >>> action = Action("XYZ567", "change_cas_to", 450.0, "human", sector="sector_xplus")
        >>> action = Action("AIR23", "descend_when_ready,level_by_fix", (200, "FIX1"), sector="sector_xplus")
        """

        if len(callsign) == 0:
            raise ValueError("Callsign must not be empty.")

        self.callsign = callsign
        self._kind = ""
        self.kind = kind
        self._value = None
        self.value = value
        self.agent = agent
        self.sector = sector

        self.voice_representation = voice_representation
        self.text_representation = text_representation

    @property
    def kind(self) -> str:
        """The Action kind"""
        return self._kind

    @kind.setter
    def kind(self, kind: str) -> None:
        """Set value and ensure validity of Action kind"""
        supported_actions = [action_kind for action_list in SUPPORTED_ACTIONS.values() for action_kind in action_list]
        if kind not in supported_actions:
            raise ValueError(f"Unrecognised Action kind: {kind}")
        self._kind = kind

    @property
    def value(self) -> int | float | str | list[str] | tuple[int, str] | None:
        """The Action value"""
        return self._value

    @value.setter
    def value(self, value: int | float | str | list[str] | tuple[int, str] | None) -> None:
        """Set value and ensure correct type"""
        # raise error if value is None for action kinds that need a value
        if value is None and self.kind in [
            "route_direct_to",
            "change_heading_to",
            "change_heading_by",
            "change_flight_level_to",
            "change_flight_level_by",
            "change_vertical_speed_to",
            "descend_when_ready,level_by_fix",
            "descend_now,level_by_fix",
            "using_speed_limit",
            "route_segment",
            "route_turn_segment",
            "heading_segment",
            "heading_turn_segment",
            "message",
            "change_heading_to_by_direction",
        ]:
            raise Exception(f"Value cannot be None for action kind {self.kind}")

        if "level_by_fix" in self.kind:
            # first, check if it is a string (from the clearance df) and convert it back to a tuple.
            # the string should be of the form "(int/float, 'str')"
            if isinstance(value, str):
                value = ast.literal_eval(value)
            if not isinstance(value, tuple):
                raise ValueError(f"Action value must be a tuple for {self.kind}. Got {type(value)}.")
            if not isinstance(value[0], int | float | np.integer | np.floating):
                raise ValueError(f"Action value[0] must be an float or int for {self.kind}. Got {type(value[0])}.")
            if not isinstance(value[1], str):
                raise ValueError(f"Action value[1] must be a string for {self.kind}. Got {type(value[1])}.")
            self._value = value
        elif self.kind in ["route_direct_to", "route_segment", "route_turn_segment"]:
            # allow single fix or list of fixes, and for route_direct_to value
            # to be a string representing a list
            if isinstance(value, list):
                self._value = [str(v) for v in value]
            elif isinstance(value, str) and "[" in value:
                # is a list represented as a string. Turn into list.
                self._value = ast.literal_eval(value)
            else:
                self._value = str(value)
        elif self.kind == "change_heading_to" or self.kind == "change_heading_by":
            self._value = int(float(value))  # accepts e.g "300.0" as input
        elif self.kind == "heading_segment" or self._kind == "heading_turn_segment":
            # system generated heading so allow for floating point headings
            self._value = float(value)
        elif self.kind == "maintain_current_heading":
            # value is irrelevant. Always set to zero for consistency
            self._value = 0
        elif self.kind == "change_flight_level_to" or self.kind == "change_flight_level_by":
            self._value = int(float(value))  # accepts e.g "300.0" as input
        elif self.kind == "change_cas_to":
            self._value = None if value in (None, "None") else int(float(value))  # None means follow own speed
        elif self.kind == "change_mach_to":
            self._value = None if value in (None, "None") else float(value)  # None means follow own speed
        elif self.kind == "change_vertical_speed_to":
            self._value = int(float(value))
        elif self.kind == "outcomm":
            self._value = None if value is None else str(value)
        elif self.kind == "squawk_ident":
            self._value = ""
        elif self.kind == "set_squawk":
            if isinstance(value, str):
                value = int(value)

            if (not isinstance(value, int)) or value < 0 or value > 7777:
                raise ValueError(
                    f"""Action value must be an integer between 0 and 7777 for {self.kind}. Got {value}
                    of type {type(value)}"""
                )
            self._value = value
        elif self.kind == "using_speed_limit":
            if not isinstance(value, str | bool):
                raise ValueError(
                    f"Action value must be a boolean (or string of boolean) for {self.kind}. Got {type(value)}."
                )
            if isinstance(value, str):
                if value not in ["True", "False"]:
                    raise ValueError(f"Action value must be a boolean for {self.kind}. Got {value}.")
                value = value == "True"
            self._value = value
        elif self.kind == "message":
            if not isinstance(value, str):
                raise ValueError(f"Action value must be a string for {self.kind}. Got {type(value)}.")
            self._value = str(value)
        elif self.kind == "change_heading_to_by_direction":
            if (
                not isinstance(value, tuple)
                or not isinstance(value[0], int | float)
                or value[1] not in ["left", "right", "shortest"]
            ):
                raise ValueError(
                    f"Action value must be a tuple[int, Literal['left', 'right', 'shortest']] for {self.kind}. Got "
                    "{value}"
                )
            self._value = value

    @staticmethod
    def from_str(s: str) -> Action:
        """
        Construct a new instance from a string representation.

        Parameters
        ----------
        s: str
            A string representation of Action.

        Returns
        --------
        Action

        Examples
        ----------
        >>> action = Action.from_str("NAX123 change_heading_by 5.0")
        >>> action.data()
        {'callsign': 'NAX123', 'kind': 'change_heading_by', 'value': 5, 'agent': None,
         'clearance': None, 'pilot_response': None, 'sector': None}
        >>> action2 = Action.from_str("NAX123 change_heading_by 5.0 human")
        >>> action2.data()
        {'callsign': 'NAX123', 'kind': 'change_heading_by', 'value': 5, 'agent': 'human',
         'clearance': None, 'pilot_response': None, 'sector': None}
        >>> action3 = Action.from_str("NAX123 change_heading_by 5.0 human on sector_xplus")
        >>> action3.data()
        {'callsign': 'NAX123', 'kind': 'change_heading_by', 'value': 5, 'agent': 'human',
         'clearance': None, 'pilot_response': None, 'sector': 'sector_xplus'}
        """

        # Naming the optional capture groups ensures None is returned if they don't match
        pattern = (
            r"(\S*) "  # CALLSIGN
            r"(\S*) "  # KIND
            r"(\(.*\)|\S*)"  # (VALUE1, VALUE2) or VALUE
            r"(?: (?P<agent>\S*))?"  # optional AGENT
            r"(?: on (?P<sector>\S*))?"  # optional on SECTOR
        )
        full_match = re.fullmatch(pattern, s.strip())

        if full_match is None:
            msg = f"from_str expected format 'CALLSIGN KIND VALUE [AGENT] [on SECTOR]': received '{s}'"
            raise ValueError(msg)

        callsign, kind, value, agent, sector = full_match.groups()

        # route_direct_to can take in a list of Fixes, e.g. FIX1>FIX2>FIX3
        if ">" in value:
            value = value.split(">")

        # change_* actions take int or float as values, not str; json.loads will convert str to int or float as needed
        if "change" in kind:
            value = json.loads(value)

        sector = None if sector == "all" or sector is None else sector.split(",")

        return Action(callsign, kind, value, agent=agent, sector=sector)

    @staticmethod
    def from_json(s: str) -> Action:
        """
        Construct a new instance from a json representation.

        Parameters
        ----------
        s: str
            A json representation of Action.

        Returns
        --------
        Action

        Examples
        ----------
        >>> Action.from_json('''
        ... {
        ...     "callsign": "NAX123",
        ...     "kind": "change_cas_to",
        ...     "value": 250
        ... }''')
        """

        data = json.loads(s)

        # json does not know about tuples and converts them to lists, so we need to convert them back
        kind = data["kind"]
        value = data["value"]

        if "level_by_fix" in kind or kind == "change_heading_to_by_direction":
            value = tuple(value)

        voice_representation: str | None = data.get("voice_representation", None)
        text_representation: str | None = data.get("text_representation", None)

        return Action(
            callsign=data["callsign"],
            kind=data["kind"],
            value=value,
            # optional
            agent=data.get("agent", None),
            voice_representation=(
                None if voice_representation is None else ClearanceAndResponse.model_validate(voice_representation)
            ),
            text_representation=(
                None if text_representation is None else ClearanceAndResponse.model_validate(text_representation)
            ),
            sector=data.get("sector", None),
        )

    def data(self) -> dict[str, typing.Any]:
        """
        Create a dictionary with key/value pairs representing the Aircraft data.

        Returns
        --------
        dict
        """
        return {
            "callsign": self.callsign,
            "kind": self.kind,
            "value": self.value,
            "agent": self.agent,
            "text_representation": (
                None if self.text_representation is None else self.text_representation.model_dump()
            ),
            "voice_representation": (
                None if self.voice_representation is None else self.voice_representation.model_dump()
            ),
            "sector": self.sector,
        }

    def to_json(self) -> str:
        """
        Serialise the instance to JSON string.

        Returns
        --------
        str
        """

        return json.dumps(self.data(), indent=4)

    def is_lateral(self) -> bool:
        """
        Determine if this action represents a lateral manoeuvre

        Returns
        ----------
        bool
            whether or not this actions represents a lateral manoeuvre
        """

        return self.kind in SUPPORTED_ACTIONS["lateral"] or self.kind in SUPPORTED_ACTIONS["system_lateral"]

    def is_vertical(self) -> bool:
        """
        Determine if this action represents a vertical manoeuvre

        Returns
        ----------
        bool
            whether or not this actions represents a vertical manoeuvre
        """

        return self.kind in SUPPORTED_ACTIONS["vertical"]

    def is_speed(self) -> bool:
        """
        Determine if this action represents a speed manoeuvre

        Returns
        ----------
        bool
            whether or not this actions represents a speed manoeuvre
        """

        return self.kind in SUPPORTED_ACTIONS["speed"]

    def is_vertical_speed(self) -> bool:
        """
        Determine if this action represents a vertical speed manoeuvre

        Returns
        ----------
        bool
            whether or not this actions represents a speed manoeuvre
        """

        return self.kind in SUPPORTED_ACTIONS["vertical_speed"]

    @staticmethod
    def load(filename: str) -> Action:
        """
        Construct a new instance from a file.

        Parameters
        ----------
        filename: str
            Path to a JSON file with an Action definition in a dictionary format.

        Returns
        --------
        Action
        """

        with open(filename) as fd:
            return Action.from_json(fd.read())

    def save(self, filename: str):
        """
        Write the instance to a file.

        Parameters
        ----------
        filename: str
            Path to file.
        """

        with open(filename, "w") as fd:
            fd.write(self.to_json())

    @override
    def __str__(self) -> str:
        """
        Create a human-readable representation of the instance. Includes the callsign, kind, value and agent, the
        latter is only shown if it is specified in the instance.
        Returns
        --------
        str
            A string representation of Action (e.g., "NAX123 change_heading_by 5.0 human")
        """

        if isinstance(self.value, float):
            value = f"{self.value:0.2f}"

        elif isinstance(self.value, int):
            value = f"{self.value:d}"

        elif isinstance(self.value, list):
            value = ">".join(self.value)

        else:
            value = self.value

        frequency_sector_string = ",".join(self.sector) if self.sector is not None else "all"

        if self.agent is not None:
            return f"{self.callsign} {self.kind} {value} {self.agent} on {frequency_sector_string}"

        return f"{self.callsign} {self.kind} {value} on {frequency_sector_string}"
