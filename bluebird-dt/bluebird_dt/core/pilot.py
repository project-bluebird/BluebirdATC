from __future__ import annotations

import json
import typing
from typing import TypedDict

from bluebird_dt.core import Action
from bluebird_dt.core.pos2d import Pos2D
from bluebird_dt.logger import logger
from bluebird_dt.utility.convert import (
    timestamp_to_string,
)
from bluebird_dt.utility.number import round_nearest
from bluebird_dt.utility.supported_actions import SUPPORTED_ACTIONS

if typing.TYPE_CHECKING:
    from bluebird_dt.core import Environment


class QueueItem(TypedDict):
    """
    Pilot Action queue item. Used to store actions, their receipt time (the time the pilot received them), and their
    processing time (the time that the actions should be processed by the pilot).
    """

    receipt_time: float
    process_time: float
    action: Action


class Pilot:
    """
    Class to pilot individual aircraft
    """

    def __init__(self, callsign: str):
        """
        Construct a new instance.

        Parameters
        ----------
        callsign: str
            The callsign of the Aircraft the Pilot is controlling.

        """

        self.action_queue: list[QueueItem] = []
        self.callsign = callsign

    def receive_actions(self, new_actions: list[Action], environment: Environment) -> list[Action]:
        """
        Receive actions for the pilots aircraft from the environment manager and store in a queue. Each queue item is
        a TypedDict defined by the class QueueItem. For the default Pilot (which processes actions with no delay or
        modification), the process_time is equal to the receipt_time.

        Currently the EnvironmentManager controls action logging. Actions that are ignored due to being uncontrollable
        are sent back to be logged in the attribute EM.actions_not_issued.

        Parameters
        ----------
        new_actions: list[Action]
            List of new Actions to be added to the pilot action queue
        environment: Environment
            The environment the Aircraft and Pilot are in

        Returns
        ----------
        ignored_actions: list[Action]
            List of actions ignored by the Pilot (currently due to being uncontrollable).
        """

        valid_actions, ignored_actions = self.check_actions(new_actions, environment)

        for action in valid_actions:
            queue_item: QueueItem = {
                "receipt_time": environment.time,
                "process_time": environment.time,
                "action": action,
            }

            self.action_queue.append(queue_item)

        return ignored_actions

    def check_actions(self, new_actions: list[Action], environment: Environment) -> tuple[list[Action], list[Action]]:
        """
        Performs checks of actions. The following checks are performed:
            - Checks that the callsign of new actions matches the Pilot's callsign
            - Checks that all new actions are in the dictionary of allowed actions (the action space)
            - Filters actions into valid and ignored action lists, based on controllable flag

        Parameters
        ----------
        new_actions: list[Action]
            List of Actions issued to the Pilot
        environment: Environment
            The environment the Aircraft and Pilot are in

        Returns
        ----------
        valid_actions: list[Action]
            List of valid actions that can be added to the Pilot action queue
        ignored_actions: list[Action]
            List of actions that were ignored by the Pilot (currently due to being uncontrollable)
        """

        valid_actions: list[Action] = []
        ignored_actions: list[Action] = []

        for action in new_actions:
            # carry out the action if the aircraft is controllable
            if environment.aircraft[self.callsign].controllable:
                if action.callsign != self.callsign:
                    raise ValueError(f"Action ({action}) erroneously issued to Pilot callsign {self.callsign}.")

                valid_actions.append(action)

            else:
                logger.warning(
                    f"Action ({action}) cannot be issued: the Aircraft is not controllable.",
                    stacklevel=2,
                )
                ignored_actions.append(action)

        return valid_actions, ignored_actions

    def process_actions(self, environment: Environment):
        """
        Process actions in the action queue.

        Parameters
        ----------
        environment: Environment
            The simulation environment at the time step
        """

        processed_items: list[QueueItem] = []

        queue_items = self.action_queue
        current_time = environment.time

        if len(queue_items) > 0:
            for item in queue_items:
                process_time = item["process_time"]
                action = item["action"]

                if process_time <= current_time:
                    if action.kind in SUPPORTED_ACTIONS["outcomm"]:
                        self.process_outcomm(action, environment)
                    elif action.kind in SUPPORTED_ACTIONS["lateral"]:
                        self.process_lateral_actions(action, environment)
                    elif action.kind in SUPPORTED_ACTIONS["speed"]:
                        self.process_speed_actions(action, environment)
                    elif action.kind in SUPPORTED_ACTIONS["vertical"]:
                        self.process_vertical_actions(action, environment)
                    elif action.kind in SUPPORTED_ACTIONS["vertical_speed"]:
                        self.process_vertical_speed_actions(action, environment)
                    elif action.kind in SUPPORTED_ACTIONS["transponder"]:
                        self.process_squawk(action, environment)
                    elif action.kind in SUPPORTED_ACTIONS["message"]:
                        pass  # no effect for message actions

                    else:
                        raise ValueError(f"Aircraft {self.callsign}: Action kind not recognised: {action.kind}")

                    processed_items.append(item)

            if len(processed_items) > 0:
                self.action_queue = [item for item in self.action_queue if item not in processed_items]

    def process_squawk(self, action: Action, environment: Environment):
        aircraft = environment.aircraft[self.callsign]

        if action.kind == "set_squawk":
            aircraft.set_squawk(action.value)
        elif action.kind == "squawk_ident":
            aircraft.ident(environment.time)

    def process_outcomm(self, action: Action, environment: Environment):
        """
        Update what Sector/Agent is in control of Aircraft. If Aircraft is
        passing out of controllable Airspace, set Aircraft status to uncontrollable.

        Parameters
        ----------
        action: Action
            Action issued to the Aircraft
        environment: Environment
            The simulation environment at the time step
        """

        aircraft = environment.aircraft[self.callsign]

        # before outcomming, delete coordinations into the current sector from the previous sector
        # or coordinations within the current sector
        environment.remove_coords_pre_outcomm(self.callsign)

        # the Aircraft has been passed to a new Sector for control
        new_sector = action.value

        # if new sector is in the airspace (or is an individual sector in the airspace)
        # then allow the incomm to that sector, else treat the new_sector as "background"
        if new_sector in environment.airspace.sectors or new_sector in environment.airspace.list_individual_sectors():
            aircraft.current_sector = new_sector
        else:
            aircraft.current_sector = "background"

    def process_lateral_actions(self, action: Action, environment: Environment):
        """
        Update `heading`, or `on_route` in cleared and selected instructions, according to the issued action.

        Parameters
        ----------
        action: Action
            Action issued to the Aircraft
        environment: Environment
            The simulation environment at the time step

        Note
        ----
        Any new lateral clearance replaces an active hold. A new hold creates
        fresh state after the old one has been discarded.
        """

        aircraft = environment.aircraft[self.callsign]

        if (
            action.kind == "route_direct_to,hold_at_location"
            and action.value.fix is not None
            and action.value.fix not in environment.airspace.fixes.places
        ):
            raise ValueError(f"Unknown holding fix: {action.value.fix}")

        # Any new lateral clearance replaces the active hold. A new hold creates
        # fresh state below after the old one has been discarded.
        aircraft.predictor_params.pop("hold", None)

        if action.kind in ["change_heading_by", "change_heading_to", "change_heading_to_by_direction"]:
            match action.kind:
                case "change_heading_by":
                    heading = aircraft.heading + action.value
                case "change_heading_to":
                    heading = action.value
                case "change_heading_to_by_direction":
                    heading = action.value[0]

            # no longer on route (if we were in the first place)
            aircraft.cleared_instructions.on_route = False
            aircraft.selected_instructions.on_route = False

            # request the heading change and store the heading to change to
            aircraft.cleared_instructions.heading = heading % 360
            aircraft.selected_instructions.heading = heading % 360

            aircraft.heading_changing_to = aircraft.selected_instructions.heading

        elif action.kind == "route_direct_to,hold_at_location":
            hold = action.value
            if hold.fix is not None:
                target_pos = environment.airspace.fixes.places[hold.fix]
            else:
                assert hold.location is not None
                target_pos = Pos2D(*hold.location)
            aircraft.cleared_instructions.on_route = False
            aircraft.selected_instructions.on_route = False
            aircraft.cleared_instructions.heading = None
            aircraft.selected_instructions.heading = None
            aircraft.heading_changing_to = aircraft.pos2d().bearing_to(target_pos)
            aircraft.predictor_params["hold"] = {
                "fix": hold.fix,
                "location": [target_pos.lat, target_pos.lon],
                "outbound_time_s": hold.outbound_time_s,
                "turn_direction": hold.turn_direction,
                "phase": "direct_to_location",
                "phase_elapsed": 0.0,
                "inbound_track": None,
            }

        elif action.kind == "route_direct_to":
            # create a new current route that starts with fix(es) provided in action
            new_current_route = [action.value] if isinstance(action.value, str) else action.value.copy()

            # aircraft is expected to continue following the rest of its filed route
            # so append remaining route fixes (if there are any)
            last_issued_fix = new_current_route[-1]
            if last_issued_fix not in aircraft.flight_plan.route.filed:
                # note: this allows for off route fixes being used as long as the last issued fix is on route
                # this behaviour is allowed as it is still guaranteed the aircraft will get to the end of its route
                logger.warning(
                    f"Aircraft {action.callsign} directed to a fix outside its filed route at"
                    f"{timestamp_to_string(environment.time)}."
                    f"\n\troute_direct_to fix(es): {new_current_route}"
                    f"\n\tfiled route: {aircraft.flight_plan.route.filed}"
                    f"\n\tcurrent_route: {aircraft.flight_plan.route.current}"
                )

            else:
                last_issued_fix_index = aircraft.flight_plan.route.filed.index(last_issued_fix)
                # note: below works even when aircraft is directed to last route fix (we call extend() with empty list)
                new_current_route.extend(aircraft.flight_plan.route.filed[last_issued_fix_index + 1 :])

                # update the route.current attribute (next_fix_index tracks where we are in route.current so set to 0)
                aircraft.flight_plan.route.current = new_current_route
                aircraft.next_fix_index = 0
                aircraft.last_passed_current_idx = None

                # target pos is the given fix
                target_fix = new_current_route[0]
                target_pos = environment.airspace.fixes.places[target_fix]

                # point towards the new target pos
                pos = aircraft.pos2d()
                heading = pos.bearing_to(target_pos)
                aircraft.heading_changing_to = heading % 360

                # turn status is now uncertain. set radius to None and let the predictor recalculate.
                aircraft.predictor_params["turn_radius"] = None

                # mark that we're on route and, therefore, have not requested a heading
                aircraft.cleared_instructions.on_route = True
                aircraft.selected_instructions.on_route = True

                aircraft.cleared_instructions.heading = None
                aircraft.selected_instructions.heading = None

        elif action.kind == "maintain_current_heading":
            # request the heading is the current heading
            aircraft.cleared_instructions.heading = aircraft.heading
            aircraft.selected_instructions.heading = aircraft.heading

            # no longer on route (if we were in the first place)
            aircraft.cleared_instructions.on_route = False
            aircraft.selected_instructions.on_route = False

            # heading not changing
            aircraft.heading_changing_to = None

        # Store last lateral action in instructions
        aircraft.cleared_instructions.lateral_action = action
        aircraft.selected_instructions.lateral_action = action

    def process_speed_actions(self, action: Action, environment: Environment):
        """
        Update `cas` or `mach` in cleared and selected instructions, according to the issued action.

        Parameters
        ----------
        action: Action
            Action issued to the Aircraft
        environment: Environment
            The simulation environment at the time step
        """

        aircraft = environment.aircraft[self.callsign]

        if action.kind == "change_cas_to":
            aircraft.cleared_instructions.cas = action.value
            aircraft.selected_instructions.cas = action.value

        elif action.kind == "change_mach_to":
            aircraft.cleared_instructions.mach = action.value
            aircraft.selected_instructions.mach = action.value

        elif action.kind == "using_speed_limit":
            aircraft.predictor_params["obey_speed_limit"] = action.value

        # Store last speed action in instructions
        aircraft.cleared_instructions.speed_action = action
        aircraft.selected_instructions.speed_action = action

    def process_vertical_actions(self, action: Action, environment: Environment):
        """
        Update `fl` in cleared and selected instructions, according to the issued action.

        Parameters
        ----------
        action: Action
            Action issued to the Aircraft
        environment: Environment
            The simulation environment at the time step
        """

        aircraft = environment.aircraft[self.callsign]

        if "level_by_fix" in action.kind:
            assert isinstance(action.value, tuple)
            cleared_fl = action.value[0]

            # In case this is True from a previous level_by_fix action
            aircraft.predictor_params["descending_to_level_by_point"] = False
        else:
            cleared_fl = aircraft.fl + action.value if action.kind == "change_flight_level_by" else action.value

        # Set cleared_fl to the nearest multiple of 10
        if cleared_fl % 10 != 0:
            rounded_cleared_fl = round_nearest(cleared_fl, 10)

            logger.warning(
                f"{self.callsign}: Cleared or selected flight level ({cleared_fl}) is not a multiple of 10. "
                f"It will be rounded to the nearest multiple of 10 ({rounded_cleared_fl}).",
                stacklevel=2,
            )

            cleared_fl = rounded_cleared_fl

        aircraft.cleared_fl = cleared_fl
        aircraft.selected_fl = cleared_fl

        # Store last vertical action in instructions
        aircraft.cleared_instructions.vertical_action = action
        aircraft.selected_instructions.vertical_action = action

    def process_vertical_speed_actions(self, action: Action, environment: Environment):
        """
        Update `vertical_speed` in cleared and selected instructions, according to issued action.

        Parameters
        ----------
        action: Action
            Action issued to the Aircraft
        environment: Environment
            The simulation environment at the time step
        """

        aircraft = environment.aircraft[self.callsign]

        aircraft.cleared_instructions.vertical_speed = action.value
        aircraft.selected_instructions.vertical_speed = action.value

        # Store last vertical speed action in instructions
        aircraft.cleared_instructions.vertical_speed_action = action
        aircraft.selected_instructions.vertical_speed_action = action

    @classmethod
    def from_json(cls, s: str) -> Pilot:
        """
        Constructs a new Pilot instance from a string in JSON format, and the Aircraft callsign.

        Parameters
        ----------
        s: str
            A string representation of Pilot data

        Returns
        ----------
        Pilot: Pilot
            Pilot instance

        Examples
        ----------
        >>> Pilot.from_json(
        >>>     '''
        >>>     {
        >>>         "pilot_type": "Pilot",
        >>>         "callsign": "AIR3"
        >>>     }
        >>>     '''
        """

        data = json.loads(s)
        callsign = data["callsign"]
        pilot_class = cls
        return pilot_class(callsign)

    def data(self) -> dict[str, str]:
        """
        Create a dictionary containing the Pilot data. Currently there is not much here.

        Returns
        ----------
        data: dict
            Dictionary containing the Pilot instance data.

        """

        return {"pilot_type": self.__class__.__name__, "callsign": self.callsign}

    def to_json(self) -> str:
        """
        Serialise the Pilot instance to JSON string.

        Returns
        ----------
        json_string: str
            A json string of the Pilot instance

        """

        return json.dumps(self.data(), indent=4)

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
