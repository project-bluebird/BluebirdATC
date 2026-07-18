from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from math import degrees, radians, tan
from typing import Literal

import numpy as np

from bluebird_dt.core import Aircraft, Fixes, Pos2D, Pos4D
from bluebird_dt.core.wind import WindField, WindVector
from bluebird_dt.utility.constants import G_ACC
from bluebird_dt.utility.convert import (
    KT_TO_MPS,
    ground_speed_from_tas,
    heading_from_ground_track,
    horizontal_tas,
)
from bluebird_dt.utility.performance import get_aircraft_key_mapping

# Default rate of turn (in degrees per second) - used for heading change turns (i.e. not route turns)
DEFAULT_RATE_OF_TURN = 1.5

# Holds are flown as standard-rate turns, independently of an aircraft's
# configured rate for ordinary heading changes.
HOLD_RATE_OF_TURN = 3.0

# Threshold difference (in degrees) between requested heading and actual heading. Below this threshold
# the heading is automatically updated, without an explicit use of a turn model.
HEADING_AUTO_THRESHOLD = 0.01

# Parameters for calculating rate of turn from turn angle and ground speed - used for route turns
BANK_RATIO = 0.4
MIN_BANK_ANGLE = 2.0
MAX_BANK_ANGLE = 30.0


class Predictor(ABC):
    """
    Base class for the trajectory predictors.
    """

    def __init__(
        self,
        dt: float,
        fix_proximity_threshold: float,
        fixes: Fixes | None,
        aircraft_mapping_path: str | None = None,
        ignore_synonym_data: bool = False,
    ):
        """
        Construct a new instance.

        Note: Currently the surface temperature differential is explicitly set to be zero in all predictors. In
        future this temperature differential could become a function of horizontal position, and thus
        its calculation/setting moved outside of the predictor class.

        Parameters
        ----------
        dt: float
            The internal step time of the predictor. The time taken between each control point within the returned
            Trajectory is dt.
        fix_proximity_threshold: float
            When an Aircraft is <= fix_proximity_threshold distance from the next Fix on its Route, the Fix is
            considered to have been passed. The next target Fix on Route for the Aircraft is then updated.
        fixes: Fixes | None
            Fixes from the airspace that aircraft are flying in. If None, the predictor will not be able to predict
            the trajectory of on-route aircraft.
        aircraft_mapping_path: str | None
            The path to an aircraft mapping file, either an aircraft type synonym data table or an aircraft weight
            mapping fallback.
        ignore_synonym_data: bool
            Defaults to False but if True it tells this constructor to not read the aircraft_mapping_path file.

        Attributes
        ----------
        delta_T: float
            The BADA atmosphere temperature differential at a pressure altitude of zero (in K)
        performance_data: dict
            A general dict object to hold aircraft performance data used in different predictors.
        synonym_data: dict
            A map between an aircraft type and a key value used for performance data lookups (either aircraft synonym
            or a weight category).
        """

        if dt <= 0.0:
            raise ValueError("dt step time must be positive.")

        if fix_proximity_threshold <= 0.0:
            raise ValueError("Fix proximity threshold must be positive.")

        self.dt = dt
        self.fixes = fixes
        self.fix_proximity_threshold = fix_proximity_threshold

        self.delta_T: float = 0.0

        self.performance_data = {}
        self.synonym_data = None if ignore_synonym_data else get_aircraft_key_mapping(path=aircraft_mapping_path)

    @abstractmethod
    def _predict(
        self,
        aircraft: Aircraft,
        time_evolve: float,
        environment_time: float = 0.0,
        wind_field: WindField | None = None,
    ) -> tuple[list[Pos4D], Aircraft]:
        """
        Calculate the trajectory of an aircraft over time_evolve time and evolve the aircraft to the last point in
        the trajectory.

        This is used in two different ways:
        (1) to evolve the aircraft by moving it to the last point in the predicted trajectory.
            In this case time_evolve is normally the radar period.
        (2) to predict a longer trajectory for use by agents, conflict detection etc.
            In this case time_evolve is normally several minutes.

        Parameters
        ----------
        aircraft: Aircraft
            The aircraft to calculate a trajectory for
        time_evolve: float
            The duration [sec] for which to calculate the trajectory and evolve the aircraft
        environment_time: float, optional
            Current time in the environment in seconds (Posix/UNIX)
        wind_field: WindField | None, optional
            The current wind field

        Returns
        -------
        list[Pos4D]
            The location(s) travelled by the aircraft
        Aircraft
            The aircraft evolved to the last point in the predicted trajectory
        """

        pass

    def predict_trajectory(
        self,
        aircraft: Aircraft,
        delta_t: float,
        environment_time: float = 0.0,
        wind_field: WindField | None = None,
        deepcopy_aircraft: bool = True,
    ) -> list[Pos4D] | None:
        """
        Calculate the trajectory of an aircraft for a time period delta_t.

        Parameters
        ----------
        aircraft: Aircraft
            The aircraft to calculate a trajectory for
        delta_t: float
            The duration [sec] for which to calculate the trajectory
        environment_time: float, optional
            Current time
        wind_field: WindField | None, optional
            The current wind field
        deepcopy_aircraft: bool, optional
            Whether to make a deepcopy of the aircraft so that the aircraft object passed by reference is unchanged,
            defaults to True.

        Returns
        -------
        List[Pos4D] | None
            The predicted trajectory, or None if there are fewer than two points in the trajectory
        """

        if delta_t <= 0.0:
            raise ValueError("delta_t must be positive.")

        if deepcopy_aircraft:
            aircraft = copy.deepcopy(aircraft)

        control_points, _ = self._predict(aircraft, delta_t, environment_time, wind_field)

        return control_points if len(control_points) >= 2 else None

    def predict_aircraft(
        self,
        aircraft: Aircraft,
        delta_t: float,
        environment_time: float = 0.0,
        wind_field: WindField | None = None,
        deepcopy_aircraft: bool = True,
    ) -> Aircraft:
        """
        Evolve an aircraft to a time delta_t in the future. Normally delta_t is the radar period.

        Parameters
        ----------
        aircraft: Aircraft
            The aircraft to evolve
        delta_t: float
            The duration [sec] over which to evolve the aircraft
        environment_time: float, optional
            Current time
        wind_field: WindField | None, optional
            The current wind field
        deepcopy_aircraft: bool, optional
            Whether to make a deepcopy of the aircraft so that the aircraft object passed by reference is unchanged,
            defaults to True.

        Returns
        -------
        Aircraft
            The aircraft evolved to a time delta_t in the future
        """

        if delta_t <= 0.0:
            raise ValueError("delta_t must be positive.")

        if deepcopy_aircraft:
            aircraft = copy.deepcopy(aircraft)

        _, evolved_aircraft = self._predict(aircraft, delta_t, environment_time, wind_field)

        return evolved_aircraft

    def get_target_pos(self, aircraft: Aircraft) -> Pos2D:
        """
        Get Pos2D of next Fix on Route.

        Parameters
        ----------
        aircraft: Aircraft
            A selected aircraft

        Returns
        -------
        target_pos: Pos2D
            Position of the next fix on the Aircraft route
        """

        if self.fixes is None:
            raise ValueError("Cannot get the position of the next fix on route without predictor.fixes")

        next_fix_index = aircraft.next_fix_index
        next_fix = aircraft.flight_plan.route.current[next_fix_index]

        return self.fixes.places[next_fix]

    def update_ground_speed(self, aircraft: Aircraft, wind_field: WindField | None):
        """
        This method updates the aircraft ground speed attributes, based on current speed_tas,
        vertical_speed and heading.

        Parameters
        ----------
        aircraft: Aircraft
            A selected aircraft
        wind_field: WindField
            The wind field the aircraft is flying through
        """
        # Find aircraft horizontal speed
        horizontal_tas_kts = horizontal_tas(aircraft.speed_tas, aircraft.vertical_speed)

        # Find wind_vector at position of aircraft
        if wind_field is not None:
            wind_vector = wind_field.get_wind_vector(
                flight_level=aircraft.fl, latitude=aircraft.lat, longitude=aircraft.lon
            )
        else:
            wind_vector = None

        # Set aircraft ground speed attributes
        aircraft.ground_speed, aircraft.ground_track_angle = ground_speed_from_tas(
            horizontal_tas=horizontal_tas_kts,
            heading=aircraft.heading,
            wind_vector=wind_vector,
        )

    def update_position(
        self,
        aircraft: Aircraft,
        time_evolve: float,
        wind_field: WindField | None,
        use_turn_model: bool = True,
    ):
        """
        Move aircraft laterally and update its heading if required.

        Parameters
        ----------
        aircraft: Aircraft
            An aircraft
        time_evolve: float
            Amount of time [sec] to evolve the Aircraft for
        wind_field: WindField | None
            The wind field to use to find ground speed and ground track angle from the true airspeed and heading
        use_turn_model: bool
            Flag indicating whether to use turn model or just instantaneously update heading. Default is True.
        """

        # This implementation of a hold explicitly requires fixed-rate turns,
        # even when a predictor was configured to make ordinary heading changes instantaneous.
        if aircraft.predictor_params.get("hold") is not None:
            use_turn_model = True

        if use_turn_model:
            self.update_position_with_turn_model(aircraft, time_evolve, wind_field)

        else:
            self.update_position_without_turn_model(aircraft, time_evolve, wind_field)

    def update_position_with_turn_model(self, aircraft: Aircraft, time_evolve: float, wind_field: WindField | None):
        """
        Move aircraft and update its heading if required using turn model.

        Parameters
        ----------
        aircraft: Aircraft
            Selected aircraft to update heading of
        time_evolve: float
            Amount of time [sec] to evolve the Aircraft for
        wind_field: WindField | None
            The wind field to use to find ground speed and ground track angle from the true airspeed and heading
        """
        horizontal_speed_kts = horizontal_tas(aircraft.speed_tas, aircraft.vertical_speed)

        wind_vector = (
            None if wind_field is None else wind_field.get_wind_vector(aircraft.fl, aircraft.lat, aircraft.lon)
        )

        ground_speed, ground_track_angle = ground_speed_from_tas(horizontal_speed_kts, aircraft.heading, wind_vector)

        hold = aircraft.predictor_params.get("hold")
        if hold is not None:
            self.update_hold_guidance(aircraft, hold, ground_track_angle, horizontal_speed_kts, wind_vector)

        # If route-following, check if we need to increment the next_fix_index.
        if aircraft.on_route:
            target_pos = self.get_target_pos(aircraft)
            distance_to_fix = aircraft.pos2d().distance(target_pos)

            last_lateral_action_kind = (
                None
                if aircraft.cleared_instructions.lateral_action is None
                else aircraft.cleared_instructions.lateral_action.kind
            )

            # if a route direct has been issued (which sets the turn radius to None, then recalculate the turn radius)
            if last_lateral_action_kind == "route_direct_to" and aircraft.predictor_params.get("turn_radius") is None:
                bearing_to_fix = aircraft.pos2d().bearing_to(target_pos)
                turn_angle = aircraft.heading - bearing_to_fix
                _, turn_radius = self.distance_and_turn_radius_from_angle(aircraft, turn_angle, ground_speed)
                aircraft.predictor_params["turn_radius"] = turn_radius

            if aircraft.next_fix_index == len(aircraft.flight_plan.route.current) - 1:
                # Next fix is the last one, so just use proximity threshold to check if we've reached it
                if distance_to_fix <= self.fix_proximity_threshold:
                    # Aircraft has reached end of route so is no longer route-following
                    aircraft.next_fix_index = None
                    aircraft.on_route = False
            else:
                # Next fix is not the last one, so calculate when to start turning towards the following fix.
                distance_before_fix_to_start_turn, turn_radius = self.distance_before_fix_to_start_turn(
                    aircraft, ground_speed
                )
                # The second condition here is just to make sure the aircraft flies to fixes with very sharp route
                # turns. The value 3 is fairly arbitrary and affects turns greater than ~145 degrees.
                if distance_to_fix <= distance_before_fix_to_start_turn and distance_to_fix <= 3 * turn_radius:
                    aircraft.next_fix_index += 1
                    # Store the route turn radius for use until the end of the turn, so that speed changes and wind
                    # don't affect the turn radius.
                    aircraft.predictor_params["turn_radius"] = turn_radius

        if aircraft.on_route:
            # If route following - update heading. If the difference between required heading and the current
            # heading is greater than the auto threshold set heading_changing_to, then trigger
            # an explicit use of the turn model. If its less than the auto threshold then just update
            # the current heading with the required heading.
            target_pos = self.get_target_pos(aircraft)
            wind_vector = (
                None if wind_field is None else wind_field.get_wind_vector(aircraft.fl, aircraft.lat, aircraft.lon)
            )
            aircraft.heading_changing_to = heading_from_ground_track(
                aircraft.pos2d().bearing_to(target_pos),
                horizontal_speed_kts,
                wind_vector,
            )

            delta_heading = abs(aircraft.heading_changing_to - aircraft.heading)
            if delta_heading > 180:
                delta_heading = 360 - delta_heading

            if delta_heading < HEADING_AUTO_THRESHOLD:
                aircraft.heading = aircraft.heading_changing_to
                aircraft.heading_changing_to = None
                aircraft.predictor_params["turn_radius"] = None

        # The turn model is triggered by aircraft.heading_changing_to having a value.
        if aircraft.heading_changing_to is not None:
            # Target and current bearings
            _, target_bearing = ground_speed_from_tas(horizontal_speed_kts, aircraft.heading_changing_to, wind_vector)
            current_bearing = ground_track_angle

            rate_of_turn = aircraft.rate_of_turn if aircraft.rate_of_turn else DEFAULT_RATE_OF_TURN
            if hold is not None and hold["phase"] in ["turn_outbound", "turn_inbound"]:
                rate_of_turn = HOLD_RATE_OF_TURN
            # If have started a route turn, then continue the turn at same radius. So calculate the rate of turn
            # from ground speed and turn radius. This means that speed changes won't affect the turn radius.
            if aircraft.on_route and aircraft.predictor_params.get("turn_radius", None) is not None:
                rate_of_turn = ground_speed / aircraft.predictor_params["turn_radius"]  # radians per hour
                rate_of_turn = degrees(rate_of_turn) / 3600  # degrees per second

            # None represents uninitialised
            turn_direction: Literal["left", "right"] | None = None

            # If aircraft is flying a heading
            if hold is not None and hold["phase"] in ["turn_outbound", "turn_inbound"]:
                turn_direction = hold["turn_direction"]
            elif (
                aircraft.cleared_instructions.lateral_action is not None
                and aircraft.cleared_instructions.lateral_action.kind == "change_heading_to_by_direction"
                and aircraft.cleared_instructions.lateral_action.value[1] != "shortest"
            ):
                turn_direction = aircraft.cleared_instructions.lateral_action.value[1]

            if turn_direction is None:
                if (current_bearing > target_bearing and current_bearing - target_bearing <= 180) or (
                    target_bearing > current_bearing and target_bearing - current_bearing > 180
                ):
                    turn_direction = "left"
                else:
                    turn_direction = "right"

            delta_phi = -1.0 * rate_of_turn * time_evolve if turn_direction == "left" else rate_of_turn * time_evolve

            new_bearing = (current_bearing + delta_phi) % 360

            # del_phi_a is the change in bearing needed to reach the target bearing
            del_phi_a = abs(target_bearing - current_bearing)
            if del_phi_a > 180:
                del_phi_a = 360 - del_phi_a

            # If we'll overshoot the target bearing, set heading directly from target bearing
            if abs(delta_phi) >= del_phi_a:
                aircraft.heading = heading_from_ground_track(target_bearing, horizontal_speed_kts, wind_vector)
                aircraft.heading_changing_to = None
                mid_point_bearing = (current_bearing + 0.5 * delta_phi * abs(del_phi_a / delta_phi)) % 360

            else:
                aircraft.heading = heading_from_ground_track(new_bearing, horizontal_speed_kts, wind_vector)
                mid_point_bearing = (current_bearing + 0.5 * delta_phi) % 360

        else:
            # Turn model not used this timestep. Use the current bearing to move the aircraft forward.
            mid_point_bearing = ground_track_angle

        # This is a bit of a roundabout way to get ground_speed.
        midpoint_heading = heading_from_ground_track(mid_point_bearing, horizontal_speed_kts, wind_vector)
        ground_speed, _ = ground_speed_from_tas(horizontal_speed_kts, midpoint_heading, wind_vector)

        pos = aircraft.pos2d()

        # Update position of the aircraft using the midpoint bearing. This gives better
        # accuracy when aircraft is turning than using a simple forward extrapolation.
        delta_s = ground_speed * (time_evolve / 3600.0)
        new_pos = pos.forward(delta_s, mid_point_bearing)

        aircraft.lat = new_pos.lat
        aircraft.lon = new_pos.lon

        if hold is not None and hold["phase"] == "outbound":
            hold["phase_elapsed"] += time_evolve

    def update_hold_guidance(
        self,
        aircraft: Aircraft,
        hold: dict,
        ground_track_angle: float,
        horizontal_speed_kts: float,
        wind_vector: WindVector | None,
    ):
        """Update lateral guidance for the current phase of a racetrack hold."""
        location = hold.get("location")
        if location is None:
            if self.fixes is None or hold.get("fix") is None:
                raise ValueError("Hold state has no resolvable location")
            target_pos = self.fixes.places[hold["fix"]]
            hold["location"] = [target_pos.lat, target_pos.lon]
        else:
            target_pos = Pos2D(*location)
        phase = hold["phase"]
        entered_turn = False

        if phase in ["direct_to_location", "inbound"]:
            distance_to_location = aircraft.pos2d().distance(target_pos)
            if distance_to_location <= self.fix_proximity_threshold:
                if hold["inbound_track"] is None:
                    hold["inbound_track"] = ground_track_angle
                hold["phase"] = "turn_outbound"
                hold["phase_elapsed"] = 0.0
                phase = "turn_outbound"
                entered_turn = True
            else:
                target_track = aircraft.pos2d().bearing_to(target_pos)
                aircraft.heading_changing_to = heading_from_ground_track(
                    target_track, horizontal_speed_kts, wind_vector
                )
                return

        if phase == "turn_outbound":
            target_track = (hold["inbound_track"] + 180.0) % 360.0
            if aircraft.heading_changing_to is None and not entered_turn:
                hold["phase"] = "outbound"
                hold["phase_elapsed"] = 0.0
                aircraft.heading = heading_from_ground_track(target_track, horizontal_speed_kts, wind_vector)
            else:
                aircraft.heading_changing_to = heading_from_ground_track(
                    target_track, horizontal_speed_kts, wind_vector
                )
            return

        if phase == "outbound":
            target_track = (hold["inbound_track"] + 180.0) % 360.0
            if hold["phase_elapsed"] >= hold["outbound_time"]:
                hold["phase"] = "turn_inbound"
                hold["phase_elapsed"] = 0.0
                target_track = hold["inbound_track"]
                aircraft.heading_changing_to = heading_from_ground_track(
                    target_track, horizontal_speed_kts, wind_vector
                )
            else:
                aircraft.heading = heading_from_ground_track(target_track, horizontal_speed_kts, wind_vector)
                aircraft.heading_changing_to = None
            return

        if phase == "turn_inbound":
            target_track = hold["inbound_track"]
            if aircraft.heading_changing_to is None:
                hold["phase"] = "inbound"
                aircraft.heading = heading_from_ground_track(target_track, horizontal_speed_kts, wind_vector)
            else:
                aircraft.heading_changing_to = heading_from_ground_track(
                    target_track, horizontal_speed_kts, wind_vector
                )
            return

        if phase != "inbound":
            raise ValueError(f"Unrecognised hold phase: {phase}")

    def update_position_without_turn_model(self, aircraft: Aircraft, time_evolve: float, wind_field: WindField | None):
        """
        Move aircraft laterally and update its heading, without using a turn model. Instead, the aircraft heading is
        determined using the bearing to the next fix position, if on route, or from the heading_changing_to attribute
        if not on route.

        Parameters
        ----------
        aircraft: Aircraft
            Selected aircraft to update heading of
        time_evolve: float
            Amount of time [sec] to evolve the Aircraft for
        wind_field: WindField
            The wind field to use to find ground speed and ground track angle from the true airspeed and heading
        """
        # Find aircraft horizontal speed
        horizontal_speed_kts = horizontal_tas(aircraft.speed_tas, aircraft.vertical_speed)

        # get aircraft position
        pos = aircraft.pos2d()

        # if route following - get ground track angle and necessary heading
        if aircraft.on_route:
            target_pos = self.get_target_pos(aircraft)
            wind_vector = (
                None if wind_field is None else wind_field.get_wind_vector(aircraft.fl, aircraft.lat, aircraft.lon)
            )
            aircraft.heading_changing_to = heading_from_ground_track(
                pos.bearing_to(target_pos), horizontal_speed_kts, wind_vector
            )
            aircraft.heading = aircraft.heading_changing_to
            aircraft.heading_changing_to = None

        # else check if the aircraft has requested a heading change
        elif aircraft.heading_changing_to is not None:
            # if so, perform the heading change and mark it as being changed
            aircraft.heading = aircraft.heading_changing_to
            aircraft.heading_changing_to = None

        # calculate the distance to travel
        wind_vector = (
            None if wind_field is None else wind_field.get_wind_vector(aircraft.fl, aircraft.lat, aircraft.lon)
        )
        ground_speed, ground_track_angle = ground_speed_from_tas(horizontal_speed_kts, aircraft.heading, wind_vector)
        distance_to_travel = ground_speed * (time_evolve / 3600.0)
        new_pos = pos.forward(distance_to_travel, ground_track_angle)

        aircraft.lat = new_pos.lat
        aircraft.lon = new_pos.lon

        # if route following - update next_fix_index. If past the last fix,
        # set the aircraft to be not route following.
        if aircraft.on_route and new_pos.distance(target_pos) <= self.fix_proximity_threshold:
            aircraft.next_fix_index += 1
            if aircraft.next_fix_index >= len(aircraft.flight_plan.route.current):
                aircraft.next_fix_index = None
                aircraft.on_route = False

    def distance_before_fix_to_start_turn(self, aircraft: Aircraft, speed_kts: float) -> tuple[float, float]:
        """
        Calculate the distance before the next fix that an aircraft should start turning to end up on the next route leg

        Parameters
        ----------
        aircraft: Aircraft
            The aircraft for which to calculate the distance
        speed_kts: float
            The aircraft's horizontal ground speed in knots

        Returns
        -------
        float
            The distance before the fix to start turning towards the following fix
        float
            The radius of the turn in nautical miles
        """
        if self.fixes is None:
            raise ValueError("Cannot calculate distance before fix to start turn without predictor.fixes being set")

        if aircraft.flight_plan is None:
            raise ValueError("Cannot calculate distance before fix to start turn without a flight plan")

        # Get positions of the next fix (fix1) and the fix after that (fix2)
        fix1_index = aircraft.next_fix_index

        if fix1_index is None:
            raise ValueError("Cannot calculate distance before fix to start turn without a next fix")

        fix2_index = fix1_index + 1

        if fix2_index >= len(aircraft.flight_plan.route.current):
            raise ValueError("Cannot calculate distance before fix to start turn without a fix after the next fix")

        fix1 = aircraft.flight_plan.route.current[fix1_index]
        fix2 = aircraft.flight_plan.route.current[fix2_index]
        fix1_pos = self.fixes.places[fix1]
        fix2_pos = self.fixes.places[fix2]

        bearing_to_fix1 = aircraft.pos2d().bearing_to(fix1_pos)
        bearing_fix1_to_fix2 = fix1_pos.bearing_to(fix2_pos)

        turn_angle = bearing_to_fix1 - bearing_fix1_to_fix2
        distance, radius_nmi = self.distance_and_turn_radius_from_angle(aircraft, turn_angle, speed_kts)
        return distance, radius_nmi

    def distance_and_turn_radius_from_angle(
        self, aircraft: Aircraft, turn_angle: float, speed_kts: float
    ) -> tuple[float, float]:
        """
        Calculate the turn radius and distance required to initiate the turn
        for a given turn angle at a given ground speed.

        Parameters
        ----------
        aircraft: Aircraft
            The aircraft executing the turn
        turn_angle: float
            The change in heading required in degrees
        speed_kts: float
            The aircraft's horizontal ground speed in knots

        Returns
        -------
        float
            The distance before the fix to start the turn
        float
            The radius of the turn in nautical miles
        """
        turn_angle = abs(turn_angle)
        turn_angle = min(turn_angle, 360.0 - turn_angle)  # Ensure turn angle is less than 180 degrees

        rate_of_turn = (
            aircraft.rate_of_turn if aircraft.rate_of_turn else self.rate_of_route_turn(speed_kts, turn_angle)
        )
        rate_of_turn = radians(rate_of_turn) * 3600  # Convert to radians per hour

        radius_nmi = speed_kts / rate_of_turn  # Radius of turn in nautical miles

        distance = radius_nmi * tan(radians(turn_angle) / 2)

        return distance, radius_nmi

    def rate_of_route_turn(self, speed_kts: float, turn_angle: float) -> float:
        """
        Calculate the rate of turn in degrees per second for an aircraft at a given speed.

        Parameters
        ----------
        speed_kts: float
            The aircraft's horizontal ground speed in knots
        turn_angle: float
            The angle of the turn in degrees

        Returns
        -------
        float
            The rate of turn in degrees per second
        """
        bank_angle = BANK_RATIO * turn_angle
        bank_angle = np.clip(bank_angle, MIN_BANK_ANGLE, MAX_BANK_ANGLE)

        speed_ms = speed_kts * KT_TO_MPS
        turn_rate = (G_ACC * tan(radians(bank_angle))) / speed_ms

        return degrees(turn_rate)

    def set_integration_step(self, internal_step_max: float) -> None:
        """
        Change the internal time integration step maximum for the Euler method.

        Parameters
        ----------
        internal_step_max: float
            The maximum internal integration time step
        """
        self.internal_step_max = internal_step_max
