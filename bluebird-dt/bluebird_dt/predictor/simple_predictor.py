from __future__ import annotations

import warnings
from math import floor

from typing_extensions import override

from bluebird_dt.core import Aircraft, Fixes, FlightState, Pos4D
from bluebird_dt.core.wind import WindField
from bluebird_dt.predictor.predictor import Predictor
from bluebird_dt.utility.convert import (
    FL_TO_FT,
    FT_TO_FL,
)


class SimplePredictor(Predictor):
    """
    Evolves Aircraft with constant speed and vertical speed along their heading.
    """

    def __init__(
        self,
        dt: float,
        fix_proximity_threshold: float,
        fixes: Fixes | None = None,
    ):
        """
        Construct a new instance.

        Parameters
        ----------
        dt: float
            The internal step time of the predictor. The time taken between each control point within the returned
            Trajectory is dt.
        fix_proximity_threshold: float
            When an Aircraft is <= fix_proximity_threshold distance from the next Fix on its Route, the Fix is
            considered to have been passed.
            The next target Fix on Route for the Aircraft is then updated.
        fixes: Fixes | None
            Fixes from the airspace that aircraft are flying in
        """

        super().__init__(dt, fix_proximity_threshold, fixes)

        # Slow descent rate for use with descend_now,level_by_fix actions
        self.slow_descent_rate = -1000.0

    @override
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
        Trajectory
            The predicted trajectory
        Aircraft
            The aircraft evolved to the last point in the predicted trajectory
        """

        # Evolve Aircraft
        control_points: list[Pos4D] = []
        time = 0.0

        # Create an array of step sizes to use for this trajectory.
        # Normally time_evolve is a multiple of dt, so the step sizes are dt, but this may not be the case if
        # the aircraft has just been added. In this case, take a smaller step first, so the remaining steps are dt.
        steps = [time_evolve % self.dt] + [self.dt] * floor(time_evolve / self.dt)

        for step in [s for s in steps if s > 0.0]:
            self.update_aircraft_state(aircraft, step, wind_field=wind_field)
            time += step

            cp = Pos4D(
                lat=aircraft.lat,
                lon=aircraft.lon,
                fl=aircraft.fl,
                time=environment_time + time,
            )
            control_points.append(cp)

        return control_points, aircraft

    def update_aircraft_state(self, aircraft: Aircraft, time_evolve: float, wind_field: WindField | None):
        """
        Update aircraft state over the period time_evolve. Currently this includes updating the aircraft speeds,
        heading, and flight level.

        Parameters
        ----------
        aircraft: Aircraft
            the aircraft to update
        time_evolve: float
            Amount of time [sec] to evolve the Aircraft for
        wind_field: WindField
            The wind field to use to find ground speed and ground track angle from the true airspeed and heading
        """

        self.update_speeds(aircraft, wind_field)
        self.update_position(aircraft, time_evolve, wind_field)
        self.update_flight_level(aircraft, time_evolve)

    def update_speeds(self, aircraft: Aircraft, wind_field: WindField | None):
        """
        Update Aircraft (total and vertical) speeds.
        The horizontal TAS is set to the cleared_cas value.

        Parameters
        ----------
        aircraft: Aircraft
            An Aircraft
        wind_field: Wind_Field
            Wind field aircraft is flying through
        """

        self.update_total_speeds_cas_is_tas(aircraft)

        # Update vertical speeds
        self.update_vertical_speeds(aircraft)

        # Update ground speed related attributes
        self.update_ground_speed(aircraft, wind_field)

    def update_total_speeds_cas_is_tas(self, aircraft: Aircraft):
        """
        Updates total speeds of a selected aircraft by writing the selected_instructions.cas value directly to the true
        airspeed (speed_tas) and calibrated airspeed (speed_cas) values. Mach values are ignored. This method is called
        when the flag "use_cas_as_tas" is set to True. If the selected_instructions.cas value is None, then an error is
        raised: this must be set to use this simplified speed modelling.

        Parameters
        ----------
        aircraft: Aircraft
            A selected aircraft for which to change the horizontal speeds.
        """

        if aircraft.selected_instructions.cas is None:
            raise ValueError("Aircraft.selected_instructions.cas must be set to use cas_is_tas speed modelling.")

        cas = aircraft.selected_instructions.cas

        aircraft.speed_tas = cas
        aircraft.speed_cas = cas

    def update_vertical_speeds(self, aircraft: Aircraft):
        """
        This method updates the aircraft vertical speed, considering the cleared vertical speed of the aircraft.

        Parameters
        ----------
        aircraft: Aircraft
            A selected aircraft
        """
        flight_state = aircraft.flight_state
        if aircraft.selected_instructions.vertical_speed is not None:
            selected_vertical_speed = aircraft.selected_instructions.vertical_speed
            if flight_state is FlightState.DESCEND:
                vertical_speed = -1.0 * abs(selected_vertical_speed)
            elif flight_state is FlightState.CLIMB:
                vertical_speed = abs(selected_vertical_speed)
            else:
                vertical_speed = 0.0
        else:
            # TODO devise some sort of tables similar to Starling's
            # For the moment, just set vertical speed to +/-2000,
            # which seems to be a common order-of-magnitude from the tables.
            if flight_state is FlightState.DESCEND:
                vertical_speed = -2000.0
            elif flight_state is FlightState.CLIMB:
                vertical_speed = 2000.0
            else:
                vertical_speed = 0.0
        aircraft.vertical_speed = vertical_speed

    def update_flight_level(self, aircraft: Aircraft, time_evolve: float):
        """
        Move Aircraft vertically and update its flight level.

        Parameters
        ----------
        aircraft: Aircraft
            An aircraft
        time_evolve: float
            Amount of time [sec] to evolve the Aircraft for
        """

        # if we do not need to change the aircraft's FL, finish.
        if aircraft.fl == aircraft.selected_fl:
            return

        # Deal with level_by_fix actions
        action = aircraft.selected_instructions.vertical_action
        if (
            action is not None
            and "level_by_fix" in action.kind
            and not aircraft.predictor_params.get("descending_to_level_by_point")
        ):
            if self.fixes is None:
                warnings.warn(
                    UserWarning("As fixes=None, the predictor cannot perform the 'level_by_fix' action."),
                    stacklevel=2,
                )
                return

            assert isinstance(action.value, tuple)
            # Use abeam fix in all cases - this reduces to distance to the fix if the aircraft is travelling towards it.
            distance_to_abeam_fix = aircraft.distance_to_abeam(self.fixes.places[action.value[1]], radius=20.0)
            # If the aircraft isn't passing within 20nm of the fix on its current ground track, distance_to_abeam
            # returns None. In this case, use the distance to the fix instead, as a rough estimate. When the
            # aircraft turns more towards the fix, distance_to_abeam will start returning a valid value.
            if distance_to_abeam_fix is None:
                distance_to_abeam_fix = aircraft.distance(self.fixes.places[action.value[1]])

            distance_for_fl_change = self.distance_for_fl_change(aircraft)

            if distance_for_fl_change is not None:
                # Add a buffer as distance_for_fl_change takes no account of varying descent rate and ground speed.
                BUFFER = 2.5  # Enough to achieve the level_by_fix condition in the unit tests.

                # Start descending at normal rate if needed.
                if distance_to_abeam_fix <= distance_for_fl_change + BUFFER:
                    aircraft.predictor_params["descending_to_level_by_point"] = True

                else:
                    if action.kind == "descend_when_ready,level_by_fix":
                        return
                    if action.kind == "descend_now,level_by_fix":
                        aircraft.vertical_speed = self.slow_descent_rate
                    else:
                        raise ValueError(f"Unknown level_by action kind: {action.kind}")

        # Descend or climb aircraft.
        # note that the vertical speed should already have the correct sign for
        # climbing/descending from update_speeds(...)
        fl_change = time_evolve * (aircraft.vertical_speed * FT_TO_FL) / 60

        # if the amount of fl change required is less than the amount to be carried
        # out this time_evolve, then snap to the cleared flight level (to avoid overshooting it)
        if abs(aircraft.fl - aircraft.selected_fl) <= abs(fl_change):
            aircraft.fl = aircraft.selected_fl

            aircraft.vertical_speed = 0.0
            aircraft.predictor_params["descending_to_level_by_point"] = False

        # otherwise, either climb/descend as required
        else:
            aircraft.fl += fl_change

    def distance_for_fl_change(self, aircraft: Aircraft) -> float | None:
        """
        Calculate the horizontal distance needed to change from the current altitude to the target altitude, using the
        current vertical speed and ground speed.

        Parameters
        ----------
        aircraft: Aircraft

        Returns
        -------
        float | None
            The horizontal distance needed for the flight level change, in nautical miles
        """
        if aircraft.vertical_speed == 0:
            warnings.warn(
                f"Aircraft {aircraft.callsign} vertical_speed == 0. Cannot calculate distance for FL change.",
                UserWarning,
                stacklevel=2,
            )
            return None

        # Both vertical_distance and vertical_speed will be negative for descent
        vertical_distance = (aircraft.selected_fl - aircraft.fl) * FL_TO_FT
        return aircraft.ground_speed * vertical_distance / aircraft.vertical_speed / 60
