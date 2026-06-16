from __future__ import annotations

import functools
from math import ceil, floor

from typing_extensions import override

from bluebird_dt.core import Aircraft, Fixes, FlightState, Pos4D
from bluebird_dt.core.wind import WindField
from bluebird_dt.logger import logger
from bluebird_dt.predictor.predictor import Predictor
from bluebird_dt.utility.convert import (
    FL_TO_FT,
    FL_TO_M,
    FT_TO_FL,
    KT_TO_MPS,
    MPS_TO_KT,
    cas_to_tas,
    mach_cas_trans_altitude,
    mach_to_tas,
)
from bluebird_dt.utility.paths import (
    AIRCRAFT_WEIGHT_MAPPING_FILE,
    SIMPLE_PERFORMANCE_PROFILE_FILE,
    SIMPLE_PERFORMANCE_UNCERTAINTY_FILE,
)
from bluebird_dt.utility.performance import (
    cas_and_mach_from_table,
    get_performance_table,
    get_performance_uncertainty_table,
    lateral_speed_keys,
    rocd_from_table,
)
from bluebird_dt.utility.sample import apply_speed_uncertainty


class LinearPredictor(Predictor):
    """
    Evolves Aircraft with constant speed and vertical speed along their heading. A turn model can be turned on or off,
    and two speed modelling modes can be used.
    """

    def __init__(
        self,
        dt: float,
        fix_proximity_threshold: float,
        fixes: Fixes | None = None,
        aircraft_mapping_path: str = AIRCRAFT_WEIGHT_MAPPING_FILE,
        performance_profile_data_path: str | None = SIMPLE_PERFORMANCE_PROFILE_FILE,
        performance_uncertainty_data_path: str | None = SIMPLE_PERFORMANCE_UNCERTAINTY_FILE,
        use_turn_model: bool = True,
        use_cas_as_tas: bool = False,
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
        aircraft_mapping_path: str, optional
            The path to aircraft type synonym data file if supplied or a fallback aircraft weight mapping file
        performance_profile_data_path: str
            The path to speed profile data file
        performance_uncertainty_data_path: str
            The path to speed uncertainty data file
        use_turn_model: boolean, default True
            Flag indicating whether turn model is in use or not. If True, a default rate of turn is used.
        use_cas_as_tas: boolean, default False
            Flag indicating whether calibrated airspeed (CAS) values are used directly as the true airspeed (TAS)
            values, with no conversion. Mach actions are ignored, and speed tables are not used, apart from finding
            an initial cas value if not specified. If False, correct speed modelling is used.
        """

        super().__init__(dt, fix_proximity_threshold, fixes, aircraft_mapping_path=aircraft_mapping_path)

        # Maximum allowed internal Euler integration step in seconds
        self.internal_step_max = 1.0

        # Set turn model and speed modelling flag values
        self.use_turn_model = use_turn_model
        self.use_cas_as_tas = use_cas_as_tas

        # Load performance data
        self.performance_data = get_performance_table(path=performance_profile_data_path)
        self.uncertainty_data = get_performance_uncertainty_table(path=performance_uncertainty_data_path)

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
            # Divide the internal trajectory time step into a series of Euler integration steps if turn model is used.
            if self.use_turn_model is True:
                num_int_steps = ceil(step / self.internal_step_max)
                time_evolve_int = step / num_int_steps

                for _ in range(int(num_int_steps)):
                    self.update_aircraft_state(aircraft, time_evolve_int, wind_field=wind_field)
            else:
                self.update_aircraft_state(aircraft, step, wind_field=wind_field)

            time += step

            cp = Pos4D(lat=aircraft.lat, lon=aircraft.lon, fl=aircraft.fl, time=environment_time + time)
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
        self.update_position(aircraft, time_evolve, wind_field, use_turn_model=self.use_turn_model)
        self.update_flight_level(aircraft, time_evolve)

    def update_speeds(self, aircraft: Aircraft, wind_field: WindField | None):
        """
        Update Aircraft (total and vertical) speeds, with consideration of the flag "use_cas_as_tas". When this
        flag is False, the default horizontal speed behaviour is used, with cleared_cas, cleared_mach, and speed tables
        considered. When True, the horizontal TAS is set to the cleared_cas value.

        Parameters
        ----------
        aircraft: Aircraft
            An Aircraft
        wind_field: Wind_Field
            Wind field aircraft is flying through
        """

        if self.use_cas_as_tas:
            self.update_total_speeds_cas_is_tas(aircraft)
        else:
            self.update_total_speeds(aircraft)

        # Update vertical speeds
        self.update_vertical_speeds(aircraft)

        # Update ground speed related attributes
        self.update_ground_speed(aircraft, wind_field)

    def update_total_speeds(self, aircraft: Aircraft):
        """
        Updates total speeds of a selected aircraft with full use of cleared CAS and cleared Mach values. This
        method is called when the flag "use_cas_as_tas" is False, and is the default behaviour.

        Parameters
        ----------
        aircraft: Aircraft
            A selected aircraft for which to change the horizontal speeds.
        """

        # Get aircraft CAS and Mach speeds
        cas_KT, mach = self.get_aircraft_cas_mach_speeds(aircraft)

        # Simple-mode tables may not include Mach values.
        if mach is None:
            tas_KT = cas_to_tas(aircraft.fl, cas_KT * KT_TO_MPS, self.delta_T) * MPS_TO_KT

        else:
            # Determine aircraft location relative to Mach/CAS transition altitude
            is_below_transition = self.is_aircraft_below_transition(aircraft)

            # If below transition, aircraft is flying on cleared_cas
            if is_below_transition:
                tas_KT = cas_to_tas(aircraft.fl, cas_KT * KT_TO_MPS, self.delta_T) * MPS_TO_KT

            # If above transition, aircraft is flying on cleared_mach
            else:
                tas_KT = mach_to_tas(aircraft.fl, mach, self.delta_T) * MPS_TO_KT

        # Set the aircraft TAS
        aircraft.speed_tas = tas_KT

    def update_total_speeds_cas_is_tas(self, aircraft: Aircraft):
        """
        Updates total speeds of a selected aircraft by writing the selected_instructions.cas value directly to the true
        airspeed (speed_tas) value. Mach values are ignored. This method is called when the flag "use_cas_as_tas" is set
        to True. If selected_instructions.cas is None, CAS is read from speed profile tables.

        Parameters
        ----------
        aircraft: Aircraft
            A selected aircraft for which to change the horizontal speeds.
        """

        cas = aircraft.selected_instructions.cas
        if cas is None:
            raise ValueError("Aircraft.selected_instructions.cas must be set to use cas_is_tas speed modelling.")

        aircraft.speed_tas = cas

    @functools.lru_cache
    def _get_performance_profile(self, key: str | None) -> dict[str, list[float | None] | float]:
        if not self.performance_data:
            raise ValueError("Missing performance data in Linear Predictor")
        # key will be a weight category in the simple case or an aircraft type if full
        # performance data has been supplied
        if key not in self.performance_data:
            if "DEFAULT" not in self.synonym_data:
                raise ValueError(f"Missing key of {key} in Linear Predictor performance data")
            return self.performance_data[self.synonym_data["DEFAULT"]]
        return self.performance_data[key]

    @functools.lru_cache
    def _get_performance_uncertainty(self, key: str | None) -> dict[str, dict[str, float]] | None:
        if not self.uncertainty_data:
            raise ValueError("Missing uncertainty data in Linear Predictor")
        # key will be a weight category in the simple case or an aircraft type if full
        # performance data has been supplied
        if key not in self.uncertainty_data:
            if "DEFAULT" not in self.synonym_data:
                raise ValueError(f"Missing key of {key} in Linear Predictor uncertainty data")
            return self.uncertainty_data[self.synonym_data["DEFAULT"]]
        return self.uncertainty_data[key]

    @functools.lru_cache
    def _get_aircraft_lookup_key(self, aircraft_type: str) -> str | None:
        if aircraft_type not in self.synonym_data:
            if "DEFAULT" not in self.synonym_data:
                raise ValueError(f"Aircraft type {aircraft_type} not found in aircraft_mapping data")
            return self.synonym_data["DEFAULT"]

        return self.synonym_data.get(aircraft_type)

    def get_aircraft_cas_mach_speeds(self, aircraft: Aircraft) -> tuple[float, float | None]:
        """
        Gets the Aircraft CAS and Mach speeds, considering their selected speeds. The possible CAS and Mach speeds are
        equivalent to the selected_instructions.cas and selected_instructions.mach speeds if not None, or read from
        BADA speed profile tables if they are None.

        Parameters
        ----------
        aircraft: Aircraft
            A selected aircraft

        Returns
        ----------
        cas: float
            The aircraft CAS value, in knots
        mach: float | None
            The aircraft Mach value, dimensionless. If unavailable from performance tables, None is returned.
        """

        if aircraft.selected_instructions.cas is None:
            cas, _ = self.speed_from_tables(aircraft)
        else:
            cas = aircraft.selected_instructions.cas

        if aircraft.selected_instructions.mach is None:
            _, mach = self.speed_from_tables(aircraft)
        else:
            mach = aircraft.selected_instructions.mach

        return cas, mach

    def update_vertical_speeds(self, aircraft: Aircraft):
        """
        This method updates the aircraft vertical speed, whilst considering the cleared vertical speed of the aircraft
        and the speed tables values.

        Parameters
        ----------
        aircraft: Aircraft
            A selected aircraft
        """

        if aircraft.selected_instructions.vertical_speed is None:
            vertical_speed = self.vertical_speed_from_tables(aircraft)

        else:
            selected_vertical_speed = aircraft.selected_instructions.vertical_speed
            flight_state = aircraft.flight_state
            if flight_state is FlightState.DESCEND:
                vertical_speed = -1.0 * abs(selected_vertical_speed)
            elif flight_state is FlightState.CLIMB:
                vertical_speed = abs(selected_vertical_speed)
            else:
                vertical_speed = 0.0

        aircraft.vertical_speed = vertical_speed

    def is_aircraft_below_transition(self, aircraft: Aircraft) -> bool:
        """
        Given an Aircraft flying at a CAS or a Mach, determine whether the aircraft is flying above or below the
        Mach/CAS transition altitude. This determines the speed regime for the aircraft.

        To calculate the transition altitude, we need to find cas_trans and mach_trans. These are equal to the cleared
        values if set (i.e. not None). Otherwise (i.e. when None), cas_trans and mach_trans are found from the speed
        tables as the max cas_cr value, and the max mach_cr value respectively.

        A vertical distance is used to calculate the is_below_transition boolean. If this transition_delta is negative,
        the aircraft is flying below the Mach/CAS transition altitude, and True is returned. If positive, the aircraft
        is above the Mach/CAS transition altitude and False is returned.

        Parameters
        ----------
        aircraft: Aircraft
            A selected aircraft

        Returns
        ----------
        is_below_transition: bool
            Boolean indicating whether Aircraft is flying below transition (True) or above transition (False)
        """
        ac_lookup_key = self._get_aircraft_lookup_key(aircraft.aircraft_type)
        performance_table = self._get_performance_profile(ac_lookup_key)

        if aircraft.selected_instructions.cas is None:
            cas_trans = max(value for value in performance_table["cas_cr"] if value is not None)
        else:
            cas_trans = aircraft.selected_instructions.cas

        if aircraft.selected_instructions.mach is None:
            mach_data = performance_table.get("mach_cr")
            if not isinstance(mach_data, list):
                return True

            valid_mach_values = [value for value in mach_data if value is not None]
            if not valid_mach_values:
                return True

            mach_trans = max(valid_mach_values)
        else:
            mach_trans = aircraft.selected_instructions.mach

        mach_cas_transition_altitude = mach_cas_trans_altitude(mach_trans, cas_trans * KT_TO_MPS, delta_T=self.delta_T)
        current_altitude_Hp = aircraft.fl * FL_TO_M
        transition_delta = current_altitude_Hp - mach_cas_transition_altitude
        return transition_delta < 0

    @functools.lru_cache
    def _speed_offset(self, lookup_key: str | None, speed_key: str, percentile_rank: float | None) -> float:
        """
        Return the speed uncertainty offset (in the speed's own units) for a given aircraft lookup key, speed key
        (e.g. "cas_cr", "mach_cl") and percentile rank.

        The offset is additive and flight-level-independent, so it is cached per
        (lookup_key, speed_key, percentile_rank) to avoid recomputing the truncated-normal draw on every step.
        Returns 0.0 when no uncertainty applies (no percentile rank, or no uncertainty data for the key).
        """
        if percentile_rank is None:
            return 0.0
        uncertainty = self._get_performance_uncertainty(lookup_key)
        if uncertainty is None:
            return 0.0
        speed_uncertainty = uncertainty.get(speed_key)
        if speed_uncertainty is None:
            return 0.0
        # apply_speed_uncertainty is additive and flight-level-independent for speeds, so evaluating it at a
        # nominal of 0.0 yields the offset to add to any nominal speed at any flight level.
        return apply_speed_uncertainty(0.0, speed_uncertainty, percentile_rank)

    def speed_from_tables(self, aircraft: Aircraft) -> tuple[float, float | None]:
        """
        Returns the predicted aircraft CAS and mach number using linear interpolation of the CAS table data, with
        the aircraft's speed uncertainty offset applied.

        Parameters
        ----------
        aircraft: Aircraft
            A selected aircraft

        Returns
        -------
        Tuple:
            Tuple of CAS (in knots) and Mach number. Mach may be None if unavailable in the speed profile.
        """

        ac_lookup_key = self._get_aircraft_lookup_key(aircraft.aircraft_type)
        performance_profile = self._get_performance_profile(ac_lookup_key)

        cas, mach = cas_and_mach_from_table(performance_profile, aircraft.fl, aircraft.flight_state)

        cas_key, mach_key = lateral_speed_keys(aircraft.flight_state)
        ranks = aircraft.percentile_rank_dict

        # Apply the flight-level-independent uncertainty offset to the nominal speeds.
        cas += self._speed_offset(ac_lookup_key, cas_key, ranks.get(cas_key))

        if mach is not None:
            # Mach falls back to the CAS percentile rank when it has no rank of its own.
            mach_rank = ranks.get(mach_key)
            if mach_rank is None:
                mach_rank = ranks.get(cas_key)
            mach += self._speed_offset(ac_lookup_key, mach_key, mach_rank)

        return cas, mach

    def vertical_speed_from_tables(self, aircraft: Aircraft) -> float:
        """
        Return the predicted aircraft vertical speed using linear interpolation on the ROCD table data

        Parameters
        ----------
        aircraft: Aircraft
            A selected aircraft

        Returns
        -------
        Float:
            The predicted vertical speed (in feet per minute)
        """

        ac_lookup_key = self._get_aircraft_lookup_key(aircraft.aircraft_type)
        performance_profile = self._get_performance_profile(ac_lookup_key)
        performance_uncertainty = self._get_performance_uncertainty(ac_lookup_key)

        return rocd_from_table(
            performance_profile,
            performance_uncertainty,
            aircraft.percentile_rank_dict,
            aircraft.fl,
            aircraft.flight_state,
        )

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
                logger.warning(
                    "As fixes=None, the predictor cannot perform the 'level_by_fix' action.",
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
            logger.debug(
                f"Aircraft {aircraft.callsign} vertical_speed == 0. Cannot calculate distance for FL change.",
                stacklevel=2,
            )
            return None

        # Both vertical_distance and vertical_speed will be negative for descent
        vertical_distance = (aircraft.selected_fl - aircraft.fl) * FL_TO_FT
        return aircraft.ground_speed * vertical_distance / aircraft.vertical_speed / 60
