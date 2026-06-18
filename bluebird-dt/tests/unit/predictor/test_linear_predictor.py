import copy
import random

import numpy as np
import pytest

from bluebird_dt.core import Action, Environment, Fixes, FlightState, Pos4D
from bluebird_dt.core.wind import WindField
from bluebird_dt.predictor import LinearPredictor
from bluebird_dt.utility.convert import (
    FL_TO_FT,
    FT_TO_FL,
    KT_TO_MPS,
    MPS_TO_KT,
    cas_to_tas,
    ground_speed_from_tas,
    heading_from_ground_track,
    horizontal_tas,
    mach_to_tas,
)
from bluebird_dt.utility.supported_actions import SUPPORTED_ACTIONS

class TestPredictorInFullMode:
    def test_initialisation_flags(self, build_full_mode_predictor):
        predictor = build_full_mode_predictor()
        assert predictor.use_cas_as_tas is False
        assert predictor.use_turn_model is True

        predictor_2 = build_full_mode_predictor(use_cas_as_tas=True, use_turn_model=False)
        assert predictor_2.use_cas_as_tas is True
        assert predictor_2.use_turn_model is False

    @pytest.mark.parametrize("callsign, fix_name", [("AIR0", "EARTH"), ("AIR1", "AIR")])
    def test_get_target_pos(self, callsign: str, fix_name: str, generate_simple_environment: Environment, build_full_mode_predictor):
        environment = generate_simple_environment
        environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
        selected_aircraft = environment.aircraft[callsign]
        predictor = build_full_mode_predictor()

        _ = predictor.predict_aircraft(selected_aircraft, 1.0, environment_time=environment.time)
        pos = predictor.get_target_pos(selected_aircraft)
        assert pos.__str__() == environment.airspace.fixes.places[fix_name].__str__()

    @pytest.mark.parametrize(
        "callsign, action_kind, action_value, heading, cleared_heading, check_heading",
        [
            ("AIR0", "change_heading_by", -90, 0, 270, 358.5),
            ("AIR0", "maintain_current_heading", 0, 0, 0, 0),
            ("AIR1", "maintain_current_heading", 0, 180, 180, 180),
            ("AIR1", "change_heading_to", 45, 180, 45, 178.5),
            ("AIR1", "route_direct_to", "AIR", 180, None, 180),
        ],
    )
    def test_process_lateral_actions(
        self,
        callsign: str,
        action_kind: str,
        action_value: float | str,
        heading: float | None,
        cleared_heading: float | None,
        check_heading: float,
        generate_simple_environment: Environment,
        build_full_mode_predictor,
    ):
        environment = generate_simple_environment
        environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
        predictor = build_full_mode_predictor()

        aircraft = environment.aircraft[callsign]
        aircraft.rate_of_turn = 1.5
        pilot = aircraft.pilot

        action = Action(callsign, action_kind, action_value)
        pilot.process_lateral_actions(action, environment)

        assert aircraft.selected_instructions.heading == cleared_heading
        assert aircraft.heading == heading
        if action_kind == "route_direct_to" or action_kind == "change_cas_to":
            assert aircraft.on_route
        else:
            assert not aircraft.on_route

        evolved = predictor.predict_aircraft(aircraft, 1.0, environment_time=environment.time)
        assert evolved.heading == check_heading
        assert evolved.selected_instructions.heading == cleared_heading

    @pytest.mark.parametrize(
        "callsign, cleared_cas, cleared_mach",
        [
            ("AIR0", 200.0, 0.75),
            ("AIR0", None, 0.78),
            ("AIR0", 175.0, None),
            ("AIR0", None, None),
        ],
    )
    def test_get_aircraft_cas_mach_speeds(
        self,
        callsign: str,
        cleared_cas: float | None,
        cleared_mach: float | None,
        generate_simple_environment: Environment,
        build_full_mode_predictor,
    ):
        environment = generate_simple_environment
        predictor = build_full_mode_predictor()
        aircraft = environment.aircraft[callsign]
        aircraft.aircraft_type = "B753"
        aircraft.fl = 140.0
        aircraft.selected_fl = 140.0
        aircraft.selected_instructions.cas = cleared_cas
        aircraft.selected_instructions.mach = cleared_mach

        evolved = predictor.predict_aircraft(aircraft, 1.0, environment_time=environment.time)
        cas, mach = predictor.get_aircraft_cas_mach_speeds(evolved)
        assert cas is not None
        assert mach is None or mach > 0.0
        if cleared_cas is not None:
            assert cas == pytest.approx(cleared_cas)
        if cleared_mach is not None:
            assert mach == pytest.approx(cleared_mach)

    @pytest.mark.parametrize(
        "selected_fl, expected_state", [(200, FlightState.CRUISE), (300, FlightState.CLIMB), (170, FlightState.DESCEND)]
    )
    def test_update_aircraft_vertical_speeds_method(
        self,
        selected_fl: float,
        expected_state: FlightState,
        generate_simple_environment: Environment,
        build_full_mode_predictor,
    ):
        predictor = build_full_mode_predictor()
        aircraft = generate_simple_environment.aircraft["AIR1"]
        aircraft.aircraft_type = "B753"
        aircraft.fl = 200.0
        aircraft.selected_fl = selected_fl
        aircraft.selected_instructions.vertical_speed = None

        predictor.update_vertical_speeds(aircraft)
        assert aircraft.flight_state is expected_state
        if expected_state is FlightState.CRUISE:
            assert aircraft.vertical_speed == pytest.approx(0.0)
        elif expected_state is FlightState.CLIMB:
            assert aircraft.vertical_speed > 0.0
        else:
            assert aircraft.vertical_speed < 0.0

    @pytest.mark.parametrize("action_value", [180.0, 250.0, 340.0])
    def test_update_cas_as_tas_speeds(
        self, action_value: float, generate_simple_environment: Environment, build_full_mode_predictor
    ):
        environment = generate_simple_environment
        predictor = build_full_mode_predictor(use_cas_as_tas=True)
        aircraft = environment.aircraft["AIR0"]
        pilot = aircraft.pilot

        with pytest.raises(ValueError):
            _ = predictor.predict_aircraft(aircraft, 1.0, environment_time=environment.time, deepcopy_aircraft=False)

        action = Action("AIR0", "change_cas_to", action_value)
        pilot.process_speed_actions(action, environment)
        evolved = predictor.predict_aircraft(aircraft, 1.0, environment_time=environment.time)
        assert evolved.speed_tas == pytest.approx(action_value)

    @pytest.mark.parametrize(
        "callsign, action_kind, action_value, initial_tas_to_check, initial_cas_check, initial_mach_check",
        [
            ("AIR0", "change_cas_to", 180.0, 360.0, None, None),
            ("AIR0", "change_cas_to", 410.0, 360.0, None, None),
            ("AIR1", "change_cas_to", 356.0, 360.0, None, None),
            ("AIR1", "change_cas_to", 550.0, 360.0, None, None),
            ("AIR0", "change_mach_to", 0.78, 360.0, None, None),
            ("AIR0", "change_mach_to", 0.68, 360.0, None, None),
            ("AIR1", "change_mach_to", 0.72, 360.0, None, None),
            ("AIR1", "change_mach_to", 0.70, 360.0, None, None),
        ],
    )
    def test_update_cas_mach_speeds(
        self,
        callsign: str,
        action_kind: str,
        action_value: float,
        initial_tas_to_check: float,
        initial_cas_check: None,
        initial_mach_check: None,
        generate_simple_environment: Environment,
        build_full_mode_predictor,
    ):
        environment = generate_simple_environment
        environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
        aircraft = environment.aircraft[callsign]
        pilot = aircraft.pilot

        assert aircraft.selected_instructions.cas is initial_cas_check
        assert aircraft.selected_instructions.mach is initial_mach_check
        assert aircraft.speed_tas == pytest.approx(initial_tas_to_check)
        assert aircraft.vertical_speed == pytest.approx(0.0)

        predictor = build_full_mode_predictor()
        _ = predictor.predict_aircraft(aircraft, 1.0, environment_time=environment.time)

        action = Action(callsign, action_kind, action_value)
        pilot.process_speed_actions(action, environment)

        if action_kind == "change_cas_to":
            assert aircraft.selected_instructions.cas == pytest.approx(action_value)
        else:
            assert aircraft.selected_instructions.mach == pytest.approx(action_value)
        assert aircraft.speed_tas == pytest.approx(initial_tas_to_check)

        predictor_aircraft = predictor.predict_aircraft(aircraft, 1.0, environment_time=environment.time)
        if action_kind == "change_cas_to":
            assert predictor_aircraft.selected_instructions.cas == pytest.approx(action_value)
        else:
            assert predictor_aircraft.selected_instructions.mach == pytest.approx(action_value)

        cas_kt, mach = predictor.get_aircraft_cas_mach_speeds(predictor_aircraft)
        if predictor.is_aircraft_below_transition(predictor_aircraft):
            tas_kt = cas_to_tas(aircraft.fl, cas_kt * KT_TO_MPS, delta_T=0.0) * MPS_TO_KT
        else:
            assert mach is not None
            tas_kt = mach_to_tas(aircraft.fl, mach, delta_T=0.0) * MPS_TO_KT
        assert predictor_aircraft.speed_tas == pytest.approx(tas_kt)

    @pytest.mark.parametrize(
        "callsign, action_kind, action_value, vspeed, change_fl",
        [
            ("AIR0", "change_vertical_speed_to", 26 * FL_TO_FT, -26 * FL_TO_FT, -50),
            ("AIR0", "change_vertical_speed_to", 26 * FL_TO_FT, 0.0, 0),
            ("AIR0", "change_vertical_speed_to", 26 * FL_TO_FT, 26 * FL_TO_FT, 50),
        ],
    )
    def test_update_vertical_speeds(
        self,
        callsign: str,
        action_kind: str,
        action_value: float,
        vspeed: float,
        change_fl: float,
        generate_simple_environment: Environment,
        build_full_mode_predictor,
    ):
        environment = generate_simple_environment
        environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
        predictor = build_full_mode_predictor()

        aircraft = environment.aircraft[callsign]
        pilot = aircraft.pilot
        action_1 = Action(callsign, action_kind, action_value)
        action_2 = Action(callsign, "change_flight_level_by", change_fl)
        action1_value = action_1.value
        action2_value = action_2.value

        assert isinstance(action1_value, (float, int))
        assert isinstance(action2_value, (float, int))
        initial_fl = aircraft.fl
        assert aircraft.vertical_speed == 0.0
        assert aircraft.selected_instructions.vertical_speed is None
        assert aircraft.cleared_instructions.vertical_speed is None

        pilot.process_vertical_speed_actions(action_1, environment)
        assert aircraft.selected_instructions.vertical_speed == action1_value
        assert aircraft.cleared_instructions.vertical_speed == action1_value

        pilot.process_vertical_actions(action_2, environment)
        assert aircraft.selected_fl == initial_fl + action2_value
        assert aircraft.cleared_fl == initial_fl + action2_value

        _ = predictor.predict_aircraft(aircraft, 1.0, environment_time=environment.time, deepcopy_aircraft=False)
        assert aircraft.vertical_speed == vspeed

    @pytest.mark.parametrize(
        "callsign, turn_model_flag, action_kind, action_value",
        [
            ("AIR1", True, "change_heading_to", 45),
            ("AIR0", True, "change_heading_by", -5),
            ("AIR1", False, "change_heading_to", 45),
            ("AIR0", False, "route_direct_to", "AIR"),
        ],
    )
    def test_update_position(
        self,
        callsign: str,
        turn_model_flag: bool,
        action_kind: str,
        action_value: float | str,
        generate_simple_environment: Environment,
        build_full_mode_predictor,
    ):
        environment = generate_simple_environment
        environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
        predictor = build_full_mode_predictor(use_turn_model=turn_model_flag)
        aircraft = environment.aircraft[callsign]
        pilot = aircraft.pilot

        action = Action(callsign, action_kind, action_value)
        pilot.process_lateral_actions(action, environment)
        evolved = predictor.predict_aircraft(aircraft, 1.0, environment_time=environment.time)
        assert evolved.heading is not None

    @pytest.mark.parametrize(
        "callsign, action_kind, action_value",
        [
            ("AIR0", "descend_when_ready,level_by_fix", (100, "AIR")),
            ("AIR0", "descend_now,level_by_fix", (120, "AIR")),
        ],
    )
    def test_update_flight_level_with_level_by_fix_actions(
        self,
        callsign: str,
        action_kind: str,
        action_value: tuple[int, str],
        generate_simple_environment: Environment,
        build_full_mode_predictor,
    ):
        environment = generate_simple_environment
        environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
        predictor = build_full_mode_predictor()
        aircraft = environment.aircraft[callsign]
        pilot = aircraft.pilot

        action = Action(callsign, action_kind, action_value)
        assert action.kind in SUPPORTED_ACTIONS["vertical"]
        pilot.process_vertical_actions(action, environment)

        start_fl = aircraft.fl
        predictor.predict_aircraft(aircraft, 6.0, environment_time=environment.time, deepcopy_aircraft=False)

        if action.kind == "descend_now,level_by_fix":
            assert aircraft.fl < start_fl
        else:
            assert aircraft.fl == start_fl

    @pytest.mark.parametrize(
        "callsign, action_kind, action_value, action_kind_2, action_value_2, check_cleared_fl, n_step",
        [
            ("AIR0", "change_flight_level_by", 0, "change_vertical_speed_to", 0.0, 150, 10),
            ("AIR1", "change_flight_level_by", -10, "change_vertical_speed_to", 0.0, 190, 10),
            ("AIR0", "change_flight_level_to", 160, "change_vertical_speed_to", 30.0 * FL_TO_FT, 160, 10),
            ("AIR1", "change_flight_level_to", 190, "change_vertical_speed_to", 25.0 * FL_TO_FT, 190, 10),
        ],
    )
    def test_update_flight_level(
        self,
        callsign: str,
        action_kind: str,
        action_value: float,
        action_kind_2: str,
        action_value_2: float,
        check_cleared_fl: float,
        n_step: float,
        generate_simple_environment: Environment,
        build_full_mode_predictor,
    ):
        environment = generate_simple_environment
        environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
        predictor = build_full_mode_predictor()
        aircraft = environment.aircraft[callsign]
        pilot = aircraft.pilot

        action_1 = Action(callsign, action_kind, action_value)
        action_2 = Action(callsign, action_kind_2, action_value_2)
        for action in [action_1, action_2]:
            if action.kind in SUPPORTED_ACTIONS["vertical"]:
                pilot.process_vertical_actions(action, environment)
            elif action.kind in SUPPORTED_ACTIONS["vertical_speed"]:
                pilot.process_vertical_speed_actions(action, environment)

        if aircraft.fl == aircraft.selected_fl:
            assert aircraft.flight_state is FlightState.CRUISE
        elif aircraft.fl < aircraft.selected_fl:
            assert aircraft.flight_state is FlightState.CLIMB
        else:
            assert aircraft.flight_state is FlightState.DESCEND

        start_fl = aircraft.fl
        predictor_aircraft = predictor.predict_aircraft(aircraft, n_step, environment_time=environment.time)
        climb_rate = (action_value_2 * FT_TO_FL) / 60
        check_vertical_speed = action_value_2
        if predictor_aircraft.flight_state is FlightState.DESCEND:
            climb_rate *= -1
            check_vertical_speed *= -1
        check_fl = start_fl + n_step * climb_rate

        assert predictor_aircraft.selected_fl == pytest.approx(check_cleared_fl)
        assert predictor_aircraft.selected_instructions.vertical_speed == pytest.approx(action_value_2)
        assert predictor_aircraft.vertical_speed == pytest.approx(check_vertical_speed)
        if predictor_aircraft.flight_state is not FlightState.CRUISE:
            assert predictor_aircraft.fl == pytest.approx(check_fl)
        else:
            assert predictor_aircraft.fl == pytest.approx(start_fl)

    def test_predict_aircraft(self, generate_simple_environment: Environment, build_full_mode_predictor):
        dt = random.choice([1.0, 3.0, 5.0])
        n_step = random.choice([15, 30, 45])

        environment = generate_simple_environment
        environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
        predictor = build_full_mode_predictor(dt=dt)
        aircraft = environment.aircraft["AIR0"]
        pilot = aircraft.pilot
        start_pos = aircraft.pos2d()

        action = Action("AIR0", "change_heading_by", -20.0)
        pilot.receive_actions([action], environment)
        pilot.process_actions(environment)
        trajectory = predictor.predict_trajectory(aircraft, n_step, environment_time=environment.time)

        assert start_pos.__str__() == aircraft.pos2d().__str__()
        assert isinstance(trajectory, list)
        assert len(trajectory) == np.ceil(n_step / dt)
        assert trajectory[-1].time == n_step

    @pytest.mark.parametrize(
        "callsign, action_kind, action_value, delta_t, n_step, next_fix_index, on_route",
        [
            ("AIR1", "route_direct_to", "AIR", 5.0, 45, 0, True),
            ("AIR0", "change_cas_to", 340.0, 5.0, 200, 2, True),
            ("AIR1", "change_cas_to", 205.0, 5.0, 150, 1, True),
            ("AIR0", "change_flight_level_by", -50, 5.0, 70, 1, True),
            ("AIR1", "change_flight_level_to", 40, 5.0, 2000, None, False),
            ("AIR0", "change_vertical_speed_to", 30.0 * FL_TO_FT, 5.0, 300, 2, True),
            ("AIR1", "change_vertical_speed_to", 28.0 * FL_TO_FT, 5.0, 65, 1, True),
            ("AIR1", "descend_when_ready,level_by_fix", (100, "EARTH"), 5.0, 300, 2, True),
            ("AIR1", "descend_now,level_by_fix", (120, "AIR"), 5.0, 65, 1, True),
            ("AIR0", "descend_now,level_by_fix", (80, "AIR"), 5.0, 2000, None, False),
            ("AIR0", "descend_when_ready,level_by_fix", (50, "EARTH"), 5.0, 150, 2, True),
        ],
    )
    def test_predict_aircraft_on_route(
        self,
        callsign: str,
        action_kind: str,
        action_value: float | str | tuple[int, str],
        delta_t: float,
        n_step: int,
        next_fix_index: int | None,
        on_route: bool,
        generate_simple_environment: Environment,
        build_full_mode_predictor,
    ):
        environment = generate_simple_environment
        environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
        predictor = build_full_mode_predictor(dt=delta_t, fix_proximity_threshold=2.0)

        selected_aircraft = environment.aircraft[callsign]
        pilot = selected_aircraft.pilot
        action = Action(callsign, action_kind, action_value)
        pilot.receive_actions([action], environment)
        pilot.process_actions(environment)

        predictor_aircraft = predictor.predict_aircraft(selected_aircraft, n_step, environment_time=environment.time)
        assert predictor_aircraft.next_fix_index == next_fix_index
        assert predictor_aircraft.on_route == on_route

    def test_speed_from_tables(self, generate_simple_environment: Environment, build_full_mode_predictor):
        predictor = build_full_mode_predictor()
        aircraft = generate_simple_environment.aircraft["AIR0"]
        aircraft.aircraft_type = "B753"

        aircraft.fl = -10.0
        aircraft.selected_fl = 20.0
        cas, mach = predictor.speed_from_tables(aircraft)
        assert cas is not None
        assert mach is not None

        aircraft.fl = 1000.0
        aircraft.selected_fl = 950.0
        cas_hi, mach_hi = predictor.speed_from_tables(aircraft)
        assert cas_hi >= cas
        assert mach_hi >= mach

    def test_vertical_speed_from_tables(self, generate_simple_environment: Environment, build_full_mode_predictor):
        predictor = build_full_mode_predictor()
        aircraft = generate_simple_environment.aircraft["AIR0"]
        aircraft.aircraft_type = "B753"

        aircraft.fl = 220.0
        aircraft.selected_fl = 300.0
        vs_climb = predictor.vertical_speed_from_tables(aircraft)
        aircraft.selected_fl = 150.0
        vs_des = predictor.vertical_speed_from_tables(aircraft)
        assert vs_climb > 0.0
        assert vs_des < 0.0

    def test_predict_trajectory_distances(self, generate_simple_environment: Environment, build_full_mode_predictor):
        environment = generate_simple_environment
        environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
        predictor = build_full_mode_predictor()
        aircraft = environment.aircraft["AIR1"]
        n_step = 15.0

        trajectory = predictor.predict_trajectory(
            aircraft, n_step, environment_time=environment.time, wind_field=environment.wind_field
        )
        start_cp = Pos4D(aircraft.lat, aircraft.lon, aircraft.fl, 0.0)
        trajectory.insert(0, start_cp)

        for index, current_cp in enumerate(trajectory[:-1]):
            next_cp = trajectory[index + 1]
            dt = next_cp.time - current_cp.time
            assert dt > 0

            test_aircraft = copy.deepcopy(aircraft)
            test_aircraft.fl = current_cp.fl
            test_aircraft.lat = current_cp.lat
            test_aircraft.lon = current_cp.lon
            predictor.update_total_speeds(test_aircraft)
            predictor.update_vertical_speeds(test_aircraft)
            horizontal_speed_kts = horizontal_tas(test_aircraft.speed_tas, test_aircraft.vertical_speed)
            wind_vector = environment.wind_field.get_wind_vector(test_aircraft.fl, test_aircraft.lat, test_aircraft.lon)
            test_aircraft.heading = heading_from_ground_track(
                horizontal_tas=horizontal_speed_kts,
                ground_track_angle=current_cp.pos3d().bearing_to(next_cp.pos3d()),
                wind_vector=wind_vector,
            )
            current_groundspeed, _ = ground_speed_from_tas(
                horizontal_tas=horizontal_speed_kts, heading=test_aircraft.heading, wind_vector=wind_vector
            )
            distance = current_cp.pos3d().distance(next_cp.pos3d())
            assert distance == pytest.approx(current_groundspeed * (dt / 3600), rel=5e-2)

    def test_predict_trajectory_alignment(self, generate_simple_environment: Environment, build_full_mode_predictor):
        environment = generate_simple_environment
        environment.wind_field = WindField.uniform(wind_speed=30.0, wind_direction=60.0)
        predictor = build_full_mode_predictor(dt=2.0)
        aircraft = environment.aircraft["AIR0"]

        for new_time in [0.5, 1.0, 1.5]:
            environment.time = new_time
            traj = predictor.predict_trajectory(aircraft, 8.0 - environment.time, environment_time=environment.time)
            traj_times = np.array([cp.time for cp in traj])
            assert np.array_equal(traj_times, np.array([2.0, 4.0, 6.0, 8.0]))

        environment.time = 0.0
        traj = predictor.predict_trajectory(aircraft, 9.5, environment_time=environment.time)
        traj_times = np.array([cp.time for cp in traj])
        assert np.array_equal(traj_times, np.array([1.5, 3.5, 5.5, 7.5, 9.5]))

    def test_full_mode_uses_user_supplied_data_files(
        self, generate_simple_environment: Environment, full_mode_type_a_paths
    ):
        performance_profile_path, performance_uncertainty_path, synonym_map_path = full_mode_type_a_paths
        predictor = LinearPredictor(
            1.0,
            2.0,
            fixes=generate_simple_environment.airspace.fixes,
            performance_profile_data_path=performance_profile_path,
            performance_uncertainty_data_path=performance_uncertainty_path,
            aircraft_mapping_path=synonym_map_path,
        )
        aircraft = generate_simple_environment.aircraft["AIR0"]

        aircraft.aircraft_type = "TYPEA"
        aircraft.fl = 200.0
        aircraft.selected_fl = 200.0
        aircraft.selected_instructions.cas = None
        aircraft.selected_instructions.mach = None

        cas, mach = predictor.speed_from_tables(aircraft)
        assert cas == pytest.approx(270.0)
        assert mach == pytest.approx(0.65)

        aircraft.selected_fl = 230.0
        assert predictor.vertical_speed_from_tables(aircraft) == pytest.approx(1400.0)

class TestPredictorInSimpleMode:
    def test_unknown_aircraft_type_raises(self, generate_simple_environment: Environment):
        predictor = LinearPredictor(1.0, 2.0, fixes=generate_simple_environment.airspace.fixes)
        aircraft = generate_simple_environment.aircraft["AIR1"]

        aircraft.aircraft_type = "UNKNOWN_TYPE"
        aircraft.fl = 200.0
        aircraft.selected_fl = 200.0
        aircraft.selected_instructions.cas = None
        aircraft.selected_instructions.mach = None

        with pytest.raises(ValueError):
            _, _ = predictor.speed_from_tables(aircraft)

    @pytest.mark.parametrize(
        ("aircraft_type", "expected_cas"),
        [
            ("A320", 310.0),
            ("B789", 320.0),
            ("C172", 200.0),
            ("A388", 320.0),
        ],
    )
    def test_builtin_tables_used_without_mach_in_default_operation(
        self, aircraft_type: str, expected_cas: float, generate_simple_environment: Environment
    ):
        predictor = LinearPredictor(1.0, 2.0, fixes=generate_simple_environment.airspace.fixes)
        aircraft = generate_simple_environment.aircraft["AIR1"]

        aircraft.aircraft_type = aircraft_type
        aircraft.fl = 200.0
        aircraft.selected_fl = 200.0
        aircraft.selected_instructions.cas = None
        aircraft.selected_instructions.mach = None

        cas, mach = predictor.speed_from_tables(aircraft)

        assert cas == pytest.approx(expected_cas)
        assert mach is None

    def test_simple_speed_uncertainty_applied_for_percentile_rank(
        self, generate_simple_environment: Environment
    ):
        predictor = LinearPredictor(1.0, 2.0, fixes=generate_simple_environment.airspace.fixes)
        aircraft = generate_simple_environment.aircraft["AIR1"]

        aircraft.aircraft_type = "MEDIUM"
        aircraft.fl = 200.0
        aircraft.selected_fl = 200.0
        aircraft.selected_instructions.cas = None
        aircraft.selected_instructions.mach = None
        aircraft.set_performance(cas_pr=75.0)

        cas, mach = predictor.speed_from_tables(aircraft)

        assert cas == pytest.approx(314.84626264893353)
        assert mach is None

    @pytest.mark.parametrize(
        ("aircraft_type", "expected_cas"),
        [
            ("A320", 310.0),
            ("B789", 320.0),
            ("C172", 200.0),
            ("A388", 320.0),
        ],
    )
    def test_cas_used_when_mach_absent(
        self, aircraft_type: str, expected_cas: float, generate_simple_environment: Environment
    ):
        predictor = LinearPredictor(1.0, 2.0, fixes=generate_simple_environment.airspace.fixes)
        aircraft = generate_simple_environment.aircraft["AIR1"]

        aircraft.aircraft_type = aircraft_type
        aircraft.fl = 200.0
        aircraft.selected_fl = 200.0
        aircraft.selected_instructions.cas = expected_cas
        aircraft.selected_instructions.mach = None

        evolved = predictor.predict_aircraft(aircraft, 1.0, deepcopy_aircraft=True)
        expected_tas = cas_to_tas(aircraft.fl, expected_cas * KT_TO_MPS, 0.0) * MPS_TO_KT

        assert evolved.speed_tas == pytest.approx(expected_tas)
        assert predictor.is_aircraft_below_transition(aircraft) is True

    @pytest.mark.parametrize(
        ("aircraft_type", "expected_climb_rocd", "expected_descent_rocd"),
        [
            ("A320", 2500.0, -2500.0),
            ("B789", 1750.0, -1750.0),
            ("C172", 0.0, -1000.0),
            ("A388", 1750.0, -1750.0),
        ],
    )
    def test_scalar_rocd_and_descent_sign(
        self,
        aircraft_type: str,
        expected_climb_rocd: float,
        expected_descent_rocd: float,
        generate_simple_environment: Environment,
    ):
        predictor = LinearPredictor(1.0, 2.0, fixes=generate_simple_environment.airspace.fixes)
        aircraft = generate_simple_environment.aircraft["AIR1"]
        aircraft.aircraft_type = aircraft_type

        aircraft.fl = 200.0
        aircraft.selected_fl = 230.0
        climb_speed = predictor.vertical_speed_from_tables(aircraft)

        aircraft.selected_fl = 170.0
        descent_speed = predictor.vertical_speed_from_tables(aircraft)

        assert climb_speed == pytest.approx(expected_climb_rocd)
        assert descent_speed == pytest.approx(expected_descent_rocd)

    def test_simple_rocd_uncertainty_applied_for_percentile_rank(
        self, generate_simple_environment: Environment
    ):
        predictor = LinearPredictor(1.0, 2.0, fixes=generate_simple_environment.airspace.fixes)
        aircraft = generate_simple_environment.aircraft["AIR1"]

        aircraft.aircraft_type = "MEDIUM"
        aircraft.fl = 200.0
        aircraft.set_performance(rocd_pr=75.0)

        aircraft.selected_fl = 230.0
        climb_speed = predictor.vertical_speed_from_tables(aircraft)

        aircraft.selected_fl = 170.0
        descent_speed = predictor.vertical_speed_from_tables(aircraft)

        assert climb_speed == pytest.approx(2742.3131324466763)
        assert descent_speed == pytest.approx(-2742.3131324466763)

    def test_no_cas_in_use_cas_mode_raises_error(
        self, generate_simple_environment: Environment
    ):
        predictor = LinearPredictor(
            1.0,
            2.0,
            fixes=generate_simple_environment.airspace.fixes,
            use_cas_as_tas=True,
        )
        aircraft = generate_simple_environment.aircraft["AIR0"]

        aircraft.aircraft_type = 'A320'
        aircraft.fl = 200.0
        aircraft.selected_fl = 200.0
        aircraft.selected_instructions.cas = None
        aircraft.selected_instructions.mach = None

        with pytest.raises(ValueError):
            predictor.predict_aircraft(aircraft, 1.0, deepcopy_aircraft=True)
