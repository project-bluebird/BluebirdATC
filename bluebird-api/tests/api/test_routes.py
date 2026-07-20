import datetime
import logging
import os

import pytest
from pydantic import ValidationError
from bluebird_api import routers
from bluebird_api import routes
from bluebird_api.models import ActionInput
from bluebird_api.runner import Runner
from bluebird_dt.core.wind import WindField
from bluebird_dt.utility.paths import LOG_DIR as REPLAY_DIR
from bluebird_dt.simulator import Simulator

class TestAPI:
    """
    Test that API calls return a successful response
    """
    class TestControl:
        """
        Test API calls corresponding to 'Control' metadata tag
        """

        def test_index(self, client):
            response = client.get("/")

            assert response.status_code == 200

        def test_save(self, client):
            response = client.post(f"/api/save")

            assert response.status_code == 200

        def test_load(self, client):
            response = client.post(f"/load/Springfield/testScenario")

            assert response.status_code == 200

        def test_tenant(self, client):
            response = client.get(f"/status")

            assert response.status_code == 200
       
        def test_close(self, client):
            response = client.post(f"/close/")

            assert response.status_code == 200

    class TestScenarios:
        """
        Test API calls corresponding to 'Scenarios' metadata tag
        """

        def test_list_scenarios(self, client):
            response = client.get("list_scenarios/Springfield")

            assert response.status_code == 200

        def test_list_scenario_categories(self, client):
            response = client.get("list_scenario_categories")

            assert response.status_code == 200

    class TestEvolve:
        """
        Test API calls corresponding to 'Evolve' metadata tag
        """
        # Ensure any previous scenario state is cleared down for this test by passing in tear_down, otherwise tenant will exist
        def test_evolve_non_existant_tenant(self, tear_down, client):
            response = client.post(f"/evolve/8.0")
            assert response.status_code == 404

        def test_evolve(self, client):
            response = client.post(f"/evolve/8.0")

            assert response.status_code == 200

        def test_set_evolve_period(self, client):
            response = client.post(f"/evolve_period/4.0")

            assert response.status_code == 200

        def test_start(self, client):
            response = client.post(f"/start/9.0")

            assert response.status_code == 200

        def test_pause(self, client):
            response = client.post(f"/pause")

            assert response.status_code == 200

    class TestState:
        """
        Test API calls corresponding to 'State' metadata tag
        """

        def test_environment(self, client):
            response = client.get(f"/environment")

            assert response.status_code == 200

        def test_wind_field(self, client):
            response = client.get(f"/wind_field")
            assert response.status_code == 200

        def test_forecast_wind_field(self, client):
            response = client.get(f"/forecast_wind_field")
            assert response.status_code == 200

        def test_static_data(self, client):
            response = client.get(f"/static_data")

            assert response.status_code == 200

        def test_dynamic_data(self, client):
            response = client.get(f"/dynamic_data/SPRINGFIELD")

            assert response.status_code == 200

        def test_get_selected_aircraft(self, client, sector_id):
            response = client.get(f"/selected_aircraft/{sector_id}")
            assert response.status_code == 200

    class TestSubmit:
        """
        Test API calls corresponding to 'Submit' metadata tag
        """

        def test_set_tick_frequency(self, client):
            response = client.post(f"/tick_frequency/243")

            assert response.status_code == 200

        def test_actions(self, client, callsign):
            action = {
                "callsign": callsign,
                "agent": "human",
                "kind": "change_flight_level_to",
                "value": "300",
                "sector": "SPRINGFIELD"
            }
            response = client.post(f"/actions", json=[action])

            assert response.status_code == 200

class TestFunctions:
    """
    Test returns and side effects of functions in routes.py
    """

    class TestControl:
        """
        Test functions corresponding to 'Control' metadata tag
        """

        @pytest.mark.asyncio
        async def test_index(self):
            """
            Test the app is runnable
            """
            received = await routers.core.index()

            assert received == "Hello, BluebirdATC!"

        @pytest.mark.asyncio
        async def test_save(self, mock_save_sim):
            """
            Test that save() returns True
            """

            received = await routers.core.save()

            assert received is True

        @pytest.mark.asyncio
        async def test_load_tenant(self):
            """
            Test that load() ends existing tenancy and creates new tenancy
            """
            received = await routes.load("Springfield", "testScenario")

            assert received is True

        @pytest.mark.asyncio
        async def test_close(self, runner: Runner):
            """
            Test that close() returns True
            """

            received = await routers.core.close(runner)

            assert received is True

        @pytest.mark.asyncio
        async def test_tenant(self, runner: Runner):
            """
            Test that tenant() returns expected state
            """
            received = await routers.core.runner_status(runner)

            expected = {
                "exists": True,
                "iterations": 0,
                "category": runner.sim.category,
                "scenario": runner.sim.scenario_name,
                "running": runner.running,
                "evolve_period": runner.evolve_period,
                "tick_frequency_period": runner.tick_frequency_period,
                "kill": runner.kill,
                "reload": runner.sim.manager.reload_environment if runner.sim is not None else False,
                "time_of_next_tick": runner.time_of_next_tick.isoformat(),
            }

            assert received == expected

            # check time of next tick separately because times will differ slightly
            received_time = datetime.datetime.fromisoformat(received["time_of_next_tick"])
            expected_time = datetime.datetime.fromisoformat('0001-01-01T00:00:00')
            assert received_time - expected_time < datetime.timedelta(seconds=1)

    class TestScenarios:
        """
        Test functions corresponding to 'Scenarios' metadata tag
        """

        @pytest.mark.asyncio
        async def test_list_scenarios(self):
            """
            Test that list_scenarios() returns list for valid scenario
            """
            received = await routers.core.list_scenarios("Springfield")

            assert isinstance(received, list)

        @pytest.mark.asyncio
        async def test_list_scenario_categories(self):
            """
            Test that list_scenario_categories() returns list
            """
            received = await routers.core.list_scenario_categories()

            assert isinstance(received, list)

    class TestEvolve:

        @pytest.mark.asyncio
        async def test_evolve_negative_time_delta(self, runner: Runner):
            """
            Test that evolve() raises an Exception if time delta is negative
            """
            with pytest.raises(Exception) as excinfo:
                await routers.core.evolve(runner, -1.0)

            assert str(excinfo.value) == "Time delta must be positive."

        @pytest.mark.asyncio
        async def test_evolve(self, runner: Runner):
            """
            Test that evolve() returns True
            """
            received = await routers.core.evolve(runner, 8.0)

            assert received is True

        @pytest.mark.asyncio
        async def test_start_negative_period(self, runner: Runner):
            """
            Test that start() raises an Exception if tick frequency period is not positive
            """
            with pytest.raises(Exception) as excinfo:
                await routers.core.start(runner, -1.0)

            assert str(excinfo.value) == "tick_frequency_period must be positive."

        @pytest.mark.asyncio
        async def test_start(self, runner: Runner):
            """
            Test that start() returns True, sets tick frequency period,
            and starts the tenant running
            """
            received = await routers.core.start(runner, 9.0)

            assert received is True
            assert runner.tick_frequency_period == 9.0
            assert runner.running is True
            assert runner.sim.manager.event_logger._sim_events[-2].event == 'clocks on'
            assert runner.sim.manager.event_logger._sim_events[-1].tick_frequency == 9.0

        @pytest.mark.asyncio
        async def test_pause(self, runner: Runner):
            """
            Test that pause() returns true and pauses the tenant
            """
            # Ensure simulation is running before pause
            await routers.core.start(runner, 10.0)

            assert runner.running is True

            received = await routers.core.pause(runner)

            assert received == True
            assert runner.running is False

        @pytest.mark.skip
        @pytest.mark.parametrize("sector_id", ["SPRINGFIELD"])
        @pytest.mark.asyncio
        async def test_rewind(self, sector_id):
            """
            xxxx
            """
            assert False

    class TestState:
        """
        Test functions corresponding to 'State' metadata tag
        """

        @pytest.mark.asyncio
        async def test_environment_default(self, runner: Runner):
            """
            Test that environment() with default arguments
            returns a dict with the expected keys
            """
            expected_keys = ["time", "start_time", "aircraft", "airspace", "coordinations"]
            received = await routers.core.environment(runner)

            assert sorted(received.keys()) == sorted(expected_keys)

        @pytest.mark.asyncio
        async def test_environment_params(self, runner: Runner):
            """
            Test that environment() with additional arguments
            returns a dict with the expected keys
            """
            expected_keys = [
                "time",
                "start_time",
                "aircraft",
                "coordinations",
                "observations",
            ]
            received = await routers.core.environment(runner, no_airspace=True, last_n_observations=5)

            assert sorted(received.keys()) == sorted(expected_keys)

        @pytest.mark.asyncio
        async def test_wind_field(self, runner: Runner):
            """
            Test that wind_field() returns a ``WindField`` instance (or ``None``)
            when a runner is loaded.
            """

            received = await routers.core.wind_field(runner)

            assert received is None or isinstance(received, WindField)

        @pytest.mark.asyncio
        async def test_wind_field_with_seeded_wind(self, runner: Runner):
            """
            Seed the environment with a known WindField and verify that wind_field()
            returns the same instance.
            """
            seeded = WindField.uniform(wind_speed=10.0, wind_direction=90.0)
            original = runner.sim.manager.environment.wind_field
            runner.sim.manager.environment.wind_field = seeded
            try:
                received = await routers.core.wind_field(runner)
            finally:
                runner.sim.manager.environment.wind_field = original

            assert isinstance(received, WindField)
            assert received == seeded

        @pytest.mark.asyncio
        async def test_forecast_wind_field(self, runner: Runner):
            """
            Test that forecast_wind_field() returns a ``WindField`` instance (or
            ``None``) when a runner is loaded.
            """

            received = await routers.core.forecast_wind_field(runner)

            assert received is None or isinstance(received, WindField)

        @pytest.mark.asyncio
        async def test_forecast_wind_field_with_seeded_wind(self, runner: Runner):
            """
            Seed the environment with a known forecast WindField and verify that
            forecast_wind_field() returns the same instance.
            """
            seeded = WindField.uniform(wind_speed=10.0, wind_direction=90.0)
            original = runner.sim.manager.environment.forecast_wind_field
            runner.sim.manager.environment.forecast_wind_field = seeded
            try:
                received = await routers.core.forecast_wind_field(runner)
            finally:
                runner.sim.manager.environment.forecast_wind_field = original

            assert isinstance(received, WindField)
            assert received == seeded

        @pytest.mark.asyncio
        async def test_static_data(self, runner: Runner):
            """
            Test that static_data() returns a dict with the expected keys,
            the data exists and the sectors are as expected
            """
            expected_keys = ["exists", "scenario_name", "bay_names", "sectors", "fixes", "projection_centre"]
            expected_sectors = {"SPRINGFIELD"}
            
            received = await routers.core.static_data(runner)

            assert sorted(received.keys()) == sorted(expected_keys)
            assert received["exists"] is True
            assert set(received["sectors"].keys()) == expected_sectors

        @pytest.mark.parametrize(
            "sector_id",
            ["SPRINGFIELD"],
        )
        @pytest.mark.asyncio
        async def test_dynamic_data_keys(self, runner: Runner, sector_id: str):
            """
            Test that dynamic_data() returns a dict with the expected keys and the data exists
            """
            expected_keys = {
                "exists",
                "time",
                "actions",
                "aircraft",
            }

            received = await routers.core.dynamic_data(runner, sector_id)
            received_keys = set(received.keys())

            assert received_keys == expected_keys, (received_keys, expected_keys)
            assert received["exists"] is True

    class TestSubmit:
        """
        Test functions corresponding to 'Submit' metadata tag
        """

        @pytest.mark.parametrize(
            ("action_kind", "expected_warning"),
            [
                ("route_direct_to", True),
                ("maintain_current_heading", False),
                ("outcomm", False),
                ("change_flight_level_to", True),
                ("change_flight_level_by", True),
                ("change_vertical_speed_to", True),
                ("change_heading_to", True),
                ("change_heading_by", True),
                ("change_cas_to", False),
                ("change_mach_to", False),
            ],
        )
        @pytest.mark.asyncio
        async def test_action_warnings(self, runner: Runner, callsign: str, action_kind: str, expected_warning: bool, caplog: pytest.LogCaptureFixture):
            """
            Test that action() raises a warning when action value is required
            but not supplied.
            """
            params = [ActionInput(agent="human", callsign=callsign,
                                  kind=action_kind, value=None,
                                  sector="Springfield")]
            
            with caplog.at_level(logging.WARN):
                await routers.core.actions(runner, params) is not None
                if expected_warning:
                    assert "An error occurred sending actions" in caplog.text
                else:
                    assert len(caplog.record_tuples) == 0

        @pytest.mark.parametrize("action_kind", ["barrel_roll", ""])
        @pytest.mark.asyncio
        async def test_unknown_action_error(self, runner: Runner, callsign: str, action_kind: str):
            """
            Test that constructing an ActionInput with an unknown action kind
            raises an Exception.
            """
            with pytest.raises(ValidationError) as excinfo:
                params = ActionInput(agent="human", callsign=callsign,
                                     kind=action_kind, value=None,
                                     sector="Springfield")

        @pytest.mark.parametrize(
            ("action_kind", "action_value"),
            [
                ("route_direct_to", "OCK"),
                ("maintain_current_heading", 0),
                ("outcomm", None),
                ("change_flight_level_to", 300),
                ("change_flight_level_by", -20),
                ("change_vertical_speed_to", 1500),
                ("change_heading_to", 300),
                ("change_heading_by", 10),
                ("change_cas_to", 250),
                ("change_mach_to", 0.82),
                ("message", "Agent unable to achieve coordinated level"),
            ],
        )
        @pytest.mark.asyncio
        async def test_actions(self, runner: Runner, callsign, action_kind: str, action_value):
            """
            Test that actions() returns True and adds the expected action to the manager actions_to_issue list
            """
            params = [ActionInput(agent="human", callsign=callsign,
                                  kind=action_kind, value=action_value,
                                  sector="Springfield")]

            received = await routers.core.actions(runner, params)
            actions = runner.sim.manager._actions_to_issue

            assert received is True

            last_action = actions[-1].data()

            assert last_action["callsign"] == callsign
            assert last_action["agent"] == "human"
            assert last_action["kind"] == action_kind
            assert last_action["value"] == action_value

        @pytest.mark.asyncio
        async def test_set_tick_frequency_negative_period(self, runner: Runner):
            """
            Test that set_tick_frequency_period() raises an Exception if tick frequency period is not positive
            """
            with pytest.raises(Exception) as excinfo:
                await routers.core.set_tick_frequency(runner, 0.0)

            assert str(excinfo.value) == "Tick frequency must be positive."

        @pytest.mark.asyncio
        async def test_set_tick_frequency(self, runner: Runner):
            """
            Test that set_tick_frequency_period() returns True and sets tick_frequency_period
            """
            received = await routers.core.set_tick_frequency(runner, 243.0)

            assert received is True
            assert runner.tick_frequency_period == 243


@pytest.mark.asyncio
async def test_runner_close():
    runner = Runner(Simulator.from_category("Springfield", "testScenario"))
    expected_logfile_name = os.path.join(REPLAY_DIR, runner.sim.manager.event_logger.log_name + ".tar.gz")
    await routers.core.close(runner)
    assert os.path.isfile(expected_logfile_name)
