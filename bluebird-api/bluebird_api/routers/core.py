import typing
from datetime import datetime, timezone

import pandas as pd
from bluebird_dt.events.event_logger import SimStartStop
from bluebird_dt.simulator.common import list_sim_scenario_categories, list_sim_scenarios
from fastapi import APIRouter

from bluebird_api.models import ActionInput, RunnerStore

from ..runnerabc import RunnerDep

core_router = APIRouter()


# explicitly store background tasks
background_tasks = set()


@core_router.get("/", tags=["Control"])
async def index() -> str:
    """
    Verify that the API is running.
    """

    return "Hello, BluebirdATC!"


@core_router.get("/list_scenario_categories", tags=["Scenarios"])
async def list_scenario_categories():  # noqa: ANN201
    """
    List the scenario categories.
    """

    return list_sim_scenario_categories()


@core_router.get("/list_scenarios/{category}", tags=["Scenarios"])
async def list_scenarios(category: str):  # noqa: ANN201
    """
    List the scenarios in a given category.
    """

    return list_sim_scenarios(category)


@core_router.post("/close", tags=["Control"])
async def close(runner: RunnerDep) -> bool:
    """
    Unload a given simulator scenario.
    """

    await runner.delete()
    RunnerStore.current_runner = None
    return True


@core_router.post("/evolve/{time_delta}", tags=["Evolve"])
async def evolve(runner: RunnerDep, time_delta: float) -> bool:
    """
    Evolve the simulation by a given time delta (in seconds).
    Note that this steps through the sim in steps of sim.evolve_period.
    """
    if time_delta <= 0.0:
        raise Exception("Time delta must be positive.")

    # evolve the sim in steps of self.evolve_period until time_delta is reached
    update_amount = runner.evolve_period

    while time_delta > 0.0:
        runner.sim.evolve(update_amount)
        time_delta -= update_amount

    return True


@core_router.post("/start/{tick_frequency_period}", tags=["Evolve"])
async def start(runner: RunnerDep, tick_frequency_period: float) -> bool:
    """
    Start the simulation running with a given tick_frequency_period (in seconds).
    """

    if tick_frequency_period <= 0.0:
        raise Exception("tick_frequency_period must be positive.")

    runner.tick_frequency_period = tick_frequency_period
    runner.sim.manager.event_logger.log_sim_event(
        SimStartStop(event="clocks on", simulation_datetime=runner.sim.manager.environment.datetime)
    )
    runner.log_simrate()
    runner.running = True

    return True


@core_router.get("/environment", tags=["State"])
async def complete_environment(  # noqa: ANN201
    runner: RunnerDep, no_airspace: bool = False, last_n_observations: int = 0
):
    """
    Get the all of the environment data.
    """
    if RunnerStore.current_runner is None or runner.sim is None:
        return {"exists": False}
    # HMI doesn't need to reload the environment again
    runner.sim.manager.reload_environment = False

    return runner.sim.environment(
        sim_time=runner.sim.manager.environment.time,
        sector_id=None,
        no_airspace=no_airspace,
        last_n_observations=last_n_observations,
    )


@core_router.get("/environment/{sector_id}", tags=["State"])
async def environment(  # noqa: ANN201
    runner: RunnerDep, sector_id: str | None = None, no_airspace: bool = False, last_n_observations: int = 0
):
    """
    Get the environment data for a given sector.
    """
    if RunnerStore.current_runner is None or runner.sim is None:
        return {"exists": False}
    # HMI doesn't need to reload the environment again
    runner.sim.manager.reload_environment = False

    return runner.sim.environment(
        sim_time=runner.sim.manager.environment.time,
        sector_id=sector_id,
        no_airspace=no_airspace,
        last_n_observations=last_n_observations,
    )


@core_router.get("/wind_field", tags=["State"])
async def wind_field(runner: RunnerDep):  # noqa: ANN201
    """
    Get the wind field data for the environment.
    """
    if RunnerStore.current_runner is None or runner.sim is None:
        return {"exists": False}
    # HMI doesn't need to reload the environment again
    runner.sim.manager.reload_environment = False

    if not runner.sim.manager.environment.wind_field:
        return None
    return runner.sim.manager.environment.wind_field.data()


@core_router.get("/static_data", tags=["State"])
async def static_data(  # noqa: ANN201
    runner: RunnerDep,
):
    """
    Get the static data for the scenario.
    """
    if RunnerStore.current_runner is None or runner.sim is None:
        return {"exists": False}
    return {"exists": True} | runner.sim.static_data(
        sim_time=runner.sim.manager.environment.time,
    )


@core_router.get("/dynamic_data/{sector_id}", tags=["State"])
async def dynamic_data(  # noqa: ANN201
    runner: RunnerDep, sector_id: str | None = None
):
    """
    Get the dynamic data for the scenario.
    """
    if RunnerStore.current_runner is None or runner.sim is None:
        return {"exists": False}
    sim_time = runner.sim.manager.environment.time
    if sector_id.lower() == "none" or sector_id.lower() == "all":
        return {"exists": True} | runner.sim.dynamic_data(sim_time)

    return {"exists": True} | runner.sim.dynamic_data(sim_time=sim_time, sector_id=sector_id)


@core_router.post("/actions", tags=["Submit"])
async def actions(runner: RunnerDep, action_input: list[ActionInput]) -> bool:
    """
    Send actions to the simulator. The POST body should be a JSON list of
    actions.
    """

    timed_actions = []

    for act in action_input:
        agent = act.agent
        callsign = act.callsign
        kind = act.kind
        value = act.value
        sector = act.sector

        action_time = (
            datetime.fromtimestamp(
                runner.sim.manager.environment.time,
                tz=timezone.utc,
            ).isoformat(timespec="microseconds")[:-6]
        ).replace("T", " ")

        print(f"{action_time} : {agent} : {callsign} -> {kind} = {value}")

        action: dict[str, typing.Any] = {
            "agent": agent,
            "callsign": callsign,
            "kind": kind,
            "value": value,
            "time": action_time,
            "sector": sector,
        }

        timed_actions.append(action)

    runner.sim.action(timed_actions)

    return True


@core_router.get("/status", tags=["Control"])
async def runner_status(runner: RunnerDep):  # noqa: ANN201
    """
    Get the current state of the current run.
    """
    if RunnerStore.current_runner is None:
        return {"exists": False}
    return {
        "exists": runner.scenario_name is not None,
        "iterations": runner.tick,
        "category": runner.category,
        "scenario": runner.scenario_name,
        "running": runner.running,
        "evolve_period": runner.evolve_period,
        "tick_frequency_period": runner.tick_frequency_period,
        "kill": runner.kill,
        "reload": runner.sim.manager.reload_environment if runner.sim is not None else False,
        "time_of_next_tick": runner.time_of_next_tick.isoformat(),
    }


@core_router.post("/save", tags=["Control"])
async def save(runner: RunnerDep) -> bool:
    """
    Save sim state to JSON file.
    """
    runner.sim.save()

    return True


@core_router.post("/evolve_period/{evolve_period}", tags=["Evolve"])
async def set_evolve_period(runner: RunnerDep, evolve_period: float) -> bool:
    """
    Change the update period.
    """

    if evolve_period <= 0.0:
        raise Exception("Evolve period must be positive.")

    runner.evolve_period = evolve_period

    return True


@core_router.post("/tick_frequency/{tick_frequency}", tags=["Evolve"])
async def set_tick_frequency(runner: RunnerDep, tick_frequency: float) -> bool:
    """
    Change the update tick frequency.
    """

    if tick_frequency <= 0.0:
        raise Exception("Tick frequency must be positive.")

    runner.tick_frequency_period = tick_frequency

    return True


@core_router.post("/pause", tags=["Evolve"])
async def pause(runner: RunnerDep) -> bool:
    """
    Pause the simulation.
    """
    runner.running = False

    return True


@core_router.post("/rewind/{new_time}", tags=["Control"])
async def rewind(runner: RunnerDep, new_time: str) -> bool:
    """
    Rewind sim to a previous time
    """

    date_format = "%Y-%m-%d %H:%M:%S"
    rewind_to_time = pd.to_datetime(new_time, format=date_format)
    runner.sim.manager.rewind_to_time(rewind_to_time)

    return True


@core_router.get("/selected_aircraft/{sector_id}", tags=["State"])
async def get_selected_aircraft(sector_id: str, runner: RunnerDep) -> dict:
    """
    Get the currently selected aircraft for a given sector.
    """

    return runner.hmi.get(sector_id, {"selected_aircraft": ""})
