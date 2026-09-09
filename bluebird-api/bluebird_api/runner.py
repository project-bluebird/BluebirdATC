"""
This module provides the interfaces used in all endpoints to obtain data from the runner and simulator that is being
requested.


This module provides the FastAPI dependency for the runner, used to access the simulator
instance, RunnerDep.
To use this dependency, create an endpoint as would be done normally.
As one of the arguments to the function, include the RunnerDep type alias as shown in the example below and
it will be available to interact with.

>>> from ..runner import RunnerDep
>>> @core_router.get("/wind_field", tags=["State"])
>>> async def wind_field(runner: RunnerDep) -> WindField | None:
>>>    return runner.sim.manager.environment.wind_field

If during resolution of the runner, for example trying to find the runner, the runner is not available,
a HTTP error 404 (Not found) will be returned before even running the function above.
"""

import asyncio
import typing
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import typing_extensions
from bluebird_dt.events.event_logger import SimRateUpdate
from bluebird_dt.logger import logger
from bluebird_dt.simulator import Simulator
from fastapi import Depends, HTTPException, Request

from bluebird_api.models import HmiRunnerInformation

TSimulator = typing_extensions.TypeVar("TSimulator", bound=Simulator, default=Simulator)


@dataclass(init=True, slots=True)
class Runner(typing.Generic[TSimulator]):
    sim: TSimulator
    running: bool = False
    evolve_period: float = 6
    tick_frequency_period: float = 6
    tick: int = 0
    kill: bool = False
    time_of_next_tick: datetime = datetime.min
    hmi: defaultdict[str, HmiRunnerInformation] = field(
        default_factory=lambda: defaultdict(lambda: HmiRunnerInformation(selected_aircraft=None))
    )

    async def close(self):
        self.kill = True
        await self.sim.async_save(autosave=False, end_save=True)
        await self.sim.async_close()
        await asyncio.sleep(3)

    def log_simrate(self):
        """
        Creates an entry in the file logs with the current tick frequency and evolve period of the runner.

        See documentation for the event_logger.log_sim_event for more information.
        """
        self.sim.manager.event_logger.log_sim_event(
            SimRateUpdate(
                simulation_datetime=self.sim.manager.environment.datetime,
                tick_frequency=self.tick_frequency_period,
                evolve_period=self.evolve_period,
            )
        )

    async def run_main(self):
        self.time_of_next_tick = datetime.now()

        while True:
            if self.kill:
                break

            if self.running and datetime.now() >= self.time_of_next_tick:
                start_time = datetime.now()
                self.time_of_next_tick = start_time + timedelta(seconds=self.tick_frequency_period)

                await self.sim.async_evolve(self.evolve_period)
                self.tick += 1
                logger.info(f"evolve time: {datetime.now() - start_time}")

            await asyncio.sleep(0.1)


TRunner = typing_extensions.TypeVar("TRunner", bound=Runner, default=Runner[Simulator])


@dataclass(init=True)
class RunnerStore(typing.Generic[TRunner, TSimulator]):
    typeof_runner: type[TRunner]
    typeof_simulator: type[TSimulator]
    current_runner: TRunner | None = None

    async def delete(self):
        if self.current_runner is not None:
            await self.current_runner.close()
            self.current_runner = None

    def initialise_from_category(self, category: str, scenario_name: str):
        self.current_runner = self.typeof_runner(self.typeof_simulator.from_category(category, scenario_name))


async def runner(request: Request) -> Runner[Simulator]:  # noqa: ARG001
    """
    Function taking the runner information from the store, and making it available for the endpoint that uses it. See
    module documentation for more details on usage, and an example.
    """
    runner = request.app.state.runner_store.current_runner

    if runner is None:
        raise HTTPException(404, "Runner instance not found")

    if not isinstance(runner, Runner):
        raise HTTPException(500, "The runner passed is not a valid type.")

    return runner


async def runner_store(request: Request) -> RunnerStore[Runner, Simulator]:
    """
    Function taking the runner store.
    """
    return request.app.state.runner_store


RunnerDep = typing.Annotated[Runner[Simulator], Depends(runner)]
RunnerStoreDep = typing.Annotated[RunnerStore[Runner[Simulator]], Depends(runner_store)]
