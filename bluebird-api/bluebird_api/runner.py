"""
This module provides the interfaces used in all endpoints to obtain data from the runner and simulator that is being
requested.

The first thing available is the RunnerABC class, which is implemented for BluebirdATC in ./runner.py.

Secondly, and most importantly, this module provides the FastAPI dependency for the runner, used to access the simulator
instance, RunnerDep.
To use this dependency, create an endpoint as would be done normally.
As one of the arguments to the function, include the RunnerDep type alias as shown in the example below and
it will be available to interact with.

>>> from ..runnerabc import RunnerDep
>>> @core_router.post("/close", tags=["Control"])
>>> async def close(runner: RunnerDep) -> bool:
>>>    await runner.delete()
>>> return True

If during resolution of the runner, for example trying to find the runner, the runner is not available,
a HTTP error 404 (Not found) will be returned before even running the function above.
"""

import asyncio
import typing
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from bluebird_dt.events.event_logger import SimRateUpdate
from bluebird_dt.logger import logger
from bluebird_dt.simulator import Simulator
from fastapi import Depends, HTTPException

from bluebird_api.models import HmiRunnerInformation

TSimulator = typing.TypeVar("TSimulator", bound=Simulator)


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

    async def delete(self):
        self.kill = True
        self.sim.save()
        self.sim.close()
        self.sim = None
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
            if self.running and datetime.now() >= self.time_of_next_tick:
                start_time = datetime.now()
                self.time_of_next_tick = start_time + timedelta(seconds=self.tick_frequency_period)

                self.sim.evolve(self.evolve_period)
                self.tick += 1
                logger.info(f"evolve time: {datetime.now() - start_time}")

            if self.kill:
                break

            await asyncio.sleep(0.1)


class RunnerStore:
    current_runner: Runner[Simulator] | None = None


async def runner() -> Runner[Simulator]:  # noqa: ARG001
    """
    Function taking the runner information from the store, and making it available for the endpoint that uses it. See
    module documentation for more details on usage, and an example.
    """
    runner = RunnerStore.current_runner

    if runner is None:
        raise HTTPException(404, "Runner instance not found")

    if not isinstance(runner, Runner):
        raise HTTPException(500, "The runner passed is not a valid type.")

    return runner


RunnerDep = typing.Annotated[Runner[Simulator], Depends(runner)]
