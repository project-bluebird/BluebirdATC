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
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta

from bluebird_dt.events.event_logger import SimRateUpdate
from bluebird_dt.logger import logger
from bluebird_dt.simulator.simulator import Simulator
from fastapi import Depends, HTTPException, Request


class RunnerABC(ABC):
    kill: bool
    category: str
    scenario_name: str
    sim: Simulator
    running: bool
    evolve_period: float
    tick_frequency_period: float
    tick: int
    hmi: dict

    @abstractmethod
    def initialise_simulator(self, *args, **kwargs) -> Simulator:  # noqa: ANN002, ANN003
        """
        Function to instantiate the simulator class, or variations of it for each use case.

        This function is designed to be a transparent function taking all the arguments passed into the constructor of
        the runner class from the load endpoint.
        """
        pass

    @abstractmethod
    async def delete(self):
        pass

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

    def __init__(self, category: str, scenario_name: str, log_name: str | None = None):
        """
        Constructor of the Runner classes.

        Although the arguments of this function are currently hardcoded, if a usecase requires different ones they will
        be replaced by args and kwargs placeholders.

        Arguments
        ---------
        category: str
            The category of scenario to load
        scenario_name: str
            The specific scenario of the category to load
        log_name: str
            The name to store the logs for the run.
        """
        self.category = category
        self.scenario_name = scenario_name
        self.sim = self.initialise_simulator(self.category, self.scenario_name, log_filename=log_name)
        self.running = False
        self.evolve_period = 6.0
        self.tick_frequency_period = 6.0
        self.kill = False
        self.time_of_next_tick = datetime.min
        self.tick = 0
        self.hmi = defaultdict(lambda: {"selected_aircraft": ""})

        # allow 'None' and 'ALL' sectors to also select aircraft
        self.hmi["None"] = {"selected_aircraft": ""}
        self.hmi["ALL"] = {"selected_aircraft": ""}
        # try a lock to avoid concurrency bugs
        self._sim_lock = asyncio.Lock()

    async def run_main(self):
        self.time_of_next_tick = datetime.now()

        while True:
            if self.running and datetime.now() >= self.time_of_next_tick:
                async with self._sim_lock:
                    start_time = datetime.now()
                    self.time_of_next_tick = start_time + timedelta(seconds=self.tick_frequency_period)

                    self.sim.evolve(self.evolve_period)
                    self.tick += 1
                    logger.info(f"evolve time: {datetime.now() - start_time}")

            if self.kill:
                self.category = None
                self.scenario_name = None
                break

            await asyncio.sleep(0.1)


async def runner(request: Request) -> RunnerABC:  # noqa: ARG001
    """
    Function taking the runner information from the state, and making it available for the endpoint that uses it. See
    module documentation for more details on usage, and an example.

    This FastAPI dependency will throw a HTTP exception if the runner isinstance is not found, therefore this does not
    need to be handled by the individual endpoint.
    """
    runner = request.state.runner

    if runner is None:
        raise HTTPException(404, "Runner instance not found")

    if not isinstance(runner, RunnerABC):
        raise HTTPException(500, "The runner passed is not a valid type.")

    request.state.runner = None
    return runner


RunnerDep = typing.Annotated[RunnerABC, Depends(runner)]
