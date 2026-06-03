"""
The runner implementation for the general BluebirdATC API. This runner is responsible for the initialisation of the
simulator class.
"""

import asyncio

from bluebird_dt.simulator import Simulator
from typing_extensions import override

from bluebird_api.runnerabc import RunnerABC


class Runner(RunnerABC):
    def __init__(self, category: str, scenario_name: str):
        super().__init__(category, scenario_name)

    @override
    def initialise_simulator(self, *args, **kwargs) -> Simulator:  # noqa: ANN002, ANN003
        return Simulator.from_category(*args, **kwargs)

    @override
    async def delete(self):
        self.kill = True
        self.sim.save()
        self.sim.close()
        self.sim = None
        await asyncio.sleep(3)
