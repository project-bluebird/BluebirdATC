"""
The routes module builds the router for the provided endpoint and adds any endpoints only available for
BluebirdATC, including loading which is implementation dependent.
"""

import asyncio

from bluebird_dt.simulator import Simulator
from fastapi import APIRouter

from bluebird_api.routers.core import background_tasks
from bluebird_api.runner import Runner, RunnerStore

from .routers import (
    core_router,
)

router = APIRouter()
router.include_router(core_router)


@router.post("/load/{category}/{scenario_name}", tags=["Control"])
async def load(category: str, scenario_name: str) -> bool:  # noqa: ARG001
    """
    End any existing run, then create a new Runner and load a given simulator scenario.
    """
    if RunnerStore.current_runner is not None:
        await RunnerStore.delete()

    RunnerStore.current_runner = Runner(Simulator.from_category(category, scenario_name))

    # start the task
    task = asyncio.create_task(RunnerStore.current_runner.run_main())

    # add the task to the background tasks set and have it auto-remove its reference from the set when done
    background_tasks.add(task)
    task.add_done_callback(background_tasks.remove)

    return True
