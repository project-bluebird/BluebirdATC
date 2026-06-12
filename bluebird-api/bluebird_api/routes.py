"""
The routes module builds the router for the provided endpoint and adds any endpoints only available for
BluebirdATC, including loading which is implementation dependent.
"""

import asyncio

from fastapi import APIRouter, Depends, Request

from bluebird_api.models import RunnerStore
from bluebird_api.routers.core import background_tasks
from bluebird_api.runner import Runner

from .routers import (
    core_router,
)


async def simulator(request: Request) -> None:
    """
    The simulator dependency, available to endpoints using the RunnerDep dependency, provides access to the runner.
    """

    request.state.runner = RunnerStore.current_runner


router: APIRouter = APIRouter(dependencies=[Depends(simulator)])

router.include_router(core_router)


@router.post("/load/{category}/{scenario_name}", tags=["Control"])
async def load(category: str, scenario_name: str) -> bool:  # noqa: ARG001
    """
    End any existing run, then create a new Runner and load a given simulator scenario.
    """

    if RunnerStore.current_runner is not None:
        await RunnerStore.current_runner.delete()

    RunnerStore.current_runner = Runner(category, scenario_name)

    # start the task
    task: asyncio.Task = asyncio.create_task(RunnerStore.current_runner.run_main())

    # add the task to the background tasks set and have it auto-remove its reference from the set when done
    background_tasks.add(task)
    task.add_done_callback(background_tasks.remove)

    return True
