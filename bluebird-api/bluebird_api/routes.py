"""
The routes module builds the router for the provided endpoint and adds any endpoints only available for
BluebirdATC, including loading which is implementation dependent.
"""

from fastapi import APIRouter

from bluebird_api.runner import RunnerStoreDep

from .routers import (
    core_router,
)

router = APIRouter()
router.include_router(core_router)


@router.post("/load/{category}/{scenario_name}", tags=["Control"])
async def load(category: str, scenario_name: str, runner_store: RunnerStoreDep) -> bool:  # noqa: ARG001
    """
    End any existing run, then create a new Runner and load a given simulator scenario.
    """
    if runner_store.current_runner is not None:
        await runner_store.delete()

    runner_store.initialise_from_category(category, scenario_name)

    runner_store.start()

    return True
