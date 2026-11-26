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
async def load(category: str, scenario_name: str, runner_store: RunnerStoreDep, use_wind: bool = True) -> bool:
    """
    End any existing run, then create a new Runner and load a given simulator scenario.

    Set use_wind to false to load the scenario without its wind and without its forecast. By default both are
    loaded, if the scenario has them.
    """
    if runner_store.current_runner is not None:
        await runner_store.delete()

    runner_store.initialise_from_category(category, scenario_name, use_wind=use_wind)

    runner_store.start()

    return True
