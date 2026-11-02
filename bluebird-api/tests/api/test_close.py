import os
import pytest
from bluebird_api.runner import Runner, RunnerStore
from bluebird_api import routers
from bluebird_dt.simulator import Simulator
from bluebird_dt.utility.paths import LOG_DIR as REPLAY_DIR

@pytest.mark.asyncio
async def test_runner_close():
    
    runner_store = RunnerStore(typeof_runner=Runner, typeof_simulator=Simulator)
    runner_store.initialise_from_category("Springfield", "testScenario")

    expected_logfile_name = os.path.join(REPLAY_DIR, runner_store.current_runner.sim.manager.event_logger.log_name + ".tar.gz")
    await routers.core.close(runner_store)
    assert os.path.isfile(expected_logfile_name)
