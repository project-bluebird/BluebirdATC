import asyncio
import pytest
from fastapi.testclient import TestClient
import datetime

from bluebird_api import app
from bluebird_api.runner import Runner, RunnerStore
from bluebird_dt.simulator import Simulator

from pytest import MonkeyPatch

@pytest.fixture(scope="module")
def runner():
    mp = MonkeyPatch()

    async def mock_delete(_):
        pass

    mp.setattr(RunnerStore, "delete", mock_delete)

    try:
        yield Runner(
            Simulator.from_category("Springfield", "testScenario")
        )
    finally:
        mp.undo()

@pytest.fixture(scope="module")
def runner_store():
    mp = MonkeyPatch()

    async def mock_delete(_):
        pass

    mp.setattr(RunnerStore, "delete", mock_delete)

    try:
        yield RunnerStore(
                typeof_runner=Runner,
                typeof_simulator=Simulator,
                    current_runner=Runner(
                Simulator.from_category("Springfield", "testScenario")
            )
        )
    finally:
        mp.undo()

@pytest.fixture(scope="module")
def sector_id():
    return "test_sector"

@pytest.fixture
def callsign():
    return "AIR01"
    
@pytest.fixture(scope="module")
def client():
    """
    Fixture to create a TestClient instance for testing
    """
    with TestClient(app) as client:
        yield client

@pytest.fixture(scope="function", autouse=True)
def setup(client):
    """
    Load scenario before running test if not currently loaded.
    Allow an opt-out for the occasional test which specifically tests prior to load
    """
    if not client.get("/status").json().get("exists"):
        client.post("/load/Springfield/testScenario")

@pytest.fixture()
def tear_down(client):
    store = client.app.state.runner_store

    if store.current_runner is not None:
        try:
            asyncio.run(store.delete())
        except Exception:
            pass
        finally:
            store.current_runner = None

    yield

@pytest.fixture
def mock_save_sim(monkeypatch):
    """
    Prevent save files being created
    """

    async def mock_save():
        return True

    monkeypatch.setattr("bluebird_api.routers.core.save", mock_save)


@pytest.fixture
def mock_evolve(monkeypatch):
    """
    Prevent simulation being evolved
    """

    async def mock_evolve(arg1, arg2):
        return True

    monkeypatch.setattr("bluebird_api.routers.core.evolve", mock_evolve)


@pytest.fixture
def mock_filter_log_list(monkeypatch):
    """
    Prevent the API trying to load logs which might not exist.
    """

    def mock(*args):
        return []

    monkeypatch.setattr("bluebird_api.routers.metrics.filter_log_list", mock)
