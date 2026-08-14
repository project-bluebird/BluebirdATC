import asyncio
import gc
import os
from unittest.mock import MagicMock, patch
import uuid
import weakref
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from bluebird_dt.core import Coordination
from bluebird_dt.logger import logger
from bluebird_dt.scenario_manager.springfield import SpringfieldScenarioManager, SpringfieldScenarioManagerConfig
from bluebird_dt.simulator import Simulator
from bluebird_dt.simulator.simconfig import SimConfig
from bluebird_dt.utility.paths import LOG_DIR

skip_cases = pytest.mark.parametrize(
    ("sim_elapsed", "real_elapsed"),
    [
        (9, 9), # Normal time
        (10, 9), # Fast time
        (9, 10), # Slow time
    ],
)
proceed_cases = pytest.mark.parametrize(
    ("sim_elapsed", "real_elapsed"),
    [
        (10, 10), # Normal time
        (11, 10), # Fast time
        (10, 11), # Slow time
    ],
)
autosave_cases = pytest.mark.parametrize("autosave", [True, False], ids=["autosave_true", "autosave_false"],)
last_save_states = pytest.mark.parametrize("last_save_task_success", [True, False, None], ids=["last_save_true", "last_save_false", "last_save_none"])


@skip_cases
@autosave_cases
def test_chunk_prepare_savedata_autosave_skip(sim_elapsed, real_elapsed, autosave):
    """
    Verify that _chunk_prepare_savedata autosave skipped when the interval has not elapsed.
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", autosave_interval=timedelta(minutes=10))

    # Fake state
    now_realtime = datetime.now(tz=UTC)
    now_simtime = sim.manager.environment.datetime
    sim.save_config.save_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.save_config.save_realtime = now_realtime - timedelta(minutes=real_elapsed)

    if autosave:
        assert sim._prepare_savedata_and_chunk(autosave=autosave) is None
    else:
        assert sim._prepare_savedata_and_chunk(autosave=autosave) is not None
        assert sim.save_config.save_simtime == now_simtime
        assert sim.save_config.save_realtime - now_realtime < timedelta(seconds=5)


@proceed_cases
@autosave_cases
def test_chunk_prepare_savedata_autosave_proceed(sim_elapsed, real_elapsed, autosave):
    """
    Verify _chunk_prepare_savedata autosave proceeds when the interval has elapsed.
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", autosave_interval=timedelta(minutes=10))

    # Fake state
    now_realtime = datetime.now(tz=UTC)
    now_simtime = sim.manager.environment.datetime
    sim.save_config.save_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.save_config.save_realtime = now_realtime - timedelta(minutes=real_elapsed)

    assert sim._prepare_savedata_and_chunk(autosave=autosave) is not None
    assert sim.save_config.save_simtime == now_simtime
    assert sim.save_config.save_realtime - now_realtime < timedelta(seconds=5)


@skip_cases
@autosave_cases
@last_save_states
def test_chunking_skipped(sim_elapsed, real_elapsed, autosave, last_save_task_success):
    """
    Verify that _chunk_prepare_savedata chunking skipped when the interval has not elapsed.
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", autosave_interval=timedelta(minutes=10), save_chunk_interval=timedelta(minutes=10))

    # Mock
    trim_logger = MagicMock()
    trim_handler = MagicMock()
    sim.manager.event_logger.trim_and_clip = trim_logger
    sim.manager.event_handler.trim = trim_handler

    # Fake state
    now_realtime = datetime.now(tz=UTC)
    now_simtime = sim.manager.environment.datetime
    sim.save_config.chunk_start_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.save_config.chunk_start_realtime = now_realtime - timedelta(minutes=real_elapsed)
    sim.save_config.save_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.save_config.save_realtime = now_realtime - timedelta(minutes=real_elapsed)
    sim.save_config.last_save_task_success = last_save_task_success
    sim.save_config.last_save_task_save_simtime = now_simtime - timedelta(minutes=sim_elapsed)

    sim._prepare_savedata_and_chunk(autosave=autosave)

    trim_logger.assert_not_called()
    trim_handler.assert_not_called()
    assert sim.save_config.save_chunk_id == 0


@proceed_cases
@autosave_cases
@last_save_states
@pytest.mark.parametrize("last_save_delayed", [True, False], ids=["last_save_delayed_true", "last_save_delayed_false"],)
def test_chunking_proceed(sim_elapsed, real_elapsed, autosave, last_save_task_success, last_save_delayed):
    """
    Verify that _chunk_prepare_savedata autosave proceed when the interval has elapsed.
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", autosave_interval=timedelta(minutes=10), save_chunk_interval=timedelta(minutes=10))

    # Mock
    trim_logger = MagicMock()
    trim_handler = MagicMock()
    sim.manager.event_logger.trim_and_clip = trim_logger
    sim.manager.event_handler.trim = trim_handler

    # Fake state
    now_realtime = datetime.now(tz=UTC)
    now_simtime = sim.manager.environment.datetime
    sim.save_config.chunk_start_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.save_config.chunk_start_realtime = now_realtime - timedelta(minutes=real_elapsed)
    sim.save_config.save_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.save_config.save_realtime = now_realtime - timedelta(minutes=real_elapsed)
    sim.save_config.last_save_task_success = last_save_task_success
    sim.save_config.last_save_task_save_simtime = now_simtime - timedelta(minutes=sim_elapsed) - timedelta(minutes=10 if last_save_delayed else 0)

    sim._prepare_savedata_and_chunk(autosave=autosave)

    if last_save_task_success and not last_save_delayed:
        trim_logger.assert_called_once()
        trim_handler.assert_called_once()
        assert sim.save_config.save_chunk_id == 1
    else:
        trim_logger.assert_not_called()
        trim_handler.assert_not_called()
        assert sim.save_config.save_chunk_id == 0


@skip_cases
@autosave_cases
def test_save_skip(sim_elapsed, real_elapsed, autosave):
    """
    Verify that _chunk_prepare_savedata autosave returns None when both simtime and realtime intervals have not elapsed.
    """
    log_filename = f"test_save_skip_{uuid.uuid4()}"

    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", autosave_interval=timedelta(minutes=10), log_filename=log_filename)
    sim.evolve(6)

    # Fake state
    now_realtime = datetime.now(tz=UTC)
    now_simtime = sim.manager.environment.datetime
    sim.save_config.save_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.save_config.save_realtime = now_realtime - timedelta(minutes=real_elapsed)

    if autosave:
        assert sim.save(autosave=autosave) is False
    else:
        assert sim.save(autosave=autosave) is True
        assert os.path.exists(os.path.join(LOG_DIR, log_filename + ".tar.gz"))
        assert os.path.exists(os.path.join(LOG_DIR, "runtime_logs", log_filename + ".log"))
        os.remove(os.path.join(LOG_DIR, log_filename + ".tar.gz"))
        os.remove(os.path.join(LOG_DIR, "runtime_logs", log_filename + ".log"))

@proceed_cases
@autosave_cases
def test_save_proceed(sim_elapsed, real_elapsed, autosave):
    """
    Verify _chunk_prepare_savedata autosave proceeds once the interval has elapsed.
    """
    log_filename = f"test_save_proceed_{uuid.uuid4()}"
    
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", autosave_interval=timedelta(minutes=10), log_filename=log_filename)
    sim.evolve(6)

    # Fake state
    now_realtime = datetime.now(tz=UTC)
    now_simtime = sim.manager.environment.datetime
    sim.save_config.save_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.save_config.save_realtime = now_realtime - timedelta(minutes=real_elapsed)

    assert sim.save(autosave=autosave) is True
    assert os.path.exists(os.path.join(LOG_DIR, log_filename + ".tar.gz"))
    assert os.path.exists(os.path.join(LOG_DIR, "runtime_logs", log_filename + ".log"))
    os.remove(os.path.join(LOG_DIR, log_filename + ".tar.gz"))
    os.remove(os.path.join(LOG_DIR, "runtime_logs", log_filename + ".log"))


@skip_cases
@autosave_cases
@pytest.mark.asyncio
async def test_async_save_skip(sim_elapsed, real_elapsed, autosave):
    """
    Verify that _chunk_prepare_savedata autosave returns None when both simtime and realtime intervals have not elapsed.
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", autosave_interval=timedelta(minutes=10))
    await sim.async_evolve(6)

    # Fake state
    now_realtime = datetime.now(tz=UTC)
    now_simtime = sim.manager.environment.datetime
    sim.save_config.save_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.save_config.save_realtime = now_realtime - timedelta(minutes=real_elapsed)

    assert sim.current_save_task is None
    if autosave:
        assert await sim.async_save(autosave=autosave) is False
        assert sim.current_save_task is None
    else:
        assert await sim.async_save(autosave=autosave) is True
        assert sim.current_save_task is not None
    assert sim.next_save_data is None


@proceed_cases
@autosave_cases
@pytest.mark.asyncio
async def test_async_save_proceed(sim_elapsed, real_elapsed, autosave):
    """
    Verify _chunk_prepare_savedata autosave proceeds once the interval has elapsed.
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", autosave_interval=timedelta(minutes=10))
    await sim.async_evolve(6)

    # Fake state
    now_realtime = datetime.now(tz=UTC)
    now_simtime = sim.manager.environment.datetime
    sim.save_config.save_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.save_config.save_realtime = now_realtime - timedelta(minutes=real_elapsed)

    assert sim.current_save_task is None
    assert await sim.async_save(autosave=autosave) is True
    assert sim.current_save_task is not None


previous_save_data = object()
next_save_data = object()
@pytest.mark.parametrize(
    ("current_task_done", "previous_save_data", "next_save_data", "expected_create_task_calls", "expected_current_task_is_none", "expected_next_save_data"),
    [
        pytest.param(True, None, next_save_data, 1, False, None, id="done_with_new_save_overrides_empty_queue"),
        pytest.param(True, None, None, 0, True, None, id="done_with_no_new_save_clears_task"),
        pytest.param(False, None, next_save_data, 0, False, next_save_data, id="active_task_keeps_new_save_in_queue"),
        pytest.param(False, None, None, 0, False, None, id="active_task_ignores_no_new_save"),
        pytest.param(None, None, next_save_data, 1, False, None, id="idle_starts_new_task_from_queue"),
        pytest.param(None, None, None, 0, True, None, id="idle_with_empty_queue_remains_none"),

        pytest.param(True, previous_save_data, next_save_data, 1, False, None, id="done_replaces_old_queue_with_new_save"),
        pytest.param(False, previous_save_data, next_save_data, 0, False, next_save_data, id="active_keeps_latest_save_over_old_queue"),
        pytest.param(None, previous_save_data, next_save_data, 1, False, None, id="idle_starts_new_task_after_old_queue"),
    ],
)
def test_update_async_save_task(
    current_task_done,
    previous_save_data,
    next_save_data,
    expected_create_task_calls,
    expected_current_task_is_none,
    expected_next_save_data,
):
    """
    Verify update_async_save_task moves queued autosave work through the current task lifecycle.
    """
    def mock_save_task(done: bool) -> MagicMock:
        task = MagicMock()
        task.done.return_value = done
        return task

    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario")
    sim.current_save_task = None if current_task_done is None else mock_save_task(current_task_done)
    sim.next_save_data = previous_save_data # Simulate after async_save execute with new save data
    sim.next_save_data = next_save_data

    new_task = MagicMock()
    new_task.done.return_value = False

    with patch("bluebird_dt.simulator.simulator.asyncio.create_task", return_value=new_task) as create_task:
        sim.update_async_save_task()

    assert create_task.call_count == expected_create_task_calls

    if expected_create_task_calls:
        scheduled_save_data = create_task.call_args.args[0].cr_frame.f_locals["save_data"]
        assert scheduled_save_data is next_save_data

    assert sim.next_save_data is expected_next_save_data

    if expected_current_task_is_none:
        assert sim.current_save_task is None
    elif expected_create_task_calls:
        assert sim.current_save_task is new_task
    else:
        assert sim.current_save_task is not None
        assert sim.current_save_task.done() is False
