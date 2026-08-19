import asyncio
import gc
import os
from unittest.mock import MagicMock, patch
import uuid
import weakref
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from bluebird_dt.core import Coordination
from bluebird_dt.logger import logger
from bluebird_dt.scenario_manager.springfield import SpringfieldScenarioManager, SpringfieldScenarioManagerConfig
from bluebird_dt.simulator import Simulator
from bluebird_dt.simulator.saver import SaveData, Saver
from bluebird_dt.simulator.simconfig import SimConfig
from bluebird_dt.simulator.simconfig import SaveConfig
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
    now_realtime = datetime.now(tz=timezone.utc)
    now_simtime = sim.manager.environment.datetime
    sim.saver.save_config.save_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.saver.save_config.save_realtime = now_realtime - timedelta(minutes=real_elapsed)

    if autosave:
        assert sim._prepare_save(autosave=autosave, end_save=False) is None
    else:
        assert sim._prepare_save(autosave=autosave, end_save=False) is not None
        assert sim.saver.save_config.save_simtime == now_simtime
        assert sim.saver.save_config.save_realtime - now_realtime < timedelta(seconds=5)


@proceed_cases
@autosave_cases
def test_chunk_prepare_savedata_autosave_proceed(sim_elapsed, real_elapsed, autosave):
    """
    Verify _chunk_prepare_savedata autosave proceeds when the interval has elapsed.
    """
    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario", autosave_interval=timedelta(minutes=10))

    # Fake state
    now_realtime = datetime.now(tz=timezone.utc)
    now_simtime = sim.manager.environment.datetime
    sim.saver.save_config.save_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.saver.save_config.save_realtime = now_realtime - timedelta(minutes=real_elapsed)

    assert sim._prepare_save(autosave=autosave, end_save=False) is not None
    assert sim.saver.save_config.save_simtime == now_simtime
    assert sim.saver.save_config.save_realtime - now_realtime < timedelta(seconds=5)


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
    now_realtime = datetime.now(tz=timezone.utc)
    now_simtime = sim.manager.environment.datetime
    sim.saver.save_config.chunk_start_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.saver.save_config.chunk_start_realtime = now_realtime - timedelta(minutes=real_elapsed)
    sim.saver.save_config.save_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.saver.save_config.save_realtime = now_realtime - timedelta(minutes=real_elapsed)
    sim.saver.save_status.last_save_task_success = last_save_task_success
    sim.saver.save_status.last_save_task_save_simtime = now_simtime - timedelta(minutes=sim_elapsed)

    sim._prepare_save(autosave=autosave, end_save=False)

    trim_logger.assert_not_called()
    trim_handler.assert_not_called()
    assert sim.saver.save_config.save_chunk_id == 0


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
    now_realtime = datetime.now(tz=timezone.utc)
    now_simtime = sim.manager.environment.datetime
    sim.saver.save_config.chunk_start_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.saver.save_config.chunk_start_realtime = now_realtime - timedelta(minutes=real_elapsed)
    sim.saver.save_config.save_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.saver.save_config.save_realtime = now_realtime - timedelta(minutes=real_elapsed)
    sim.saver.save_status.last_save_task_success = last_save_task_success
    sim.saver.save_status.last_save_task_save_simtime = now_simtime - timedelta(minutes=sim_elapsed) - timedelta(minutes=10 if last_save_delayed else 0)

    sim._prepare_save(autosave=autosave, end_save=False)

    if last_save_task_success and not last_save_delayed:
        trim_logger.assert_called_once()
        trim_handler.assert_called_once()
        assert sim.saver.save_config.save_chunk_id == 1
    else:
        trim_logger.assert_not_called()
        trim_handler.assert_not_called()
        assert sim.saver.save_config.save_chunk_id == 0


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
    now_realtime = datetime.now(tz=timezone.utc)
    now_simtime = sim.manager.environment.datetime
    sim.saver.save_config.save_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.saver.save_config.save_realtime = now_realtime - timedelta(minutes=real_elapsed)

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
    now_realtime = datetime.now(tz=timezone.utc)
    now_simtime = sim.manager.environment.datetime
    sim.saver.save_config.save_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.saver.save_config.save_realtime = now_realtime - timedelta(minutes=real_elapsed)

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
    now_realtime = datetime.now(tz=timezone.utc)
    now_simtime = sim.manager.environment.datetime
    sim.saver.save_config.save_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.saver.save_config.save_realtime = now_realtime - timedelta(minutes=real_elapsed)

    assert sim.saver.current_save_task is None
    if autosave:
        assert await sim.async_save(autosave=autosave) is False
        assert sim.saver.current_save_task is None
    else:
        assert await sim.async_save(autosave=autosave) is True
        assert sim.saver.current_save_task is not None
    assert sim.saver.next_save_data is None


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
    now_realtime = datetime.now(tz=timezone.utc)
    now_simtime = sim.manager.environment.datetime
    sim.saver.save_config.save_simtime = now_simtime - timedelta(minutes=sim_elapsed)
    sim.saver.save_config.save_realtime = now_realtime - timedelta(minutes=real_elapsed)

    assert sim.saver.current_save_task is None
    assert await sim.async_save(autosave=autosave) is True
    assert sim.saver.current_save_task is not None


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
def test_saver_update(
    current_task_done,
    previous_save_data,
    next_save_data,
    expected_create_task_calls,
    expected_current_task_is_none,
    expected_next_save_data,
):
    """
    Verify Saver.update moves queued autosave work through the current task lifecycle.
    """
    def mock_save_task(done: bool) -> MagicMock:
        task = MagicMock()
        task.done.return_value = done
        return task

    sim = Simulator.from_category(category="Springfield", scenario_name="example-scenario")
    sim.saver.current_save_task = None if current_task_done is None else mock_save_task(current_task_done)
    sim.saver.next_save_data = previous_save_data
    sim.saver.next_save_data = next_save_data

    new_task = MagicMock()
    new_task.done.return_value = False

    with patch("bluebird_dt.simulator.saver.asyncio.create_task", return_value=new_task) as create_task:
        asyncio.run(sim.saver.update())

    assert create_task.call_count == expected_create_task_calls

    if expected_create_task_calls:
        scheduled_coroutine = create_task.call_args.args[0]
        scheduled_save_data = scheduled_coroutine.cr_frame.f_locals["save_data"]
        assert scheduled_save_data is next_save_data
        scheduled_coroutine.close()

    assert sim.saver.next_save_data is expected_next_save_data

    if expected_current_task_is_none:
        assert sim.saver.current_save_task is None
    elif expected_create_task_calls:
        assert sim.saver.current_save_task is new_task
    else:
        assert sim.saver.current_save_task is not None
        assert sim.saver.current_save_task.done() is False


def make_saver(
    *, autosave_interval: timedelta | None = timedelta(minutes=10), save_chunk_interval: timedelta | None = None
) -> Saver:
    now = datetime.now(tz=timezone.utc)
    return Saver(
        SaveConfig(
            save_csv=False,
            autosave_interval=autosave_interval,
            save_chunk_interval=save_chunk_interval,
            load_simtime=now,
            load_realtime=now,
            save_chunk_id=0 if save_chunk_interval is not None else None,
        )
    )


@pytest.mark.parametrize(
    ("autosave", "sim_elapsed", "real_elapsed", "expected"),
    [
        pytest.param(True, 9, 10, False, id="autosave_skips_when_simtime_is_short"),
        pytest.param(True, 10, 9, False, id="autosave_skips_when_realtime_is_short"),
        pytest.param(True, 10, 10, True, id="autosave_proceeds_when_both_intervals_elapsed"),
        pytest.param(False, 0, 0, True, id="manual_save_always_proceeds"),
    ],
)
def test_saver_should_save(autosave, sim_elapsed, real_elapsed, expected):
    saver = make_saver()
    now = datetime.now(tz=timezone.utc)
    saver.save_config.save_simtime = now - timedelta(minutes=sim_elapsed)
    saver.save_config.save_realtime = now - timedelta(minutes=real_elapsed)

    assert saver.should_save(autosave, now, now) is expected


@pytest.mark.parametrize(
    ("last_save_task_success", "last_save_task_save_simtime", "current_task_done", "expected"),
    [
        pytest.param(True, 10, None, True, id="successful_matching_save_allows_chunk"),
        pytest.param(True, 9, None, False, id="delayed_save_blocks_chunk"),
        pytest.param(False, 10, None, False, id="failed_save_blocks_chunk"),
        pytest.param(True, 10, False, False, id="active_save_task_blocks_chunk"),
    ],
)
def test_saver_should_chunk(last_save_task_success, last_save_task_save_simtime, current_task_done, expected):
    saver = make_saver(save_chunk_interval=timedelta(minutes=10))
    now = datetime.now(tz=timezone.utc)
    saver.save_config.chunk_start_simtime = now - timedelta(minutes=10)
    saver.save_config.chunk_start_realtime = now - timedelta(minutes=10)
    saver.save_config.save_simtime = now
    saver.save_status.last_save_task_success = last_save_task_success
    saver.save_status.last_save_task_save_simtime = now - timedelta(minutes=10 - last_save_task_save_simtime)
    if current_task_done is not None:
        saver.current_save_task = MagicMock()
        saver.current_save_task.done.return_value = current_task_done

    assert saver.should_chunk(now, now) is expected


@pytest.mark.asyncio
async def test_saver_force_dispatch_writes_save_data(tmp_path):
    saver = make_saver()
    save_time = datetime.now(tz=timezone.utc)
    saver.save_config.save_simtime = save_time
    savedata = SaveData(b"save contents", "direct-saver-test", saver.save_config.model_copy(deep=True))

    with patch("bluebird_dt.simulator.saver.LOG_DIR", str(tmp_path)):
        assert await saver.dispatch(savedata, force=True) is True

    assert (tmp_path / "direct-saver-test.tar.gz").read_bytes() == b"save contents"
    assert saver.save_status.last_save_task_success is True
    assert saver.save_status.last_save_task_save_simtime == save_time
    await saver.close()


@pytest.mark.asyncio
async def test_saver_async_save_task_records_failure():
    saver = make_saver()
    save_time = datetime.now(tz=timezone.utc)
    savedata = SaveData(b"save contents", "failed-saver-test", saver.save_config.model_copy(deep=True))

    with patch("bluebird_dt.simulator.saver.os.makedirs", side_effect=OSError("disk unavailable")):
        await saver.async_save_task(savedata)

    assert saver.save_status.last_save_task_success is False
    assert saver.save_status.last_save_task_save_simtime == savedata.save_config.save_simtime
