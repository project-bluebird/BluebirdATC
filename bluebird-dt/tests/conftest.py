import logging
import os
import shutil
from tempfile import mkdtemp

import pytest

_log_dir = None

# Keep generated scenario logs out of user data directories during tests.
# This must run before bluebird_dt.utility.paths is first imported because
# that module resolves LOG_DIR once at import time.
def pytest_sessionstart(session):
    global _log_dir

    if "BLUEBIRD_LOG_DIR" not in os.environ:
        _log_dir = mkdtemp(prefix="bluebird_test_logs_")
        os.environ["BLUEBIRD_LOG_DIR"] = _log_dir


# Clean up
def pytest_sessionfinish(session, exitstatus):
    if _log_dir and os.path.isdir(_log_dir):
        shutil.rmtree(_log_dir, ignore_errors=True)

# Each Simulator attaches a FileHandler (and a ContextFilter) to the bluebird_dt logger,
# which are only released by Simulator.close(). Most tests don't call close(), so without
# this the open log files accumulate and exhaust the file descriptor limit (256 on macOS).
@pytest.fixture(autouse=True)
def _release_simulator_log_handlers():
    bluebird_logger = logging.getLogger("bluebird_dt.logger")
    handlers_before = list(bluebird_logger.handlers)
    filters_before = list(bluebird_logger.filters)

    yield

    for handler in [h for h in bluebird_logger.handlers if h not in handlers_before]:
        bluebird_logger.removeHandler(handler)
        handler.close()
    for log_filter in [f for f in bluebird_logger.filters if f not in filters_before]:
        bluebird_logger.removeFilter(log_filter)
