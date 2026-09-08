import os
import shutil
from tempfile import mkdtemp

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