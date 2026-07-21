import os
import tempfile

# Keep generated scenario logs out of user data directories during tests.
# This must run before bluebird_dt.utility.paths is first imported because
# that module resolves LOG_DIR once at import time.
if "BLUEBIRD_LOG_DIR" not in os.environ:
    os.environ["BLUEBIRD_LOG_DIR"] = tempfile.mkdtemp(prefix="bluebird_test_logs_")
