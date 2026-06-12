import os
import tempfile

# Scenario logs are, by default, written to a per-user OS data directory.
# During tests we redirect them to a throwaway temp directory
# This must happen before bluebird_dt.utility.paths is first imported, because that module
# resolves LOG_DIR once at import time. 
os.environ.setdefault("BLUEBIRD_LOG_DIR", tempfile.mkdtemp(prefix="bluebird_test_logs_"))
