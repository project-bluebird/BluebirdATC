import importlib.resources
import os

import platformdirs

ROOT_DIR = importlib.resources.files("bluebird_dt")
# We have the data in the bluebird_dt package itself.
BASE_DATA_DIR = ROOT_DIR.joinpath("scenario_data")
SPRINGFIELD_DIR: str = os.path.join(BASE_DATA_DIR, "Springfield")

# Environment variable used to override where scenario logs / replay files are written.
LOG_DIR_ENV_VAR = "BLUEBIRD_LOG_DIR"

# The application name used to derive the per-user data directory via platformdirs.
_APP_NAME = "bluebird"


def get_log_dir(app_subdir: str = "") -> str:
    """
    Resolve the directory used for scenario logs and replay files.

    By default it ensures that logs do not live inside the installed package
    (e.g. the virtual environment / site-packages). All applications share a
    single base location (overridable via ``BLUEBIRD_LOG_DIR``); pass
    ``app_subdir`` to give each application its own subdirectory underneath it
    so their logs are co-located but distinguishable (e.g. ``"bluebird_dt"``,
    ``"starling"``).

    Parameters
    ----------
    app_subdir
        Optional per-application subdirectory placed under the shared base.

    Returns
    -------
    str
        The absolute path to the directory in which scenario logs should be written/read.
    """
    base = os.environ.get(LOG_DIR_ENV_VAR) or os.path.join(
        platformdirs.user_data_dir(_APP_NAME), "scenario_logs"
    )
    return os.path.join(base, app_subdir) if app_subdir else base


# Resolved once at import for backwards compatibility with existing ``LOG_DIR`` imports.
LOG_DIR: str = get_log_dir("bluebird_dt")

AIRCRAFT_DATA_DIR = ROOT_DIR.joinpath("aircraft_data")
AIRCRAFT_WEIGHT_MAPPING_FILE: str = os.path.join(AIRCRAFT_DATA_DIR, "aircraft_weight_map.json")
SIMPLE_PERFORMANCE_PROFILE_FILE = os.path.join(AIRCRAFT_DATA_DIR, "simple_performance_profile_data.json")
SIMPLE_PERFORMANCE_UNCERTAINTY_FILE = os.path.join(AIRCRAFT_DATA_DIR, "simple_performance_uncertainty_data.json")
