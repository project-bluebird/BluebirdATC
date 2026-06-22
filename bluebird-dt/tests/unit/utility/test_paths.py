import os

import platformdirs

from bluebird_dt.utility.paths import LOG_DIR_ENV_VAR, get_log_dir


def test_get_log_dir_uses_env_override(tmp_path, monkeypatch):
    """
    When BLUEBIRD_LOG_DIR is set, it is used as the log directory. 
    conftest.py has set this up for use prior to the path import
    """
    override = str(tmp_path / "my_logs")
    monkeypatch.setenv(LOG_DIR_ENV_VAR, override)

    assert get_log_dir() == override


def test_get_log_dir_defaults_to_user_data_dir(monkeypatch):
    """
    Test that without an override, the logs resolve to the per-user OS data directory.
    """
    monkeypatch.delenv(LOG_DIR_ENV_VAR, raising=False)

    expected = os.path.join(platformdirs.user_data_dir("bluebird"), "scenario_logs")
    assert get_log_dir() == expected


def test_get_log_dir_default_is_not_inside_installed_package(monkeypatch):
    """
    Regression test: the default log location must not live inside the
    installed bluebird_dt package (i.e. the virtual environment / site-packages), otherwise
    logs are hard to find and get wiped when the environment is rebuilt.
    """
    monkeypatch.delenv(LOG_DIR_ENV_VAR, raising=False)
    import bluebird_dt

    package_dir = os.path.dirname(os.path.abspath(bluebird_dt.__file__))
    assert not get_log_dir().startswith(package_dir)


def test_empty_env_override_falls_back_to_default(monkeypatch):
    """ Test that an empty BLUEBIRD_LOG_DIR is ignored in favour of the default location."""
    monkeypatch.setenv(LOG_DIR_ENV_VAR, "")

    expected = os.path.join(platformdirs.user_data_dir("bluebird"), "scenario_logs")
    assert get_log_dir() == expected


def test_get_log_dir_app_subdir_is_nested_under_base(monkeypatch):
    """
    An app_subdir is placed underneath the shared base so that each application's
    logs are co-located (same base) but distinguishable (own subdirectory).
    """
    monkeypatch.delenv(LOG_DIR_ENV_VAR, raising=False)

    base = os.path.join(platformdirs.user_data_dir("bluebird"), "scenario_logs")
    assert get_log_dir("starling") == os.path.join(base, "starling")
    assert get_log_dir("bluebird_dt") == os.path.join(base, "bluebird_dt")


def test_get_log_dir_app_subdir_respects_env_override(tmp_path, monkeypatch):
    """The app_subdir is nested under BLUEBIRD_LOG_DIR when the override is set."""
    override = str(tmp_path / "shared_logs")
    monkeypatch.setenv(LOG_DIR_ENV_VAR, override)

    assert get_log_dir("starling") == os.path.join(override, "starling")
