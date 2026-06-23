import os

from bluebird_dt.airspace_generator.airspace_loader import AirspaceLoader
from bluebird_dt.scenario_manager.springfield import SpringfieldScenarioManager
from bluebird_dt.utility.paths import LOG_DIR


def list_sim_scenario_categories() -> list[str]:
    """
    List the available scenario categories.
    """

    return ["Two Aircraft", "Regular", "Custom", "Infinite", "Springfield", "Flight School", "Replay"]


def list_sim_scenarios(category: str) -> list[str]:
    """
    List the scenarios in a given category.
    """

    # make directory for replay files if it doesn't exist already
    os.makedirs(LOG_DIR, exist_ok=True)

    if category == "Springfield":
        return SpringfieldScenarioManager.list_scenarios()

    if category in ["Two Aircraft", "Regular", "Custom", "Infinite"]:
        # These scenario categories can use any of the artificial airspaces.
        return AirspaceLoader.list_airspaces()

    if category == "Flight School":
        return ["Xplus-Sector"]

    if category == "Replay":
        return sorted(
            [
                file.removesuffix(".tar.gz")
                for file in os.listdir(LOG_DIR)
                if file.endswith(".tar.gz") and not file.startswith(".")
            ]
        )

    raise ValueError(f"Unknown scenario category: {category}")
