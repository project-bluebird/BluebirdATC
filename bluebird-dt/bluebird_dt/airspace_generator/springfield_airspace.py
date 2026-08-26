import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from bluebird_dt.airspace_generator.airspace_generator import AirspaceGenerator
from bluebird_dt.core import Airspace, Airway, Fixes, Route, Sector
from bluebird_dt.utility.airspace_data import create_sector, load_fixes
from bluebird_dt.utility.paths import SPRINGFIELD_DIR


@dataclass(init=True)
class SpringfieldAirspaceGenerator(AirspaceGenerator):
    """
    Springfield airspace generator.
    """

    fixes_path: str = os.path.join(SPRINGFIELD_DIR, "fixes.csv")
    sector_path: str = os.path.join(SPRINGFIELD_DIR, "sectors")
    airways_path: str = os.path.join(SPRINGFIELD_DIR, "airways.json")
    routes_path: str = os.path.join(SPRINGFIELD_DIR, "routes_and_exits.json")

    def __init__(
        self,
    ):
        """
        Construct a new instance.

        Parameters
        ----------

        """

    def generate_airspace(self) -> tuple[Airspace, list[Route]]:
        """
        Generate an Airspace.

        Returns
        ----------
        tuple[Airspace, list[Route]]
            A tuple containing the new airspace object and its corresponding routes.
        """
        # Generate airspace and set up geo_helper
        airspace = self._airspace_init()
        routes = self._routes_init()

        return airspace, routes

    @staticmethod
    def _sectors_init() -> dict[str, Sector]:
        """
        Loads sectors for Springfield from all geojson files in
        :attr:`~springfield.SpringfieldScenarioManager.sector_path`,
        ignoring all hidden files (prefixed with a '.').

        Returns
        -------
        dict[str, Sector]
            A dictionary with the sector names as the keys and its sector object as the corresponding value.

        """

        sector_paths = [
            os.path.join(SpringfieldAirspaceGenerator.sector_path, f)
            for f in os.listdir(SpringfieldAirspaceGenerator.sector_path)
            if os.path.isfile(os.path.join(SpringfieldAirspaceGenerator.sector_path, f))
        ]

        sectors: dict[str, Sector] = {}
        sector_aor: dict[str, Sector] = {}

        for sector_path in sector_paths:
            sector_name = Path(sector_path).stem

            # Ignore hidden files
            if sector_name[0] == ".":
                continue

            sector = create_sector(sector_path)

            if (sector_name_regex := re.search(r"^([A-Z]+)_aor", sector_name)) is not None:
                sector_aor[sector_name_regex.group(1)] = sector
            else:
                sectors[sector_name] = sector

        for key, value in sector_aor.items():
            if (parent_sector := sectors.get(key)) is not None:
                parent_sector.area_of_responsibility = value.volumes
            else:
                raise KeyError(f"Sector with {key}_aor does not exists therefore {key} cannot be matched to it.")

        return sectors

    @staticmethod
    def _fixes_init() -> Fixes:
        """
        Loads fixes for Springfield from :attr:~'springfield.SpringfieldScenarioManager.fixes_path'.

        Returns
        -------
        Fixes
            An object which stores all the loaded fixes.
        """
        return load_fixes(fixes_path=SpringfieldAirspaceGenerator.fixes_path)

    @staticmethod
    def _airways_init(fixes: Fixes) -> dict[str, Airway]:
        """
        Loads airways for Springfield from :attr:`~springfield.SpringfieldScenarioManager.airways_path`.

        Returns
        -------
        dict[str, Airway]
            Returns a dictionary of airways where the key is its identifier as a string,
            and the value the corresponding Airway object.
        """

        with open(SpringfieldAirspaceGenerator.airways_path) as f:
            airway_data = json.load(f)

        airways: dict[str, Airway] = {}

        for key, val in airway_data.items():
            airways[key] = Airway.from_list_of_fixes(key, val["fixes"], 0, 660, fixes)

        return airways

    @staticmethod
    def _airspace_init() -> Airspace:
        """
        Function to initialise an Airspace object for Springfield

        Returns
        -------
        airspace: Airspace, the Springfield airspace, with some sectors bandboxed.
        """
        fixes = SpringfieldAirspaceGenerator._fixes_init()

        return Airspace(
            sectors=SpringfieldAirspaceGenerator._sectors_init(),
            fixes=fixes,
            airways=SpringfieldAirspaceGenerator._airways_init(fixes),
        )

    @staticmethod
    def _routes_init() -> list[Route]:
        """
        Loads routes for Springfield from a json file in
        :attr:`~springfield.SpringfieldScenarioManager.routes_path`.

        Returns
        -------
        list[Route]
            A list of possible routes within the Springfield airspace.
        """
        with open(SpringfieldAirspaceGenerator.routes_path) as f:
            routes_data = json.load(f)

        return [Route(r["route"]) for r in routes_data["mats_routes"]]
