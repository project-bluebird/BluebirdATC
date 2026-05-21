from bluebird_dt.airspace_generator.artificial_airspace import ArtificialAirspace
from bluebird_dt.airspace_generator.springfield_airspace import SpringfieldAirspace
from bluebird_dt.core import Airspace, Route


class AirspaceLoader:
    @classmethod
    def load(cls, scenario_name: str) -> tuple[Airspace, list[Route], str]:
        """
        For scenario categories such as "Infinite" where the scenario_name
        defines the airspace, use this function to return the necessary information
        for the scenario manager, which can then remain airspace-agnostic.
        """
        match scenario_name:
            case "I-Sector":
                airspace, routes = ArtificialAirspace("i").generate_airspace()
                return airspace, routes, "sector_i"
            case "Y-Sector":
                airspace, routes = ArtificialAirspace("y").generate_airspace()
                return airspace, routes, "sector_y"
            case "X-Sector":
                airspace, routes = ArtificialAirspace("x").generate_airspace()
                return airspace, routes, "sector_x"
            case "Xplus-Sector":
                airspace, routes = ArtificialAirspace("xplus").generate_airspace()
                return airspace, routes, "sector_xplus"
            case "Two Sector":
                airspace, routes = ArtificialAirspace("two").generate_airspace()
                return airspace, routes, "sector_1"
            case "Springfield":
                airspace, routes = SpringfieldAirspace().generate_airspace()
                return airspace, routes, "SPRINGFIELD"
            case _:
                raise ValueError(f"Scenario name {scenario_name} not recognized.")
