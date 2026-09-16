from bluebird_dt.airspace_generator.artificial_airspace import ArtificialAirspace
from bluebird_dt.airspace_generator.springfield_airspace import SpringfieldAirspaceGenerator
from bluebird_dt.core import Airspace, Route


class AirspaceLoader:
    @classmethod
    def load(
        cls,
        scenario_name: str,
        **airspace_kwargs: int | float | tuple[int, int] | tuple[float, float],
    ) -> tuple[Airspace, list[Route], str]:
        """
        For scenario categories such as "Infinite" where the scenario_name
        defines the airspace, use this function to return the necessary information
        for the scenario manager, which can then remain airspace-agnostic.

        Parameters
        ----------
        scenario_name: str
            The name of the airspace to load (see `list_airspaces`).
        **airspace_kwargs
            Forwarded to `ArtificialAirspace` (e.g. `width`, `height`, `fl_limits`,
            `alpha`, `origin`) for artificial sectors, so callers can override its
            defaults. Ignored for "Springfield", which has no configurable geometry.
        """
        match scenario_name:
            case "I-Sector":
                airspace, routes = ArtificialAirspace("i", **airspace_kwargs).generate_airspace()
                return airspace, routes, "sector_i"
            case "Y-Sector":
                airspace, routes = ArtificialAirspace("y", **airspace_kwargs).generate_airspace()
                return airspace, routes, "sector_y"
            case "X-Sector":
                airspace, routes = ArtificialAirspace("x", **airspace_kwargs).generate_airspace()
                return airspace, routes, "sector_x"
            case "Xplus-Sector":
                airspace, routes = ArtificialAirspace("xplus", **airspace_kwargs).generate_airspace()
                return airspace, routes, "sector_xplus"
            case "Two Sector":
                airspace, routes = ArtificialAirspace("two", **airspace_kwargs).generate_airspace()
                return airspace, routes, "sector_1"
            case "Springfield":
                airspace, routes = SpringfieldAirspaceGenerator().generate_airspace()
                return airspace, routes, "SPRINGFIELD"
            case _:
                raise ValueError(f"Scenario name {scenario_name} not recognized.")

    @staticmethod
    def list_airspaces() -> list[str]:
        return ["I-Sector", "X-Sector", "Xplus-Sector", "Y-Sector", "Two Sector", "Springfield"]
