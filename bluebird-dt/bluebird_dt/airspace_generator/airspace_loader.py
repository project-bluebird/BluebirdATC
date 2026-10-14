import inspect
from collections.abc import Mapping
from typing import Any

from bluebird_dt.airspace_generator.artificial_airspace import ArtificialAirspace
from bluebird_dt.airspace_generator.sector_xplus import SectorXPlus
from bluebird_dt.airspace_generator.springfield_airspace import SpringfieldAirspaceGenerator
from bluebird_dt.core import Airspace, Route
from bluebird_dt.utility.geo_helper import GeoHelper


def _filtered_config(target: type, config: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the keys in `config` that the target's constructor accepts."""

    valid_keys = set(inspect.signature(target.__init__).parameters)
    return {key: value for key, value in config.items() if key in valid_keys}


class AirspaceLoader:
    @classmethod
    def load(
        cls,
        scenario_name: str,
        airspace_config: Mapping[str, Any] | None = None,
    ) -> tuple[Airspace, list[Route], str]:
        """
        For scenario categories such as "Infinite" where the scenario_name
        defines the airspace, use this function to return the necessary information
        for the scenario manager, which can then remain airspace-agnostic.

        Parameters
        ----------
        scenario_name: str
            The name of the airspace to load (see `list_airspaces`).
        airspace_config: Mapping, optional
            Airspace configuration (e.g. a gymnasium env's `airspace_config`).
            Only the keys relevant to the target sector generator's constructor are
            used; anything else (e.g. `exit_window_width`) is ignored. `origin`, if
            present, is read as `(lat, lon)` and translated to the `(lon, lat)` order
            the generators expect.
        """
        config = dict(airspace_config or {})
        origin = config.get("origin")  # (lat, lon), as callers store it
        if origin is not None:
            config["origin"] = (origin[1], origin[0])  # generators expect (lon, lat)

        match scenario_name:
            case "I-Sector":
                filtered_kwargs = _filtered_config(ArtificialAirspace, config)
                airspace, routes = ArtificialAirspace("i", **filtered_kwargs).generate_airspace()
                sector_name = "sector_i"
            case "Y-Sector":
                filtered_kwargs = _filtered_config(ArtificialAirspace, config)
                airspace, routes = ArtificialAirspace("y", **filtered_kwargs).generate_airspace()
                sector_name = "sector_y"
            case "X-Sector":
                filtered_kwargs = _filtered_config(ArtificialAirspace, config)
                airspace, routes = ArtificialAirspace("x", **filtered_kwargs).generate_airspace()
                sector_name = "sector_x"
            case "Xplus-Sector":
                filtered_kwargs = _filtered_config(SectorXPlus, config)
                airspace, routes = SectorXPlus(**filtered_kwargs).generate_airspace()
                sector_name = "sector_xplus"
            case "Two Sector":
                filtered_kwargs = _filtered_config(ArtificialAirspace, config)
                airspace, routes = ArtificialAirspace("two", **filtered_kwargs).generate_airspace()
                sector_name = "sector_1"
            case "Springfield":
                airspace, routes = SpringfieldAirspaceGenerator().generate_airspace()
                return airspace, routes, "SPRINGFIELD"
            case _:
                raise ValueError(f"Scenario name {scenario_name} not recognized.")

        if origin is not None:
            # the generators attach their own `geo_helper` using their own (lon, lat)
            # -first convention; re-anchor it to the (lat, lon) origin callers use, so
            # `airspace.geo_helper` agrees with the `airspace_config` they passed in.
            airspace.geo_helper = GeoHelper(origin)

        return airspace, routes, sector_name

    @staticmethod
    def list_airspaces() -> list[str]:
        return ["I-Sector", "X-Sector", "Xplus-Sector", "Y-Sector", "Two Sector", "Springfield"]
