from __future__ import annotations

from datetime import timedelta
from typing import Literal

import pandas as pd
from bluebird_dt.core import Aircraft, Coordination, FlightPlan, Route
from bluebird_dt.events.event_handler import EventHandler
from bluebird_dt.scenario_manager.tactical import Tactical
from pydantic import BaseModel, Field
from typing_extensions import override

from bluebird_gymnasium.envs import SCENARIO_CLS


class FixedSequenceScenarioManagerConfig(BaseModel):
    """Configuration for the fixed-sequence curriculum scenario manager."""

    scenario_manager: Literal["fixed_sequence"] = Field(default="fixed_sequence")


class FixedSequenceTactical(Tactical):
    """Tactical scenario manager with explicit per-aircraft route/time/speed specs.

    This is intended for curriculum stages where the notebook needs deterministic
    traffic geometry instead of the stock random route sampling used by Tactical.
    """

    def __init__(
        self,
        airspace: object,
        routes: list[Route],
        aircraft_specs: list[dict],
        start_time: int = 0,
        vertical_buffer_distance: float | int = 500,
        lateral_buffer_distance: float | int = 20,
        initialise_with_event_handler: bool = True,
    ) -> None:
        super().__init__(
            num_aircraft=len(aircraft_specs),
            airspace=airspace,
            routes=routes,
            balance=[0.0, 0.0, 1.0],
            speed_range=[400.0, 400.0],
            time_entry_gap=0.0,
            lateral_offset=None,
            env_manager_class=None,
            start_time=start_time,
            vertical_buffer_distance=vertical_buffer_distance,
            lateral_buffer_distance=lateral_buffer_distance,
            initialise_with_event_handler=initialise_with_event_handler,
        )
        self.aircraft_specs = aircraft_specs

    def _resolve_route(self, route_spec: dict) -> Route:
        if "route_filed" in route_spec:
            return Route(route_spec["route_filed"])

        route_index = route_spec.get("route_index")
        if route_index is None:
            raise ValueError("Each aircraft spec must define either route_filed or route_index.")

        base_route = self.routes[route_index]
        if route_spec.get("reverse", False):
            return Route(base_route.filed[::-1])

        return base_route

    @override
    def create_event_handler(self) -> EventHandler:
        sector_name = next(iter(self.airspace.sectors.keys()))
        event_handler = EventHandler(ignore=self.event_handler_ignore_flags)

        for index, spec in enumerate(self.aircraft_specs):
            route = self._resolve_route(spec)
            callsign = spec.get("callsign", f"AIR{index}")
            entry_fl = float(spec["entry_fl"])
            exit_fl = float(spec.get("exit_fl", entry_fl))
            speed_tas = float(spec["speed_tas"])
            start_time_seconds = float(spec.get("start_time_seconds", self.start_time))

            start_fix = self.airspace.fixes.places[route.filed[0]]
            next_fix = self.airspace.fixes.places[route.filed[1]]
            heading = start_fix.bearing_to(next_fix)

            flight_plan = FlightPlan(route)
            pos = start_fix.pos3d(entry_fl)

            aircraft = Aircraft(
                pos.lat,
                pos.lon,
                pos.fl,
                heading,
                flight_plan,
                callsign,
                selected_fl=pos.fl,
                current_sector=None,
            )
            aircraft.speed_tas = speed_tas
            aircraft.simulated = True
            aircraft.selected_instructions.cas = speed_tas

            coordination_entry = Coordination(
                callsign=callsign,
                from_sector="background",
                to_sector=sector_name,
                fl=entry_fl,
                fix=route.filed[0],
                direction="Horizontal",
            )
            coordination_exit = Coordination(
                callsign=callsign,
                from_sector=sector_name,
                to_sector="background",
                fl=exit_fl,
                fix=route.filed[-1],
                direction="Horizontal",
            )

            event_start_time = pd.to_datetime(start_time_seconds, unit="s")
            event_handler.add_coordination(
                event_start_time - timedelta(seconds=1),
                coordination_exit,
            )
            event_handler.add_coordination(
                event_start_time - timedelta(seconds=1),
                coordination_entry,
            )
            event_handler.add_aircraft(event_start_time, aircraft)

        return event_handler

    @override
    def config(self) -> FixedSequenceScenarioManagerConfig:
        return FixedSequenceScenarioManagerConfig()


def register_curriculum_scenario_managers() -> None:
    """Register notebook-specific scenario managers with the gymnasium env lookup."""

    SCENARIO_CLS["fixed_sequence"] = FixedSequenceTactical
