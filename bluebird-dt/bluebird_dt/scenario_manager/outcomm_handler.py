from collections import defaultdict
from dataclasses import dataclass, field

import typing_extensions
from shapely.geometry import LineString

from bluebird_dt.core import Action, Coordination, Fixes, FlightPlan
from bluebird_dt.core.pos2d import Pos2D
from bluebird_dt.logger import logger
from bluebird_dt.manager import EnvironmentManager


@dataclass(frozen=True, slots=True)
class ExitGate:
    """A sector-boundary gate associated with the coordinated exit fix."""

    start_position: Pos2D
    end_position: Pos2D

    @classmethod
    def new(
        cls,
        start_position: Pos2D,
        end_position: Pos2D,
    ) -> typing_extensions.Self:
        return cls(
            start_position=start_position,
            end_position=end_position,
        )

    def flip(self) -> typing_extensions.Self:
        """
        Return a copy of the gate with its direction reversed.

        Returns
        -------
        Self
            A new gate with the start and end positions reversed.
        """
        return type(self)(
            start_position=self.end_position,
            end_position=self.start_position,
        )

    def orient_exit_gate(
        self,
        flight_plan: FlightPlan,
        coordination: Coordination,
        fixes: Fixes,
    ) -> typing_extensions.Self:
        """Orient an exit gate according to the aircraft's planned route."""
        legs = next(
            (segment for segment in flight_plan.route.segments() if segment[1] == coordination.fix),
            None,
        )

        if legs is None:
            return self

        approach_fix = fixes.get(legs[0])

        if approach_fix is None:
            return self

        return self.oriented_for_approach(approach_fix)

    def oriented_for_approach(
        self,
        approach_position: Pos2D,
    ) -> typing_extensions.Self:
        """Return the gate oriented for an aircraft approaching from a position."""
        approach_side = (self.end_position.lon - self.start_position.lon) * (
            approach_position.lat - self.start_position.lat
        ) - (self.end_position.lat - self.start_position.lat) * (approach_position.lon - self.start_position.lon)

        if approach_side == 0.0:
            raise ValueError(
                "Cannot orient the exit gate because the route approach position is collinear with the gate"
            )

        return self if approach_side > 0.0 else self.flip()

    def is_crossed(
        self,
        previous_position: Pos2D,
        current_position: Pos2D,
    ) -> bool:
        """
        Return whether the aircraft crosses the gate in the exit direction.

        This will intentionally treat lats and longs as a cartesian coordinate system,
        to align with the implementation of Sector.contains_laterally
        """
        start = (self.start_position.lon, self.start_position.lat)
        end = (self.end_position.lon, self.end_position.lat)
        previous = (previous_position.lon, previous_position.lat)
        current = (current_position.lon, current_position.lat)

        gate = LineString([start, end])
        movement = LineString([previous, current])

        if not movement.crosses(gate):
            return False

        previous_side = (end[0] - start[0]) * (previous[1] - start[1]) - (end[1] - start[1]) * (previous[0] - start[0])

        return previous_side > 0.0


@dataclass(slots=True)
class OutcommHandler:
    """
    Outcomm aircraft that cross the sector-boundary gate nearest to their
    coordinated exit fix.

    Parameters
    ----------
    aircraft_in_sector:
        Used to keep track of aircraft observed inside each sector.
    require_exit_fl:
        Whether the aircraft must be at the coordinated exit flight level.
    """

    previously_aircraft_in_sector: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    require_exit_fl: bool = True

    _previous_positions: dict[str, Pos2D] = field(default_factory=dict)

    def update(self, env_manager: EnvironmentManager) -> None:
        """
        Update internal aircraft positions and outcomm aircraft crossing
        their coordinated sector-boundary gate.
        """
        environment = env_manager.environment

        for aircraft in environment.aircraft.values():
            if aircraft.current_sector == "background" or aircraft.callsign is None:
                continue

            callsign = aircraft.callsign
            current_position = aircraft.pos2d()
            current_sector_name = aircraft.current_sector

            previous_position = self._previous_positions.get(callsign)
            self._previous_positions[callsign] = current_position

            current_sector = environment.airspace.sectors[current_sector_name]

            if current_sector.contains_laterally(current_position):
                self.previously_aircraft_in_sector[current_sector_name].add(callsign)
                continue

            coordination = environment.exit_coordination(
                current_sector_name,
                callsign,
            )

            if coordination is None:
                if callsign in self.previously_aircraft_in_sector[current_sector_name]:
                    self._outcomm(env_manager, current_sector_name, callsign)

                continue

            exit_fix = environment.airspace.fixes.get(coordination.fix)

            if exit_fix is None:
                logger.warning(
                    f"Cannot check the exit gate for aircraft "
                    f"{callsign}: coordination fix "
                    f"{coordination.fix} was not found"
                )
                continue

            if previous_position is None:
                continue

            at_exit_fl = aircraft.fl == coordination.fl or not self.require_exit_fl

            if not at_exit_fl:
                continue

            gate_segment = current_sector.volumes[0].area.nearest_segment_to_point(exit_fix)

            if gate_segment is None:
                logger.warning(
                    f"Cannot check the exit gate for aircraft because there was no gate segment"
                    f"{callsign}: coordination fix "
                    f"{coordination.fix} was not found"
                )
                continue

            gate = ExitGate.new(
                start_position=Pos2D(lon=gate_segment[0][0], lat=gate_segment[0][1]),
                end_position=Pos2D(lon=gate_segment[1][0], lat=gate_segment[1][1]),
            )

            if aircraft.flight_plan is not None:
                gate = gate.orient_exit_gate(
                    flight_plan=aircraft.flight_plan,
                    coordination=coordination,
                    fixes=environment.airspace.fixes,
                )

            crossed_gate = gate.is_crossed(
                previous_position=previous_position,
                current_position=current_position,
            )

            if crossed_gate and at_exit_fl:
                self._outcomm(env_manager=env_manager, sector_name=current_sector_name, callsign=callsign)

    def _outcomm(
        self,
        env_manager: EnvironmentManager,
        sector_name: str,
        callsign: str,
    ) -> None:
        self._forget_aircraft(
            sector_name=sector_name,
            callsign=callsign,
        )

        env_manager.receive_actions(
            [
                Action(
                    callsign,
                    "outcomm",
                    "background",
                )
            ]
        )

    def _forget_aircraft(
        self,
        sector_name: str,
        callsign: str,
    ) -> None:
        if sector_name in self.previously_aircraft_in_sector:
            self.previously_aircraft_in_sector[sector_name].discard(callsign)

        self._previous_positions.pop(callsign, None)
