from collections import defaultdict
from dataclasses import dataclass, field

import typing_extensions
from shapely.geometry import LineString

from bluebird_dt.core import Action, Aircraft, Coordination, Fixes, Sector
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
    require_exit_fl:
        Whether the aircraft must be at the coordinated exit flight level.
    """

    require_exit_fl: bool = True

    _previously_aircraft_in_sector: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    _previous_positions: dict[str, Pos2D] = field(default_factory=dict)

    def _should_outcomm_aircraft(
        self,
        aircraft: Aircraft,
        previous_position: Pos2D,
        current_sector: Sector,
        current_sector_name: str,
        coordination: Coordination | None,
        fixes: Fixes,
    ) -> bool:

        callsign = aircraft.callsign
        current_position = aircraft.pos2d()

        if aircraft.current_sector == "background" or callsign is None:
            return False

        if current_sector.contains_laterally(current_position):
            self._previously_aircraft_in_sector[current_sector_name].add(callsign)
            return False

        if coordination is not None:
            return self._should_outcomm_aircraft_based_on_coordination(
                aircraft, previous_position, current_sector, coordination, fixes
            )

        return callsign in self._previously_aircraft_in_sector[current_sector_name]

    def _should_outcomm_aircraft_based_on_coordination(
        self,
        aircraft: Aircraft,
        previous_position: Pos2D,
        current_sector: Sector,
        coordination: Coordination,
        fixes: Fixes,
    ) -> bool:

        callsign = aircraft.callsign
        current_position = aircraft.pos2d()
        exit_fix = fixes.get(coordination.fix)

        if exit_fix is None:
            logger.warning(
                f"Cannot check the exit gate for aircraft {callsign}: coordination fix {coordination.fix} was not found"
            )
            return False

        if self.require_exit_fl and aircraft.fl != coordination.fl:
            return False

        gate_segment = current_sector.volumes[0].area.nearest_segment_to_point(exit_fix)

        if gate_segment is None:
            logger.warning(
                f"Cannot check the exit gate for aircraft because there was no gate segment"
                f"{callsign}: coordination fix "
                f"{coordination.fix} was not found"
            )
            return False

        gate = ExitGate.new(
            start_position=Pos2D(lon=gate_segment[0][0], lat=gate_segment[0][1]),
            end_position=Pos2D(lon=gate_segment[1][0], lat=gate_segment[1][1]),
        )

        return gate.is_crossed(
            previous_position=previous_position,
            current_position=current_position,
        )

    def update(self, env_manager: EnvironmentManager) -> None:
        """
        Update internal aircraft positions and outcomm aircraft crossing
        their coordinated sector-boundary gate.
        """
        environment = env_manager.environment

        for aircraft in environment.aircraft.values():
            if aircraft.current_sector == "background" or aircraft.callsign is None:
                continue

            current_sector_name = aircraft.current_sector
            current_sector = environment.airspace.sectors.get(current_sector_name)

            previous_position = self._previous_positions.get(aircraft.callsign)
            self._previous_positions[aircraft.callsign] = aircraft.pos2d()

            if current_sector is None or previous_position is None:
                continue

            coordination = environment.exit_coordination(
                current_sector_name,
                aircraft.callsign,
            )

            if self._should_outcomm_aircraft(
                aircraft,
                previous_position,
                current_sector,
                current_sector_name,
                coordination,
                env_manager.environment.airspace.fixes,
            ):
                self._outcomm(env_manager, current_sector_name, aircraft.callsign)

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
        if sector_name in self._previously_aircraft_in_sector:
            self._previously_aircraft_in_sector[sector_name].discard(callsign)

        self._previous_positions.pop(callsign, None)
