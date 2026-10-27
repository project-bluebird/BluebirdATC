from collections import defaultdict
from dataclasses import dataclass, field

from bluebird_dt.core import Action
from bluebird_dt.core.pos2d import Pos2D
from bluebird_dt.logger import logger
from bluebird_dt.manager import EnvironmentManager


@dataclass(frozen=True, slots=True)
class ExitGate:
    """A sector-boundary gate associated with the coordinated exit fix."""

    start_position: Pos2D
    end_position: Pos2D

    @staticmethod
    def _orientation(
        a: Pos2D,
        b: Pos2D,
        c: Pos2D,
    ) -> float:
        return (b.lon - a.lon) * (c.lat - a.lat) - (b.lat - a.lat) * (c.lon - a.lon)

    def is_crossed(
        self,
        previous_position: Pos2D,
        current_position: Pos2D,
    ) -> bool:
        """Return whether movement between the positions crosses the gate."""

        o1 = self._orientation(
            self.start_position,
            self.end_position,
            previous_position,
        )
        o2 = self._orientation(
            self.start_position,
            self.end_position,
            current_position,
        )
        o3 = self._orientation(
            previous_position,
            current_position,
            self.start_position,
        )
        o4 = self._orientation(
            previous_position,
            current_position,
            self.end_position,
        )

        return (o1 * o2 < 0) and (o3 * o4 < 0)


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

    aircraft_in_sector: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
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
            sector_name = aircraft.current_sector
            current_position = aircraft.pos2d()
            previous_position = self._previous_positions.get(callsign)

            self._previous_positions[callsign] = current_position

            current_sector = environment.airspace.sectors[sector_name]
            aircraft_in_sector = current_sector.contains_laterally(current_position)

            if aircraft_in_sector:
                self.aircraft_in_sector[sector_name].add(callsign)

            coordination = environment.exit_coordination(
                sector_name,
                callsign,
            )

            if coordination is not None:
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

                gate_segment = current_sector.volumes[0].area.nearest_segment_to_point(exit_fix)

                if gate_segment is None:
                    logger.warning(
                        f"Cannot check the exit gate for aircraft because there was no gate segment"
                        f"{callsign}: coordination fix "
                        f"{coordination.fix} was not found"
                    )
                    continue

                gate = ExitGate(
                    start_position=Pos2D(lon=gate_segment[0][0], lat=gate_segment[0][1]),
                    end_position=Pos2D(lon=gate_segment[1][0], lat=gate_segment[1][1]),
                )

                crossed_gate = gate.is_crossed(
                    previous_position=previous_position,
                    current_position=current_position,
                )

                at_exit_fl = aircraft.fl == coordination.fl or not self.require_exit_fl

                if crossed_gate and at_exit_fl:
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

                continue

            if not aircraft_in_sector and callsign in self.aircraft_in_sector[sector_name]:
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
        if sector_name in self.aircraft_in_sector:
            self.aircraft_in_sector[sector_name].discard(callsign)

        self._previous_positions.pop(callsign, None)
