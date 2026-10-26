from dataclasses import dataclass, field

import typing_extensions

from bluebird_dt.core import Action, Coordination, Fixes, Route
from bluebird_dt.core.pos2d import Pos2D
from bluebird_dt.logger import logger
from bluebird_dt.manager import EnvironmentManager


@dataclass(frozen=True, slots=True)
class ExitGate:
    """A gate perpendicular to the planned route at the coordinated exit fix."""

    start_position: Pos2D
    end_position: Pos2D

    @staticmethod
    def _orientation(
        a: Pos2D,
        b: Pos2D,
        c: Pos2D,
    ) -> float:
        """
        Return the orientation of the ordered points ``a``, ``b`` and ``c``.

        A positive or negative result indicates which side of line ``a-b``
        point ``c`` lies on. Zero indicates collinearity.
        """
        return (b.lon - a.lon) * (c.lat - a.lat) - (b.lat - a.lat) * (c.lon - a.lon)

    @classmethod
    def new(
        cls,
        route: Route,
        coordination: Coordination,
        fixes: Fixes,
        gate_width_nm: float,
    ) -> typing_extensions.Self | None:
        """
        Construct the coordinated exit gate from the planned route.

        The gate is centred on the coordination fix and perpendicular to the
        first route segment that ends at that fix.
        """
        coordination_fix = fixes.get(coordination.fix)

        if coordination_fix is None:
            logger.warning(
                f"Cannot construct exit gate for aircraft "
                f"{coordination.callsign}: coordination fix "
                f"{coordination.fix} was not found"
            )
            return None

        # Use the first planned-route segment ending at the coordination fix.
        route_segment = next(
            (segment for segment in route.segments() if segment[1] == coordination.fix),
            None,
        )

        if route_segment is None:
            logger.warning(
                f"Cannot construct exit gate for aircraft "
                f"{coordination.callsign}: coordination fix "
                f"{coordination.fix} is not in the planned route"
            )
            return None

        previous_fix_designator, exit_fix_designator = route_segment

        previous_fix = fixes.get(previous_fix_designator)
        exit_fix = fixes.get(exit_fix_designator)

        if previous_fix is None:
            logger.warning(
                f"Cannot construct exit gate for aircraft "
                f"{coordination.callsign}: route fix "
                f"{previous_fix_designator} was not found"
            )
            return None

        if exit_fix is None:
            logger.warning(
                f"Cannot construct exit gate for aircraft "
                f"{coordination.callsign}: route fix "
                f"{exit_fix_designator} was not found"
            )
            return None

        route_angle = previous_fix.bearing_to(exit_fix)
        half_width_nm = gate_width_nm / 2

        return cls(
            start_position=coordination_fix.forward(
                half_width_nm,
                route_angle + 90,
            ),
            end_position=coordination_fix.forward(
                half_width_nm,
                route_angle - 90,
            ),
        )

    def is_crossed(self, previous_position: Pos2D, current_position: Pos2D) -> bool:
        """
        Return the orientation of the ordered points ``a``, ``b`` and ``c``.

        A positive or negative result indicates which side of line ``a-b``
        point ``c`` lies on. Zero indicates collinearity.
        """

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
    Outcomm aircraft when they cross their coordinated exit gate at the
    coordinated flight level.

    The exit gate uses the same definition as the coordination-gate metric:

    - its centre is the coordinated exit fix;
    - its orientation is perpendicular to the inbound planned-route segment;
    - its width is ``gate_width_nm``.

    Aircraft without an exit coordination are outcommed when they leave a
    sector in which they were previously observed.

    Parameters
    ----------
    gate_width_nm:
        Total width of the exit gate in nautical miles.
    aircraft_in_sector:
        Callsigns observed inside each sector.
    previous_positions:
        Previous aircraft positions, used to detect gate crossings.
    """

    require_exit_fl: bool = True
    gate_width_nm: float = 10

    _previous_positions: dict[str, Pos2D] = field(default_factory=dict)

    def update(self, env_manager: EnvironmentManager) -> None:
        """
        Update aircraft positions and outcomm aircraft that cross their
        coordinated exit gates at the coordinated flight level.
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

            coordination = environment.exit_coordination(
                sector_name,
                callsign,
            )

            if coordination is not None:
                if aircraft.flight_plan is None:
                    logger.warning(f"Cannot check the exit gate for aircraft {callsign}: it has no flight plan")
                    continue

                if previous_position is None:
                    continue

                gate = ExitGate.new(
                    route=aircraft.flight_plan.route,
                    coordination=coordination,
                    fixes=environment.airspace.fixes,
                    gate_width_nm=self.gate_width_nm,
                )

                if gate is None:
                    continue

                crossed_gate = gate.is_crossed(
                    previous_position=previous_position,
                    current_position=current_position,
                )

                if crossed_gate and (aircraft.fl == coordination.fl or not self.require_exit_fl):
                    self._previous_positions.pop(callsign, None)

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
