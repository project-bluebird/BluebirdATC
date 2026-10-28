from collections import defaultdict
from dataclasses import dataclass, field

import typing_extensions
from shapely.geometry import LineString

from bluebird_dt.core import Action, Coordination, Fixes, FlightPlan
from bluebird_dt.core.pos2d import Pos2D
from bluebird_dt.logger import logger
from bluebird_dt.manager import EnvironmentManager
from bluebird_dt.utility.geo_helper import GeoHelper


@dataclass(frozen=True, slots=True)
class ExitGate:
    """A sector-boundary gate associated with the coordinated exit fix."""

    start_position: Pos2D
    end_position: Pos2D
    midpoint: Pos2D
    projection: GeoHelper

    @classmethod
    def new(cls, start_position: Pos2D, end_position: Pos2D) -> typing_extensions.Self:
        midpoint = start_position.bisect(end_position)
        projection = GeoHelper(origin=(midpoint.lat, midpoint.lon))

        return cls(start_position=start_position, end_position=end_position, midpoint=midpoint, projection=projection)

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
            midpoint=self.midpoint,
            projection=self.projection,
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

    def oriented_for_approach(self, approach_position: Pos2D) -> typing_extensions.Self:
        """Return the gate oriented for an aircraft approaching *from* a position.

        The gate is projected onto a local Cartesian plane centred on its
        geodesic midpoint. The sign of the two-dimensional cross product

        ``(end - start) x (approach - start)``

        identifies which side of the directed gate contains the approach
        position.

        ``is_crossed()`` defines a valid exit as movement from the positive,
        or left, side of the directed gate to the negative, or right, side.
        The gate is therefore flipped when the approach position lies on the
        negative side.

        Parameters
        ----------
        approach_position:
            A position on the aircraft's planned route before the gate,
            normally the fix preceding the coordinated exit fix.

        Returns
        -------
        Self
            This gate when already correctly oriented, otherwise a new gate
            with its endpoints reversed.

        Raises
        ------
        ValueError
            If the approach position is collinear with the gate, leaving the
            required gate direction ambiguous.
        """

        start = self.projection.projection(
            self.start_position.lon,
            self.start_position.lat,
        )
        end = self.projection.projection(
            self.end_position.lon,
            self.end_position.lat,
        )
        approach = self.projection.projection(
            approach_position.lon,
            approach_position.lat,
        )

        approach_side = (end[0] - start[0]) * (approach[1] - start[1]) - (end[1] - start[1]) * (approach[0] - start[0])

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
        """A unidirectional sector-boundary gate associated with an exit fix.

        The gate is represented by the directed line segment from
        ``start_position`` to ``end_position``. An aircraft crosses the gate when
        the line segment formed by its previous and current positions intersects
        the gate.

        Because latitude and longitude are angular coordinates on an ellipsoid,
        the intersection is not calculated directly in geographic coordinates.
        Instead, a stereographic projection centred on the geodesic midpoint of
        the gate transforms the gate and aircraft positions into a local
        two-dimensional Cartesian coordinate system. The intersection can then
        be evaluated using planar geometry.

        The direction of the gate is defined by the ordering of its endpoints.
        After establishing that the aircraft movement intersects the gate, the
        side of the directed gate on which the previous position lies is
        determined using the two-dimensional cross product::

            (end - start) x (previous - start)

        or, in Cartesian coordinates::

            (end.x - start.x) * (previous.y - start.y)
            - (end.y - start.y) * (previous.x - start.x)

        The sign of this value describes the orientation of the previous position
        relative to the directed gate:

        * positive: the previous position lies to the left of the gate;
        * negative: the previous position lies to the right of the gate;
        * zero: the previous position lies on the gate.

        A valid exit crossing is defined as an intersection in which the aircraft
        approaches from the left side of the directed gate. Since an intersection
        has already been established, this corresponds to movement from the left
        side to the right side.

        Reversing ``start_position`` and ``end_position`` therefore reverses the
        permitted crossing direction.
        """
        start = self.projection.projection(
            self.start_position.lon,
            self.start_position.lat,
        )
        end = self.projection.projection(
            self.end_position.lon,
            self.end_position.lat,
        )
        previous = self.projection.projection(
            previous_position.lon,
            previous_position.lat,
        )
        current = self.projection.projection(
            current_position.lon,
            current_position.lat,
        )

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
