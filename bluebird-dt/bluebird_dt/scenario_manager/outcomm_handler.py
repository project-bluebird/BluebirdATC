from collections import defaultdict
from dataclasses import dataclass, field

from bluebird_dt.core import Action
from bluebird_dt.logger import logger
from bluebird_dt.manager import EnvironmentManager


@dataclass(slots=True)
class OutcommHandler:
    """
    A handler, designed to be used to with a ScenarioManager, which outcomms aircraft given at exit:
    - they leave through the gate closest to their exit fix
    - they leave at the correct exit flight levels

    Parameters
    ----------
    aircraft_in_sector: dict[str, set[str]]
        Used to keep track of aircraft in each sector.
    """

    aircraft_in_sector: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    require_exit_fl: bool = True

    def update(self, env_manager: EnvironmentManager):
        """
        Update the internal state and outcom any required aircraft given at exit:
        - they leave through the gate closest to their exit fix
        - they leave at the correct exit flight levels

        Parameters
        ----------
        env_manager: EnvironmentManager
            The environment manager to consider for updating
        """
        for aircraft in env_manager.environment.aircraft.values():
            if (
                # Aircraft is not incommed into a sector
                aircraft.current_sector == "background"
                # Aircraft callsign is None
                or aircraft.callsign is None
            ):
                continue

            current_sector = env_manager.environment.airspace.sectors[aircraft.current_sector]
            if (
                # Aircraft is physically in the sector
                current_sector.contains_laterally(aircraft.pos2d())
            ):
                self.aircraft_in_sector[aircraft.current_sector].add(aircraft.callsign)

            # Aircraft used to be in the sector
            elif aircraft.callsign in self.aircraft_in_sector[aircraft.current_sector]:
                self.aircraft_in_sector[aircraft.current_sector].remove(aircraft.callsign)

                # If aircraft has an exit coordination
                if (
                    exit_coordination := env_manager.environment.exit_coordination(
                        aircraft.current_sector, aircraft.callsign
                    )
                ) is not None:
                    exit_fix = env_manager.environment.airspace.fixes.get(exit_coordination.fix)

                    logger.debug(
                        f"Checking if aircraft {aircraft.callsign} needs to be outcommed based on the exit conditions"
                        "against fix {exit_fix}."
                    )
                    if (
                        # Aircraft is at exit flight level
                        ((aircraft.fl == exit_coordination.fl) or (not self.require_exit_fl))
                        # And either the aircraft exit fix failed to be identified or, if identified, it is exiting
                        # through the same gate
                        and (
                            exit_fix is None
                            or current_sector.volumes[0].area.nearest_segment_to_point(aircraft.pos2d())
                            == current_sector.volumes[0].area.nearest_segment_to_point(exit_fix)
                        )
                    ):
                        env_manager.receive_actions([Action(aircraft.callsign, "outcomm", "background")])

                # Aircraft doesn't have an exit coordination
                else:
                    env_manager.receive_actions([Action(aircraft.callsign, "outcomm", "background")])
