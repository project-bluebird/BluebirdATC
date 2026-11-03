from bluebird_gymnasium.envs import ViewType
from bluebird_gymnasium.envs.sector_i import SectorIEnv
from bluebird_gymnasium.utils.constants import DISTANCE_AWAY_FROM_EXIT_THRESHOLD
from bluebird_gymnasium.utils.types import PositionStatus


def test_step_archives_aircraft_beyond_exit_threshold():
    """An aircraft due to be archived is moved from `ac_tracker` to `ac_archived`.

    The subject aircraft is set to EXIT_REACHED and placed beyond its exit fix
    by more than `DISTANCE_AWAY_FROM_EXIT_THRESHOLD`, so `step` should add it
    to `callsigns_to_move` and archive it.
    """

    config = SectorIEnv.get_default_env_config(ViewType.CENTRALIZED)
    gym_env = SectorIEnv(config=config)
    gym_env.reset()

    # step until at least one aircraft is being tracked. set a maximum number of
    # steps to run.
    for _ in range(150):
        gym_env.step(0)  # NOOP action
        if len(gym_env.ac_tracker) > 0:
            callsign = next(iter(gym_env.ac_tracker))
            aircraft_tracked_data = gym_env.ac_tracker[callsign]
            aircraft = gym_env.simulator_env.aircraft[callsign]

            # the subject aircraft has reached its exit
            aircraft_tracked_data.pos_status = PositionStatus.EXIT_REACHED
            aircraft_tracked_data.incomm_status = True
            aircraft_tracked_data.incorrect_sector_exit_pos = None

            # place it at its exit fix, then move it forward (away from the sector)
            # beyond the threshold distance
            exit_pos = aircraft_tracked_data.sector_exit_pos
            heading = aircraft_tracked_data.sector_entry_pos.bearing_to(exit_pos)
            new_pos = exit_pos.forward(DISTANCE_AWAY_FROM_EXIT_THRESHOLD + 10.0, heading)
            aircraft.set_position(new_pos.lat, new_pos.lon)
            aircraft.set_attributes({"heading": heading})

            gym_env.step(0)

            assert callsign not in gym_env.ac_tracker
            assert callsign in gym_env.ac_archived
            archived = gym_env.ac_archived[callsign]
            assert archived.pos_status == PositionStatus.EXIT_REACHED
            assert archived.dist_away_from_sector_exit > DISTANCE_AWAY_FROM_EXIT_THRESHOLD
            break
