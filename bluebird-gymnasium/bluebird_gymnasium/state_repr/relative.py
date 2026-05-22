from __future__ import annotations

import typing

import numpy as np
from bluebird_dt.utility import convert

from bluebird_gymnasium.state_repr import (
    StateReprClipper as SRC,
)
from bluebird_gymnasium.state_repr import (
    StateReprScaler as SRS,
)
from bluebird_gymnasium.state_repr.base import BaseRepresentation
from bluebird_gymnasium.utils.constants import DIST_EXIT_LEVEL_THRESHOLD
from bluebird_gymnasium.utils.geo_utils import (
    angle_diff,
    left_right_check,
    segment_in_sector_interp,
)
from bluebird_gymnasium.utils.simulator_utils import (
    distance_to_target_pos_along_route,
    get_n_forward_fixes,
)
from bluebird_gymnasium.utils.simulator_utils import (
    get_aircraft_selected_heading as _selected_heading,
)
from bluebird_gymnasium.utils.types import PositionStatus

if typing.TYPE_CHECKING:
    import numpy.typing as npt

    from bluebird_gymnasium.envs.base import BaseEnv
    from bluebird_gymnasium.utils.types import Line


class RelativeRepresentationRaw(BaseRepresentation):
    """Representation of aircraft state in the simulator.

    Raw feature values

    +-----+-------------------------------+------+------+---------------------+
    | Num |         Observation           | Min  | Max  |        Unit         |
    +-----+-------------------------------+------+------+---------------------+
    |   0 | Aircraft flight level and exit| -inf |  inf | Flight level (FL)   |
    |     | flight level difference       |      |      |                     |
    |   1 | Aircraft selected flight level| -inf |  inf | Flight level (FL)   |
    |     | and exit flight level         |      |      |                     |
    |     | difference                    |      |      |                     |
    |   2 | Aircraft entry flight level   | -inf |  inf | Flight level (FL)   |
    |     | and exit flight level         |      |      |                     |
    |     | difference                    |      |      |                     |
    |   3 | Aircraft true air speed (tas) |  0.0 |  inf | Nautical miles per  |
    |     |                               |      |      | second              |
    |   4 | Aircraft distance from the    | -inf |  inf | Nautical Miles (NM) |
    |     | filed route centreline        |      |      |                     |
    |   5 | Aircraft distance from the    | -inf |  inf | Nautical Miles (NM) |
    |     | current route centreline      |      |      |                     |
    |   6 | Difference betweeen aircraft  | -inf |  inf | Nautical Miles (NM) |
    |     | track distance (current route)|      |      |                     |
    |     | to its exit position and its  |      |      |                     |
    |     | track distance to its exit    |      |      |                     |
    |     | flight level                  |      |      |                     |
    |   7 | Difference between aircraft   | -inf |  inf | Nautical Miles (NM) |
    |     | previous and current timestep |      |      |                     |
    |     | track (current route) distance|      |      |                     |
    |     | to its sector's exit          |      |      |                     |
    |   8 | Aircraft distance to sector   |  0.0 |  inf | Nautical Miles (NM) |
    |     | entry, pre-incomm. Once incomm|      |      |                     |
    |     | this is set as 0.0 onwards.   |      |      |                     |
    |   9 | Aircraft distance from sector |  0.0 |  inf | Nautical Miles (NM) |
    |     | exit, after outcomm. This is  |      |      |                     |
    |     | set as 0.0 until outcomm.     |      |      |                     |
    |   10| Aircraft route following (ff) | -1.0 |  1.0 | Discrete:           |
    |     | status/flag                   |      |      | -1 no route ff.     |
    |     |                               |      |      |  1 route ff.        |
    |   11| Aircraft position status      | -1.0 |  1.0 | Discrete:           |
    |     |                               |      |      | -1 out of sector    |
    |     |                               |      |      |  1 inside sector    |
    |   12| Aircraft position status:     |  -pi |   pi | Angle (radians)     |
    |     | difference between current    |      |      |                     |
    |     | heading and the bearing from  |      |      |                     |
    |     | aircraft position to the      |      |      |                     |
    |     | nearest boundary point of its |      |      |                     |
    |     | current sector (360 deg look) |      |      |                     |
    |   13| Aircraft position status:     |  -pi |   pi | Angle (radians)     |
    |     | difference between selected   |      |      |                     |
    |     | heading and the bearing from  |      |      |                     |
    |     | aircraft position to the      |      |      |                     |
    |     | nearest boundary point of its |      |      |                     |
    |     | current sector (360 deg look) |      |      |                     |
    |   14| Aircraft position status:     | -inf |  inf | Nautical Miles (NM) |
    |     | lateral distance to the       |      |      |                     |
    |     | nearest boundary point of its |      |      |                     |
    |     | current sector (360 deg look) |      |      |                     |
    |   15| Aircraft position status:     | -inf |  inf | Nautical Miles (NM) |
    |     | lateral distance to the       |      |      |                     |
    |     | nearest boundary point of its |      |      |                     |
    |     | current sector (forward look) |      |      |                     |
    |   16| Aircraft position status:     | -inf |  inf | Flight level (FL)   |
    |     | difference between aircraft   |      |      |                     |
    |     | selected flight level and a   |      |      |                     |
    |     | minimum flight level of its   |      |      |                     |
    |     | current sector                |      |      |                     |
    |   17| Aircraft position status:     | -inf |  inf | Flight level (FL)   |
    |     | difference between aircraft   |      |      |                     |
    |     | selected flight level and a   |      |      |                     |
    |     | maximum flight level of its   |      |      |                     |
    |     | current sector                |      |      |                     |
    |   18| Aircraft position status:     | -inf |  inf | Flight level (FL)   |
    |     | difference between aircraft   |      |      |                     |
    |     | current flight level and a    |      |      |                     |
    |     | minimum flight level of its   |      |      |                     |
    |     | current sector                |      |      |                     |
    |   19| Aircraft position status:     | -inf |  inf | Flight level (FL)   |
    |     | difference between aircraft   |      |      |                     |
    |     | current flight level and a    |      |      |                     |
    |     | maximum flight level of its   |      |      |                     |
    |     | current sector                |      |      |                     |
    +-----+-------------------------------+------+------+---------------------+

    If num_forward_fixes > 0, then for each forward fix, the following
    representation is concatenated to the state:
    +-----+-------------------------------+------+------+---------------------+
    | Num |         Observation           | Min  | Max  |        Unit         |
    +-----+-------------------------------+------+------+---------------------+
    |   0 | Relative angular difference   | -pi  |  pi  | Angle (radians)     |
    |     | between aircraft heading and  |      |      |                     |
    |     | the bearing from aircraft     |      |      |                     |
    |     | position to fix               |      |      |                     |
    |   1 | Relative angular difference   | -pi  |  pi  | Angle (radians)     |
    |     | between aircraft selected     |      |      |                     |
    |     | heading and the bearing from  |      |      |                     |
    |     | aircraft position to fix      |      |      |                     |
    |   2 | Relative angular difference   | -pi  |  pi  | Angle (radians)     |
    |     | between aircraft heading and  |      |      |                     |
    |     | the bearing from the previous |      |      |                     |
    |     | fix to the fix                |      |      |                     |
    |   3 | Relative angular difference   | -pi  |  pi  | Angle (radians)     |
    |     | between aircraft selected     |      |      |                     |
    |     | heading and the bearing from  |      |      |                     |
    |     | the previous fix to the fix   |      |      |                     |
    |   4 | Aircraft linear distance to   |  0.0 |  inf | Nautical Miles (NM) |
    |     | fix                           |      |      |                     |
    |   5 | Aircraft route distance to    |  0.0 |  inf | Nautical Miles (NM) |
    |     | fix                           |      |      |                     |
    |   6 | Flag: is aircraft to next fix | -1.0 |  1.0 | +1.0 if in sector   |
    |     | segment within the sector     |      |      | -1.0 otherwise      |
    +-----+-------------------------------+------+------+---------------------+

    If knn > 0, then for each nearby aircraft, the following representation is
    concatenated to the state:
    +-----+-------------------------------+-----+-----+-----------------------+
    | Num |         Observation           | Min | Max |        Unit           |
    +-----+-------------------------------+-----+-----+-----------------------+
    |   0 | Relative angular (non-reflex) | -pi | +pi | Angle (radians)       |
    |     | difference between aircraft   |     |     |                       |
    |     | heading and the heading of the|     |     |                       |
    |     | neighbour aircraft, including |     |     |                       |
    |     | the direction of turn (sign). |     |     |                       |
    |     | (helps agent to determine     |     |     |                       |
    |     | interaction category: same,   |     |     |                       |
    |     | cross or opposite track)      |     |     |                       |
    |   1 | Relative angular (non-reflex) | -pi | +pi | Angle (radians)       |
    |     | difference between aircraft   |     |     |                       |
    |     | selected heading and the      |     |     |                       |
    |     | selected heading of the       |     |     |                       |
    |     | neighbour aircraft, including |     |     |                       |
    |     | the direction of turn (sign). |     |     |                       |
    |     | (helps agent to determine     |     |     |                       |
    |     | interaction category: same,   |     |     |                       |
    |     | cross or opposite track)      |     |     |                       |
    |   2 | Category of the distance      | -1.0| +1.0| N/A. discrete feature |
    |     | between aircraft and its      |     |     | -1, 0, and +1         |
    |     | neighbour: either reducing,   |     |     |                       |
    |     | same, or growing              |     |     |                       |
    |   3 | Distance between aircraft     | 0.0 | inf | Nautical Miles (NM)   |
    |     | and neighbour aircraft        |     |     |                       |
    |   4 | Flight level difference       | -inf| inf | Flight level          |
    |     | between aircraft and          |     |     |                       |
    |     | neighbour aircraft            |     |     |                       |
    |   5 | Selected flight level         | -inf| inf | Flight level          |
    |     | difference between aircraft   |     |     |                       |
    |     | and neighbour aircraft        |     |     |                       |
    |   6 | Speed difference between      | -inf| inf | Flight level          |
    |     | the aircraft and neighbour    |     |     |                       |
    |     | aircraft                      |     |     |                       |
    |   7 | Flag to indicate whether or   | -1.0| +1.0| Discrete:             |
    |     | not the neighbour aircraft is |     |     | -1 not controllable   |
    |     | controllable. an aircraft is  |     |     |  1 controllable       |
    |     | not controllable if its       |     |     |                       |
    |     | controllable flag attribute is|     |     |                       |
    |     | set to False in the simulator.|     |     |                       |
    |     | even if it set to True, the   |     |     |                       |
    |     | aircraft may not controllable |     |     |                       |
    |     | based on defined logic such as|     |     |                       |
    |     | - if aircraft hasn't incommed |     |     |                       |
    |     | - if aircraft has outcommed   |     |     |                       |
    |     | - if aircraft is out of sector|     |     |                       |
    |   8 | difference between aircraft   |  0.0| inf | Nautical Miles (NM)   |
    |     | pair relative centreline      |     |     |                       |
    |     | distance, depending on the    |     |     |                       |
    |     | interaction cateogry          |     |     |                       |
    |     |                               |     |     |                       |
    +-----+------------------------------ +-----+-----+-----------------------+

    Args:
        knn: the number of nearest aircraft to consider when representing
            state for an aircraft. Defaults to 0.
        num_forward_fixes: the number of forward fixes to encode in the state
            representation of the aircraft. Defaults to 1.
        use_filed_route: specifies whether to get the forward fixes from the
            aircraft's filed (`True`) or current (`False`) route.
            Defaults to `False`.
        num_actions: the number of actions for an aircraft. Defaults to
            `None` if it is not used in the aircraft's state representation.

    Note for users when training agents based on neural network policies:
    while this representation could be proccessed directly by a neural network,
    only consider using it only when additional pre-processing has been
    done before it is fed to a network as input. This is because the scales
    and the range of values for each feature is signficantly different.
    An example of an additional processing is the use of normalization (and
    clipping) strategies to standardize the input based on running mean and
    standard deviation. This dynamically adjusts the scaling metrics unlike
    the features in the next class below that is scaled and clipped using
    handpicked metrics.
    """

    def __init__(
        self,
        knn: int = 0,
        num_forward_fixes: int = 1,
        use_filed_route: bool = True,
        num_actions: int | None = None,
    ):
        super().__init__(knn, num_forward_fixes, use_filed_route, num_actions)

        ####### base features range
        base_feats_low = [
            -np.inf,
            -np.inf,
            -np.inf,
            0.0,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            0.0,
            0.0,
            -1.0,
            -1.0,
            -np.pi,
            -np.pi,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
            -np.inf,
        ]
        base_feats_high = [
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            1.0,
            1.0,
            np.pi,
            np.pi,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
        ]

        ####### fixes features range
        if self.num_forward_fixes > 0:
            fixes_feats_low = [
                -np.pi,
                -np.pi,
                -np.pi,
                -np.pi,
                0.0,
                0.0,
                -1.0,
            ]
            fixes_feats_high = [
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                np.inf,
                np.inf,
                1.0,
            ]
            self.num_features_per_fix = len(fixes_feats_low)

            fixes_feats_low *= self.num_forward_fixes
            fixes_feats_high *= self.num_forward_fixes

        else:
            fixes_feats_low = []
            fixes_feats_high = []

        ####### neighbours features range
        if self.knn > 0:
            neighbours_feats_low = [
                -np.pi,
                -np.pi,
                -1.0,
                0.0,
                -np.inf,
                -np.inf,
                -np.inf,
                -1.0,
                0.0,
            ]
            neighbours_feats_high = [
                np.pi,
                np.pi,
                1.0,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                1.0,
                np.inf,
            ]
            self.num_features_per_neighbour = len(neighbours_feats_low)

            neighbours_feats_low *= self.knn
            neighbours_feats_high *= self.knn

        else:
            neighbours_feats_low = []
            neighbours_feats_high = []

        self.low = base_feats_low + fixes_feats_low + neighbours_feats_low
        self.high = base_feats_high + fixes_feats_high + neighbours_feats_high
        self.low = np.asarray(self.low, dtype=np.float32)
        self.high = np.asarray(self.high, dtype=np.float32)

    def repr(self, gym_env: BaseEnv, callsign: str) -> npt.NDArray[np.float32]:
        """Generate a vectorised representation of an aircraft's state.

        Args:
            gym_env: defines the gymnasium environment.
            callsign: the identifier of the aircraft.

        Returns:
            numpy array which represents the aircraft's current state as
            a vector.
        """

        simulator_env = gym_env.get_simulator_env()
        tracked_data = gym_env.get_tracked_aircraft_data()
        prev_tracked_data = gym_env.get_tracked_aircraft_data_previous()
        if prev_tracked_data is None:
            # gym_env was just reset and this is the first step.
            # hence, use the current tracked data as the previous
            # timestep tracked data (as a proxy)
            prev_tracked_data = tracked_data
        airspace_sector = gym_env.get_active_airspace_sector()
        aircraft = simulator_env.aircraft[callsign]
        airspace = simulator_env.airspace

        ####### utils: get useful information for computing base features
        # get entry and exit flight levels from coordination data
        entry_fl = tracked_data[callsign].entry_coords[airspace_sector].fl
        exit_fl = tracked_data[callsign].exit_coords[airspace_sector].fl

        # relative flight level: difference beween aircraft flight level
        # and exit flight level
        fl_diff_ac_exit = aircraft.fl - exit_fl

        # relative flight level: difference beween aircraft
        # selected flight level and exit flight level
        fl_diff_ac_exit_selected = aircraft.selected_fl - exit_fl

        # relative flight level: difference between aircraft
        # entry flight level and exit flight level
        fl_diff_ac_entry_exit = entry_fl - exit_fl

        # centreline distance (filed route)
        _dist, turn_dir, _ = tracked_data[callsign].centreline_info_fr
        ac_centre_dist_fr = _dist * turn_dir

        # centreline distance (current route)
        _dist, turn_dir, _ = tracked_data[callsign].centreline_info_cr
        ac_centre_dist_cr = _dist * turn_dir

        # difference between track (current route) distance to exit
        # and distance to exit flight level
        ## aircraft to exit location distance, along its planned route/track
        dist_ac_exit = tracked_data[callsign].track_dist_to_exit_cr

        ## aircraft to exit flight level distance
        dist_ac_level = tracked_data[callsign].dist_to_target_fl

        ## finally the difference
        dist_diff_exit_pos_level = dist_ac_exit - dist_ac_level
        dist_diff_exit_pos_level -= DIST_EXIT_LEVEL_THRESHOLD

        # difference between the previous and current track (current route)
        # distance to exit
        if callsign in prev_tracked_data:
            prev_dist_ac_exit = prev_tracked_data[callsign].track_dist_to_exit_cr
        else:
            # aircraft tracking just started (either was just spawned or
            # arrived at the point before sector entry where tracking begins).
            # so, there isn't any tracked data from the previous timestep.
            # use the current state as the previous.
            prev_dist_ac_exit = tracked_data[callsign].track_dist_to_exit_cr

        curr_dist_ac_exit = tracked_data[callsign].track_dist_to_exit_cr
        dist_diff_prev_curr_to_exit = curr_dist_ac_exit - prev_dist_ac_exit

        # aircraft distance to the sector entry (pre-incomm).
        # only measured before aircraft reaches the sector.
        # if aircraft has already incommed, then it set to 0.0
        dist_to_sector_entry = tracked_data[callsign].dist_to_sector_entry

        # aircraft distance away from the sector exit (after outcomm)
        # only measured after the aircraft has exited the sector.
        # if aircraft is still in the sector, then it set to 0.0
        dist_away_sector_exit = tracked_data[callsign].dist_away_from_sector_exit

        # aircraft route following status: indicates whether or not the
        # aircraft is following its defined flight plan route, or not (in this
        # case, flying on a heading issued by the agent).
        route_ff_status = 1.0 if aircraft.on_route else -1.0

        # aircraft position indicator: in-sector or out of sector
        pos_status = 1.0 if tracked_data[callsign].pos_status == PositionStatus.IN_SECTOR else -1.0

        # aircraft position status measurement:
        bearing_ac_nb_360 = tracked_data[callsign].nearest_360_boundary_bear

        # 1. difference between aircraft heading and the bearing from the
        # aircraft position to the nearest boundary (nb) point.
        angle_diff_ac_nb_360 = angle_diff(aircraft.heading, bearing_ac_nb_360)
        turn_dir = left_right_check(aircraft.heading, bearing_ac_nb_360)
        angle_diff_ac_nb_360 *= turn_dir

        # 2. difference between aircraft selected heading and the bearing from
        # the aircraft position to the nearest boundary (nb) point.
        angle_diff_ac_nb_360_sh = angle_diff(_selected_heading(aircraft), bearing_ac_nb_360)
        turn_dir = left_right_check(_selected_heading(aircraft), bearing_ac_nb_360)
        angle_diff_ac_nb_360_sh *= turn_dir

        # aircraft position status measurements:
        # 3. lateral distance between aircraft (ac) position to the nearest
        #    boundary (nb) point (360 degrees look) of the sector.
        # 4. lateral distance between aircraft (ac) position to the nearest
        #    forward boundary (nb) point (forward look only) of the sector.
        # 5. difference between aircraft selected flight level and the minimum
        #    flight level of the sector.
        # 6. difference between aircraft selected flight level and the maximum
        #    flight level of the sector.
        # 7. difference between aircraft current flight level and the minimum
        #    flight level of the sector.
        # 8. difference between aircraft current flight level and the minimum
        #    flight level of the sector.

        if tracked_data[callsign].pos_status == PositionStatus.IN_SECTOR:
            dist_ac_nb_360 = tracked_data[callsign].nearest_360_boundary_dist
            # it could be None
            dist_ac_nb_fwd = tracked_data[callsign].nearest_forward_boundary_dist
            if dist_ac_nb_fwd is None:
                # set it to an arbitrarily large feature value
                dist_ac_nb_fwd = 1e3

            ### find the volume of the sector that the aircraft is
            ### located in. the aircraft should be in one of them because we
            ### have previously ascertained that the aircraft is in the sector
            found_idx = -1
            airspace = simulator_env.airspace
            _sector = airspace.sectors[airspace_sector]
            main_volumes = _sector.volumes
            cond_volumes = _sector.get_conditional_volumes_for_aircraft(aircraft)
            cond_volumes = list(cond_volumes.values())
            volumes = main_volumes + cond_volumes

            for i, volume in enumerate(volumes):
                if volume.contains(aircraft):
                    found_idx = i
                    break
            min_fl = volumes[found_idx].min_fl
            max_fl = volumes[found_idx].max_fl

            fl_diff_ac_selected_sector_min = aircraft.selected_fl - min_fl
            fl_diff_ac_selected_sector_max = aircraft.selected_fl - max_fl
            fl_diff_ac_current_sector_min = aircraft.fl - min_fl
            fl_diff_ac_current_sector_max = aircraft.fl - max_fl

        else:
            dist_ac_nb_360 = -tracked_data[callsign].nearest_360_boundary_dist
            # it could be None
            dist_ac_nb_fwd = tracked_data[callsign].nearest_forward_boundary_dist
            if dist_ac_nb_fwd is None:
                # set it to an arbitrarily large feature value
                dist_ac_nb_fwd = -1e3
            else:
                dist_ac_nb_fwd *= -1.0

            fl_diff_ac_selected_sector_min = 0.0
            fl_diff_ac_selected_sector_max = 0.0
            fl_diff_ac_current_sector_min = 0.0
            fl_diff_ac_current_sector_max = 0.0

        ####### base features: the current aircraft state
        base_feats = [
            fl_diff_ac_exit,
            fl_diff_ac_exit_selected,
            fl_diff_ac_entry_exit,
            aircraft.speed_tas,
            ac_centre_dist_fr,
            ac_centre_dist_cr,
            dist_diff_exit_pos_level,
            dist_diff_prev_curr_to_exit,
            dist_to_sector_entry,
            dist_away_sector_exit,
            route_ff_status,
            pos_status,
            angle_diff_ac_nb_360 * convert.DEG_TO_RAD,
            angle_diff_ac_nb_360_sh * convert.DEG_TO_RAD,
            dist_ac_nb_360,
            dist_ac_nb_fwd,
            fl_diff_ac_selected_sector_min,
            fl_diff_ac_selected_sector_max,
            fl_diff_ac_current_sector_min,
            fl_diff_ac_current_sector_max,
        ]
        base_feats = np.asarray(base_feats, dtype=np.float32)

        ####### fixes features
        fixes_feats = self.generate_forward_fixes_features(gym_env, callsign)

        ####### neighbours features
        neighbours_feats = self.generate_neighbours_features(gym_env, callsign)

        feats_list = [base_feats, *fixes_feats, *neighbours_feats]
        return np.concatenate(feats_list, dtype=np.float32)

    def generate_forward_fixes_features(self, gym_env: BaseEnv, callsign: str) -> list[npt.NDArray[np.float32]]:
        """Generate features for N forward fixes.

        The forward fixes are derived using the aircraft's filed route
        if `.use_filed_route` is `True`. Otherwise, the current route is used.

        Args:
            gym_env: defines the gymnasium environment.
            callsign: the identifier of the aircraft.

        Returns:
            list of representations for each fix or an empty list if
            `.num_forward_fixes` is 0.
        """

        if self.num_forward_fixes == 0:
            return []

        # features related to aircraft's relationship to N fixes. For each
        # fix, compute:
        # - relative angular difference between aircraft heading and the
        #   bearing of the aircraft position to the next fix (current route)
        # - relative angular difference between aircraft selected heading and
        #   the bearing of the aircraft position to the next fix (current
        #   route)
        # - relative angular difference between aircraft heading and the
        #   bearing from the previous fix to the next fix (current route)
        # - relative angular difference between aircraft selected heading and
        #   the bearing from the previous fix to the next fix (current route)
        # - aircraft to next fix linear distance
        # - aircraft to next fix route/track distance
        # - flag: indicates whether the linear segment from aircraft's
        #         position to fix position is within the sector, or crosses
        #         out of the sector.

        simulator_env = gym_env.get_simulator_env()
        tracked_data = gym_env.get_tracked_aircraft_data()
        airspace_sector = gym_env.get_active_airspace_sector()
        aircraft = simulator_env.aircraft[callsign]
        airspace = simulator_env.airspace

        ####### utils: get useful information for computing fixes features
        # forward fixes position: in *current route* (cr)
        route = aircraft.flight_plan.route.current
        _next_fix = tracked_data[callsign].next_fix_cr
        fixes = get_n_forward_fixes(route, start_from=_next_fix, n=self.num_forward_fixes)
        next_fixes_pos = [simulator_env.airspace.fixes.places[fix] for fix in fixes if fix is not None]
        num_none_fixes = self.num_forward_fixes - len(next_fixes_pos)

        # previous fixes position (relative to each forward fix): in
        # *current route* (cr)
        _next_fix_idx = route.index(_next_fix)
        if _next_fix_idx == 0:
            prev_fixes_pos = [tracked_data[callsign].pos_at_last_route_direct, *next_fixes_pos[:-1]]
        else:
            _prev_fix = route[_next_fix_idx - 1]
            prev_fixes_pos = [simulator_env.airspace.fixes.places[_prev_fix], *next_fixes_pos[:-1]]

        # generate the features for each forward fix
        curr_route_start_pos = tracked_data[callsign].pos_at_last_route_direct
        ac_pos = aircraft.pos2d()

        angles_diff_ac_nfs = []
        angles_diff_ac_sh_nfs = []
        angles_diff_ac_pf_nfs = []
        angles_diff_ac_sh_pf_nfs = []
        linear_dists_ac_nfs = []
        track_dists_ac_nfs = []
        in_sector_segments_ac_nfs = []
        for prev_fix_pos, fix_pos in zip(prev_fixes_pos, next_fixes_pos, strict=True):
            # relative angular diff b/w heading and bearing to next fix
            bearing_ac_nf = ac_pos.bearing_to(fix_pos)
            angle_diff_ac_nf = angle_diff(aircraft.heading, bearing_ac_nf)
            turn_dir = left_right_check(aircraft.heading, bearing_ac_nf)
            angle_diff_ac_nf *= turn_dir
            angles_diff_ac_nfs.append(angle_diff_ac_nf)

            # relative angular diff b/w selected heading and bearing to next fix
            angle_diff_ac_sh_nf = angle_diff(_selected_heading(aircraft), bearing_ac_nf)
            turn_dir = left_right_check(_selected_heading(aircraft), bearing_ac_nf)
            angle_diff_ac_sh_nf *= turn_dir
            angles_diff_ac_sh_nfs.append(angle_diff_ac_sh_nf)

            # relative angular diff b/w heading and
            # bearing from prev to next fix
            bearing_pf_nf = prev_fix_pos.bearing_to(fix_pos)
            angle_diff_ac_pf_nf = angle_diff(aircraft.heading, bearing_pf_nf)
            turn_dir = left_right_check(aircraft.heading, bearing_pf_nf)
            angle_diff_ac_pf_nf *= turn_dir
            angles_diff_ac_pf_nfs.append(angle_diff_ac_pf_nf)

            # relative angular diff b/w selected heading and
            # bearing from prev to next fix
            angle_diff_ac_sh_pf_nf = angle_diff(_selected_heading(aircraft), bearing_pf_nf)
            turn_dir = left_right_check(_selected_heading(aircraft), bearing_pf_nf)
            angle_diff_ac_sh_pf_nf *= turn_dir
            angles_diff_ac_sh_pf_nfs.append(angle_diff_ac_sh_pf_nf)

            # aircraft to next fix linear distance
            linear_dists_ac_nfs.append(aircraft.pos2d().distance(fix_pos))

            # aircraft to next fix route/track distance
            _dist = distance_to_target_pos_along_route(
                aircraft.pos2d(),
                fix_pos,
                aircraft.flight_plan.route.current,
                airspace,
                curr_route_start_pos,
            )
            _dist = _dist if _dist > 0.0 else 0.0
            track_dists_ac_nfs.append(_dist)

            # segment in sector flag
            segment: Line = (ac_pos, fix_pos)
            # setting the epsilon to 0.015 helps to deal with issues of
            # entry/exit fixes being classed as being outside the sector
            # due to the low tolerance value of 1e-10.
            epsilon = 1e-10
            ret = segment_in_sector_interp(
                segment,
                airspace,
                airspace_sector,
                epsilon=epsilon,
                num_interp_positions=5,
            )
            in_sector_segments_ac_nfs.append(1.0 if ret else -1.0)

        ####### fixes features
        fixes_feats = []
        for idx in range(len(next_fixes_pos)):
            tmp = np.asarray(
                [
                    angles_diff_ac_nfs[idx] * convert.DEG_TO_RAD,
                    angles_diff_ac_sh_nfs[idx] * convert.DEG_TO_RAD,
                    angles_diff_ac_pf_nfs[idx] * convert.DEG_TO_RAD,
                    angles_diff_ac_sh_pf_nfs[idx] * convert.DEG_TO_RAD,
                    linear_dists_ac_nfs[idx],
                    track_dists_ac_nfs[idx],
                    in_sector_segments_ac_nfs[idx],
                ],
                dtype=np.float32,
            )
            fixes_feats.append(tmp)

        # zero padding
        for _ in range(num_none_fixes):
            tmp = np.zeros(self.num_features_per_fix, dtype=np.float32)
            fixes_feats.append(tmp)

        return fixes_feats

    def generate_neighbours_features(self, gym_env: BaseEnv, callsign: str) -> list[npt.NDArray[np.float32]]:
        """Generate features for N neighbour aircraft.

        Args:
            gym_env: defines the gymnasium environment.
            callsign: the identifier of the aircraft.

        Returns:
            list of representations for each neighbour or an empty list if
            `.knn` is 0.
        """

        if self.knn == 0:
            return []

        tracked_data = gym_env.get_tracked_aircraft_data()

        ####### utils: get useful information for neighbour features
        # sorted based on aircraft distance to other aircraft
        interactions = self._knn_repr(callsign, gym_env, True)

        _num_other_ac_retrieved = len(interactions)
        if _num_other_ac_retrieved < self.knn:
            add_count = _num_other_ac_retrieved
            balance_count = self.knn - _num_other_ac_retrieved
        else:
            add_count = self.knn
            balance_count = 0

        ####### neighbours features
        interactions_features = []
        for i in range(add_count):
            callsign_other = interactions[i].other_callsign
            dist_ac_other = interactions[i].dist_ac_other
            angle_diff_ac_other = interactions[i].angle_diff_ac_other
            turn_dir_ac_other = interactions[i].turn_dir_ac_other
            angle_diff_ac_other_sh = interactions[i].angle_diff_ac_other_sh
            turn_dir_ac_other_sh = interactions[i].turn_dir_ac_other_sh
            dist_type_ac_other = interactions[i].dist_type_ac_other
            fl_diff_ac_other = interactions[i].fl_diff_ac_other
            selected_fl_diff_ac_other = interactions[i].selected_fl_diff_ac_other
            speed_diff_ac_other = interactions[i].speed_diff_ac_other
            pair_distance = interactions[i].centreline_dist_diff_cr

            controllable = None
            _cs = callsign_other
            if _cs not in tracked_data:
                controllable = -1.0
            elif (
                tracked_data[_cs].pos_status == PositionStatus.IN_SECTOR
                and tracked_data[_cs].incomm_status is True
                and tracked_data[_cs].outcomm_status is False
            ):
                controllable = 1.0
            else:
                controllable = -1.0

            tmp = np.asarray(
                [
                    (angle_diff_ac_other * convert.DEG_TO_RAD * turn_dir_ac_other),
                    (angle_diff_ac_other_sh * convert.DEG_TO_RAD * turn_dir_ac_other_sh),
                    dist_type_ac_other,
                    dist_ac_other,
                    fl_diff_ac_other,
                    selected_fl_diff_ac_other,
                    speed_diff_ac_other,
                    controllable,
                    pair_distance,
                ],
                dtype=np.float32,
            )
            interactions_features.append(tmp)

        # zero padding
        for _ in range(balance_count):
            tmp = np.zeros(self.num_features_per_neighbour, dtype=np.float32)
            interactions_features.append(tmp)

        return interactions_features


class RelativeRepresentation(BaseRepresentation):
    """Representation of aircraft state in the simulator.

    Scaled and clipped version, using handpicked (manual) metrics.

    +-----+-------------------------------+------+------+---------------------+
    | Num |         Observation           | Min  | Max  |        Unit         |
    +-----+-------------------------------+------+------+---------------------+
    |   0 | Aircraft flight level and exit| -3.0 |  3.0 | Flight level        |
    |     | flight level difference       |      |      | (scaled by 20)      |
    |   1 | Aircraft selected flight level| -3.0 |  3.0 | Flight level        |
    |     | and exit flight level         |      |      |                     |
    |     | difference                    |      |      |                     |
    |   2 | Aircraft entry flight level   | -3.0 |  3.0 | Flight level        |
    |     | and exit flight level         |      |      |                     |
    |     | difference                    |      |      |                     |
    |   3 | Aircraft true air speed (tas) |  0.0 |  3.0 | Nautical miles per  |
    |     |                               |      |      | second              |
    |     |                               |      |      | (scaled by 200)     |
    |   4 | Aircraft distance from the    | -3.0 |  3.0 | Nautical Miles (NM) |
    |     | filed route centreline        |      |      |                     |
    |   5 | Aircraft distance from the    | -3.0 |  3.0 | Nautical Miles (NM) |
    |     | current route centreline      |      |      |                     |
    |   6 | Difference betweeen aircraft  | -3.0 |  3.0 | Nautical Miles (NM) |
    |     | track distance (current route)|      |      |                     |
    |     | to its exit position and its  |      |      |                     |
    |     | track distance to its exit    |      |      |                     |
    |     | flight level                  |      |      |                     |
    |   7 | Difference between aircraft   | -3.0 |  3.0 | Nautical Miles (NM) |
    |     | previous and current timestep |      |      |                     |
    |     | track (current route) distance|      |      |                     |
    |     | to its sector's exit          |      |      |                     |
    |   8 | Aircraft distance to sector   |  0.0 |  3.0 | Nautical Miles (NM) |
    |     | entry, pre-incomm. Once incomm|      |      |                     |
    |     | this is set as 0.0 onwards.   |      |      |                     |
    |   9 | Aircraft distance from sector |  0.0 |  3.0 | Nautical Miles (NM) |
    |     | exit, after outcomm. This is  |      |      |                     |
    |     | set as 0.0 until outcomm.     |      |      |                     |
    |   10| Aircraft route following (ff) | -1.0 |  1.0 | Discrete:           |
    |     | status/flag                   |      |      | -1 no route ff.     |
    |     |                               |      |      |  1 route ff.        |
    |   11| Aircraft position status      | -1.0 |  1.0 | Discrete:           |
    |     |                               |      |      | -1 out of sector    |
    |     |                               |      |      |  1 inside sector    |
    |   12| Aircraft position status:     |  -pi |   pi | Angle (radians)     |
    |     | difference between current    |      |      |                     |
    |     | heading and the bearing from  |      |      |                     |
    |     | aircraft position to the      |      |      |                     |
    |     | nearest boundary point of its |      |      |                     |
    |     | current sector (360 deg look) |      |      |                     |
    |   13| Aircraft position status:     |  -pi |   pi | Angle (radians)     |
    |     | difference between selected   |      |      |                     |
    |     | heading and the bearing from  |      |      |                     |
    |     | aircraft position to the      |      |      |                     |
    |     | nearest boundary point of its |      |      |                     |
    |     | current sector (360 deg look) |      |      |                     |
    |   14| Aircraft position status:     | -3.0 |  3.0 | Nautical Miles (NM) |
    |     | lateral distance to the       |      |      | (scaled)            |
    |     | nearest boundary point of its |      |      |                     |
    |     | current sector (360 deg look) |      |      |                     |
    |   15| Aircraft position status:     | -3.0 |  3.0 | Nautical Miles (NM) |
    |     | lateral distance to the       |      |      |                     |
    |     | nearest boundary point of its |      |      |                     |
    |     | current sector (forward look) |      |      |                     |
    |   16| Aircraft position status:     | -3.0 |  3.0 | Flight level (FL)   |
    |     | difference between aircraft   |      |      | (scaled)            |
    |     | selected flight level and a   |      |      |                     |
    |     | minimum flight level of its   |      |      |                     |
    |     | current sector                |      |      |                     |
    |   17| Aircraft position status:     | -3.0 |  3.0 | Flight level (FL)   |
    |     | difference between aircraft   |      |      | (scaled)            |
    |     | selected flight level and a   |      |      |                     |
    |     | maximum flight level of its   |      |      |                     |
    |     | current sector                |      |      |                     |
    |   18| Aircraft position status:     | -3.0 |  3.0 | Flight level (FL)   |
    |     | difference between aircraft   |      |      | (scaled)            |
    |     | current flight level and a    |      |      |                     |
    |     | minimum flight level of its   |      |      |                     |
    |     | current sector                |      |      |                     |
    |   19| Aircraft position status:     | -3.0 |  3.0 | Flight level (FL)   |
    |     | difference between aircraft   |      |      | (scaled)            |
    |     | current flight level and a    |      |      |                     |
    |     | maximum flight level of its   |      |      |                     |
    |     | current sector                |      |      |                     |
    +-----+-------------------------------+------+------+---------------------+

    If num_forward_fixes > 0, then for each forward fix, the following
    representation is concatenated to the state:
    +-----+-------------------------------+------+------+---------------------+
    | Num |         Observation           | Min  | Max  |        Unit         |
    +-----+-------------------------------+------+------+---------------------+
    |   0 | Relative angular difference   | -pi  |  pi  | Angle (radians)     |
    |     | between aircraft heading and  |      |      |                     |
    |     | the bearing from aircraft     |      |      |                     |
    |     | position to fix               |      |      |                     |
    |   1 | Relative angular difference   | -pi  |  pi  | Angle (radians)     |
    |     | between aircraft selected     |      |      |                     |
    |     | heading and the bearing from  |      |      |                     |
    |     | aircraft position to fix      |      |      |                     |
    |   2 | Relative angular difference   | -pi  |  pi  | Angle (radians)     |
    |     | between aircraft heading and  |      |      |                     |
    |     | the bearing from the previous |      |      |                     |
    |     | fix to the fix                |      |      |                     |
    |   3 | Relative angular difference   | -pi  |  pi  | Angle (radians)     |
    |     | between aircraft selected     |      |      |                     |
    |     | heading and the bearing from  |      |      |                     |
    |     | the previous fix to the fix   |      |      |                     |
    |   4 | Aircraft distance to fix      |  0.0 |  3.0 | Nautical Miles (NM) |
    |   5 | Aircraft route distance to    |  0.0 |  3.0 | Nautical Miles (NM) |
    |     | fix                           |      |      |                     |
    |   6 | Flag: is aircraft to next fix | -1.0 |  1.0 | +1.0 if in sector   |
    |     | segment within the sector     |      |      | -1.0 otherwise      |
    +-----+-------------------------------+------+------+---------------------+

    If knn > 0, then for each nearby aircraft, the following representation is
    concatenated to the state:
    +-----+-------------------------------+-----+-----+-----------------------+
    | Num |         Observation           | Min | Max |        Unit           |
    +-----+-------------------------------+-----+-----+-----------------------+
    |   0 | Relative angular (non-reflex) | -pi | +pi | Angle (radians)       |
    |     | difference between aircraft   |     |     |                       |
    |     | heading and the heading of the|     |     |                       |
    |     | neighbour aircraft, including |     |     |                       |
    |     | the direction of turn (sign). |     |     |                       |
    |     | (helps agent to determine     |     |     |                       |
    |     | interaction category: same,   |     |     |                       |
    |     | cross or opposite track)      |     |     |                       |
    |   1 | Relative angular (non-reflex) | -pi | +pi | Angle (radians)       |
    |     | difference between aircraft   |     |     |                       |
    |     | selected heading and the      |     |     |                       |
    |     | selected heading of the       |     |     |                       |
    |     | neighbour aircraft, including |     |     |                       |
    |     | the direction of turn (sign). |     |     |                       |
    |     | (helps agent to determine     |     |     |                       |
    |     | interaction category: same,   |     |     |                       |
    |     | cross or opposite track)      |     |     |                       |
    |   2 | Category of the distance      | -1.0| +1.0| N/A. discrete         |
    |     | between aircraft and its      |     |     | category              |
    |     | neighbour: either reducing,   |     |     |                       |
    |     | same, or growing              |     |     |                       |
    |   3 | Distance between aircraft     | 0.0 | 3.0 | Nautical Miles (NM)   |
    |     | and neighbour aircraft        |     |     | (scaled by 50)        |
    |   4 | Flight level difference       | -3.0| 3.0 | Flight level          |
    |     | between aircraft and          |     |     | (scaled by 20)        |
    |     | neighbour aircraft            |     |     |                       |
    |   5 | Selected flight level         | -3.0| 3.0 | Flight level          |
    |     | difference between aircraft   |     |     | (scaled by 20)        |
    |     | and neighbour aircraft        |     |     |                       |
    |   6 | Speed difference between      | -3.0| 3.0 | Flight level          |
    |     | the aircraft and neighbour    |     |     | (scaled by 30)        |
    |     | aircraft                      |     |     |                       |
    |   7 | Flag to indicate whether or   | -1.0| 1.0 | Discrete:             |
    |     | not the neighbour aircraft is |     |     | -1 not controllable   |
    |     | controllable. an aircraft is  |     |     |  1 controllable       |
    |     | not controllable if its       |     |     |                       |
    |     | controllable flag attribute is|     |     |                       |
    |     | set to False in the simulator.|     |     |                       |
    |     | even if it set to True, the   |     |     |                       |
    |     | aircraft may not controllable |     |     |                       |
    |     | based on defined logic such as|     |     |                       |
    |     | - if aircraft hasn't incommed |     |     |                       |
    |     | - if aircraft has outcommed   |     |     |                       |
    |     | - if aircraft is out of sector|     |     |                       |
    |   8 | difference between aircraft   |  0.0| 3.0 | Nautical Miles (NM)   |
    |     | pair relative centreline      |     |     |                       |
    |     | distance, depending on the    |     |     |                       |
    |     | interaction cateogry          |     |     |                       |
    |     |                               |     |     |                       |
    +-----+------------------------------ +-----+-----+-----------------------+

    Args:
        knn: the number of nearest aircraft to consider when representing
            state for an aircraft. Defaults to 0.
        num_forward_fixes: the number of forward fixes to encode in the state
            representation of the aircraft. Defaults to 1.
        use_filed_route: specifies whether to get the forward fixes from the
            aircraft's filed (`True`) or current (`False`) route.
            Defaults to `False`.
        num_actions: the number of actions for an aircraft. Defaults to
            `None` if it is not used in the aircraft's state representation.
    """

    def __init__(
        self,
        knn: int = 0,
        num_forward_fixes: int = 1,
        use_filed_route: bool = True,
        num_actions: int | None = None,
    ):
        super().__init__(knn, num_forward_fixes, use_filed_route, num_actions)

        ####### base features range
        base_feats_low = [
            -3.0,
            -3.0,
            -3.0,
            0.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            0.0,
            0.0,
            -1.0,
            -1.0,
            -np.pi,
            -np.pi,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
        ]
        base_feats_high = [
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            1.0,
            1.0,
            np.pi,
            np.pi,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
        ]

        ####### fixes features range
        if self.num_forward_fixes > 0:
            fixes_feats_low = [
                -np.pi,
                -np.pi,
                -np.pi,
                -np.pi,
                0.0,
                0.0,
                -1.0,
            ]
            fixes_feats_high = [
                np.pi,
                np.pi,
                np.pi,
                np.pi,
                3.0,
                3.0,
                1.0,
            ]
            self.num_features_per_fix = len(fixes_feats_low)

            fixes_feats_low *= self.num_forward_fixes
            fixes_feats_high *= self.num_forward_fixes

        else:
            fixes_feats_low = []
            fixes_feats_high = []

        ####### neighbours features range
        if self.knn > 0:
            neighbours_feats_low = [
                -np.pi,
                -np.pi,
                -1.0,
                0.0,
                -3.0,
                -3.0,
                -3.0,
                -1.0,
                0.0,
            ]
            neighbours_feats_high = [
                np.pi,
                np.pi,
                1.0,
                3.0,
                3.0,
                3.0,
                3.0,
                1.0,
                3.0,
            ]
            self.num_features_per_neighbour = len(neighbours_feats_low)

            neighbours_feats_low *= self.knn
            neighbours_feats_high *= self.knn
        else:
            neighbours_feats_low = []
            neighbours_feats_high = []

        self.low = base_feats_low + fixes_feats_low + neighbours_feats_low
        self.high = base_feats_high + fixes_feats_high + neighbours_feats_high
        self.low = np.asarray(self.low, dtype=np.float32)
        self.high = np.asarray(self.high, dtype=np.float32)

    def repr(self, gym_env: BaseEnv, callsign: str) -> npt.NDArray[np.float32]:
        """Generate a vectorised representation of an aircraft's state.

        Args:
            gym_env: defines the gymnasium environment.
            callsign: the identifier of the aircraft.

        Returns:
            numpy array which represents the aircraft's current state as
            a vector.
        """

        simulator_env = gym_env.get_simulator_env()
        tracked_data = gym_env.get_tracked_aircraft_data()
        prev_tracked_data = gym_env.get_tracked_aircraft_data_previous()
        if prev_tracked_data is None and gym_env.timestep == 0:
            # gym_env was just reset and this is the first step.
            # hence, use the current tracked data as the previous
            # timestep tracked data (as a proxy)
            prev_tracked_data = tracked_data
        airspace_sector = gym_env.get_active_airspace_sector()
        aircraft = simulator_env.aircraft[callsign]
        airspace = simulator_env.airspace

        ####### utils: get useful information for computing base features
        # get entry and exit flight levels from coordination data
        entry_fl = tracked_data[callsign].entry_coords[airspace_sector].fl
        exit_fl = tracked_data[callsign].exit_coords[airspace_sector].fl

        # relative flight level: difference beween aircraft flight level
        # and exit flight level
        fl_diff_ac_exit = aircraft.fl - exit_fl

        # relative flight level: difference beween aircraft
        # selected flight level and exit flight level
        fl_diff_ac_exit_selected = aircraft.selected_fl - exit_fl

        # relative flight level: difference between aircraft
        # entry flight level and exit flight level
        fl_diff_ac_entry_exit = entry_fl - exit_fl

        # centreline distance (filed route)
        _dist, turn_dir, _ = tracked_data[callsign].centreline_info_fr
        _dist = np.clip(_dist, 0.0, SRC.CLIP_DIST)
        ac_centre_dist_fr = _dist * turn_dir

        # centreline distance (current route)
        _dist, turn_dir, _ = tracked_data[callsign].centreline_info_cr
        _dist = np.clip(_dist, 0.0, SRC.CLIP_DIST)
        ac_centre_dist_cr = _dist * turn_dir

        # difference between track (current route) distance to exit
        # and distance to exit flight level
        ## aircraft to exit location distance, along its planned route/track
        dist_ac_exit = tracked_data[callsign].track_dist_to_exit_cr

        ## aircraft to exit flight level distance
        dist_ac_level = tracked_data[callsign].dist_to_target_fl

        ## finally the difference
        dist_diff_exit_pos_level = dist_ac_exit - dist_ac_level
        dist_diff_exit_pos_level -= DIST_EXIT_LEVEL_THRESHOLD

        # difference between the previous and current track (current route)
        # distance to exit
        if callsign in prev_tracked_data:
            prev_dist_ac_exit = prev_tracked_data[callsign].track_dist_to_exit_cr
        else:
            # aircraft tracking just started (either was just spawned or
            # arrived at the point before sector entry where tracking begins).
            # so, there isn't any tracked data from the previous timestep.
            # use the current state as the previous.
            prev_dist_ac_exit = tracked_data[callsign].track_dist_to_exit_cr

        curr_dist_ac_exit = tracked_data[callsign].track_dist_to_exit_cr
        dist_diff_prev_curr_to_exit = curr_dist_ac_exit - prev_dist_ac_exit

        # aircraft distance to the sector entry (pre-incomm).
        # only measured before aircraft reaches the sector.
        # if aircraft has already incommed, then it set to 0.0
        dist_to_sector_entry = tracked_data[callsign].dist_to_sector_entry

        # aircraft distance away from the sector exit (after outcomm)
        # only measured after the aircraft has exited the sector.
        # if aircraft is still in the sector, then it set to 0.0
        dist_away_sector_exit = tracked_data[callsign].dist_away_from_sector_exit

        # aircraft route following status: indicates whether or not the
        # aircraft is following its defined flight plan route, or not (in this
        # case, flying on a heading issued by the agent).
        route_ff_status = 1.0 if aircraft.on_route else -1.0

        # aircraft position indicator: in-sector or out of sector
        pos_status = 1.0 if tracked_data[callsign].pos_status == PositionStatus.IN_SECTOR else -1.0

        # aircraft position status measurement:
        bearing_ac_nb_360 = tracked_data[callsign].nearest_360_boundary_bear

        # 1. difference between aircraft heading and the bearing from the
        # aircraft position to the nearest boundary (nb) point.
        bearing_ac_nb_360 = tracked_data[callsign].nearest_360_boundary_bear
        angle_diff_ac_nb_360 = angle_diff(aircraft.heading, bearing_ac_nb_360)
        turn_dir = left_right_check(aircraft.heading, bearing_ac_nb_360)
        angle_diff_ac_nb_360 *= turn_dir

        # 2. difference between aircraft selected heading and the bearing from
        # the aircraft position to the nearest boundary (nb) point.
        angle_diff_ac_nb_360_sh = angle_diff(_selected_heading(aircraft), bearing_ac_nb_360)
        turn_dir = left_right_check(_selected_heading(aircraft), bearing_ac_nb_360)
        angle_diff_ac_nb_360_sh *= turn_dir

        # aircraft position status measurements:
        # 3. lateral distance between aircraft (ac) position to the nearest
        #    boundary (nb) point (360 degrees look) of the sector.
        # 4. lateral distance between aircraft (ac) position to the nearest
        #    forward boundary (nb) point (forward look only) of the sector.
        # 5. difference between aircraft selected flight level and the minimum
        #    flight level of the sector.
        # 6. difference between aircraft selected flight level and the maximum
        #    flight level of the sector.
        # 7. difference between aircraft current flight level and the minimum
        #    flight level of the sector.
        # 8. difference between aircraft current flight level and the minimum
        #    flight level of the sector.

        if tracked_data[callsign].pos_status == PositionStatus.IN_SECTOR:
            dist_ac_nb_360 = tracked_data[callsign].nearest_360_boundary_dist
            # it could be None
            dist_ac_nb_fwd = tracked_data[callsign].nearest_forward_boundary_dist
            if dist_ac_nb_fwd is None:
                # set it to an arbitrarily large feature value
                dist_ac_nb_fwd = 1e3

            ### find the volume of the sector that the aircraft is
            ### located in. the aircraft should be in one of them because we
            ### have previously ascertained that the aircraft is in the sector
            found_idx = -1
            airspace = simulator_env.airspace
            _sector = airspace.sectors[airspace_sector]
            main_volumes = _sector.volumes
            cond_volumes = _sector.get_conditional_volumes_for_aircraft(aircraft)
            cond_volumes = list(cond_volumes.values())
            volumes = main_volumes + cond_volumes

            for i, volume in enumerate(volumes):
                if volume.contains(aircraft):
                    found_idx = i
                    break
            min_fl = volumes[found_idx].min_fl
            max_fl = volumes[found_idx].max_fl

            fl_diff_ac_selected_sector_min = aircraft.selected_fl - min_fl
            fl_diff_ac_selected_sector_max = aircraft.selected_fl - max_fl
            fl_diff_ac_current_sector_min = aircraft.fl - min_fl
            fl_diff_ac_current_sector_max = aircraft.fl - max_fl

        else:
            dist_ac_nb_360 = -tracked_data[callsign].nearest_360_boundary_dist
            # it could be None
            dist_ac_nb_fwd = tracked_data[callsign].nearest_forward_boundary_dist
            if dist_ac_nb_fwd is None:
                # set it to an arbitrarily large feature value
                dist_ac_nb_fwd = -1e3
            else:
                dist_ac_nb_fwd *= -1.0

            fl_diff_ac_selected_sector_min = 0.0
            fl_diff_ac_selected_sector_max = 0.0
            fl_diff_ac_current_sector_min = 0.0
            fl_diff_ac_current_sector_max = 0.0

        ####### base features: the current aircraft state
        base_feats = [
            np.clip(fl_diff_ac_exit, -SRC.CLIP_FL_DIFF, SRC.CLIP_FL_DIFF) / SRS.SCALER_FL_DIFF,
            np.clip(fl_diff_ac_exit_selected, -SRC.CLIP_FL_DIFF, SRC.CLIP_FL_DIFF) / SRS.SCALER_FL_DIFF,
            np.clip(fl_diff_ac_entry_exit, -SRC.CLIP_FL_DIFF, SRC.CLIP_FL_DIFF) / SRS.SCALER_FL_DIFF,
            np.clip(aircraft.speed_tas, 0.0, SRC.CLIP_SPEED) / SRS.SCALER_SPEED,
            ac_centre_dist_fr / SRS.SCALER_CENTRELINE_DIST,  # already clipped
            ac_centre_dist_cr / SRS.SCALER_CENTRELINE_DIST,  # already clipped
            np.clip(
                dist_diff_exit_pos_level,
                -SRC.CLIP_DIST_DIFF,
                SRC.CLIP_DIST_DIFF,
            )
            / SRS.SCALER_DIST_DIFF,
            np.clip(dist_diff_prev_curr_to_exit, -3.0, 3.0),  # no scaling
            np.clip(
                dist_to_sector_entry,
                -SRC.CLIP_INCOMM_DIST,
                SRC.CLIP_INCOMM_DIST,
            )
            / SRS.SCALER_INCOMM_DIST,
            np.clip(
                dist_away_sector_exit,
                -SRC.CLIP_OUTCOMM_DIST,
                SRC.CLIP_OUTCOMM_DIST,
            )
            / SRS.SCALER_OUTCOMM_DIST,
            route_ff_status,  # a discrete feature, no scale/clip required
            pos_status,  # a discrete feature, no scale/clip required
            angle_diff_ac_nb_360 * convert.DEG_TO_RAD,
            angle_diff_ac_nb_360_sh * convert.DEG_TO_RAD,
            np.clip(dist_ac_nb_360, -SRC.CLIP_AC_NB_DIST, SRC.CLIP_AC_NB_DIST) / SRS.SCALER_AC_NB_DIST,
            np.clip(dist_ac_nb_fwd, -SRC.CLIP_AC_NB_DIST, SRC.CLIP_AC_NB_DIST) / SRS.SCALER_AC_NB_DIST,
            np.clip(
                fl_diff_ac_selected_sector_min,
                -SRC.CLIP_FL_DIFF,
                SRC.CLIP_FL_DIFF,
            )
            / SRS.SCALER_FL_DIFF,
            np.clip(
                fl_diff_ac_selected_sector_max,
                -SRC.CLIP_FL_DIFF,
                SRC.CLIP_FL_DIFF,
            )
            / SRS.SCALER_FL_DIFF,
            np.clip(
                fl_diff_ac_current_sector_min,
                -SRC.CLIP_FL_DIFF,
                SRC.CLIP_FL_DIFF,
            )
            / SRS.SCALER_FL_DIFF,
            np.clip(
                fl_diff_ac_current_sector_max,
                -SRC.CLIP_FL_DIFF,
                SRC.CLIP_FL_DIFF,
            )
            / SRS.SCALER_FL_DIFF,
        ]
        base_feats = np.asarray(base_feats, dtype=np.float32)

        ####### fixes features
        fixes_feats = self.generate_forward_fixes_features(gym_env, callsign)

        ####### neighbours features
        neighbours_feats = self.generate_neighbours_features(gym_env, callsign)

        feats_list = [base_feats, *fixes_feats, *neighbours_feats]
        return np.concatenate(feats_list, dtype=np.float32)

    def generate_forward_fixes_features(self, gym_env: BaseEnv, callsign: str) -> list[npt.NDArray[np.float32]]:
        """Generate features for N forward fixes.

        The forward fixes are derived using the aircraft's filed route
        if `.use_filed_route` is `True`. Otherwise, the current route is used.

        Args:
            gym_env: defines the gymnasium environment.
            callsign: the identifier of the aircraft.

        Returns:
            list of representations for each fix or an empty list if
            `.num_forward_fixes` is 0.
        """

        if self.num_forward_fixes == 0:
            return []

        # features related to aircraft's relationship to N fixes. For each
        # fix, compute:
        # - relative angular difference between aircraft heading and the
        #   bearing of the aircraft position to the next fix (current route)
        # - relative angular difference between aircraft selected heading and
        #   the bearing of the aircraft position to the next fix (current
        #   route)
        # - relative angular difference between aircraft heading and the
        #   bearing from the previous fix to the next fix (current route)
        # - relative angular difference between aircraft selected heading and
        #   the bearing from the previous fix to the next fix (current route)
        # - aircraft to next fix linear distance
        # - aircraft to next fix route/track distance
        # - flag: indicates whether the linear segment from aircraft's
        #         position to fix position is within the sector, or crosses
        #         out of the sector.

        simulator_env = gym_env.get_simulator_env()
        tracked_data = gym_env.get_tracked_aircraft_data()
        airspace_sector = gym_env.get_active_airspace_sector()
        aircraft = simulator_env.aircraft[callsign]
        airspace = simulator_env.airspace

        ####### utils: get useful information for computing fixes features
        # forward fixes position: in *current route* (cr)
        route = aircraft.flight_plan.route.current
        _next_fix = tracked_data[callsign].next_fix_cr
        fixes = get_n_forward_fixes(route, start_from=_next_fix, n=self.num_forward_fixes)
        next_fixes_pos = [simulator_env.airspace.fixes.places[fix] for fix in fixes if fix is not None]
        num_none_fixes = self.num_forward_fixes - len(next_fixes_pos)

        # previous fixes position (relative to each forward fix): in
        # *current route* (cr)
        _next_fix_idx = route.index(_next_fix)
        if _next_fix_idx == 0:
            prev_fixes_pos = [tracked_data[callsign].pos_at_last_route_direct, *next_fixes_pos[:-1]]
        else:
            _prev_fix = route[_next_fix_idx - 1]
            prev_fixes_pos = [simulator_env.airspace.fixes.places[_prev_fix], *next_fixes_pos[:-1]]

        # generate the features for each forward fix
        curr_route_start_pos = tracked_data[callsign].pos_at_last_route_direct
        ac_pos = aircraft.pos2d()

        angles_diff_ac_nfs = []
        angles_diff_ac_sh_nfs = []
        angles_diff_ac_pf_nfs = []
        angles_diff_ac_sh_pf_nfs = []
        linear_dists_ac_nfs = []
        track_dists_ac_nfs = []
        in_sector_segments_ac_nfs = []
        for prev_fix_pos, fix_pos in zip(prev_fixes_pos, next_fixes_pos, strict=True):
            # relative angular diff b/w heading and bearing to next fix
            bearing_ac_nf = ac_pos.bearing_to(fix_pos)
            angle_diff_ac_nf = angle_diff(aircraft.heading, bearing_ac_nf)
            turn_dir = left_right_check(aircraft.heading, bearing_ac_nf)
            angle_diff_ac_nf *= turn_dir
            angles_diff_ac_nfs.append(angle_diff_ac_nf)

            # relative angular diff b/w selected heading and bearing to next fix
            angle_diff_ac_sh_nf = angle_diff(_selected_heading(aircraft), bearing_ac_nf)
            turn_dir = left_right_check(_selected_heading(aircraft), bearing_ac_nf)
            angle_diff_ac_sh_nf *= turn_dir
            angles_diff_ac_sh_nfs.append(angle_diff_ac_sh_nf)

            # relative angular diff b/w heading and
            # bearing from prev to next fix
            bearing_pf_nf = prev_fix_pos.bearing_to(fix_pos)
            angle_diff_ac_pf_nf = angle_diff(aircraft.heading, bearing_pf_nf)
            turn_dir = left_right_check(aircraft.heading, bearing_pf_nf)
            angle_diff_ac_pf_nf *= turn_dir
            angles_diff_ac_pf_nfs.append(angle_diff_ac_pf_nf)

            # relative angular diff b/w selected heading and
            # bearing from prev to next fix
            angle_diff_ac_sh_pf_nf = angle_diff(_selected_heading(aircraft), bearing_pf_nf)
            turn_dir = left_right_check(_selected_heading(aircraft), bearing_pf_nf)
            angle_diff_ac_sh_pf_nf *= turn_dir
            angles_diff_ac_sh_pf_nfs.append(angle_diff_ac_sh_pf_nf)

            # aircraft to next fix linear distance
            linear_dists_ac_nfs.append(aircraft.pos2d().distance(fix_pos))

            # aircraft to next fix route/track distance
            _dist = distance_to_target_pos_along_route(
                aircraft.pos2d(),
                fix_pos,
                aircraft.flight_plan.route.current,
                airspace,
                curr_route_start_pos,
            )
            _dist = _dist if _dist > 0.0 else 0.0
            track_dists_ac_nfs.append(_dist)

            # segment in sector flag
            segment: Line = (ac_pos, fix_pos)
            # setting the epsilon to 0.015 helps to deal with issues of
            # entry/exit fixes being classed as being outside the sector
            # due to the low tolerance value of 1e-10.
            epsilon = 1e-10
            ret = segment_in_sector_interp(
                segment,
                airspace,
                airspace_sector,
                epsilon=epsilon,
                num_interp_positions=5,
            )
            in_sector_segments_ac_nfs.append(1.0 if ret else -1.0)

        ####### fixes features
        fixes_feats = []
        for idx in range(len(next_fixes_pos)):
            tmp = np.asarray(
                [
                    angles_diff_ac_nfs[idx] * convert.DEG_TO_RAD,
                    angles_diff_ac_sh_nfs[idx] * convert.DEG_TO_RAD,
                    angles_diff_ac_pf_nfs[idx] * convert.DEG_TO_RAD,
                    angles_diff_ac_sh_pf_nfs[idx] * convert.DEG_TO_RAD,
                    (np.clip(linear_dists_ac_nfs[idx], 0.0, SRC.CLIP_DIST) / SRS.SCALER_AC_NF_DIST),
                    (np.clip(track_dists_ac_nfs[idx], 0.0, SRC.CLIP_DIST) / SRS.SCALER_AC_NF_DIST),
                    # a discrete feature, no scale/clip required
                    in_sector_segments_ac_nfs[idx],
                ],
                dtype=np.float32,
            )

            fixes_feats.append(tmp)

        # zero padding
        for _ in range(num_none_fixes):
            tmp = np.zeros(self.num_features_per_fix, dtype=np.float32)
            fixes_feats.append(tmp)

        return fixes_feats

    def generate_neighbours_features(self, gym_env: BaseEnv, callsign: str) -> list[npt.NDArray[np.float32]]:
        """Generate features for N neighbour aircraft.

        Args:
            gym_env: defines the gymnasium environment.
            callsign: the identifier of the aircraft.

        Returns:
            list of representations for each neighbour or an empty list if
            `.knn` is 0.
        """

        if self.knn == 0:
            return []

        tracked_data = gym_env.get_tracked_aircraft_data()

        ####### utils: get useful information for neighbour features
        # sorted based on aircraft distance to other aircraft
        interactions = self._knn_repr(callsign, gym_env, True)

        _num_other_ac_retrieved = len(interactions)
        if _num_other_ac_retrieved < self.knn:
            add_count = _num_other_ac_retrieved
            balance_count = self.knn - _num_other_ac_retrieved
        else:
            add_count = self.knn
            balance_count = 0

        ####### neighbours features
        interactions_features = []
        for i in range(add_count):
            callsign_other = interactions[i].other_callsign
            dist_ac_other = interactions[i].dist_ac_other
            angle_diff_ac_other = interactions[i].angle_diff_ac_other
            turn_dir_ac_other = interactions[i].turn_dir_ac_other
            angle_diff_ac_other_sh = interactions[i].angle_diff_ac_other_sh
            turn_dir_ac_other_sh = interactions[i].turn_dir_ac_other_sh
            dist_type_ac_other = interactions[i].dist_type_ac_other
            fl_diff_ac_other = interactions[i].fl_diff_ac_other
            selected_fl_diff_ac_other = interactions[i].selected_fl_diff_ac_other
            speed_diff_ac_other = interactions[i].speed_diff_ac_other
            pair_distance = interactions[i].centreline_dist_diff_cr

            controllable = None
            _cs = callsign_other
            if _cs not in tracked_data:
                controllable = -1.0
            elif (
                tracked_data[_cs].pos_status == PositionStatus.IN_SECTOR
                and tracked_data[_cs].incomm_status is True
                and tracked_data[_cs].outcomm_status is False
            ):
                controllable = 1.0
            else:
                controllable = -1.0

            tmp = np.asarray(
                [
                    (angle_diff_ac_other * convert.DEG_TO_RAD * turn_dir_ac_other),
                    (angle_diff_ac_other_sh * convert.DEG_TO_RAD * turn_dir_ac_other_sh),
                    dist_type_ac_other,
                    np.clip(dist_ac_other, 0.0, SRC.CLIP_DIST) / SRS.SCALER_AC_OTHER_DIST,
                    np.clip(
                        fl_diff_ac_other,
                        -SRC.CLIP_FL_DIFF,
                        SRC.CLIP_FL_DIFF,
                    )
                    / SRS.SCALER_FL_DIFF,
                    np.clip(
                        selected_fl_diff_ac_other,
                        -SRC.CLIP_FL_DIFF,
                        SRC.CLIP_FL_DIFF,
                    )
                    / SRS.SCALER_FL_DIFF,
                    np.clip(
                        speed_diff_ac_other,
                        -SRC.CLIP_SPEED_DIFF,
                        SRC.CLIP_SPEED_DIFF,
                    )
                    / SRS.SCALER_SPEED_DIFF,
                    controllable,  # a discrete feature, no scale/clip required
                    np.clip(pair_distance, 0.0, SRC.CLIP_CENTRELINE_DIST_DIFF) / SRS.SCALER_CENTRELINE_DIST_DIFF,
                ],
                dtype=np.float32,
            )
            interactions_features.append(tmp)

        # zero padding
        for _ in range(balance_count):
            tmp = np.zeros(self.num_features_per_neighbour, dtype=np.float32)
            interactions_features.append(tmp)

        return interactions_features
