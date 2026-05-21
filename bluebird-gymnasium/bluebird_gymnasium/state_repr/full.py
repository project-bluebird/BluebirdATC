from __future__ import annotations

import numpy as np
from bluebird_dt.utility import convert

from bluebird_gymnasium.envs.base import BaseEnv
from bluebird_gymnasium.state_repr import (
    StateReprClipper as SRC,
    StateReprScaler as SRS,
)
from bluebird_gymnasium.state_repr.base import BaseRepresentation
from bluebird_gymnasium.utils.simulator_utils import prev_next_fixes
from bluebird_gymnasium.utils.types import PositionStatus
from bluebird_gymnasium.utils.simulator_utils import get_n_forward_fixes

import typing

if typing.TYPE_CHECKING:
    import numpy.typing as npt


class FullRepresentationRaw(BaseRepresentation):
    """Representation of aircraft state in the simulator.

    Raw feature values

    +-----+-------------------------------+------+------+---------------------+
    | Num |         Observation           | Min  | Max  |        Unit         |
    +-----+-------------------------------+------+------+---------------------+
    |   0 | Aircraft entry fix latitude   | -pi/2| +pi/2| Angle (radians)     |
    |   1 | Aircraft entry fix longitude  | -pi  | +pi  | Angle (radians)     |
    |   2 | Aircraft exit fix latitude    | -pi/2| +pi/2| Angle (radians)     |
    |   3 | Aircraft exit fix longitude   | -pi  | +pi  | Angle (radians)     |
    |   4 | Aircraft position latitude    | -pi/2| +pi/2| Angle (radians)     |
    |   5 | Aircraft position longitude   | -pi  | +pi  | Angle (radians)     |
    |   6 | Aircraft heading              |  0.0 |  2*pi| Angle (radians)     |
    |   7 | Aircraft entry flight level   |  0.0 |  inf | Flight level        |
    |   8 | Aircraft exit flight level    |  0.0 |  inf | Flight level        |
    |   9 | Aircraft current  flight level|  0.0 |  inf | Flight level        |
    |  10 | Aircraft selected flight level|  0.0 |  inf | Flight level        |
    |  11 | Aircraft true air speed (tas) |  0.0 |  inf | Nautical miles per  |
    |     |                               |      |      | second              |
    |  12 | Aircraft distance from the    | -inf |  inf | Nautical Miles (NM) |
    |     | filed route centreline        |      |      |                     |
    |  13 | Aircraft route following (ff) | -1.0 |  1.0 | Discrete:           |
    |     | status/flag                   |      |      | -1 no route ff.     |
    |     |                               |      |      |  1 route ff.        |
    |  14 | Aircraft nearest boundary     | -pi/2| +pi/2| Angle (radians)     |
    |     | point, latitude (360 deg look)|      |      |                     |
    |  15 | Aircraft nearest boundary     | -pi  | +pi  | Angle (radians)     |
    |     | point, longitude(360 deg look)|      |      |                     |
    |  16 | Minimum flight level in the   |  0.0 |  inf | Flight level        |
    |     | volume the aircraft is located|      |      |                     |
    |     | in the sector                 |      |      |                     |
    |  17 | Maximum flight level in the   |  0.0 |  inf | Flight level        |
    |     | volume the aircraft is located|      |      |                     |
    |     | in the sector                 |      |      |                     |
    |  18 | Aircraft previous fix latitude| -pi/2| +pi/2| Angle (radians)     |
    |  19 | Aircraft previous fix long.   | -pi  | +pi  | Angle (radians)     |
    +-----+-------------------------------+------+------+---------------------+

    If num_forward_fixes > 0, then for each forward fix, the following
    representation is concatenated to the state:
    +-----+-------------------------------+------+------+---------------------+
    | Num |         Observation           | Min  | Max  |        Unit         |
    +-----+-------------------------------+------+------+---------------------+
    |   0 | Aircraft next fix latitude    | -pi/2| +pi/2| Angle (radians)     |
    |   1 | Aircraft next fix longitude   | -pi  | +pi  | Angle (radians)     |
    |   2 | Previous fix (of aircraft)    |  0.0 |  2*pi| Angle (radians)     |
    |     | bearing to next fix           |      |      |                     |
    |   3 | Aircraft bearing to next fix  |  0.0 |  2*pi| Angle (radians)     |
    +-----+-------------------------------+------+------+---------------------+

    If knn > 0, then for each nearby aircraft, the following representation is
    concatenated to the state:
    +-----+------------------------------+------+------+---------------------+
    | Num |         Observation          | Min  | Max  |        Unit         |
    +-----+------------------------------+------+------+---------------------+
    |   0 | Other aircraft latitude (lat)| -pi/2| +pi/2| Angle (radians)     |
    |   1 | Other aircraft longitude(lon)| -pi  | +pi  | Angle (radians)     |
    |   2 | Other aircraft next fix lat  | -pi/2| +pi/2| Angle (radians)     |
    |   3 | Other aircraft next fix lon  | -pi  | +pi  | Angle (radians)     |
    |   4 | Other aircraft heading       |  0.0 | 2*pi | Angle (radians)     |
    |   5 | Other aircraft flight level  |  0.0 | inf  | Flight level        |
    |   6 | Other aircraft speed         |  0.0 | inf  | Nautical miles per  |
    |     |                              |      |      | second              |
    |   7 | Other aircraft               | -1.0 | 1.0  | Discete             |
    |     | controllability status/flag  |      |      | -1 no route ff.     |
    |     |                              |      |      |  1 route ff.        |
    +-----+------------------------------+------+------+---------------------+

    Args:
        knn: the number of nearest aircraft to consider when representing
            state for an aircraft. Defaults to 0.
        num_forward_fixes: the number of forward fixes to encode in the state
            representation of the aircraft. Defaults to 1.
        use_filed_route: specifies whether to get the forward fixes from the
            aircraft's filed (`True`) or current (`False`) route.
            Defaults to `True`.
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
        super(FullRepresentationRaw, self).__init__(knn, num_forward_fixes, use_filed_route, num_actions)

        ####### base features range
        base_feats_low = [
            -np.pi / 2,
            -np.pi,
            -np.pi / 2,
            -np.pi,
            -np.pi / 2,
            -np.pi,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -np.inf,
            -1.0,
            -np.pi / 2,
            -np.pi,
            0.0,
            0.0,
            -np.pi / 2,
            -np.pi,
        ]
        base_feats_high = [
            np.pi / 2,
            np.pi,
            np.pi / 2,
            np.pi,
            np.pi / 2,
            np.pi,
            2 * np.pi,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            1.0,
            np.pi / 2,
            np.pi,
            np.inf,
            np.inf,
            np.pi / 2,
            np.pi,
        ]
        self.num_features_base = len(base_feats_low)

        ####### fixes features range
        if self.num_forward_fixes > 0:
            fixes_feats_low = [
                -np.pi / 2,
                -np.pi,
                0.0,
                0.0,
            ]
            fixes_feats_high = [
                np.pi / 2,
                np.pi,
                2 * np.pi,
                2 * np.pi,
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
                -np.pi / 2,
                -np.pi,
                -np.pi / 2,
                -np.pi,
                0.0,
                0.0,
                0.0,
                -1.0,
            ]
            neighbours_feats_high = [
                np.pi / 2,
                np.pi,
                np.pi / 2,
                np.pi,
                2 * np.pi,
                np.inf,
                np.inf,
                1.0,
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

    def repr(self, gym_env: BaseEnv, callsign: str) -> np.ndarray:
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
        airspace_sector = gym_env.get_active_airspace_sector()
        aircraft = simulator_env.aircraft[callsign]

        ####### utils: get useful information for computing base features
        # entry fix
        entry_fix = tracked_data[callsign].entry_coords[airspace_sector].fix
        entry_fix_pos = simulator_env.airspace.fixes.places[entry_fix]

        # exit fix
        exit_fix = tracked_data[callsign].exit_coords[airspace_sector].fix
        exit_fix_pos = simulator_env.airspace.fixes.places[exit_fix]

        # entry and exit flight level
        entry_fl = tracked_data[callsign].entry_coords[airspace_sector].fl
        exit_fl = tracked_data[callsign].exit_coords[airspace_sector].fl

        # centreline distance
        ac_centre_dist, turn_dir, _ = tracked_data[callsign].centreline_info_fr
        ac_centre_dist *= turn_dir

        # aircraft route following status: indicates whether or not the
        # aircraft is following its defined flight plan route, or not (in this
        # case, flying on a heading issued by the agent).
        if aircraft.on_route is True:
            route_ff_status = 1.0
        else:
            route_ff_status = -1.0

        # position of the nearest boundary point of the sector to the aircraft
        pos2d_ac_nb = tracked_data[callsign].nearest_360_boundary_pos

        if tracked_data[callsign].pos_status == PositionStatus.IN_SECTOR:
            ### find the volume of the sector that the aircraft is
            ### located in. the aircraft should be in one of them because we
            ### have previously ascertained that the aircraft is in the sector
            found_idx = -1
            airspace = simulator_env.airspace
            for i, volume in enumerate(airspace.sectors[airspace_sector].volumes):
                if volume.contains(aircraft):
                    found_idx = i
                    break
            min_fl = airspace.sectors[airspace_sector].volumes[found_idx].min_fl
            max_fl = airspace.sectors[airspace_sector].volumes[found_idx].max_fl

        else:
            min_fl = 0.0
            max_fl = 0.0

        # previous fix position
        prev_fix = tracked_data[callsign].previous_fix_fr
        prev_fix_pos = simulator_env.airspace.fixes.places[prev_fix]

        ####### base features: the current aircraft state
        base_feats = np.asarray(
            [
                entry_fix_pos.lat * convert.DEG_TO_RAD,
                entry_fix_pos.lon * convert.DEG_TO_RAD,
                exit_fix_pos.lat * convert.DEG_TO_RAD,
                exit_fix_pos.lon * convert.DEG_TO_RAD,
                aircraft.pos2d().lat * convert.DEG_TO_RAD,
                aircraft.pos2d().lon * convert.DEG_TO_RAD,
                aircraft.heading * convert.DEG_TO_RAD,
                entry_fl,
                exit_fl,
                aircraft.fl,
                aircraft.selected_fl,
                aircraft.speed_tas,
                ac_centre_dist,
                route_ff_status,
                pos2d_ac_nb.lat * convert.DEG_TO_RAD,
                pos2d_ac_nb.lon * convert.DEG_TO_RAD,
                min_fl,
                max_fl,
                prev_fix_pos.lat * convert.DEG_TO_RAD,
                prev_fix_pos.lon * convert.DEG_TO_RAD,
            ],
            dtype=np.float32,
        )

        ####### fixes features
        fixes_feats = self.generate_forward_fixes_features(gym_env, callsign)

        ####### neighbours features
        neighbours_feats = self.generate_neighbours_features(gym_env, callsign)

        feats_list = [base_feats] + fixes_feats + neighbours_feats
        return np.concatenate(feats_list, dtype=np.float32)

    def generate_forward_fixes_features(self, gym_env, callsign) -> list[npt.NDArray[np.float32]]:
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

        simulator_env = gym_env.get_simulator_env()
        tracked_data = gym_env.get_tracked_aircraft_data()
        aircraft = simulator_env.aircraft[callsign]

        ####### utils: get useful information for computing fixes features
        # forward fixes position: in *filed route* (fr)
        route = aircraft.flight_plan.route.filed
        prev_fix = tracked_data[callsign].previous_fix_fr
        prev_fix_pos = simulator_env.airspace.fixes.places[prev_fix]

        _next_fix = tracked_data[callsign].next_fix_fr
        fixes = get_n_forward_fixes(route, start_from=_next_fix, n=self.num_forward_fixes)
        next_fixes_pos = [simulator_env.airspace.fixes.places[fix] for fix in fixes if fix is not None]
        num_none_fixes = self.num_forward_fixes - len(next_fixes_pos)

        ac_pos = aircraft.pos2d()
        bearings_pf_nf = []
        bearings_ac_nf = []
        prev_pos = prev_fix_pos  # initialise prev_pos
        for fix_pos in next_fixes_pos:
            # bearing from previous fix (pf) to next_fix (nf)
            bearings_pf_nf.append(prev_pos.bearing_to(fix_pos))
            prev_pos = fix_pos

            # bearing of aircraft position to the next fix
            bearings_ac_nf.append(ac_pos.bearing_to(fix_pos))

        ####### fixes features
        fixes_feats = []
        for idx in range(len(next_fixes_pos)):
            tmp = np.asarray(
                [
                    next_fixes_pos[idx].lat * convert.DEG_TO_RAD,
                    next_fixes_pos[idx].lon * convert.DEG_TO_RAD,
                    bearings_pf_nf[idx] * convert.DEG_TO_RAD,
                    bearings_ac_nf[idx] * convert.DEG_TO_RAD,
                ],
                dtype=np.float32,
            )
            fixes_feats.append(tmp)

        # zero padding
        for idx in range(num_none_fixes):
            tmp = np.zeros(self.num_features_per_fix, dtype=np.float32)
            fixes_feats.append(tmp)

        return fixes_feats

    def generate_neighbours_features(self, gym_env, callsign) -> list[npt.NDArray[np.float32]]:
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

        simulator_env = gym_env.get_simulator_env()
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
            other_callsign = interactions[i].other_callsign
            other_aircraft = simulator_env.aircraft[other_callsign]

            if (
                other_callsign in tracked_data.keys()
                and tracked_data[other_callsign].previous_fix_fr is not None
                and tracked_data[other_callsign].next_fix_fr is not None
            ):
                _prev_fix = tracked_data[other_callsign].previous_fix_fr
                _next_fix = tracked_data[other_callsign].next_fix_fr
            else:
                ret = prev_next_fixes(other_callsign, simulator_env)
                _prev_fix, _next_fix = ret
            other_next_fix_pos = simulator_env.airspace.fixes.places[_next_fix]

            controllable = None
            _pos_status = tracked_data[other_callsign].pos_status
            if other_callsign not in tracked_data.keys():
                controllable = -1.0
            elif (
                _pos_status == PositionStatus.IN_SECTOR
                and tracked_data[other_callsign].incomm_status is True
                and tracked_data[other_callsign].outcomm_status is False
            ):
                controllable = 1.0
            else:
                controllable = -1.0

            tmp = np.asarray(
                [
                    other_aircraft.pos2d().lat * convert.DEG_TO_RAD,
                    other_aircraft.pos2d().lon * convert.DEG_TO_RAD,
                    other_next_fix_pos.lat * convert.DEG_TO_RAD,
                    other_next_fix_pos.lon * convert.DEG_TO_RAD,
                    other_aircraft.heading * convert.DEG_TO_RAD,
                    other_aircraft.fl,
                    other_aircraft.speed_tas,
                    controllable,
                ],
                dtype=np.float32,
            )
            interactions_features.append(tmp)

        # zero padding
        for i in range(balance_count):
            tmp = np.zeros(self.num_features_per_neighbour, dtype=np.float32)
            interactions_features.append(tmp)

        return interactions_features


class FullRepresentation(BaseRepresentation):
    """Representation of aircraft state in the simulator.

    Scaled and clipped version, using handpicked (manual) metrics.

    +-----+-------------------------------+------+------+---------------------+
    | Num |         Observation           | Min  | Max  |        Unit         |
    +-----+-------------------------------+------+------+---------------------+
    |   0 | Aircraft entry fix latitude   | -pi/2| +pi/2| Angle (radians)     |
    |   1 | Aircraft entry fix longitude  | -pi  | +pi  | Angle (radians)     |
    |   2 | Aircraft exit fix latitude    | -pi/2| +pi/2| Angle (radians)     |
    |   3 | Aircraft exit fix longitude   | -pi  | +pi  | Angle (radians)     |
    |   4 | Aircraft position latitude    | -pi/2| +pi/2| Angle (radians)     |
    |   5 | Aircraft position longitude   | -pi  | +pi  | Angle (radians)     |
    |   6 | *Aircraft heading             | -pi  | +pi  | Angle (radians)     |
    |   7 | Aircraft entry flight level   |  0.0 |  3.0 | Flight level        |
    |     |                               |      |      | (scaled by 160)     |
    |   8 | Aircraft exit flight level    |  0.0 |  3.0 | Flight level        |
    |     |                               |      |      | (scaled by 160)     |
    |   9 | Aircraft current flight level |  0.0 |  3.0 | Flight level        |
    |     |                               |      |      | (scaled by 160)     |
    |  10 | Aircraft selected flight level|  0.0 |  3.0 | Flight level        |
    |     |                               |      |      | (scaled by 160)     |
    |  11 | Aircraft true air speed (tas) |  0.0 |  3.0 | Nautical miles per  |
    |     |                               |      |      | second              |
    |     |                               |      |      | (scaled by 200)     |
    |  12 | Aircraft distance from the    | -3.0 |  3.0 | Nautical Miles (NM) |
    |     | filed route centreline        |      |      |                     |
    |  13 | Aircraft route following (ff) | -1.0 |  1.0 | Discrete:           |
    |     | status/flag                   |      |      | -1 no route ff.     |
    |     |                               |      |      |  1 route ff.        |
    |  14 | Aircraft nearest boundary     | -pi/2| +pi/2| Angle (radians)     |
    |     | point, latitude (360 deg look)|      |      |                     |
    |  15 | Aircraft nearest boundary     | -pi  | +pi  | Angle (radians)     |
    |     | point, longitude(360 deg look)|      |      |                     |
    |  16 | Minimum flight level in the   |  0.0 |  3.0 | Flight level        |
    |     | volume the aircraft is located|      |      |                     |
    |     | in the sector                 |      |      |                     |
    |  17 | Maximum flight level in the   |  0.0 |  3.0 | Flight level        |
    |     | volume the aircraft is located|      |      |                     |
    |     | in the sector                 |      |      |                     |
    |  18 | Aircraft previous fix latitude| -pi/2| +pi/2| Angle (radians)     |
    |  19 | Aircraft previous fix long.   | -pi  | +pi  | Angle (radians)     |
    +-----+-------------------------------+------+------+---------------------+

    If num_forward_fixes > 0, then for each forward fix, the following
    representation is concatenated to the state:
    +-----+-------------------------------+------+------+---------------------+
    | Num |         Observation           | Min  | Max  |        Unit         |
    +-----+-------------------------------+------+------+---------------------+
    |   0 | Aircraft next fix latitude    | -pi/2| +pi/2| Angle (radians)     |
    |   1 | Aircraft next fix longitude   | -pi  | +pi  | Angle (radians)     |
    |   2 | *Previous fix (of aircraft)   | -pi  | +pi  | Angle (radians)     |
    |     | bearing to next fix           |      |      |                     |
    |   3 | *Aircraft bearing to next fix | -pi  | +pi  | Angle (radians)     |
    +-----+-------------------------------+------+------+---------------------+

    If knn > 0, then for each nearby aircraft, the following representation is
    concatenated to the state:
    +-----+------------------------------+------+------+---------------------+
    | Num |         Observation          | Min  | Max  |        Unit         |
    +-----+------------------------------+------+------+---------------------+
    |   0 | Other aircraft latitude (lat)| -pi/2| +pi/2| Angle (radians)     |
    |   1 | Other aircraft longitude(lon)| -pi  | +pi  | Angle (radians)     |
    |   2 | Other aircraft next fix lat  | -pi/2| +pi/2| Angle (radians)     |
    |   3 | Other aircraft next fix lon  | -pi  | +pi  | Angle (radians)     |
    |   4 | *Other aircraft heading      | -pi  | +pi  | Angle (radians)     |
    |   5 | Other aircraft flight level  |  0.0 | 3.0  | Flight level        |
    |     |                              |      |      | (scaled by 160)     |
    |   6 | Other aircraft speed         |  0.0 | 3.0  | Nautical miles per  |
    |     |                              |      |      | second              |
    |     |                              |      |      | (scaled by 200)     |
    |   7 | Other aircraft               | -1.0 | 1.0  | Discete             |
    |     | controllability status/flag  |      |      | -1 no route ff.     |
    |     |                              |      |      |  1 route ff.        |
    +-----+------------------------------+------+------+---------------------+

    Note:
    * denote that an angular feature that is naturally of range [0, 2pi]
      but scaled to [-pi, pi].

    Args:
        knn: the number of nearest aircraft to consider when representing
            state for an aircraft. Defaults to 0.
        num_forward_fixes: the number of forward fixes to encode in the state
            representation of the aircraft. Defaults to 1.
        use_filed_route: specifies whether to get the forward fixes from the
            aircraft's filed (`True`) or current (`False`) route.
            Defaults to `True`.
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
        super(FullRepresentation, self).__init__(knn, num_forward_fixes, use_filed_route, num_actions)

        ####### base features range
        base_feats_low = [
            -np.pi / 2,
            -np.pi,
            -np.pi / 2,
            -np.pi,
            -np.pi / 2,
            -np.pi,
            -np.pi,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -3.0,
            -1.0,
            -np.pi / 2,
            -np.pi,
            0.0,
            0.0,
            -np.pi / 2,
            -np.pi,
        ]
        base_feats_high = [
            np.pi / 2,
            np.pi,
            np.pi / 2,
            np.pi,
            np.pi / 2,
            np.pi,
            np.pi,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            1.0,
            np.pi / 2,
            np.pi,
            3.0,
            3.0,
            np.pi / 2,
            np.pi,
        ]
        self.num_features_base = len(base_feats_low)

        ####### fixes features range
        if self.num_forward_fixes > 0:
            fixes_feats_low = [
                -np.pi / 2,
                -np.pi,
                -np.pi,
                -np.pi,
            ]
            fixes_feats_high = [
                np.pi / 2,
                np.pi,
                np.pi,
                np.pi,
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
                -np.pi / 2,
                -np.pi,
                -np.pi / 2,
                -np.pi,
                -np.pi,
                0.0,
                0.0,
                -1.0,
            ]
            neighbours_feats_high = [
                np.pi / 2,
                np.pi,
                np.pi / 2,
                np.pi,
                np.pi,
                3.0,
                3.0,
                1.0,
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

    def repr(self, gym_env: BaseEnv, callsign: str) -> np.ndarray:
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
        airspace_sector = gym_env.get_active_airspace_sector()
        aircraft = simulator_env.aircraft[callsign]

        ####### utils: get useful information for computing base features
        # entry fix
        entry_fix = tracked_data[callsign].entry_coords[airspace_sector].fix
        entry_fix_pos = simulator_env.airspace.fixes.places[entry_fix]

        # exit fix
        exit_fix = tracked_data[callsign].exit_coords[airspace_sector].fix
        exit_fix_pos = simulator_env.airspace.fixes.places[exit_fix]

        # entry and exit flight level
        entry_fl = tracked_data[callsign].entry_coords[airspace_sector].fl
        exit_fl = tracked_data[callsign].exit_coords[airspace_sector].fl

        # centreline distance
        ac_centre_dist, turn_dir, _ = tracked_data[callsign].centreline_info_fr
        ac_centre_dist = np.clip(ac_centre_dist, 0.0, SRC.CLIP_DIST)
        ac_centre_dist *= turn_dir

        # aircraft route following status: indicates whether or not the
        # aircraft is following its defined flight plan route, or not (in this
        # case, flying on a heading issued by the agent).
        if aircraft.on_route is True:
            route_ff_status = 1.0
        else:
            route_ff_status = -1.0

        # position of the nearest boundary point of the sector to the aircraft
        pos2d_ac_nb = tracked_data[callsign].nearest_360_boundary_pos

        if tracked_data[callsign].pos_status == PositionStatus.IN_SECTOR:
            ### find the volume of the sector that the aircraft is
            ### located in. the aircraft should be in one of them because we
            ### have previously ascertained that the aircraft is in the sector
            found_idx = -1
            airspace = simulator_env.airspace
            for i, volume in enumerate(airspace.sectors[airspace_sector].volumes):
                if volume.contains(aircraft):
                    found_idx = i
                    break
            min_fl = airspace.sectors[airspace_sector].volumes[found_idx].min_fl
            max_fl = airspace.sectors[airspace_sector].volumes[found_idx].max_fl

        else:
            min_fl = 0.0
            max_fl = 0.0

        # previous fix position
        prev_fix = tracked_data[callsign].previous_fix_fr
        prev_fix_pos = simulator_env.airspace.fixes.places[prev_fix]

        ####### base features: the current aircraft state
        base_feats = np.asarray(
            [
                entry_fix_pos.lat * convert.DEG_TO_RAD,
                entry_fix_pos.lon * convert.DEG_TO_RAD,
                exit_fix_pos.lat * convert.DEG_TO_RAD,
                exit_fix_pos.lon * convert.DEG_TO_RAD,
                aircraft.pos2d().lat * convert.DEG_TO_RAD,
                aircraft.pos2d().lon * convert.DEG_TO_RAD,
                (aircraft.heading * convert.DEG_TO_RAD) - np.pi,
                np.clip(entry_fl, 0.0, SRC.CLIP_FL) / SRS.SCALER_FL,
                np.clip(exit_fl, 0.0, SRC.CLIP_FL) / SRS.SCALER_FL,
                np.clip(aircraft.fl, 0.0, SRC.CLIP_FL) / SRS.SCALER_FL,
                np.clip(aircraft.selected_fl, 0.0, SRC.CLIP_FL) / SRS.SCALER_FL,
                np.clip(aircraft.speed_tas, 0.0, SRC.CLIP_SPEED) / SRS.SCALER_SPEED,
                ac_centre_dist / SRS.SCALER_CENTRELINE_DIST,  # already clipped
                route_ff_status,  # no need for clip/scale
                pos2d_ac_nb.lat * convert.DEG_TO_RAD,
                pos2d_ac_nb.lon * convert.DEG_TO_RAD,
                np.clip(min_fl, 0.0, SRC.CLIP_FL) / SRS.SCALER_FL,
                np.clip(max_fl, 0.0, SRC.CLIP_FL) / SRS.SCALER_FL,
                prev_fix_pos.lat * convert.DEG_TO_RAD,
                prev_fix_pos.lon * convert.DEG_TO_RAD,
            ],
            dtype=np.float32,
        )

        ####### fixes features
        fixes_feats = self.generate_forward_fixes_features(gym_env, callsign)

        ####### neighbours features
        neighbours_feats = self.generate_neighbours_features(gym_env, callsign)

        feats_list = [base_feats] + fixes_feats + neighbours_feats
        return np.concatenate(feats_list, dtype=np.float32)

    def generate_forward_fixes_features(self, gym_env, callsign) -> list[npt.NDArray[np.float32]]:
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

        simulator_env = gym_env.get_simulator_env()
        tracked_data = gym_env.get_tracked_aircraft_data()
        aircraft = simulator_env.aircraft[callsign]

        ####### utils: get useful information for computing fixes features
        # forward fixes position: in *filed route* (fr)
        route = aircraft.flight_plan.route.filed
        prev_fix = tracked_data[callsign].previous_fix_fr
        prev_fix_pos = simulator_env.airspace.fixes.places[prev_fix]

        _next_fix = tracked_data[callsign].next_fix_fr
        fixes = get_n_forward_fixes(route, start_from=_next_fix, n=self.num_forward_fixes)
        next_fixes_pos = [simulator_env.airspace.fixes.places[fix] for fix in fixes if fix is not None]
        num_none_fixes = self.num_forward_fixes - len(next_fixes_pos)

        ac_pos = aircraft.pos2d()
        bearings_pf_nf = []
        bearings_ac_nf = []
        prev_pos = prev_fix_pos  # initialise prev_pos
        for fix_pos in next_fixes_pos:
            # bearing from previous fix (pf) to next_fix (nf)
            bearings_pf_nf.append(prev_pos.bearing_to(fix_pos))
            prev_pos = fix_pos

            # bearing of aircraft position to the next fix
            bearings_ac_nf.append(ac_pos.bearing_to(fix_pos))

        ####### fixes features
        fixes_feats = []
        for idx in range(len(next_fixes_pos)):
            tmp = np.asarray(
                [
                    next_fixes_pos[idx].lat * convert.DEG_TO_RAD,
                    next_fixes_pos[idx].lon * convert.DEG_TO_RAD,
                    (bearings_pf_nf[idx] * convert.DEG_TO_RAD) - np.pi,
                    (bearings_ac_nf[idx] * convert.DEG_TO_RAD) - np.pi,
                ],
                dtype=np.float32,
            )
            fixes_feats.append(tmp)

        # zero padding
        for idx in range(num_none_fixes):
            tmp = np.zeros(self.num_features_per_fix, dtype=np.float32)
            fixes_feats.append(tmp)

        return fixes_feats

    def generate_neighbours_features(self, gym_env, callsign) -> list[npt.NDArray[np.float32]]:
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

        simulator_env = gym_env.get_simulator_env()
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
            other_callsign = interactions[i].other_callsign
            other_aircraft = simulator_env.aircraft[other_callsign]

            if (
                other_callsign in tracked_data.keys()
                and tracked_data[other_callsign].previous_fix_fr is not None
                and tracked_data[other_callsign].next_fix_fr is not None
            ):
                _prev_fix = tracked_data[other_callsign].previous_fix_fr
                _next_fix = tracked_data[other_callsign].next_fix_fr
            else:
                ret = prev_next_fixes(other_callsign, simulator_env)
                _prev_fix, _next_fix = ret
            other_next_fix_pos = simulator_env.airspace.fixes.places[_next_fix]

            controllable = None
            _pos_status = tracked_data[other_callsign].pos_status
            if other_callsign not in tracked_data.keys():
                controllable = -1.0
            elif (
                _pos_status == PositionStatus.IN_SECTOR
                and tracked_data[other_callsign].incomm_status is True
                and tracked_data[other_callsign].outcomm_status is False
            ):
                controllable = 1.0
            else:
                controllable = -1.0

            _fl = np.clip(other_aircraft.fl, 0.0, SRC.CLIP_FL)
            _speed_tas = np.clip(other_aircraft.speed_tas, 0.0, SRC.CLIP_SPEED)
            tmp = np.asarray(
                [
                    other_aircraft.pos2d().lat * convert.DEG_TO_RAD,
                    other_aircraft.pos2d().lon * convert.DEG_TO_RAD,
                    other_next_fix_pos.lat * convert.DEG_TO_RAD,
                    other_next_fix_pos.lon * convert.DEG_TO_RAD,
                    (other_aircraft.heading * convert.DEG_TO_RAD) - np.pi,
                    _fl / SRS.SCALER_FL,
                    _speed_tas / SRS.SCALER_SPEED,
                    controllable,
                ],
                dtype=np.float32,
            )
            interactions_features.append(tmp)

        # zero padding
        for i in range(balance_count):
            tmp = np.zeros(self.num_features_per_neighbour, dtype=np.float32)
            interactions_features.append(tmp)

        return interactions_features
