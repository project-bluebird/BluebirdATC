from __future__ import annotations

import numpy as np
from bluebird_dt.utility import convert

from bluebird_gymnasium.envs.base import BaseEnv
from bluebird_gymnasium.state_repr import (
    StateReprClipper as SRC,
    StateReprScaler as SRS,
)
from bluebird_gymnasium.state_repr.base import BaseRepresentation
from bluebird_gymnasium.utils.geo_utils import angle_diff, left_right_check
from bluebird_gymnasium.utils.simulator_utils import get_n_forward_fixes

import typing

if typing.TYPE_CHECKING:
    import numpy.typing as npt


class ExtraMinimalRepresentationRaw(BaseRepresentation):
    """Representation of aircraft state in the simulator.

    Raw feature values

    +-----+-------------------------------+-----+------+---------------------+
    | Num |         Observation           | Min | Max  |        Unit         |
    +-----+-------------------------------+-----+------+---------------------+
    |   0 | Aircraft distance from the    |-inf | inf  | Nautical Miles (NM) |
    |     | filed route centreline        |     |      |                     |
    +-----+-------------------------------+-----+------+---------------------+

    If num_forward_fixes > 0, then for each forward fix, the following
    representation is concatenated to the state:
    +-----+-------------------------------+-----+------+---------------------+
    | Num |         Observation           | Min | Max  |        Unit         |
    +-----+-------------------------------+-----+------+---------------------+
    |   0 | Relative angular difference   |-pi  | pi   | Angle (radians)     |
    |     | between aircraft selected     |     |      |                     |
    |     | heading and the bearing from  |     |      |                     |
    |     | aircraft position to the next |     |      |                     |
    |     | fix in the filed route        |     |      |                     |
    +-----+-------------------------------+-----+------+---------------------+

    If knn > 0, then for each nearby aircraft, the following representation is
    concatenated to the state:

    +-----+-------------------------------+-----+-----+---------------------+
    | Num |         Observation           | Min | Max |        Unit         |
    +-----+-------------------------------+-----+-----+---------------------+
    |   0 | Relative angular (non-reflex) | -pi | +pi | Angle (radians)     |
    |     | difference between aircraft   |     |     |                     |
    |     | heading and the heading of the|     |     |                     |
    |     | neighbour aircraft, including |     |     |                     |
    |     | the direction of turn (sign)  |     |     |                     |
    |   1 | Distance between aircraft     | 0.0 | inf | Nautical Miles (NM) |
    |     | and neighbour aircraft        |     |     |                     |
    +-----+------------------------------ +-----+-----+---------------------+

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
    while this representation could be processed directly by a neural network,
    only consider using it only when additional pre-processing has been
    done before it is fed to a network as input. This is because the scales
    and the range of values for each feature is significantly different.
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
        super(ExtraMinimalRepresentationRaw, self).__init__(
            knn, num_forward_fixes, use_filed_route, num_actions
        )

        ####### base features range
        base_feats_low = [
            -np.inf,
        ]
        base_feats_high = [
            np.inf,
        ]
        self.num_features_base = len(base_feats_low)

        ####### fixes features range
        if self.num_forward_fixes > 0:
            fixes_feats_low = [
                -np.pi,
            ]
            fixes_feats_high = [
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
            neighbours_feats_low = [-np.pi, 0.0]
            neighbours_feats_high = [np.pi, np.inf]
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

        tracked_data = gym_env.get_tracked_aircraft_data()

        ####### utils: get useful information for computing base features
        # centreline distance
        ac_centre_dist, turn_dir, _ = tracked_data[callsign].centreline_info_fr
        ac_centre_dist *= turn_dir

        # step since action
        # Uncomment for use in training with action penalty memory
        # steps_since_action = tracked_data[callsign].steps_since_action

        ####### base features: the current aircraft state
        base_feats = np.asarray(
            [
                ac_centre_dist,
                # steps_since_action / STEPS_SINCE_ACTION_MAX - 1,
            ],
            dtype=np.float32,
        )

        ####### fixes features
        fixes_feats = self.generate_forward_fixes_features(gym_env, callsign)

        ####### neighbours features
        neighbours_feats = self.generate_neighbours_features(gym_env, callsign)

        feats_list = [base_feats] + fixes_feats + neighbours_feats
        return np.concatenate(feats_list, dtype=np.float32)

    def generate_forward_fixes_features(
        self, gym_env, callsign
    ) -> list[npt.NDArray[np.float32]]:
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
        _next_fix = tracked_data[callsign].next_fix_fr
        fixes = get_n_forward_fixes(
            route, start_from=_next_fix, n=self.num_forward_fixes
        )
        next_fixes_pos = [
            simulator_env.airspace.fixes.places[fix]
            for fix in fixes
            if fix is not None
        ]
        num_none_fixes = self.num_forward_fixes - len(next_fixes_pos)

        # aircraft heading
        if aircraft.cleared_instructions.heading is not None:
            hdg = aircraft.cleared_instructions.heading
        else:
            hdg = aircraft.heading

        ac_pos = aircraft.pos2d()
        angles_diff_ac_nfs = []
        for fix_pos in next_fixes_pos:
            # relative angular difference between aircraft heading
            # and bearing of aircraft position to the next fix
            bearing_ac_nf = ac_pos.bearing_to(fix_pos)
            angle_diff_ac_nf = angle_diff(hdg, bearing_ac_nf)
            turn_dir = left_right_check(hdg, bearing_ac_nf)
            angle_diff_ac_nf *= turn_dir
            angles_diff_ac_nfs.append(angle_diff_ac_nf)

        ####### fixes features
        fixes_feats = []
        for idx in range(len(next_fixes_pos)):
            tmp = np.asarray(
                [
                    angles_diff_ac_nfs[idx] * convert.DEG_TO_RAD,
                ],
                dtype=np.float32,
            )
            fixes_feats.append(tmp)

        # zero padding
        for idx in range(num_none_fixes):
            tmp = np.zeros(self.num_features_per_fix, dtype=np.float32)
            fixes_feats.append(tmp)

        return fixes_feats

    def generate_neighbours_features(
        self, gym_env, callsign
    ) -> list[npt.NDArray[np.float32]]:
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
            dist_ac_other = interactions[i].dist_ac_other
            angle_diff_ac_other = interactions[i].angle_diff_ac_other
            turn_dir_ac_other = interactions[i].turn_dir_ac_other

            tmp = np.asarray(
                [
                    (
                        angle_diff_ac_other
                        * convert.DEG_TO_RAD
                        * turn_dir_ac_other
                    ),
                    dist_ac_other,
                ],
                dtype=np.float32,
            )
            interactions_features.append(tmp)

        # zero padding
        for i in range(balance_count):
            tmp = np.zeros(self.num_features_per_neighbour, dtype=np.float32)
            interactions_features.append(tmp)

        return interactions_features


class ExtraMinimalRepresentation(BaseRepresentation):
    """Representation of aircraft state in the simulator.

    Scaled and clipped version, using handpicked (manual) metrics.

    +-----+-------------------------------+-----+------+---------------------+
    | Num |         Observation           | Min | Max  |        Unit         |
    +-----+-------------------------------+-----+------+---------------------+
    |   0 | Aircraft distance from the    |-3.0 | 3.0  | Nautical Miles (NM) |
    |     | filed route centreline        |     |      | (scaled by 50)      |
    +-----+-------------------------------+-----+------+---------------------+

    If num_forward_fixes > 0, then for each forward fix, the following
    representation is concatenated to the state:
    +-----+-------------------------------+-----+------+---------------------+
    | Num |         Observation           | Min | Max  |        Unit         |
    +-----+-------------------------------+-----+------+---------------------+
    |   0 | Relative angular difference   |-pi  | pi   | Angle (radians)     |
    |     | between aircraft selected     |     |      |                     |
    |     | heading and the bearing from  |     |      |                     |
    |     | aircraft position to the next |     |      |                     |
    |     | fix in the filed route        |     |      |                     |
    +-----+-------------------------------+-----+------+---------------------+

    If knn > 0, then for each nearby aircraft, the following representation is
    concatenated to the state:

    +-----+-------------------------------+-----+-----+---------------------+
    | Num |         Observation           | Min | Max |        Unit         |
    +-----+-------------------------------+-----+-----+---------------------+
    |   0 | Relative angular (non-reflex) | -pi | +pi | Angle (radians)     |
    |     | difference between aircraft   |     |     |                     |
    |     | heading and the heading of the|     |     |                     |
    |     | neighbour aircraft, including |     |     |                     |
    |     | the direction of turn (sign)  |     |     |                     |
    |   1 | Distance between aircraft     | 0.0 | 3.0 | Nautical Miles (NM) |
    |     | and neighbour aircraft        |     |     | (scaled by 50)      |
    +-----+------------------------------ +-----+-----+---------------------+

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
        super(ExtraMinimalRepresentation, self).__init__(
            knn, num_forward_fixes, use_filed_route, num_actions
        )

        ####### base features range
        base_feats_low = [
            -3.0,
        ]
        base_feats_high = [
            3.0,
        ]
        self.num_features_base = len(base_feats_low)

        ####### fixes features range
        if self.num_forward_fixes > 0:
            fixes_feats_low = [
                -np.pi,
            ]
            fixes_feats_high = [
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
            neighbours_feats_low = [-np.pi, 0.0]
            neighbours_feats_high = [np.pi, 3.0]
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

        tracked_data = gym_env.get_tracked_aircraft_data()

        ####### utils: get useful information for computing base features
        # centreline distance
        ac_centre_dist, turn_dir, _ = tracked_data[callsign].centreline_info_fr
        ac_centre_dist = np.clip(ac_centre_dist, 0.0, SRC.CLIP_DIST)
        ac_centre_dist *= turn_dir

        # step since action
        # Uncomment for use in training with action penalty memory
        # steps_since_action = tracked_data[callsign].steps_since_action

        ####### base features: the current aircraft state
        base_feats = np.asarray(
            [
                ac_centre_dist / SRS.SCALER_CENTRELINE_DIST,  # already clipped
                # steps_since_action / STEPS_SINCE_ACTION_MAX - 1,
            ],
            dtype=np.float32,
        )

        ####### fixes features
        fixes_feats = self.generate_forward_fixes_features(gym_env, callsign)

        ####### neighbours features
        neighbours_feats = self.generate_neighbours_features(gym_env, callsign)

        feats_list = [base_feats] + fixes_feats + neighbours_feats
        return np.concatenate(feats_list, dtype=np.float32)

    def generate_forward_fixes_features(
        self, gym_env, callsign
    ) -> list[npt.NDArray[np.float32]]:
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
        _next_fix = tracked_data[callsign].next_fix_fr
        fixes = get_n_forward_fixes(
            route, start_from=_next_fix, n=self.num_forward_fixes
        )
        next_fixes_pos = [
            simulator_env.airspace.fixes.places[fix]
            for fix in fixes
            if fix is not None
        ]
        num_none_fixes = self.num_forward_fixes - len(next_fixes_pos)

        # aircraft heading
        if aircraft.cleared_instructions.heading is not None:
            hdg = aircraft.cleared_instructions.heading
        else:
            hdg = aircraft.heading

        ac_pos = aircraft.pos2d()
        angles_diff_ac_nfs = []
        for fix_pos in next_fixes_pos:
            # relative angular difference between aircraft heading
            # and bearing of aircraft position to the next fix
            bearing_ac_nf = ac_pos.bearing_to(fix_pos)
            angle_diff_ac_nf = angle_diff(hdg, bearing_ac_nf)
            turn_dir = left_right_check(hdg, bearing_ac_nf)
            angle_diff_ac_nf *= turn_dir
            angles_diff_ac_nfs.append(angle_diff_ac_nf)

        ####### fixes features
        fixes_feats = []
        for idx in range(len(next_fixes_pos)):
            tmp = np.asarray(
                [
                    angles_diff_ac_nfs[idx] * convert.DEG_TO_RAD,
                ],
                dtype=np.float32,
            )
            fixes_feats.append(tmp)

        # zero padding
        for idx in range(num_none_fixes):
            tmp = np.zeros(self.num_features_per_fix, dtype=np.float32)
            fixes_feats.append(tmp)

        return fixes_feats

    def generate_neighbours_features(
        self, gym_env, callsign
    ) -> list[npt.NDArray[np.float32]]:
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
            dist_ac_other = interactions[i].dist_ac_other
            angle_diff_ac_other = interactions[i].angle_diff_ac_other
            turn_dir_ac_other = interactions[i].turn_dir_ac_other

            _dist = np.clip(dist_ac_other, 0.0, SRC.CLIP_DIST)
            tmp = np.asarray(
                [
                    (
                        angle_diff_ac_other
                        * convert.DEG_TO_RAD
                        * turn_dir_ac_other
                    ),
                    _dist / SRS.SCALER_AC_OTHER_DIST,
                ],
                dtype=np.float32,
            )
            interactions_features.append(tmp)

        # zero padding
        for i in range(balance_count):
            tmp = np.zeros(self.num_features_per_neighbour, dtype=np.float32)
            interactions_features.append(tmp)

        return interactions_features
