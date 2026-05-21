from __future__ import annotations

import numpy as np
from bluebird_dt.utility import convert

from bluebird_gymnasium.state_repr import (
    StateReprClipper as SRC,
    StateReprScaler as SRS,
)
from bluebird_gymnasium.state_repr.base import BaseRepresentation
from bluebird_gymnasium.utils.simulator_utils import east_north_ground_speed

import typing

if typing.TYPE_CHECKING:
    import numpy.typing as npt
    from bluebird_dt.core import Environment as SimulatorEnv
    from bluebird_gymnasium.envs.base import BaseEnv
    from bluebird_gymnasium.utils.types import ACStateTracker


# helper function
def _latlon_differences(
    simulator_env: SimulatorEnv,
    callsign: str,
    tracked_data: dict[str, ACStateTracker],
    num_fixes: int,
) -> tuple[list[float], list[float]]:
    """Compute the position differences between aircraft and fixes positions."""

    aircraft = simulator_env.aircraft[callsign]

    ac_pos = aircraft.pos2d()
    lon_differences = []
    lat_differences = []
    route = aircraft.flight_plan.route.filed
    next_fix = tracked_data[callsign].next_fix_fr
    next_fix_idx = route.index(next_fix)
    fixes = route[next_fix_idx : next_fix_idx + num_fixes]
    for fix in fixes:
        fix_pos = simulator_env.airspace.fixes.places[fix]
        lon_differences.append((ac_pos.lon - fix_pos.lon) * convert.DEG_TO_RAD)
        lat_differences.append((ac_pos.lat - fix_pos.lat) * convert.DEG_TO_RAD)
    for _ in range(num_fixes - len(fixes)):
        # pad blank (lat/lon difference) features with zeros
        lon_differences.append(0.0)
        lat_differences.append(0.0)

    return lat_differences, lon_differences


class DrlanRepresentationRaw(BaseRepresentation):
    """Representation of aircraft state in the simulator.

    Raw feature values

    +-----+-------------------------------+------+------+---------------------+
    | Num |         Observation           | Min  | Max  |        Unit         |
    +-----+-------------------------------+------+------+---------------------+
    |   0 | *Aircraft heading             |  0.0 |  2*pi| Angle (radians)     |
    |   1 | Aircraft current  flight level|  0.0 |  inf | Flight level        |
    |   2 | Aircraft horizontal airspeed  |  0.0 |  inf | Nautical miles per  |
    |     |                               |      |      | second              |
    |   3 | Aircraft vertical speed       | -inf |  inf | Nautical miles per  |
    |     |                               |      |      | second              |
    |   4 | Aircraft east ground speed    |  0.0 |  inf | Nautical miles per  |
    |     |                               |      |      | second              |
    |   5 | Aircraft north ground speed   |  0.0 |  inf | Nautical miles per  |
    |     |                               |      |      | second              |
    |   6 | Time of day                   |  0.0 |  23  | Hour                |
    |   7 | Previous step action of the   |  0.0 |  1.0 | one-hot vectorized  |
    |     | aircraft                      |      |      | representation of   |
    |     |                               |      |      | size n, the number  |
    |     |                               |      |      | of actions          |
    |   8 | Longitude difference between  | -pi  | +pi  | Angle (radians)     |
    |     | aircraft's n* fixes/waypoint  |      |      | vectorized          |
    |     | and the aircraft's current    |      |      | representation of   |
    |     | position                      |      |      | size n              |
    |   9 | Latitude difference between   | -pi/2| +pi/2| Angle (radians)     |
    |     | aircraft's n* fixes/waypoint  |      |      | vectorized          |
    |     | and the aircraft's current    |      |      | representation of   |
    |     | position                      |      |      | size n              |
    +-----+-------------------------------+------+------+---------------------+

    If knn > 0, then for each nearby aircraft, the following representation is
    concatenated to the state:

    +-----+-------------------------------+------+------+---------------------+
    | Num |         Observation           | Min  | Max  |        Unit         |
    +-----+-------------------------------+------+------+---------------------+
    |   0 | Relative angular (non-reflex) | -pi  | +pi  | Angle (radians)     |
    |     | difference between aircraft   |      |      |                     |
    |     | heading and the heading of the|      |      |                     |
    |     | neighbour aircraft, including |      |      |                     |
    |     | the direction of turn (sign)  |      |      |                     |
    |   1 | Flight level difference       | -inf |  inf | Flight level        |
    |     | between aircraft and neighbour|      |      |                     |
    |   2 | Neighbour horizontal airspeed |  0.0 |  inf | Nautical miles per  |
    |     |                               |      |      | second              |
    |   3 | Neighbour vertical speed      |  0.0 |  inf | Nautical miles per  |
    |     |                               |      |      | second              |
    |   4 | Neighbour east ground speed   |  0.0 |  inf | Nautical miles per  |
    |     |                               |      |      | second              |
    |   5 | Neighbour north ground speed  |  0.0 |  inf | Nautical miles per  |
    |     |                               |      |      | second              |
    |   6 | Bearing from aircraft to the  |  0.0 |  inf | Nautical Miles (NM) |
    |     | neighbour aircraft            |      |      |                     |
    |   7 | Distance between aircraft     |  0.0 |  inf | Nautical Miles (NM) |
    |     | and neighbour                 |      |      |                     |
    |   8 | Previous step action of the   |  0.0 |  1.0 | one-hot vectorized  |
    |     | neighbour aircraft            |      |      | representation of   |
    |     |                               |      |      | size n, the number  |
    |     |                               |      |      | of actions          |
    |   9 | Longitude difference between  | -pi  | +pi  | Angle (radians)     |
    |     | aircraft's n* fixes/waypoint  |      |      | vectorized          |
    |     | and the neighbour aircraft's  |      |      | representation of   |
    |     | current position              |      |      | size n              |
    |  10 | Latitude difference between   | -pi/2| +pi/2| Angle (radians)     |
    |     | aircraft's n* fixes/waypoint  |      |      | vectorized          |
    |     | and the neighbour aircraft's  |      |      | representation of   |
    |     | current position              |      |      | size n              |
    +-----+-------------------------------+------+------+---------------------+

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
    while this presentation could be proccessed directly by a neural network,
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
        num_forward_fixes: int = 3,
        use_filed_route: bool = False,
        num_actions: int | None = None,
    ):
        if num_actions is None or num_actions < 1:
            raise ValueError(
                "`num_actions` argument must be set to an integer value >= 1."
            )

        super(DrlanRepresentationRaw, self).__init__(
            knn, num_forward_fixes, use_filed_route, num_actions
        )

        # range for previous action feature vector
        low_prev_action = [
            0.0,
        ] * self.num_actions
        high_prev_action = [
            1.0,
        ] * self.num_actions

        # range for longitude difference feature vector
        low_lon_diff = [
            -np.pi,
        ] * self.num_forward_fixes
        high_lon_diff = [
            np.pi,
        ] * self.num_forward_fixes

        # range for latitude difference feature vector
        low_lat_diff = [
            -np.pi / 2,
        ] * self.num_forward_fixes
        high_lat_diff = [
            np.pi / 2,
        ] * self.num_forward_fixes

        ####### base features range
        base_feats_low = [
            0.0,
            0.0,
            0.0,
            -np.inf,
            0.0,
            0.0,
            0.0,
            *low_prev_action,  # * to unroll to the list/vector
            *low_lon_diff,
            *low_lat_diff,
        ]
        base_feats_high = [
            2 * np.pi,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            np.inf,
            *high_prev_action,  # * to unroll to the list/vector
            *high_lon_diff,
            *high_lat_diff,
        ]
        self.num_features_base = len(base_feats_low)

        ####### neighbours features range
        if knn > 0:
            neighbours_feats_low = [
                -np.pi,
                -np.inf,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                *low_prev_action,  # * to unroll to the list/vector
                *low_lon_diff,
                *low_lat_diff,
            ]

            neighbours_feats_high = [
                +np.pi,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                np.inf,
                *high_prev_action,  # * to unroll to the list/vector
                *high_lon_diff,
                *high_lat_diff,
            ]
            self.num_features_per_neighbour = len(neighbours_feats_low)

            neighbours_feats_low *= self.knn
            neighbours_feats_high *= self.knn
        else:
            neighbours_feats_low = []
            neighbours_feats_high = []

        self.low = base_feats_low + neighbours_feats_low
        self.high = base_feats_high + neighbours_feats_high
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
        airspace_sector = gym_env.get_active_airspace_sector()
        aircraft = simulator_env.aircraft[callsign]

        ####### utils: get useful information for computing base features
        # horizontal airspeed
        horizontal_airspeed = convert.horizontal_tas(
            aircraft.speed_tas, aircraft.vertical_speed
        )

        # ground speed: east and north ground speed
        east_ground_speed, north_ground_speed = east_north_ground_speed(
            callsign, simulator_env
        )

        # time of day
        time_of_day = simulator_env.datetime.hour

        # previous step action
        _action_int = tracked_data[callsign].action
        previous_step_action = np.zeros(self.num_actions, dtype=np.float32)
        previous_step_action[_action_int] = 1.0

        # difference between aircraft position (lat/lon) and the position
        # of the next `self.num_forward_fixes` fixes.
        lat_differences, lon_differences = _latlon_differences(
            simulator_env, callsign, tracked_data, self.num_forward_fixes
        )

        ####### base features: the current aircraft state
        base_feats = np.asarray(
            [
                aircraft.heading * convert.DEG_TO_RAD,
                aircraft.fl,
                horizontal_airspeed,  # knots: nautical miles per hour
                aircraft.vertical_speed,  # feet per minute
                east_ground_speed,  # knots
                north_ground_speed,  # knots
                time_of_day,
                *previous_step_action,  # unroll one-hot vector
                *lon_differences,  # unroll vector (already in radians)
                *lat_differences,  # unroll vector (already in radians)
            ],
            dtype=np.float32,
        )

        ####### neighbours features
        neighbours_feats = self.generate_neighbours_features(gym_env, callsign)

        feats_list = [
            base_feats,
        ] + neighbours_feats
        return np.concatenate(feats_list, dtype=np.float32)

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

        simulator_env = gym_env.get_simulator_env()
        tracked_data = gym_env.get_tracked_aircraft_data()
        airspace_sector = gym_env.get_active_airspace_sector()
        aircraft = simulator_env.aircraft[callsign]

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
            other_aircraft = simulator_env.aircraft[callsign_other]

            # distance from aircraft to other aircraft
            dist_ac_other = interactions[i].dist_ac_other
            # bearing from aircraft to other aircraft
            bearing_ac_other = interactions[i].bearing_ac_other
            # heading (angle) difference between aircraft to other aircraft
            angle_diff_ac_other = interactions[i].angle_diff_ac_other
            turn_dir_ac_other = interactions[i].turn_dir_ac_other
            # flight level difference between aircraft to other aircraft
            fl_diff_ac_other = interactions[i].fl_diff_ac_other

            # horizontal airspeed
            other_horizontal_airspeed = convert.horizontal_tas(
                aircraft.speed_tas, aircraft.vertical_speed
            )

            # ground speed: east and north ground speed
            other_east_ground_speed, other_north_ground_speed = (
                east_north_ground_speed(callsign, simulator_env)
            )

            # previous step action
            _action_int = tracked_data[callsign_other].action
            other_previous_step_action = np.zeros(
                self.num_actions, dtype=np.float32
            )
            other_previous_step_action[_action_int] = 1.0

            # difference between aircraft position (lat/lon) and the position
            # of the next `self.num_forward_fixes` fixes.
            other_lat_differences, other_lon_differences = _latlon_differences(
                simulator_env,
                callsign_other,
                tracked_data,
                self.num_forward_fixes,
            )

            tmp = np.asarray(
                [
                    (
                        angle_diff_ac_other
                        * convert.DEG_TO_RAD
                        * turn_dir_ac_other
                    ),
                    fl_diff_ac_other,
                    other_horizontal_airspeed,
                    other_aircraft.vertical_speed,
                    other_east_ground_speed,
                    other_north_ground_speed,
                    bearing_ac_other * convert.DEG_TO_RAD,
                    dist_ac_other,
                    *other_previous_step_action,  # unroll one-hot vector
                    *other_lon_differences,  # unroll vector already in radians
                    *other_lat_differences,  # unroll vector already in radians
                ],
                dtype=np.float32,
            )
            interactions_features.append(tmp)

        # zero padding
        for i in range(balance_count):
            # number of each neighbour (other) aircraft features.
            _num_features = (
                8  # eight scalar features
                + self.num_actions  # other_previous_action
                + self.num_forward_fixes  # other_lon_differences
                + self.num_forward_fixes  # other_lat_differences
            )
            tmp = np.zeros(_num_features, dtype=np.float32)
            interactions_features.append(tmp)

        return interactions_features


class DrlanRepresentation(BaseRepresentation):
    """Representation of aircraft state in the simulator.

    Scaled and clipped version, using handpicked (manual) metrics.

    +-----+-------------------------------+------+------+---------------------+
    | Num |         Observation           | Min  | Max  |        Unit         |
    +-----+-------------------------------+------+------+---------------------+
    |   0 | *Aircraft heading             | -pi  |  +pi | Angle (radians)     |
    |   1 | Aircraft current  flight level|  0.0 |  3.0 | Flight level        |
    |   2 | Aircraft horizontal airspeed  |  0.0 |  3.0 | Nautical miles per  |
    |     |                               |      |      | second              |
    |   3 | Aircraft vertical speed       | -3.0 |  3.0 | Nautical miles per  |
    |     |                               |      |      | second              |
    |   4 | Aircraft east ground speed    |  0.0 |  3.0 | Nautical miles per  |
    |     |                               |      |      | second              |
    |   5 | Aircraft north ground speed   |  0.0 |  3.0 | Nautical miles per  |
    |     |                               |      |      | second              |
    |   6 | Time of day                   |  0.0 |  3.0 | 8 Hour block        |
    |   7 | Previous step action of the   |  0.0 |  1.0 | one-hot vectorized  |
    |     | aircraft                      |      |      | representation of   |
    |     |                               |      |      | size n, the number  |
    |     |                               |      |      | of actions          |
    |   8 | Longitude difference between  | -pi  | +pi  | Angle (radians)     |
    |     | aircraft's n* fixes/waypoint  |      |      | vectorized          |
    |     | and the aircraft's current    |      |      | representation of   |
    |     | position                      |      |      | size n              |
    |   9 | Latitude difference between   | -pi/2| +pi/2| Angle (radians)     |
    |     | aircraft's n* fixes/waypoint  |      |      | vectorized          |
    |     | and the aircraft's current    |      |      | representation of   |
    |     | position                      |      |      | size n              |
    +-----+-------------------------------+------+------+---------------------+

    If knn > 0, then for each nearby aircraft, the following representation is
    concatenated to the state:

    +-----+-------------------------------+------+------+---------------------+
    | Num |         Observation           | Min  | Max  |        Unit         |
    +-----+-------------------------------+------+------+---------------------+
    |   0 | Relative angular (non-reflex) | -pi  | +pi  | Angle (radians)     |
    |     | difference between aircraft   |      |      |                     |
    |     | heading and the heading of the|      |      |                     |
    |     | neighbour aircraft, including |      |      |                     |
    |     | the direction of turn (sign)  |      |      |                     |
    |   1 | Flight level difference       | -3.0 |  3.0 | Flight level        |
    |     | between aircraft and neighbour|      |      | (scaled by 20)      |
    |   2 | Neighbour horizontal airspeed |  0.0 |  3.0 | Nautical miles per  |
    |     |                               |      |      | second              |
    |   3 | Neighbour vertical speed      | -3.0 |  3.0 | Nautical miles per  |
    |     |                               |      |      | second              |
    |   4 | Neighbour east ground speed   |  0.0 |  3.0 | Nautical miles per  |
    |     |                               |      |      | second              |
    |   5 | Neighbour north ground speed  |  0.0 |  3.0 | Nautical miles per  |
    |     |                               |      |      | second              |
    |   6 | Bearing from aircraft to the  | -pi  | +pi  | Nautical Miles (NM) |
    |     | neighbour aircraft            |      |      |                     |
    |   7 | Distance between aircraft     |  0.0 |  3.0 | Nautical Miles (NM) |
    |     | and neighbour                 |      |      |                     |
    |   8 | Previous step action of the   |  0.0 |  1.0 | one-hot vectorized  |
    |     | neighbour aircraft            |      |      | representation of   |
    |     |                               |      |      | size n, the number  |
    |     |                               |      |      | of actions          |
    |   9 | Longitude difference between  | -pi  | +pi  | Angle (radians)     |
    |     | aircraft's n* fixes/waypoint  |      |      | vectorized          |
    |     | and the neighbour aircraft's  |      |      | representation of   |
    |     | current position              |      |      | size n              |
    |  10 | Latitude difference between   | -pi/2| +pi/2| Angle (radians)     |
    |     | aircraft's n* fixes/waypoint  |      |      | vectorized          |
    |     | and the neighbour aircraft's  |      |      | representation of   |
    |     | current position              |      |      | size n              |
    +-----+-------------------------------+------+------+---------------------+

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
            Defaults to `False`.
        num_actions: the number of actions for an aircraft. Defaults to
            `None` if it is not used in the aircraft's state representation.
    """

    def __init__(
        self,
        knn: int,
        num_forward_fixes: int = 3,
        use_filed_route: bool = False,
        num_actions: int | None = None,
    ):
        if num_actions is None:
            raise ValueError("`num_actions` argument must be set.")

        super(DrlanRepresentation, self).__init__(
            knn, num_forward_fixes, use_filed_route, num_actions
        )

        # range for previous action feature vector
        low_prev_action = [
            0.0,
        ] * self.num_actions
        high_prev_action = [
            1.0,
        ] * self.num_actions

        # range for longitude difference feature vector
        low_lon_diff = [
            -np.pi,
        ] * self.num_forward_fixes
        high_lon_diff = [
            np.pi,
        ] * self.num_forward_fixes

        # range for latitude difference feature vector
        low_lat_diff = [
            -np.pi / 2,
        ] * self.num_forward_fixes
        high_lat_diff = [
            np.pi / 2,
        ] * self.num_forward_fixes

        ####### base features range
        base_feats_low = [
            -np.pi,
            0.0,
            0.0,
            -3.0,
            0.0,
            0.0,
            0.0,
            *low_prev_action,  # * to unroll to the list/vector
            *low_lon_diff,
            *low_lat_diff,
        ]
        base_feats_high = [
            +np.pi,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            3.0,
            *high_prev_action,  # * to unroll to the list/vector
            *high_lon_diff,
            *high_lat_diff,
        ]
        self.num_features_base = len(base_feats_low)

        ####### neighbours features range
        if self.knn > 0:
            neighbours_feats_low = [
                -np.pi,
                -3.0,
                0.0,
                -3.0,
                0.0,
                0.0,
                -np.pi,
                0.0,
                *low_prev_action,  # * to unroll to the list/vector
                *low_lon_diff,
                *low_lat_diff,
            ]

            neighbours_feats_high = [
                +np.pi,
                3.0,
                3.0,
                3.0,
                3.0,
                3.0,
                +np.pi,
                3.0,
                *high_prev_action,  # * to unroll to the list/vector
                *high_lon_diff,
                *high_lat_diff,
            ]
            self.num_features_per_neighbour = len(neighbours_feats_low)

            neighbours_feats_low *= self.knn
            neighbours_feats_high *= self.knn
        else:
            neighbours_feats_low = []
            neighbours_feats_high = []

        self.low = base_feats_low + neighbours_feats_low
        self.high = base_feats_high + neighbours_feats_high
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
        airspace_sector = gym_env.get_active_airspace_sector()
        aircraft = simulator_env.aircraft[callsign]

        ####### utils: get useful information for computing base features
        # horizontal airspeed
        horizontal_airspeed = convert.horizontal_tas(
            aircraft.speed_tas, aircraft.vertical_speed
        )

        # ground speed: east and north ground speed
        east_ground_speed, north_ground_speed = east_north_ground_speed(
            callsign, simulator_env
        )

        # time of day
        time_of_day = simulator_env.datetime.hour
        time_of_day /= 7.67  # 7.67 scale the max hour (i.e., 23) to approx. 3

        # previous step action
        _action_int = tracked_data[callsign].action
        previous_step_action = np.zeros(self.num_actions, dtype=np.float32)
        previous_step_action[_action_int] = 1.0

        # difference between aircraft position (lat/lon) and the position
        # of the next `self.num_forward_fixes` fixes.
        lat_differences, lon_differences = _latlon_differences(
            simulator_env, callsign, tracked_data, self.num_forward_fixes
        )

        ####### base features: the current aircraft state
        base_feats = np.asarray(
            [
                (aircraft.heading * convert.DEG_TO_RAD) - np.pi,
                np.clip(aircraft.fl, 0.0, SRC.CLIP_FL) / SRS.SCALER_FL,
                np.clip(horizontal_airspeed, 0.0, SRC.CLIP_SPEED)
                / SRS.SCALER_SPEED,
                np.clip(
                    aircraft.vertical_speed,
                    -SRC.CLIP_VERTICAL_SPEED,
                    SRC.CLIP_VERTICAL_SPEED,
                )
                / SRS.SCALER_VERTICAL_SPEED,
                np.clip(east_ground_speed, 0.0, SRC.CLIP_SPEED)
                / SRS.SCALER_SPEED,
                np.clip(north_ground_speed, 0.0, SRC.CLIP_SPEED)
                / SRS.SCALER_SPEED,
                time_of_day,
                *previous_step_action,  # unroll one-hot vector
                *lon_differences,  # unroll vector already in radians
                *lat_differences,  # unroll vector already in radians
            ],
            dtype=np.float32,
        )

        ####### neighbours features
        neighbours_feats = self.generate_neighbours_features(gym_env, callsign)

        feats_list = [
            base_feats,
        ] + neighbours_feats
        return np.concatenate(feats_list, dtype=np.float32)

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

        simulator_env = gym_env.get_simulator_env()
        tracked_data = gym_env.get_tracked_aircraft_data()
        airspace_sector = gym_env.get_active_airspace_sector()
        aircraft = simulator_env.aircraft[callsign]

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
            other_aircraft = simulator_env.aircraft[callsign_other]

            # distance from aircraft to other aircraft
            dist_ac_other = interactions[i].dist_ac_other
            # bearing from aircraft to other aircraft
            bearing_ac_other = interactions[i].bearing_ac_other
            # heading (angle) difference between aircraft to other aircraft
            angle_diff_ac_other = interactions[i].angle_diff_ac_other
            turn_dir_ac_other = interactions[i].turn_dir_ac_other
            # flight level difference between aircraft to other aircraft
            fl_diff_ac_other = interactions[i].fl_diff_ac_other

            # horizontal airspeed
            other_horizontal_airspeed = convert.horizontal_tas(
                aircraft.speed_tas, aircraft.vertical_speed
            )

            # ground speed: east and north ground speed
            other_east_ground_speed, other_north_ground_speed = (
                east_north_ground_speed(callsign, simulator_env)
            )

            # previous step action
            _action_int = tracked_data[callsign_other].action
            other_previous_step_action = np.zeros(
                self.num_actions, dtype=np.float32
            )
            other_previous_step_action[_action_int] = 1.0

            # difference between aircraft position (lat/lon) and the position
            # of the next `self.num_forward_fixes` fixes.
            other_lat_differences, other_lon_differences = _latlon_differences(
                simulator_env,
                callsign_other,
                tracked_data,
                self.num_forward_fixes,
            )

            tmp = np.asarray(
                [
                    (
                        angle_diff_ac_other
                        * convert.DEG_TO_RAD
                        * turn_dir_ac_other
                    ),
                    np.clip(
                        fl_diff_ac_other,
                        -SRC.CLIP_FL_DIFF,
                        SRC.CLIP_FL_DIFF,
                    )
                    / SRS.SCALER_FL_DIFF,
                    np.clip(other_horizontal_airspeed, 0.0, SRC.CLIP_SPEED)
                    / SRS.SCALER_SPEED,
                    np.clip(
                        other_aircraft.vertical_speed,
                        -SRC.CLIP_VERTICAL_SPEED,
                        SRC.CLIP_VERTICAL_SPEED,
                    )
                    / SRS.SCALER_VERTICAL_SPEED,
                    np.clip(other_east_ground_speed, 0.0, SRC.CLIP_SPEED)
                    / SRS.SCALER_SPEED,
                    np.clip(other_north_ground_speed, 0.0, SRC.CLIP_SPEED)
                    / SRS.SCALER_SPEED,
                    (bearing_ac_other * convert.DEG_TO_RAD) - np.pi,
                    np.clip(dist_ac_other, 0.0, SRC.CLIP_DIST)
                    / SRS.SCALER_AC_OTHER_DIST,
                    *other_previous_step_action,  # unroll one-hot vector
                    *other_lon_differences,  # unroll vector already in radians
                    *other_lat_differences,  # unroll vector already in radians
                ],
                dtype=np.float32,
            )
            interactions_features.append(tmp)

        # zero padding
        for i in range(balance_count):
            # number of each neighbour (other) aircraft features.
            _num_features = (
                8  # eight scalar features
                + self.num_actions  # other_previous_action
                + self.num_forward_fixes  # other_lon_differences
                + self.num_forward_fixes  # other_lat_differences
            )
            tmp = np.zeros(_num_features, dtype=np.float32)
            interactions_features.append(tmp)

        return interactions_features
