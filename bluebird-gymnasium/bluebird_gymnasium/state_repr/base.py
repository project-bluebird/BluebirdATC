from __future__ import annotations

import numpy as np

import typing

if typing.TYPE_CHECKING:
    import numpy.typing as npt

    from bluebird_gymnasium.envs.base import BaseEnv
    from bluebird_gymnasium.utils.types import InteractionInfo


class BaseRepresentation:
    """Base class for defining state representation.

    Base class for defining state representation based on the aircraft state
    in the simulator.

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
        if knn < 0:
            raise ValueError("`knn` should be set to an integer value >= 0")

        if num_forward_fixes < 0:
            raise ValueError(
                "`num_forward_fixes` should be set to an integer value >= 0"
            )

        if num_actions is not None and num_actions < 1:
            raise ValueError(
                "`num_actions` should be set to `None` or integer value >= 1"
            )

        self.knn = knn
        self.num_forward_fixes = num_forward_fixes
        self.use_filed_route = use_filed_route
        self.num_actions = num_actions

        # employed if a future trajectory is not provided for
        # aircraft. it is set via the gym env that holds an
        # instance of a state representation class.
        self.rollout_predictor = None

        # others: set in each child class instance.
        self.num_features_base = None  # for the specific aircraft
        self.num_features_per_fix = None
        self.num_features_per_neighbour = None
        self.low = None
        self.high = None

    def repr(self, gym_env: BaseEnv, callsign: str) -> np.ndarray:
        """Generate a vectorised representation of an aircraft's state.

        Args:
            gym_env: defines the gymnasium environment.
            callsign: the identifier of the aircraft.

        Returns:
            numpy array which represents the aircraft's current state as
            a vector.
        """

        raise NotImplementedError

    def _one_hot(self, size: int, idx: int) -> list[float]:
        ret = [0.0] * size
        ret[idx] = 1.0
        return ret

    def _knn_repr(
        self,
        callsign: str,
        gym_env: BaseEnv,
        relevant_interactions_only=True,
    ) -> list[InteractionInfo]:
        """Gets statistics about relevant aircraft interaction with others."""

        traffic_monitor = gym_env.get_traffic_monitor()
        if relevant_interactions_only:
            return traffic_monitor.get_relevant_traffic(callsign)

        else:
            return traffic_monitor.get_aircraft_interaction_info(
                callsign, gym_env
            )

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

        raise NotImplementedError

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

        raise NotImplementedError
