import gymnasium as gym
from bluebird_dt.core import Pos4D

from bluebird_gymnasium.envs.base import BaseEnv


class TrajectoryOverlayWrapper(gym.Wrapper):
    def __init__(self, env: BaseEnv, *args, **kwargs): # noqa: ANN002, ANN003, ARG002
        super().__init__(env)
        self.traj_dict = None

    def save_trajectory(self, traj_dict: dict[str, list[Pos4D]]):
        self.traj_dict = traj_dict

    def render(self) -> None:
        env_ = self.env
        while not isinstance(env_, gym.Env):
            env_ = env_.env
        return env_.render_w_overlay_trajectory(self.traj_dict)
