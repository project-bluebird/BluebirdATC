from __future__ import annotations

import shutil
import sys

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt

from bluebird_gymnasium.envs import registry_env

import typing

if typing.TYPE_CHECKING:
    from bluebird_gymnasium.envs.base import BaseEnv


# sort out macOS matplotlib backend issues
if sys.platform == "darwin":
    matplotlib.use("macOSX")  # backend TKAgg also works


def visualize_airspace(env: BaseEnv, render_mode: str = "rgb_array", clean_up: bool = True):
    """Visualize the airspace of an environment

    This is useful to give a pictorial view of an airspace without
    necessarily running a full experiment.

    Args:
        env: specifies the name (gym key) of the environment or an already
            instantiated environment from one of the environments defined in
            `bluebird_gymnasium/envs`.
        render_mode: specifies the preferred render_mode used by the gym
            environment, set to one of the following ['rgb_array', 'human',
            'file'].
        clean_up: specifies whether to clean up the directory structure after
            visualization is completed (as the airspace image is saved to disk
            default). Note, when `render_mode` is set to 'file', this
            argument is ignored regardless of its set value.
    """
    if render_mode == "file" and clean_up is True:
        gym.logger.warn("When `render_mode` is set to 'file', `clean_up` is ignored even when set to True.")

    # set up env if necessary
    if isinstance(env, str):
        if env not in registry_env.keys():
            raise ValueError("{0} not found. Specify one of the following: {1}".format(env, list(registry_env.keys())))
        env = registry_env.get(env)(render_mode=render_mode)
        prev_mode = None
    else:
        prev_mode = env.get_render_mode()

    # visualize
    if render_mode == "file":
        env.render()
    else:
        img = env.render()
        plt.imshow(img)
        plt.axis("off")
        plt.show()

        if clean_up:
            shutil.rmtree(env.config.radar_config["render_dir"])

    # restore env back to previous render mode
    env.set_render_mode(prev_mode)


# Example usage
if __name__ == "__main__":
    visualize_airspace("SectorXEnv-v0", "rgb_array")
