"""Minimal policy-gradient agent for Bluebird Gymnasium.

This file is an educational example that connects the core RL concepts:

- observation vectors
- a neural-network policy
- logits and stochastic action sampling
- rewards and discounted returns
- loss calculation
- backpropagation
- gradient-descent-based parameter updates

It implements a very small REINFORCE-style training loop. This is useful for
understanding the mechanics, but it is not intended to be a strong baseline.
For serious experiments, use a more stable algorithm such as PPO.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Allow this example to be run from the monorepo without installing packages.
THIS_FILE = Path(__file__).resolve()
BLUEBIRD_GYMNASIUM_ROOT = THIS_FILE.parents[1]
BLUEBIRD_DT_ROOT = THIS_FILE.parents[2] / "bluebird-dt"
sys.path.insert(0, str(BLUEBIRD_GYMNASIUM_ROOT))
sys.path.insert(0, str(BLUEBIRD_DT_ROOT))

from bluebird_gymnasium.envs import EnvConfig, ViewType  # noqa: E402
from bluebird_gymnasium.envs.sector_i import SectorIEnv  # noqa: E402


class PolicyNetwork(nn.Module):
    """Neural network that maps one aircraft observation to action logits."""

    def __init__(
        self,
        observation_dimension: int,
        number_of_actions: int,
        hidden_units: int = 128,
    ) -> None:
        super().__init__()

        # A unit is one neuron in a layer. This network has two hidden layers,
        # each with hidden_units neurons.
        self.layers = nn.Sequential(
            nn.Linear(observation_dimension, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, number_of_actions),
        )

    def forward(self, observation_batch: torch.Tensor) -> torch.Tensor:
        """Return raw action scores, also called logits."""
        return self.layers(observation_batch)


class SharedPolicyAgent:
    """One shared policy network reused for every aircraft."""

    def __init__(
        self,
        observation_dimension: int,
        number_of_actions: int,
        learning_rate: float = 1e-3,
        hidden_units: int = 128,
    ) -> None:
        self.policy_network = PolicyNetwork(
            observation_dimension=observation_dimension,
            number_of_actions=number_of_actions,
            hidden_units=hidden_units,
        )

        # Adam is a gradient-descent-based optimizer. It updates the network's
        # weights and biases using gradients computed by backpropagation.
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate,
        )

    def sample_training_actions(
        self,
        observation_by_callsign: dict[str, np.ndarray],
    ) -> tuple[dict[str, int], dict[str, torch.Tensor]]:
        """Choose stochastic actions and keep log-probabilities for learning.

        During training, we sample from the action distribution instead of
        always choosing the largest logit. This is stochastic training: the
        agent explores actions that are not currently its top choice.
        """
        action_by_callsign: dict[str, int] = {}
        log_probability_by_callsign: dict[str, torch.Tensor] = {}

        for callsign, observation_vector in observation_by_callsign.items():
            # Convert the NumPy observation vector into a PyTorch tensor.
            # Shape changes from (observation_dimension,) to
            # (1, observation_dimension). The leading 1 is the batch dimension.
            observation_tensor = torch.tensor(
                observation_vector,
                dtype=torch.float32,
            ).unsqueeze(0)

            # The network outputs logits: raw, unconstrained action scores.
            # If there are three actions, this might look like:
            # [[0.2, 1.1, -0.5]]
            action_logits = self.policy_network(observation_tensor)

            # Categorical(logits=...) applies softmax internally to create a
            # probability distribution over discrete actions.
            action_distribution = torch.distributions.Categorical(
                logits=action_logits,
            )

            # Sample one action from the distribution. This is what enables
            # exploration during training.
            sampled_action = action_distribution.sample()

            # The log-probability is needed for the policy-gradient loss.
            # If an action later receives high return, the loss will push the
            # network to increase this log-probability next time.
            sampled_action_log_probability = action_distribution.log_prob(
                sampled_action,
            )

            action_by_callsign[callsign] = sampled_action.item()
            log_probability_by_callsign[callsign] = (
                sampled_action_log_probability.squeeze()
            )

        return action_by_callsign, log_probability_by_callsign

    def choose_evaluation_actions(
        self,
        observation_by_callsign: dict[str, np.ndarray],
    ) -> dict[str, int]:
        """Choose deterministic actions for evaluation or deployment."""
        action_by_callsign: dict[str, int] = {}

        # No gradients are needed when evaluating a trained policy.
        with torch.no_grad():
            for callsign, observation_vector in observation_by_callsign.items():
                observation_tensor = torch.tensor(
                    observation_vector,
                    dtype=torch.float32,
                ).unsqueeze(0)

                action_logits = self.policy_network(observation_tensor)

                # argmax chooses the highest-scoring action.
                chosen_action = torch.argmax(action_logits, dim=-1).item()
                action_by_callsign[callsign] = chosen_action

        return action_by_callsign

    def update_policy_from_episode(
        self,
        log_probability_per_step: list[torch.Tensor],
        reward_per_step: list[float],
        discount_factor_gamma: float = 0.99,
    ) -> float | None:
        """Turn episode rewards into a loss, then update network parameters."""
        if not log_probability_per_step:
            return None

        discounted_return_per_step: list[float] = []
        running_discounted_return = 0.0

        # Compute discounted returns in reverse:
        # G_t = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
        for reward in reversed(reward_per_step):
            running_discounted_return = (
                reward + discount_factor_gamma * running_discounted_return
            )
            discounted_return_per_step.append(running_discounted_return)

        discounted_return_per_step.reverse()

        returns_tensor = torch.tensor(
            discounted_return_per_step,
            dtype=torch.float32,
        )

        # Normalizing returns often makes simple policy-gradient training less
        # numerically erratic. This changes scale, not the episode ordering.
        if returns_tensor.numel() > 1:
            returns_std = returns_tensor.std(unbiased=False)
            if returns_std > 1e-8:
                returns_tensor = (
                    (returns_tensor - returns_tensor.mean())
                    / (returns_std + 1e-8)
                )

        # REINFORCE objective:
        # - If return is high, increase probability of the sampled action.
        # - If return is low/negative, decrease probability of that action.
        policy_loss_terms: list[torch.Tensor] = []
        for action_log_probability, discounted_return in zip(
            log_probability_per_step,
            returns_tensor,
        ):
            policy_loss_terms.append(
                -action_log_probability * discounted_return,
            )

        loss = torch.stack(policy_loss_terms).sum()

        # Clear gradients from the previous update.
        self.optimizer.zero_grad()

        # Backpropagation: compute gradients of loss with respect to every
        # trainable weight and bias in the policy network.
        loss.backward()

        # Gradient descent update: Adam uses the gradients to modify the
        # policy network's parameters.
        self.optimizer.step()

        return loss.item()


def make_sector_i_training_config() -> EnvConfig:
    """Create a small, easy Bluebird environment configuration."""
    config = SectorIEnv.get_default_env_config(ViewType.DECENTRALIZED)

    # State encoding: convert raw simulator state into compact numeric vectors.
    config.state_repr_config = {
        "encoder_cls": "extra_minimal",
        "k_nearest_aircraft": 1,
    }

    # Keep the first action space small: no-op, left turn, right turn.
    config.action_config = {
        "simple_heading_left": True,
        "simple_heading_right": True,
        "simple_fl_climb": False,
        "simple_fl_descent": False,
        "simple_fl_exit": False,
    }

    # Reward components define what behavior the agent is encouraged to learn.
    config.reward_config = {
        "fns": [
            "position_status_const",
            "lateral_centreline_distance_shaped",
            "safety_simple_avoidance_exp",
        ],
        "coeffs": [1.0, 1.0, 1.2],
    }

    # Decentralized mode means obs/action/reward/done are dicts keyed by
    # aircraft callsign.
    config.view_config = {
        "type": ViewType.DECENTRALIZED.value,
        "decentralized_params": {},
    }

    # Start with one aircraft. Increase difficulty only after this works.
    config.scenario_config = {
        "cls": "tactical",
        "args": {
            "num_aircraft": 1,
            "balance": [0.0, 0.0, 1.0],
        },
    }

    return config


def run_one_training_episode(
    environment: SectorIEnv,
    agent: SharedPolicyAgent,
    random_seed: int,
    discount_factor_gamma: float,
) -> tuple[float, int, float | None]:
    """Run one episode, then update the policy from the collected trajectory."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    observation_by_callsign, _info = environment.reset(seed=random_seed)

    episode_is_done = False
    episode_step_count = 0
    episode_total_reward = 0.0

    log_probability_per_step: list[torch.Tensor] = []
    reward_per_step: list[float] = []

    while not episode_is_done:
        action_by_callsign, log_probability_by_callsign = (
            agent.sample_training_actions(observation_by_callsign)
        )

        (
            next_observation_by_callsign,
            reward_by_callsign,
            done_by_callsign,
            truncated_by_callsign,
            _info,
        ) = environment.step(action_by_callsign)

        # For this simple example, aggregate per-aircraft rewards into one
        # scalar reward for the timestep. With one aircraft, this is just that
        # aircraft's reward.
        timestep_reward = (
            float(sum(reward_by_callsign.values()))
            if reward_by_callsign
            else 0.0
        )

        # Aggregate log-probabilities for all aircraft active this step. With
        # one aircraft, this is just that aircraft's sampled action log-prob.
        if log_probability_by_callsign:
            timestep_log_probability = torch.stack(
                list(log_probability_by_callsign.values()),
            ).sum()
            log_probability_per_step.append(timestep_log_probability)
            reward_per_step.append(timestep_reward)

        episode_total_reward += timestep_reward

        # In decentralized mode, done is also per aircraft.
        # truncated_by_callsign is not used separately here because Bluebird
        # sets done when the episode is time-truncated in this path.
        _ = truncated_by_callsign
        episode_is_done = (
            all(done_by_callsign.values()) if done_by_callsign else True
        )

        observation_by_callsign = next_observation_by_callsign
        episode_step_count += 1

    loss_value = agent.update_policy_from_episode(
        log_probability_per_step=log_probability_per_step,
        reward_per_step=reward_per_step,
        discount_factor_gamma=discount_factor_gamma,
    )

    return episode_total_reward, episode_step_count, loss_value


def run_one_evaluation_episode(
    environment: SectorIEnv,
    agent: SharedPolicyAgent,
    random_seed: int,
) -> tuple[float, int]:
    """Run one episode without updating network weights."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    observation_by_callsign, _info = environment.reset(seed=random_seed)

    episode_is_done = False
    episode_step_count = 0
    episode_total_reward = 0.0

    while not episode_is_done:
        action_by_callsign = agent.choose_evaluation_actions(
            observation_by_callsign,
        )

        (
            next_observation_by_callsign,
            reward_by_callsign,
            done_by_callsign,
            truncated_by_callsign,
            _info,
        ) = environment.step(action_by_callsign)

        _ = truncated_by_callsign
        episode_total_reward += (
            float(sum(reward_by_callsign.values()))
            if reward_by_callsign
            else 0.0
        )
        episode_is_done = (
            all(done_by_callsign.values()) if done_by_callsign else True
        )
        observation_by_callsign = next_observation_by_callsign
        episode_step_count += 1

    return episode_total_reward, episode_step_count


def main() -> None:
    config = make_sector_i_training_config()
    environment = SectorIEnv(config=config)

    # In decentralized mode, this is the length of one aircraft's observation
    # vector. For extra_minimal with k_nearest_aircraft=1, it is typically 4.
    observation_dimension = environment.observation_space.shape[0]

    # This is the number of discrete actions available to each aircraft.
    number_of_actions = environment.action_space.n

    agent = SharedPolicyAgent(
        observation_dimension=observation_dimension,
        number_of_actions=number_of_actions,
        learning_rate=1e-3,
        hidden_units=128,
    )

    print(
        "environment shapes:",
        f"observation_dimension={observation_dimension}",
        f"number_of_actions={number_of_actions}",
    )

    discount_factor_gamma = 0.99

    for episode_index in range(20):
        random_seed = 100 + episode_index
        total_reward, step_count, loss_value = run_one_training_episode(
            environment=environment,
            agent=agent,
            random_seed=random_seed,
            discount_factor_gamma=discount_factor_gamma,
        )

        print(
            "[train]",
            f"episode={episode_index:02d}",
            f"seed={random_seed}",
            f"reward={total_reward:.3f}",
            f"steps={step_count}",
            f"loss={loss_value}",
        )

    for random_seed in [200, 201, 202]:
        total_reward, step_count = run_one_evaluation_episode(
            environment=environment,
            agent=agent,
            random_seed=random_seed,
        )

        print(
            "[eval]",
            f"seed={random_seed}",
            f"reward={total_reward:.3f}",
            f"steps={step_count}",
        )


if __name__ == "__main__":
    main()
