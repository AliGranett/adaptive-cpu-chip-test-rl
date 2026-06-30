"""Training pipeline for the tabular Q-learning agent.

Run as a module::

    python -m src.training.train_qlearning --episodes 20000 \\
        --dataset full_stage_v1 --run-name full_stage_v1
"""

from __future__ import annotations

import argparse

import numpy as np

from src.agents.q_learning_agent import QLearningAgent
from src.config import CONFIG, Config, config_for_profile
from src.environment.factory import DEFAULT_DATASET, MULTI_STAGE, load_dataset_bundle, make_env
from src.utils.helpers import get_logger, save_json, set_global_seed
from src.utils.plotting import plot_reward_curve

logger = get_logger(__name__)


def train_q_learning(
    config: Config = CONFIG,
    *,
    n_episodes: int | None = None,
    reward_profile: str = "full_stage_v1",
    run_name: str | None = None,
    dataset: str = DEFAULT_DATASET,
    environment: str = MULTI_STAGE,
) -> QLearningAgent:
    """Train a tabular Q-learning agent on the multi-stage chip-testing environment."""
    run_config = config_for_profile(reward_profile, config)
    run_paths = config.paths.run_paths(run_name)
    run_paths.ensure()

    set_global_seed(run_config.seed)
    bundle = load_dataset_bundle(dataset, environment, run_config)
    env = make_env(bundle, "train", run_config, reward_config=run_config.reward)
    logger.info(
        "Q-learning | dataset=%s | reward_profile=%s | run_name=%s",
        dataset,
        reward_profile,
        run_name or "default",
    )

    agent = QLearningAgent(n_features=env.n_features, config=run_config)
    episodes = n_episodes or config.qlearning.n_episodes
    episode_rewards: list[float] = []

    logger.info("Starting Q-learning training for %d episodes", episodes)
    for episode in range(episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action = agent.act(obs, explore=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            total_reward += reward
        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (episode + 1) % max(1, episodes // 20) == 0:
            recent = np.mean(episode_rewards[-100:])
            logger.info(
                "Episode %d/%d | avg100 reward %.3f | epsilon %.3f | states %d",
                episode + 1,
                episodes,
                recent,
                agent.epsilon,
                agent.n_states,
            )

    model_path = run_paths.models / "qlearning.pkl"
    agent.save(model_path)
    save_json(
        {
            "episode_rewards": episode_rewards,
            "n_episodes": episodes,
            "n_states": agent.n_states,
            "final_epsilon": agent.epsilon,
            "reward_profile": reward_profile,
            "run_name": run_name,
            "dataset": dataset,
            "environment": environment,
        },
        run_paths.metrics / "qlearning_training.json",
    )
    plot_reward_curve(
        episode_rewards,
        run_paths.figures / "qlearning_reward_curve.png",
        title="Q-Learning Training Reward",
    )
    logger.info("Saved Q-learning model to %s", model_path)
    return agent


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train the Q-learning agent")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--reward-profile", type=str, default="full_stage_v1")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--environment", type=str, default=MULTI_STAGE)
    args = parser.parse_args()
    train_q_learning(
        n_episodes=args.episodes,
        reward_profile=args.reward_profile,
        run_name=args.run_name,
        dataset=args.dataset,
        environment=args.environment,
    )


if __name__ == "__main__":
    main()
