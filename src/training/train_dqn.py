"""Training pipeline for the Deep Q-Network (DQN) agent (Stable-Baselines3).

Run as a module::

    python -m src.training.train_dqn --timesteps 200000
    python -m src.training.train_dqn --timesteps 200000 \\
        --reward-profile safety_reward_v1 --run-name safety_reward_v1
"""

from __future__ import annotations

import argparse

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.agents.dqn_agent import DQNAgent
from src.config import CONFIG, Config, config_for_profile
from src.environment.factory import SINGLE_STAGE, load_dataset_bundle, make_env
from src.utils.helpers import get_logger, save_json, set_global_seed
from src.utils.plotting import plot_reward_curve

logger = get_logger(__name__)


class EpisodeRewardCallback(BaseCallback):
    """Collect per-episode rewards emitted by a :class:`Monitor` wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self.episode_rewards.append(float(episode["r"]))
        return True


def train_dqn(
    config: Config = CONFIG,
    *,
    total_timesteps: int | None = None,
    reward_profile: str = "baseline",
    run_name: str | None = None,
    dataset: str = "baseline",
    environment: str = SINGLE_STAGE,
) -> DQNAgent:
    """Train a DQN agent on the chip-testing environment.

    Args:
        config: Project configuration.
        total_timesteps: Optional override for the training-step budget.
        reward_profile: Named reward profile to train under (see
            :data:`~src.config.REWARD_PROFILES`).
        run_name: Optional experiment name; outputs are isolated under
            ``results/runs/<run_name>/`` when set.
        dataset: Dataset name (``"baseline"`` or ``"full_stage_v1"``).
        environment: ``"single_stage"`` or ``"multi_stage"``.

    Returns:
        The trained :class:`DQNAgent`.
    """
    run_config = config_for_profile(reward_profile, config)
    run_paths = config.paths.run_paths(run_name)
    run_paths.ensure()

    set_global_seed(run_config.seed)
    bundle = load_dataset_bundle(dataset, environment, run_config)
    env = Monitor(make_env(bundle, "train", run_config, reward_config=run_config.reward))
    logger.info(
        "DQN | dataset=%s | environment=%s | reward_profile=%s | run_name=%s",
        dataset,
        environment,
        reward_profile,
        run_name or "baseline",
    )

    agent = DQNAgent(run_config)
    agent.build(env)
    callback = EpisodeRewardCallback()
    agent.train(env, total_timesteps=total_timesteps, callback=callback)

    model_path = run_paths.models / "dqn"
    agent.save(model_path)
    save_json(
        {
            "episode_rewards": callback.episode_rewards,
            "total_timesteps": total_timesteps or run_config.dqn.total_timesteps,
            "n_episodes": len(callback.episode_rewards),
            "reward_profile": reward_profile,
            "run_name": run_name or "baseline",
            "dataset": dataset,
            "environment": environment,
        },
        run_paths.metrics / "dqn_training.json",
    )
    if callback.episode_rewards:
        plot_reward_curve(
            callback.episode_rewards,
            run_paths.figures / "dqn_reward_curve.png",
            title="DQN Training Reward",
        )
    logger.info("Saved DQN model to %s.zip", model_path)
    return agent


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Train the DQN agent")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--reward-profile", type=str, default="baseline")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="baseline")
    parser.add_argument(
        "--environment", type=str, default=SINGLE_STAGE, choices=[SINGLE_STAGE, "multi_stage"]
    )
    args = parser.parse_args()
    train_dqn(
        total_timesteps=args.timesteps,
        reward_profile=args.reward_profile,
        run_name=args.run_name,
        dataset=args.dataset,
        environment=args.environment,
    )


if __name__ == "__main__":
    main()
