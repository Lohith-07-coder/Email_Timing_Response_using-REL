import argparse
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agent.base import BaseAgent
from agent.dqn import DQNAgent
from agent.q_learning import QLearningAgent
from config import Config
from environment.base import BaseEnvironment
from environment.email_env import EmailEnvironment
from monitoring.mlflow_tracker import (
    Timer,
    log_agent_params,
    log_config,
    log_dqn_model,
    log_env_params,
    log_episode,
    log_eval,
    log_pkl_file,
    log_reward_history,
    log_summary,
    log_weights_file,
    start_run,
)
from simulation.simulator import EmailSimulator
from simulation.sources.enron import EnronSource
from simulation.sources.synthetic import SyntheticEmailSource

EVAL_EVERY = 200
EVAL_EPISODES = 5


def _source_from_name(source_name: str):
    if source_name == "synthetic":
        return SyntheticEmailSource(seed=Config.SEED)
    if source_name == "enron":
        dataset_path = os.getenv("ENRON_DATASET_PATH", "data/enron")
        return EnronSource(dataset_path)
    raise ValueError(f"Unknown source: {source_name}")


def _build_env(source: str) -> EmailEnvironment:
    simulator = EmailSimulator(_source_from_name(source))
    return EmailEnvironment(simulator, max_steps=Config.MAX_STEPS)


def _dqn_params() -> dict[str, Any]:
    return {
        "state_size": Config.STATE_SIZE,
        "action_size": Config.ACTION_SIZE,
        "dqn_alpha": Config.DQN_ALPHA,
        "dqn_gamma": Config.DQN_GAMMA,
        "dqn_epsilon": Config.DQN_EPSILON,
        "dqn_epsilon_min": Config.DQN_EPSILON_MIN,
        "dqn_epsilon_decay": Config.DQN_EPSILON_DECAY,
        "dqn_batch_size": Config.DQN_BATCH_SIZE,
        "dqn_target_update": Config.DQN_TARGET_UPDATE,
        "dqn_buffer_cap": Config.DQN_BUFFER_CAP,
    }


def _ql_params() -> dict[str, Any]:
    return {
        "state_size": Config.STATE_SIZE,
        "action_size": Config.ACTION_SIZE,
        "ql_alpha": Config.QL_ALPHA,
        "ql_gamma": Config.QL_GAMMA,
        "ql_epsilon": Config.QL_EPSILON,
        "ql_epsilon_min": Config.QL_EPSILON_MIN,
        "ql_epsilon_decay": Config.QL_EPSILON_DECAY,
    }


def _make_dqn_agent() -> DQNAgent:
    return DQNAgent(
        state_size=Config.STATE_SIZE,
        action_size=Config.ACTION_SIZE,
        alpha=Config.DQN_ALPHA,
        gamma=Config.DQN_GAMMA,
        epsilon=Config.DQN_EPSILON,
        epsilon_min=Config.DQN_EPSILON_MIN,
        epsilon_decay=Config.DQN_EPSILON_DECAY,
        batch_size=Config.DQN_BATCH_SIZE,
        target_update=Config.DQN_TARGET_UPDATE,
        buffer_capacity=Config.DQN_BUFFER_CAP,
    )


def _make_ql_agent() -> QLearningAgent:
    return QLearningAgent(
        state_size=Config.STATE_SIZE,
        action_size=Config.ACTION_SIZE,
        alpha=Config.QL_ALPHA,
        gamma=Config.QL_GAMMA,
        epsilon=Config.QL_EPSILON,
        epsilon_min=Config.QL_EPSILON_MIN,
        epsilon_decay=Config.QL_EPSILON_DECAY,
    )


def _run_episode(
    env: BaseEnvironment, agent: BaseAgent, train: bool = True
) -> tuple[float, int]:
    state = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        if train:
            agent.learn(state, action, reward, next_state, done)
        total_reward += reward
        steps += 1
        state = next_state

    return total_reward, steps


def _evaluate(
    agent: BaseAgent,
    source: str,
    episodes: int = EVAL_EPISODES,
) -> tuple[list[float], dict[int, int]]:
    previous_epsilon = getattr(agent, "epsilon", None)
    if previous_epsilon is not None:
        agent.epsilon = 0.0

    env = _build_env(source)
    rewards = []
    action_counts: Counter[int] = Counter()

    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(state)
            action_counts[action] += 1
            state, reward, done = env.step(action)
            total_reward += reward
        rewards.append(total_reward)

    if previous_epsilon is not None:
        agent.epsilon = previous_epsilon

    return rewards, dict(action_counts)


def _save_dqn_checkpoint(agent: DQNAgent, path: str, episodes: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "policy_state_dict": agent._policy_net.state_dict(),
        "target_state_dict": agent._target_net.state_dict(),
        "epsilon": agent.epsilon,
        "state_size": Config.STATE_SIZE,
        "action_size": Config.ACTION_SIZE,
        "episodes_trained": episodes,
    }
    torch.save(checkpoint, path)


def _save_q_learning(agent: QLearningAgent, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dict(agent.q_table), f)


class Trainer:
    def __init__(
        self,
        env: BaseEnvironment,
        agent: BaseAgent,
        episodes: int = 1000,
        log_every: int = 100,
        checkpoint_every: int = 0,
        checkpoint_dir: str = "saved_models",
    ):
        self._env = env
        self._agent = agent
        self._episodes = episodes
        self._log_every = log_every
        self._checkpoint_every = checkpoint_every
        self._checkpoint_dir = checkpoint_dir

    def run(self) -> list[float]:
        reward_history = []
        for episode in range(1, self._episodes + 1):
            reward, _ = _run_episode(self._env, self._agent, train=True)
            reward_history.append(reward)
            if hasattr(self._agent, "decay_epsilon"):
                self._agent.decay_epsilon()
            if self._log_every and episode % self._log_every == 0:
                print(
                    f"episode={episode} reward={reward:.2f} "
                    f"epsilon={getattr(self._agent, 'epsilon', 0.0):.4f}"
                )
            if self._checkpoint_every and episode % self._checkpoint_every == 0:
                path = Path(self._checkpoint_dir) / f"checkpoint_ep{episode}.pkl"
                self.save(str(path))
        return reward_history

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._agent, f)

    @staticmethod
    def load(path: str) -> BaseAgent:
        with open(path, "rb") as f:
            return pickle.load(f)


def _maybe_log_best_eval(
    agent: BaseAgent,
    source: str,
    episode: int,
    best_eval_reward: float | None,
) -> tuple[float | None, bool]:
    rewards, action_counts = _evaluate(agent, source)
    log_eval(episode, rewards, action_counts)
    eval_reward = sum(rewards) / len(rewards)

    if best_eval_reward is None or eval_reward > best_eval_reward:
        return eval_reward, True
    return best_eval_reward, False


def train_dqn(
    episodes: int = Config.EPISODES,
    run_name: str | None = None,
    source: str = "synthetic",
) -> dict[str, Any]:
    agent = _make_dqn_agent()
    env = _build_env(source)
    rewards = []
    total_steps = 0
    best_eval_reward = None
    timer = Timer()

    with start_run(run_name, "dqn") as run:
        log_agent_params("dqn", _dqn_params())
        log_env_params({"source": source, "max_steps": Config.MAX_STEPS})
        log_config()

        for episode in range(1, episodes + 1):
            reward, steps = _run_episode(env, agent, train=True)
            agent.decay_epsilon()
            rewards.append(reward)
            total_steps += steps
            log_episode(
                episode=episode,
                reward=reward,
                epsilon=agent.epsilon,
                steps=steps,
                buffer_size=len(agent._buffer),
            )

            if episode % EVAL_EVERY == 0:
                best_eval_reward, improved = _maybe_log_best_eval(
                    agent, source, episode, best_eval_reward
                )
                if improved:
                    _save_dqn_checkpoint(agent, Config.DQN_WEIGHTS_PATH, episode)

        if best_eval_reward is None:
            best_eval_reward, _ = _maybe_log_best_eval(
                agent, source, episodes, best_eval_reward
            )
            _save_dqn_checkpoint(agent, Config.DQN_WEIGHTS_PATH, episodes)

        log_weights_file(Config.DQN_WEIGHTS_PATH)
        input_example = np.zeros((1, Config.STATE_SIZE), dtype=np.float32)
        log_dqn_model(agent._policy_net, input_example=input_example)
        log_reward_history(rewards)
        log_summary(episodes, total_steps, timer.elapsed(), best_eval_reward)

        return {
            "agent": agent,
            "agent_type": "dqn",
            "best_eval_reward": best_eval_reward,
            "reward_history": rewards,
            "run_id": run.info.run_id,
            "total_steps": total_steps,
        }


def train_q_learning(
    episodes: int = Config.EPISODES,
    run_name: str | None = None,
    source: str = "synthetic",
) -> dict[str, Any]:
    agent = _make_ql_agent()
    env = _build_env(source)
    rewards = []
    total_steps = 0
    best_eval_reward = None
    pkl_path = Config.DQN_MODEL_PATH
    timer = Timer()

    with start_run(run_name, "q_learning") as run:
        log_agent_params("q_learning", _ql_params())
        log_env_params({"source": source, "max_steps": Config.MAX_STEPS})
        log_config()

        for episode in range(1, episodes + 1):
            reward, steps = _run_episode(env, agent, train=True)
            agent.decay_epsilon()
            rewards.append(reward)
            total_steps += steps
            log_episode(
                episode=episode,
                reward=reward,
                epsilon=agent.epsilon,
                steps=steps,
                buffer_size=len(agent.q_table),
            )

            if episode % EVAL_EVERY == 0:
                best_eval_reward, improved = _maybe_log_best_eval(
                    agent, source, episode, best_eval_reward
                )
                if improved:
                    _save_q_learning(agent, pkl_path)

        if best_eval_reward is None:
            best_eval_reward, _ = _maybe_log_best_eval(
                agent, source, episodes, best_eval_reward
            )
            _save_q_learning(agent, pkl_path)

        log_pkl_file(pkl_path)
        log_reward_history(rewards)
        log_summary(episodes, total_steps, timer.elapsed(), best_eval_reward)

        return {
            "agent": agent,
            "agent_type": "q_learning",
            "best_eval_reward": best_eval_reward,
            "reward_history": rewards,
            "run_id": run.info.run_id,
            "total_steps": total_steps,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the email timing RL agent.")
    parser.add_argument("--agent", choices=["dqn", "q_learning"], default="dqn")
    parser.add_argument("--episodes", type=int, default=Config.EPISODES)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--source", choices=["synthetic", "enron"], default="synthetic")
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    if args.agent == "dqn":
        return train_dqn(args.episodes, args.run_name, args.source)
    return train_q_learning(args.episodes, args.run_name, args.source)


if __name__ == "__main__":
    main()
