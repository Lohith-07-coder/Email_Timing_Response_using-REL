import matplotlib.pyplot as plt
import numpy as np

from agent.base import BaseAgent
from environment.base import BaseEnvironment


class Evaluator:
    """
    Evaluates a trained agent and produces:
        1. Average reward over a test run
        2. Decision accuracy per priority level
        3. Reward-vs-episodes learning curve plot
    """

    OPTIMAL_ACTIONS = {3: 0, 2: 2, 1: 3}  # priority → best action
    ACTION_LABELS = {
        0: "reply_now",
        1: "delay_reply",
        2: "mark_important",
        3: "archive",
    }

    def __init__(self, env: BaseEnvironment, agent: BaseAgent):
        self._env = env
        self._agent = agent

    # 1. test run

    def evaluate(self, episodes: int = 100) -> dict:
        """
        Run the agent greedily (no exploration) and collect metrics.
        Returns a dict of results so the caller decides what to display.
        """
        saved_epsilon = self._agent.epsilon
        self._agent.epsilon = 0.0  # pure exploitation during eval

        rewards, correct, total = [], 0, 0

        for _ in range(episodes):
            state = self._env.reset()
            ep_reward, done = 0.0, False

            while not done:
                action = self._agent.select_action(state)
                next_state, reward, done = self._env.step(action)
                priority = int(state[0])
                correct += int(action == self.OPTIMAL_ACTIONS[priority])
                total += 1
                ep_reward += reward
                state = next_state

            rewards.append(ep_reward)

        self._agent.epsilon = saved_epsilon  # restore

        return {
            "mean_reward": round(float(np.mean(rewards)), 2),
            "accuracy": round(correct / total * 100, 1),
            "min_reward": round(float(np.min(rewards)), 2),
            "max_reward": round(float(np.max(rewards)), 2),
        }

    def print_results(self, results: dict) -> None:
        print("\n── Evaluation Results")
        print(f"  Mean reward   : {results['mean_reward']:+.2f}")
        print(f"  Decision acc  : {results['accuracy']}%")
        print(
            f"  Reward range  : {results['min_reward']:+.2f}  "
            f"→  {results['max_reward']:+.2f}"
        )

    # 2. learning curve

    def plot_rewards(
        self,
        reward_history: list[float],
        window: int = 50,
        save_path: str = "reward_curve.png",
    ) -> None:
        """
        Plot raw rewards + a rolling average (smoothed curve).
        The smoothed line is what shows the agent is learning.

        WHY ROLLING AVERAGE?
            Raw rewards are noisy episode-to-episode.
            A window=50 rolling mean reveals the upward trend clearly.
        """
        episodes = range(1, len(reward_history) + 1)
        rolling = self._rolling_mean(reward_history, window)

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            episodes,
            reward_history,
            color="#94a3b8",
            alpha=0.4,
            linewidth=0.8,
            label="Raw reward",
        )
        ax.plot(
            episodes[window - 1 :],
            rolling,
            color="#38bdf8",
            linewidth=2.2,
            label=f"Rolling avg (window={window})",
        )

        ax.axhline(0, color="#475569", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Episode", fontsize=12)
        ax.set_ylabel("Total Reward", fontsize=12)
        ax.set_title(
            "Q-Learning Agent — Reward vs Episodes",
            fontsize=14,
            pad=14,
        )
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved → {save_path}")

    # helper

    @staticmethod
    def _rolling_mean(values: list[float], window: int) -> np.ndarray:
        return np.convolve(values, np.ones(window) / window, mode="valid")
