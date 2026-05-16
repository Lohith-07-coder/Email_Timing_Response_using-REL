import pickle
from collections import defaultdict

import numpy as np

from agent.base import BaseAgent


class QLearningAgent(BaseAgent):

    def __init__(
        self,
        state_size: int = 5,
        action_size: int = 4,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(action_size))

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return int(np.argmax(self.q_table[self._key(state)]))

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        s, s_ = self._key(state), self._key(next_state)
        future_value = 0.0 if done else np.max(self.q_table[s_])
        td_error = (reward + self.gamma * future_value) - self.q_table[s][action]
        self.q_table[s][action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)
        print(f"✅ Model saved → {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.action_size), data)
        self.epsilon = self.epsilon_min  # loaded model should exploit, not explore
        print(f"✅ Model loaded ← {path}")

    @staticmethod
    def _key(state: np.ndarray) -> tuple:
        return tuple(state.astype(int).tolist())

    @property
    def q_table_size(self) -> int:
        return len(self.q_table)
