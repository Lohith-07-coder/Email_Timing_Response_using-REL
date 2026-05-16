import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.base import BaseAgent


class QNetwork(nn.Module):
    """
    64-64 network — matches exactly what was trained in ONE_CELL_TRAIN.py.
    DO NOT change layer sizes or the saved weights will not load.
    """

    def __init__(self, state_size: int = 5, action_size: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self._buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done=False) -> None:
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple:
        batch = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)


class DQNAgent(BaseAgent):
    def __init__(
        self,
        state_size: int = 5,
        action_size: int = 4,
        alpha: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
        batch_size: int = 64,
        target_update: int = 100,
        buffer_capacity: int = 20_000,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self._episode = 0

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._policy_net = QNetwork(state_size, action_size).to(self._device)
        self._target_net = QNetwork(state_size, action_size).to(self._device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        self._optimizer = optim.Adam(self._policy_net.parameters(), lr=alpha)
        self._buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self._device)
            return int(self._policy_net(t).argmax().item())

    def learn(self, state, action, reward, next_state, done) -> None:
        self._buffer.push(state, action, reward, next_state, done)
        if len(self._buffer) < self.batch_size:
            return
        self._train_step()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._episode += 1
        if self._episode % self.target_update == 0:
            self._target_net.load_state_dict(self._policy_net.state_dict())

    def _train_step(self) -> None:
        states, actions, rewards, next_states, dones = [
            x.to(self._device) for x in self._buffer.sample(self.batch_size)
        ]
        current_q = self._policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: policy net selects the action, target net evaluates it.
            # Reduces Q-value overestimation vs vanilla DQN (max from same net).
            best_acts = self._policy_net(next_states).argmax(1)  # action selection
            max_next_q = (
                self._target_net(next_states)
                .gather(1, best_acts.unsqueeze(1))
                .squeeze(1)
            )  # value evaluation
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        loss = nn.SmoothL1Loss()(current_q, target_q)
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._policy_net.parameters(), 1.0)
        self._optimizer.step()

    @property
    def q_table_size(self) -> int:
        return len(self._buffer)
