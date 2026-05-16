"""
Unit tests for the DQN agent and Q-Network.

Run: pytest tests/test_model.py -v
"""

import numpy as np
import pytest
import torch

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def q_network():
    from agent.dqn import QNetwork

    return QNetwork(state_size=5, action_size=4)


@pytest.fixture
def dqn_agent():
    from agent.dqn import DQNAgent

    return DQNAgent(
        state_size=5,
        action_size=4,
        buffer_capacity=200,
        batch_size=16,
        epsilon=1.0,
    )


@pytest.fixture
def dummy_state():
    return np.array([0.5, 0.5, 0.2, 0.5, 0.5], dtype=np.float32)


# ── QNetwork tests ────────────────────────────────────────────────────────────


class TestQNetwork:
    def test_output_shape(self, q_network, dummy_state):
        t = torch.tensor(dummy_state).unsqueeze(0)
        out = q_network(t)
        assert out.shape == (1, 4), "Q-values should have shape (batch, action_size)"

    def test_output_is_finite(self, q_network, dummy_state):
        t = torch.tensor(dummy_state).unsqueeze(0)
        out = q_network(t)
        assert torch.isfinite(out).all(), "Q-values must not contain NaN/Inf"

    def test_batch_inference(self, q_network):
        batch = torch.rand(32, 5)
        out = q_network(batch)
        assert out.shape == (32, 4)


# ── DQNAgent tests ────────────────────────────────────────────────────────────


class TestDQNAgent:
    def test_select_action_range(self, dqn_agent, dummy_state):
        for _ in range(50):
            a = dqn_agent.select_action(dummy_state)
            assert 0 <= a < 4, f"action {a} out of valid range [0, 3]"

    def test_greedy_action_deterministic(self, dqn_agent, dummy_state):
        """With epsilon=0 the agent should always pick the same action."""
        dqn_agent.epsilon = 0.0
        actions = {dqn_agent.select_action(dummy_state) for _ in range(20)}
        assert len(actions) == 1, "Greedy policy must be deterministic"

    def test_learn_fills_buffer(self, dqn_agent, dummy_state):
        next_state = np.zeros(5, dtype=np.float32)
        for i in range(20):
            dqn_agent.learn(dummy_state, i % 4, 1.0, next_state, False)
        assert len(dqn_agent._buffer) == 20

    def test_learn_trains_when_buffer_full(self, dqn_agent, dummy_state):
        """After batch_size experiences, learn() should trigger a gradient step."""
        next_state = np.zeros(5, dtype=np.float32)
        params_before = [p.clone() for p in dqn_agent._policy_net.parameters()]
        for i in range(dqn_agent.batch_size + 5):
            dqn_agent.learn(dummy_state, i % 4, float(i % 3), next_state, i % 10 == 0)
        params_after = list(dqn_agent._policy_net.parameters())
        changed = any(
            not torch.equal(b, a) for b, a in zip(params_before, params_after)
        )
        assert changed, "Parameters should update after a full batch"

    def test_epsilon_decay(self, dqn_agent):
        initial_eps = dqn_agent.epsilon
        dqn_agent.decay_epsilon()
        assert (
            dqn_agent.epsilon < initial_eps
            or dqn_agent.epsilon == dqn_agent.epsilon_min
        )

    def test_epsilon_never_below_min(self, dqn_agent):
        for _ in range(10_000):
            dqn_agent.decay_epsilon()
        assert dqn_agent.epsilon >= dqn_agent.epsilon_min


# ── ReplayBuffer tests ────────────────────────────────────────────────────────


class TestReplayBuffer:
    def test_capacity(self):
        from agent.dqn import ReplayBuffer

        buf = ReplayBuffer(capacity=10)
        s = np.zeros(5, dtype=np.float32)
        for _ in range(20):
            buf.push(s, 0, 1.0, s, False)
        assert len(buf) == 10, "Buffer should not exceed capacity"

    def test_sample_shapes(self):
        from agent.dqn import ReplayBuffer

        buf = ReplayBuffer(capacity=50)
        s = np.zeros(5, dtype=np.float32)
        for _ in range(20):
            buf.push(s, 0, 1.0, s, False)
        states, actions, rewards, next_states, dones = buf.sample(8)
        assert states.shape == (8, 5)
        assert actions.shape == (8,)
        assert rewards.shape == (8,)
