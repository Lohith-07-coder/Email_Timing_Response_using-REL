"""
Unit tests for data classes and environment.

Run: pytest tests/test_data.py -v
"""

import os
import sys

import numpy as np
import pytest

from data.email_data import Email
from environment.email_env import EmailEnvironment
from environment.reward import RewardCalculator
from simulation.simulator import EmailSimulator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Email dataclass ───────────────────────────────────────────────────────────


class TestEmailDataclass:
    @pytest.fixture
    def sample_email(self):
        return Email(
            subject="Urgent: Server down",
            sender="ops@company.com",
            priority=3,
            sender_importance=3,
            waiting_time=5,
            workload=2,
            time_of_day=10,
        )

    def test_to_state_vector_shape(self, sample_email):
        vec = sample_email.to_state_vector()
        assert vec.shape == (5,), "State vector must be length 5"

    def test_to_state_vector_normalized(self, sample_email):
        vec = sample_email.to_state_vector()
        assert np.all(vec >= 0.0), "All features must be >= 0"
        assert np.all(vec <= 1.0), "All features must be <= 1 (normalized)"

    def test_to_state_vector_dtype(self, sample_email):
        vec = sample_email.to_state_vector()
        assert vec.dtype == np.float32

    def test_edge_case_min_values(self):
        e = Email(
            "a",
            "b",
            priority=1,
            sender_importance=1,
            waiting_time=0,
            workload=1,
            time_of_day=0,
        )
        vec = e.to_state_vector()
        assert np.all(vec == 0.0), "Min-value email should map to all zeros"

    def test_edge_case_max_values(self):
        e = Email(
            "a",
            "b",
            priority=3,
            sender_importance=3,
            waiting_time=1440,
            workload=3,
            time_of_day=23,
        )
        vec = e.to_state_vector()
        assert np.all(vec == 1.0), "Max-value email should map to all ones"

    def test_waiting_time_clipped_at_1440(self):
        e1 = Email("a", "b", 2, 2, waiting_time=1440, workload=2, time_of_day=12)
        e2 = Email("a", "b", 2, 2, waiting_time=9999, workload=2, time_of_day=12)
        vec1 = e1.to_state_vector()
        vec2 = e2.to_state_vector()
        assert vec1[2] == vec2[2] == 1.0, "waiting_time should be clipped at 1440"


# ── RewardCalculator ──────────────────────────────────────────────────────────


class TestRewardCalculator:
    @pytest.fixture
    def calc(self):
        return RewardCalculator()

    @pytest.fixture
    def high_priority_email(self):
        return Email("Urgent", "boss@co", 3, 3, 120, 1, 9)

    @pytest.fixture
    def low_priority_email(self):
        return Email("Newsletter", "promo@co", 1, 1, 0, 1, 22)

    def test_reward_is_numeric(self, calc, high_priority_email):
        r = calc.calculate(high_priority_email, action=0)
        assert isinstance(r, (int, float))

    def test_reply_now_high_priority_positive(self, calc, high_priority_email):
        r = calc.calculate(high_priority_email, action=0)  # reply_now
        assert (
            r > 0
        ), "Replying immediately to high-priority email should yield positive reward"

    def test_archive_low_priority_positive(self, calc, low_priority_email):
        r = calc.calculate(low_priority_email, action=3)  # archive
        assert r >= 0, "Archiving low-priority email should not be penalized"


# ── EmailEnvironment ──────────────────────────────────────────────────────────


class TestEmailEnvironment:
    @pytest.fixture
    def env(self):
        from simulation.sources.synthetic import SyntheticEmailSource

        sim = EmailSimulator(SyntheticEmailSource(seed=42))
        return EmailEnvironment(sim, max_steps=10)

    def test_reset_returns_state(self, env):
        state = env.reset()
        assert state.shape == (5,)
        assert np.all(np.isfinite(state))

    def test_step_returns_tuple(self, env):
        env.reset()
        next_state, reward, done = env.step(0)
        assert next_state.shape == (5,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_episode_terminates_at_max_steps(self, env):
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, done = env.step(np.random.randint(4))
            steps += 1
        assert steps == 10, "Episode should end at max_steps=10"

    def test_action_labels(self, env):
        labels = env.action_labels
        assert set(labels.values()) == {
            "reply_now",
            "delay_reply",
            "mark_important",
            "archive",
        }
