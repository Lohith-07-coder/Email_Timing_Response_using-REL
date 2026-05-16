import io
import os
import pickle
import sys

import torch

from agent.dqn import DQNAgent

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_HERE, "models")


def get_trained_agent(mode: str = "dqn", force_retrain: bool = False):
    weights_path = os.path.join(_MODELS, "dqn_weights.pt")
    pkl_path = os.path.join(_MODELS, "dqn.pkl")

    # ── Strategy 1: state_dict load (ALWAYS works GPU→CPU) ───────────────────
    if not force_retrain and os.path.exists(weights_path):
        print("  Loading dqn_weights.pt (CPU-safe)...")
        try:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            agent = DQNAgent(
                state_size=ckpt.get("state_size", 5),
                action_size=ckpt.get("action_size", 4),
            )
            agent._policy_net.load_state_dict(ckpt["policy_state_dict"])
            agent._target_net.load_state_dict(ckpt["target_state_dict"])
            agent._policy_net.eval()
            agent._target_net.eval()
            agent._device = torch.device("cpu")
            agent._policy_net = agent._policy_net.to("cpu")
            agent._target_net = agent._target_net.to("cpu")
            agent.epsilon = ckpt.get("epsilon", 0.0)
            print("  Agent loaded from weights. Device: cpu")
            return agent
        except Exception as e:
            print(f"  weights load failed ({e}), trying pkl...")

    # ── Strategy 2: full pickle with CPU remapping (fallback) ─────────────────
    if not force_retrain and os.path.exists(pkl_path):
        print("  Loading dqn.pkl with CPU remapping...")
        try:

            class _CPUUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == "torch.storage" and name == "_load_from_bytes":
                        return lambda b: torch.load(
                            io.BytesIO(b), map_location="cpu", weights_only=False
                        )
                    return super().find_class(module, name)

            with open(pkl_path, "rb") as f:
                data = _CPUUnpickler(f).load()

            # pkl saved as state_dict directly in ONE_CELL_TRAIN
            if isinstance(data, dict) and "policy_state_dict" not in data:
                # it's a raw state_dict
                agent = DQNAgent()
                agent._policy_net.load_state_dict(data)
                agent._target_net.load_state_dict(data)
            elif isinstance(data, DQNAgent):
                agent = data
                agent._device = torch.device("cpu")
                agent._policy_net = agent._policy_net.to("cpu")
                agent._target_net = agent._target_net.to("cpu")
                # fix old buffer attribute name if needed
                if hasattr(agent._buffer, "_buf") and not hasattr(
                    agent._buffer, "_buffer"
                ):
                    agent._buffer._buffer = agent._buffer._buf
            else:
                agent = DQNAgent()
                agent._policy_net.load_state_dict(data)

            agent._policy_net.eval()
            agent._target_net.eval()
            agent._device = torch.device("cpu")
            agent._policy_net = agent._policy_net.to("cpu")
            agent._target_net = agent._target_net.to("cpu")
            agent.epsilon = agent.epsilon_min
            print("  Agent loaded from pkl. Device: cpu")
            return agent
        except Exception as e:
            print(f"  pkl load failed: {e}")

    raise FileNotFoundError(
        "No trained model found!\n"
        "Place dqn_weights.pt (and dqn.pkl) in the models/ folder.\n"
        "Run ONE_CELL_TRAIN.py in Colab to generate them."
    )
