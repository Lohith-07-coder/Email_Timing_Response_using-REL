<<<<<<< HEAD
"""
End-to-end training pipeline with MLflow experiment tracking.

Wires together: simulator → environment → agent → trainer → model persistence.
All hyper-parameters, episode metrics, and model artefacts are logged to MLflow.

Usage:
    python pipelines/training_pipeline.py
    python pipelines/training_pipeline.py --episodes 5000 --source synthetic
"""

import argparse
import os
import sys
import time

import mlflow
import torch

from agent.dqn import DQNAgent
from config import Config
from environment.email_env import EmailEnvironment
from mlflow_config import MLflowConfig, init_mlflow
from monitoring.logging_config import get_logger
from monitoring.mlflow_logger import (
    log_episode_metrics,
    log_model_artifact,
    log_reward_curve,
    log_training_params,
)
from simulation.simulator import EmailSimulator
from simulation.sources.enron import EnronSource
from simulation.sources.synthetic import SyntheticEmailSource
from training.trainer import Trainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


logger = get_logger(__name__)
=======
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from monitoring.model_registry import (
    DQN_MODEL_NAME,
    QL_MODEL_NAME,
    compare_runs,
    promote_model,
    register_dqn_model,
    register_ql_model,
)
from training.trainer import train_dqn, train_q_learning
>>>>>>> 729dc55db9c02ec9a9c0305798c8c49c755f0b64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Email Timing Response training pipeline"
    )
<<<<<<< HEAD
    p.add_argument("--output-dir", default=Config.MODEL_DIR)
    p.add_argument("--checkpoint-every", type=int, default=Config.CHECKPOINT_EVERY)
    p.add_argument("--log-every", type=int, default=Config.LOG_EVERY)
    p.add_argument("--tag", default=None, help="Optional version tag for saved model")
    p.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking for this run",
    )
    return p.parse_args()
=======
    parser.add_argument("--agent", choices=["dqn", "q_learning"], default="dqn")
    parser.add_argument("--episodes", type=int, default=Config.EPISODES)
    parser.add_argument("--source", choices=["synthetic", "enron"], default="synthetic")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--promote", action="store_true")
    return parser.parse_args()
>>>>>>> 729dc55db9c02ec9a9c0305798c8c49c755f0b64


def _register_model(agent_type: str, run_id: str):
    if agent_type == "dqn":
        return register_dqn_model(run_id)
    return register_ql_model(run_id)

<<<<<<< HEAD
    use_mlflow = not getattr(args, "no_mlflow", False)

    logger.info("=== Training Pipeline Started ===")
    logger.info("episodes=%d  source=%s  output=%s", args.episodes, args.source, args.output_dir)

    # ── Initialise MLflow ─────────────────────────────────────────────────────
    if use_mlflow:
        init_mlflow(MLflowConfig.EXPERIMENT_DQN)

    # ── Build pipeline ────────────────────────────────────────────────────────
    if args.source == "synthetic":
        source = SyntheticEmailSource()
    elif args.source == "enron":
        source = EnronSource()
=======

def run(args: argparse.Namespace | None = None):
    args = args or parse_args()
    if args.agent == "dqn":
        result = train_dqn(args.episodes, args.run_name, args.source)
        model_name = DQN_MODEL_NAME
>>>>>>> 729dc55db9c02ec9a9c0305798c8c49c755f0b64
    else:
        result = train_q_learning(args.episodes, args.run_name, args.source)
        model_name = QL_MODEL_NAME

    registered_model = None
    if args.register:
        registered_model = _register_model(args.agent, result["run_id"])
        print(
            "Registered model "
            f"{registered_model.name} version {registered_model.version}"
        )

    if args.promote:
        if registered_model is None:
            registered_model = _register_model(args.agent, result["run_id"])
        promote_model(model_name, registered_model.version)
        print(f"Promoted {model_name} version {registered_model.version} to Production")

<<<<<<< HEAD
    # ── MLflow run context ────────────────────────────────────────────────────
    version_tag = args.tag or time.strftime("%Y%m%d_%H%M%S")

    mlflow_ctx = mlflow.start_run(run_name=f"dqn-{version_tag}") if use_mlflow else _noop_ctx()

    with mlflow_ctx:
        # Log all hyper-parameters
        if use_mlflow:
            log_training_params(
                {
                    "algorithm": "Double-DQN",
                    "episodes": args.episodes,
                    "source": args.source,
                    "state_size": Config.STATE_SIZE,
                    "action_size": Config.ACTION_SIZE,
                    "alpha": Config.DQN_ALPHA,
                    "gamma": Config.DQN_GAMMA,
                    "epsilon_start": Config.DQN_EPSILON,
                    "epsilon_min": Config.DQN_EPSILON_MIN,
                    "epsilon_decay": Config.DQN_EPSILON_DECAY,
                    "batch_size": Config.DQN_BATCH_SIZE,
                    "target_update": Config.DQN_TARGET_UPDATE,
                    "buffer_capacity": Config.DQN_BUFFER_CAP,
                    "max_steps_per_episode": Config.MAX_STEPS,
                }
            )

        # ── Train ─────────────────────────────────────────────────────────────
        t0 = time.time()
        reward_history = trainer.run()
        elapsed = (time.time() - t0) / 60

        # ── Log episode-level metrics to MLflow ───────────────────────────────
        if use_mlflow:
            # Log sampled episodes to avoid excessive data (every log_every)
            for ep_idx in range(0, len(reward_history), max(1, args.log_every)):
                window = reward_history[max(0, ep_idx - 500) : ep_idx + 1]
                avg_r = sum(window) / len(window) if window else 0.0
                log_episode_metrics(
                    episode=ep_idx + 1,
                    reward=reward_history[ep_idx],
                    epsilon=max(
                        Config.DQN_EPSILON_MIN,
                        Config.DQN_EPSILON * (Config.DQN_EPSILON_DECAY ** (ep_idx + 1)),
                    ),
                    avg_reward=avg_r,
                )

        # ── Save versioned weights ────────────────────────────────────────────
        os.makedirs(args.output_dir, exist_ok=True)
        weights_path = os.path.join(args.output_dir, f"dqn_weights_{version_tag}.pt")

        checkpoint = {
            "policy_state_dict": agent._policy_net.state_dict(),
            "target_state_dict": agent._target_net.state_dict(),
            "epsilon": agent.epsilon,
            "state_size": Config.STATE_SIZE,
            "action_size": Config.ACTION_SIZE,
            "episodes_trained": args.episodes,
            "version": version_tag,
            "reward_history_tail": reward_history[-100:],
        }
        torch.save(checkpoint, weights_path)
        logger.info("Weights saved → %s", weights_path)

        # Also overwrite the default path for the inference service
        torch.save(checkpoint, Config.DQN_WEIGHTS_PATH)
        logger.info("Default weights updated → %s", Config.DQN_WEIGHTS_PATH)

        avg_reward = sum(reward_history[-500:]) / min(500, len(reward_history))

        # ── Log final metrics and artefacts to MLflow ─────────────────────────
        if use_mlflow:
            mlflow.log_metrics(
                {
                    "final_avg_reward_500": avg_reward,
                    "final_epsilon": agent.epsilon,
                    "training_minutes": round(elapsed, 2),
                    "total_episodes": args.episodes,
                }
            )
            log_model_artifact(weights_path)
            log_reward_curve(reward_history)
            mlflow.set_tag("version", version_tag)
            mlflow.set_tag("source", args.source)

        logger.info(
            "=== Training Complete | %.1f min | avg_reward(last500)=%.2f ===",
            elapsed,
            avg_reward,
        )
        return reward_history


class _noop_ctx:
    """Dummy context manager when MLflow is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
=======
    print(compare_runs(top_n=5))
    return result
>>>>>>> 729dc55db9c02ec9a9c0305798c8c49c755f0b64


if __name__ == "__main__":
    run()
