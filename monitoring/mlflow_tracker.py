import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import mlflow
    import mlflow.pytorch
except ImportError:  # pragma: no cover - only used when deps are missing.
    mlflow = None


EXPERIMENT_NAME = "email_timing_response"


def _require_mlflow():
    if mlflow is None:
        raise ImportError(
            "MLflow is required for tracking. Install dependencies with "
            "`pip install -r requirements.txt`."
        )
    return mlflow


def set_tracking_uri() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    if "://" not in tracking_uri and Path(tracking_uri).is_absolute():
        tracking_uri = Path(tracking_uri).as_uri()
    _require_mlflow().set_tracking_uri(tracking_uri)


def get_or_create_experiment() -> str:
    set_tracking_uri()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(EXPERIMENT_NAME)


@contextmanager
def start_run(run_name: str | None, agent_type: str):
    experiment_id = get_or_create_experiment()
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        mlflow.set_tag("agent_type", agent_type)
        yield run


def log_agent_params(agent_type: str, params_dict: dict[str, Any]) -> None:
    mlflow.log_param("agent_type", agent_type)
    mlflow.log_params(params_dict)


def log_env_params(params_dict: dict[str, Any]) -> None:
    mlflow.log_params({f"env_{key}": value for key, value in params_dict.items()})


def log_episode(
    episode: int,
    reward: float,
    epsilon: float,
    steps: int,
    loss: float | None = None,
    buffer_size: int | None = None,
) -> None:
    mlflow.log_metric("episode_reward", reward, step=episode)
    mlflow.log_metric("epsilon", epsilon, step=episode)
    mlflow.log_metric("episode_steps", steps, step=episode)
    if loss is not None:
        mlflow.log_metric("loss", loss, step=episode)
    if buffer_size is not None:
        mlflow.log_metric("buffer_size", buffer_size, step=episode)


def log_eval(
    episode: int, rewards_list: list[float], action_counts_dict: dict[int | str, int]
) -> None:
    if rewards_list:
        avg_reward = sum(rewards_list) / len(rewards_list)
        mlflow.log_metric("eval_reward_mean", avg_reward, step=episode)
        mlflow.log_metric("eval_reward_min", min(rewards_list), step=episode)
        mlflow.log_metric("eval_reward_max", max(rewards_list), step=episode)
    for action, count in action_counts_dict.items():
        mlflow.log_metric(f"eval_action_{action}_count", count, step=episode)


def log_summary(
    total_episodes: int,
    total_steps: int,
    elapsed_s: float,
    best_eval_reward: float | None,
) -> None:
    mlflow.log_metric("total_episodes", total_episodes)
    mlflow.log_metric("total_steps", total_steps)
    mlflow.log_metric("elapsed_s", elapsed_s)
    if best_eval_reward is not None:
        mlflow.log_metric("best_eval_reward", best_eval_reward)


def log_dqn_model(policy_net, input_example=None) -> None:
    mlflow.pytorch.log_model(
        pytorch_model=policy_net,
        artifact_path="dqn_policy",
        input_example=input_example,
    )


def log_weights_file(path: str | Path) -> None:
    mlflow.log_artifact(str(path), artifact_path="model_weights")


def log_pkl_file(path: str | Path) -> None:
    mlflow.log_artifact(str(path), artifact_path="model_weights")


def log_reward_history(rewards: list[float]) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "reward_history.json"
        path.write_text(json.dumps(rewards, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(path), artifact_path="training")


def log_config(config_path: str = "config.py") -> None:
    path = Path(config_path)
    if path.exists():
        mlflow.log_artifact(str(path), artifact_path="config")


class Timer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start
