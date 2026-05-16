from monitoring.mlflow_tracker import _require_mlflow, get_or_create_experiment

DQN_MODEL_NAME = "EmailTiming_DQN"
QL_MODEL_NAME = "EmailTiming_QLearning"


def register_dqn_model(run_id: str):
    mlflow = _require_mlflow()
    return mlflow.register_model(f"runs:/{run_id}/dqn_policy", DQN_MODEL_NAME)


def register_ql_model(run_id: str):
    mlflow = _require_mlflow()
    return mlflow.register_model(f"runs:/{run_id}/model_weights", QL_MODEL_NAME)


def promote_model(
    model_name: str,
    version: str | int,
    stage: str = "Production",
    archive_existing: bool = True,
):
    mlflow = _require_mlflow()
    client = mlflow.tracking.MlflowClient()
    return client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=archive_existing,
    )


def load_production_dqn(device: str = "cpu"):
    mlflow = _require_mlflow()
    model_uri = f"models:/{DQN_MODEL_NAME}/Production"
    return mlflow.pytorch.load_model(model_uri, map_location=device)


def compare_runs(
    experiment_name: str = "email_timing_response",
    metric: str = "best_eval_reward",
    top_n: int = 5,
):
    mlflow = _require_mlflow()
    get_or_create_experiment()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    return mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=top_n,
    )
