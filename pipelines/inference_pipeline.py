"""
Batch inference pipeline with MLflow experiment tracking.

Loads the trained DQN agent and runs it over a stream of emails,
writing action recommendations to a JSON Lines output file.
Logs inference metrics and drift reports to MLflow.

Usage:
    python pipelines/inference_pipeline.py \
        --input emails.jsonl --output predictions.jsonl
    python pipelines/inference_pipeline.py --demo   # runs 20 synthetic emails
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from config import Config
from data.email_data import Email
from monitoring.drift_detection import DriftDetector
from monitoring.logging_config import get_logger
from monitoring.metrics import MetricsTracker

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


logger = get_logger(__name__)

ACTION_LABELS = {0: "reply_now", 1: "delay_reply", 2: "mark_important", 3: "archive"}


def load_agent():
    from agent.dqn import DQNAgent  # noqa

    weights_path = Config.DQN_WEIGHTS_PATH
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model not found: {weights_path}. Train first.")

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    agent = DQNAgent(
        state_size=ckpt.get("state_size", Config.STATE_SIZE),
        action_size=ckpt.get("action_size", Config.ACTION_SIZE),
    )
    agent._policy_net.load_state_dict(ckpt["policy_state_dict"])
    agent._policy_net.eval()
    agent.epsilon = 0.0  # pure greedy for inference
    logger.info("Agent loaded. Version: %s", ckpt.get("version", "unknown"))
    return agent, ckpt.get("version", "unknown")


def predict_email(agent, email: Email) -> dict:
    state = email.to_state_vector()
    with torch.no_grad():
        t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_vals = agent._policy_net(t).squeeze(0).numpy()

    action_id = int(np.argmax(q_vals))
    exp_q = np.exp(q_vals - q_vals.max())
    confidence = float(exp_q[action_id] / exp_q.sum())

    return {
        "action_id": action_id,
        "action_label": ACTION_LABELS[action_id],
        "confidence": round(confidence, 4),
        "state_vector": state.tolist(),
        "q_values": q_vals.tolist(),
    }


def run_batch(agent, emails: list, output_path: str, version: str, use_mlflow: bool = True):
    metrics = MetricsTracker()
    drift = DriftDetector.default()
    results = []

    for i, email_dict in enumerate(emails):
        email = Email(
            **{
                k: email_dict[k]
                for k in [
                    "subject",
                    "sender",
                    "priority",
                    "sender_importance",
                    "waiting_time",
                    "workload",
                    "time_of_day",
                ]
            }
        )
        t0 = time.time()
        pred = predict_email(agent, email)
        latency_ms = (time.time() - t0) * 1000

        metrics.record_prediction(pred["action_id"], pred["confidence"], latency_ms)
        drift.update(np.array(pred["state_vector"], dtype=np.float32))

        record = {
            "index": i,
            "email": email_dict,
            "prediction": pred,
            "model_version": version,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        results.append(record)

        if (i + 1) % 50 == 0:
            logger.info("Processed %d emails", i + 1)

    # Write results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    snap = metrics.snapshot()
    drift_report = drift.report()
    logger.info("Batch complete. Metrics: %s", snap)
    logger.info("Drift report: %s", drift_report.get("status"))

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    if use_mlflow:
        try:
            import mlflow

            from mlflow_config import MLflowConfig, init_mlflow
            from monitoring.mlflow_logger import log_drift_report, log_inference_metrics

            init_mlflow(MLflowConfig.EXPERIMENT_INFERENCE)
            with mlflow.start_run(run_name=f"inference-batch-{time.strftime('%Y%m%d_%H%M%S')}"):
                mlflow.set_tag("model_version", version)
                mlflow.set_tag("batch_size", len(emails))
                mlflow.log_param("num_emails", len(emails))
                mlflow.log_param("model_version", version)

                log_inference_metrics(snap)
                log_drift_report(drift_report)

                # Log output file as artifact
                if os.path.exists(output_path):
                    mlflow.log_artifact(output_path, artifact_path="predictions")

        except Exception as e:
            logger.warning("MLflow logging failed (non-fatal): %s", e)

    return results, snap


def demo_run():
    """Run 20 synthetic emails and print results to stdout."""
    from simulation.simulator import EmailSimulator

    sim = EmailSimulator()
    emails = []
    for _ in range(20):
        e = sim.next_email()
        emails.append(
            {
                "subject": e.subject,
                "sender": e.sender,
                "priority": e.priority,
                "sender_importance": e.sender_importance,
                "waiting_time": e.waiting_time,
                "workload": e.workload,
                "time_of_day": e.time_of_day,
            }
        )
    agent, version = load_agent()
    results, _ = run_batch(agent, emails, "logs/demo_predictions.jsonl", version)
    print(f"\n{'─'*60}")
    print(f"  Demo Inference Results (model v{version})")
    print(f"{'─'*60}")
    for r in results:
        p = r["prediction"]
        print(
            f"  [{p['action_label']:15s}] conf={p['confidence']:.2f}  "
            f"subject={r['email']['subject'][:30]}"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="Path to input JSONL file")
    p.add_argument("--output", default="logs/predictions.jsonl")
    p.add_argument("--demo", action="store_true", help="Run on 20 synthetic emails")
    p.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking for this run",
    )
    args = p.parse_args()

    use_mlflow = not args.no_mlflow

    if args.demo or args.input is None:
        demo_run()
    else:
        agent, version = load_agent()
        with open(args.input) as f:
            emails = [json.loads(line) for line in f]
        run_batch(agent, emails, args.output, version, use_mlflow=use_mlflow)
        print(f"Results written to {args.output}")
