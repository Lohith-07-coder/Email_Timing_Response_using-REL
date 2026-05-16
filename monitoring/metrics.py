"""
Lightweight in-process metrics tracker for the inference service.

Tracked:
  - Prediction counts by action label
  - Confidence distribution (rolling mean)
  - Request latency histogram buckets

Expose via /metrics endpoint or dump to a JSON file for Prometheus
file-based scraping if you later add the pushgateway.
"""

import json
import os
import time
from collections import defaultdict
from threading import Lock
from typing import Dict

from monitoring.logging_config import get_logger

logger = get_logger(__name__)

ACTION_LABELS = {
    0: "reply_now",
    1: "delay_reply",
    2: "mark_important",
    3: "archive",
}


class MetricsTracker:
    """Thread-safe accumulator for inference-time metrics."""

    def __init__(self, dump_path: str = None):
        self._lock = Lock()
        self._counts: Dict[str, int] = defaultdict(int)
        self._confidence_sum: float = 0.0
        self._confidence_n: int = 0
        self._latencies: list = []
        self._dump_path = dump_path or os.environ.get(
            "METRICS_PATH", "logs/metrics.json"
        )
        self._start_ts = time.time()

    # ── public API ────────────────────────────────────────────────────────────

    def record_prediction(
        self, action_id: int, confidence: float, latency_ms: float = 0.0
    ):
        label = ACTION_LABELS.get(action_id, f"action_{action_id}")
        with self._lock:
            self._counts[label] += 1
            self._confidence_sum += confidence
            self._confidence_n += 1
            if latency_ms:
                self._latencies.append(latency_ms)

    def snapshot(self) -> dict:
        with self._lock:
            total = sum(self._counts.values())
            avg_conf = (
                round(self._confidence_sum / self._confidence_n, 4)
                if self._confidence_n
                else 0.0
            )
            lat_sorted = sorted(self._latencies)
            return {
                "uptime_seconds": round(time.time() - self._start_ts, 1),
                "total_predictions": total,
                "action_distribution": dict(self._counts),
                "avg_confidence": avg_conf,
                "latency_p50_ms": _percentile(lat_sorted, 50),
                "latency_p95_ms": _percentile(lat_sorted, 95),
            }

    def dump(self):
        """Write a JSON snapshot to disk for external scrapers."""
        snap = self.snapshot()
        os.makedirs(os.path.dirname(self._dump_path) or ".", exist_ok=True)
        with open(self._dump_path, "w") as f:
            json.dump(snap, f, indent=2)
        logger.debug("Metrics dumped → %s", self._dump_path)
        return snap


# ── helpers ───────────────────────────────────────────────────────────────────


def _percentile(sorted_vals: list, pct: int) -> float:
    if not sorted_vals:
        return 0.0
    idx = int(len(sorted_vals) * pct / 100)
    return round(sorted_vals[min(idx, len(sorted_vals) - 1)], 2)
