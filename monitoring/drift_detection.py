"""
Data-drift detection for incoming email state vectors.

Compares a rolling window of live feature distributions against a
reference distribution captured at training time. Raises alerts when
Population Stability Index (PSI) or mean-shift thresholds are exceeded.

Usage:
    detector = DriftDetector.from_reference_stats(ref_means, ref_stds)
    detector.update(state_vector)          # call per prediction
    report = detector.report()             # periodic check
"""

import json
import os
from collections import deque

import numpy as np

from monitoring.logging_config import get_logger

logger = get_logger(__name__)

FEATURE_NAMES = [
    "priority_norm",
    "sender_importance_norm",
    "waiting_time_norm",
    "workload_norm",
    "time_of_day_norm",
]

# Alert thresholds
PSI_WARNING_THRESHOLD = 0.10  # mild shift
PSI_CRITICAL_THRESHOLD = 0.25  # significant shift
MEAN_SHIFT_SIGMA = 2.5  # std-deviations from reference mean


class DriftDetector:
    """
    Monitors feature-level distribution drift using a rolling window
    compared to a reference (training-time) distribution.
    """

    def __init__(
        self,
        ref_means: np.ndarray,
        ref_stds: np.ndarray,
        window_size: int = 500,
        n_bins: int = 10,
    ):
        assert len(ref_means) == len(FEATURE_NAMES), "ref_means length mismatch"
        self._ref_means = np.array(ref_means, dtype=np.float32)
        self._ref_stds = np.clip(ref_stds, 1e-6, None).astype(np.float32)
        self._window: deque = deque(maxlen=window_size)
        self._n_bins = n_bins
        self._alerts: list = []

    # public API

    def update(self, state_vector: np.ndarray):
        """Add a new observation to the rolling window."""
        self._window.append(np.array(state_vector, dtype=np.float32))

    def report(self) -> dict:
        """
        Compute per-feature drift statistics over the current window.
        Returns a dict with PSI scores, mean-shift z-scores, and alert list.
        """
        if len(self._window) < 50:
            return {"status": "insufficient_data", "n_samples": len(self._window)}

        obs = np.stack(list(self._window))  # (N, 5)
        drift_report = {
            "n_samples": len(self._window),
            "features": {},
            "alerts": [],
        }

        for i, feat in enumerate(FEATURE_NAMES):
            col = obs[:, i]
            live_mean = float(col.mean())
            live_std = float(col.std())
            z_score = (live_mean - self._ref_means[i]) / self._ref_stds[i]
            psi = self._psi(self._ref_means[i], self._ref_stds[i], col)

            level = "ok"
            if psi >= PSI_CRITICAL_THRESHOLD or abs(z_score) >= MEAN_SHIFT_SIGMA * 1.5:
                level = "critical"
            elif psi >= PSI_WARNING_THRESHOLD or abs(z_score) >= MEAN_SHIFT_SIGMA:
                level = "warning"

            drift_report["features"][feat] = {
                "live_mean": round(live_mean, 4),
                "live_std": round(live_std, 4),
                "ref_mean": round(float(self._ref_means[i]), 4),
                "z_score": round(float(z_score), 3),
                "psi": round(psi, 4),
                "status": level,
            }
            if level != "ok":
                msg = (
                    f"[{level.upper()}] Feature '{feat}' drift: "
                    f"PSI={psi:.3f}, z={z_score:.2f}"
                )
                drift_report["alerts"].append(msg)
                logger.warning(msg)

        drift_report["status"] = (
            "critical"
            if any("critical" in a.lower() for a in drift_report["alerts"])
            else "warning" if drift_report["alerts"] else "ok"
        )
        return drift_report

    def dump(self, path: str = "logs/drift_report.json"):
        report = self.report()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        return report

    # helpers

    def _psi(self, ref_mean: float, ref_std: float, live_col: np.ndarray) -> float:
        """
        Approximate PSI using equal-width bins in [0, 1] (normalized space).
        """
        bins = np.linspace(0.0, 1.0, self._n_bins + 1)
        ref_sample = np.clip(np.random.normal(ref_mean, ref_std, size=1000), 0, 1)
        ref_counts, _ = np.histogram(ref_sample, bins=bins)
        live_counts, _ = np.histogram(live_col, bins=bins)

        # Avoid division by zero
        ref_pct = (ref_counts + 1e-6) / ref_counts.sum()
        live_pct = (live_counts + 1e-6) / live_counts.sum()
        psi = float(np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct)))
        return max(psi, 0.0)

    # factory

    @classmethod
    def from_reference_stats(cls, ref_means, ref_stds, **kwargs) -> "DriftDetector":
        return cls(np.array(ref_means), np.array(ref_stds), **kwargs)

    @classmethod
    def default(cls) -> "DriftDetector":
        """
        Sensible defaults based on the normalized [0,1] feature space
        used by Email.to_state_vector().
        """
        # Approximate training-distribution statistics (update after real training)
        ref_means = [0.5, 0.5, 0.2, 0.5, 0.5]
        ref_stds = [0.3, 0.3, 0.2, 0.3, 0.3]
        return cls.from_reference_stats(ref_means, ref_stds)
