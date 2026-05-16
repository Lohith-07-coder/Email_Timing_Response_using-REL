"""
sender_memory.py
Session-scoped adaptive sender importance tracker.

Starts with the rule-based prior from NLPEmailExtractor (_classify_sender),
then nudges the score slightly based on agent decisions.

Design choices (per reviewer feedback):
  - Small alpha (0.05) → slow updates, prevent self-reinforcing feedback loops
  - Only update on positive reward → agent must have done well to influence memory
  - Scores clamped to [1.0, 3.0] — same scale as Email.sender_importance
  - Session-scoped (in-memory dict), resets per server restart
"""


class SenderMemory:
    """
    Adaptive sender reputation tracker for one server session.

    Score scale:
        1.0 = Promotional / low-trust
        2.0 = Normal professional
        3.0 = Academic / Gov / high-trust

    Update rule:
        reply_now on positive reward  → slight increase  (+ALPHA)
        archive   on positive reward  → slight decrease  (-ALPHA)
        other actions                 → no change
        negative reward               → no update (agent made a mistake, don't learn it)
    """

    ALPHA = 0.05  # Slow nudge rate — prevents feedback loops
    DECAY = 0.995  # Slow forgetting — prevents false importance inflation

    def __init__(self) -> None:
        self._scores: dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def get_importance(self, sender: str, rule_prior: int) -> int:
        """
        Return adapted sender importance (1–3).
        Falls back to rule_prior (from NLPEmailExtractor) if sender is new.
        """
        score = self._scores.get(sender, float(rule_prior))
        return int(round(min(3.0, max(1.0, score))))

    def update(self, sender: str, action: int, reward: float) -> None:
        """
        Nudge sender score based on agent's decision outcome.
        Only updates when reward > 0 (safe update rule from reviewer).

        Also applies a global decay across all tracked senders so that
        stale importance scores slowly return toward neutral (2.0).
        This prevents a one-off good reward from permanently inflating
        a promo sender's importance.

        action encoding:
            0 = reply_now      -> raise importance
            1 = delay_reply    -> no change
            2 = mark_important -> no change
            3 = archive        -> lower importance
        """
        # Global decay step — applied before the nudge so every update
        # slightly pulls ALL scores toward the neutral range.
        for s in list(self._scores):
            self._scores[s] = round(
                min(
                    3.0, max(1.0, self._scores[s] * self.DECAY + 2.0 * (1 - self.DECAY))
                ),
                4,
            )

        if reward <= 0:
            return  # Don't learn from bad decisions

        current = self._scores.get(sender, 2.0)
        delta = {0: +self.ALPHA, 3: -self.ALPHA}.get(action, 0.0)
        if delta == 0.0:
            return

        self._scores[sender] = round(min(3.0, max(1.0, current + delta)), 4)

    def snapshot(self) -> dict:
        """Return a copy of current sender scores for debugging / UI display."""
        return dict(self._scores)

    def reset(self) -> None:
        """Clear all learned scores (e.g., new session)."""
        self._scores.clear()
