import re
from datetime import datetime

from data.email_data import Email


class NLPEmailExtractor:

    _PROMO_SENDER = [
        "noreply",
        "no-reply",
        "donotreply",
        "newsletter",
        "promo",
        "marketing",
        "updates@",
        "notify@",
        "notification",
        "mailer",
        "digest",
        "deals",
        "offers",
        "alert@",
        "info@",
        "support@",
        "news@",
        "hello@",
        "team@",
    ]
    _ACADEMIC_SENDER = [
        ".edu",
        ".ac.in",
        ".ac.uk",
        ".ac.",
        ".gov",
        "university",
        "college",
        "institute",
        "school",
        "hospital",
    ]
    _HIGH_PHRASES = [
        "action required",
        "response needed",
        "urgent",
        "critical issue",
        "asap",
        "time sensitive",
        "immediate attention",
        "due today",
        "final notice",
        "overdue",
        "respond immediately",
        "emergency",
        "system down",
        "server down",
        "outage",
        "production issue",
        "security breach",
        "data loss",
        "deadline today",
        # College / institution specific
        "come and get",
        "come and collect",
        "collect your",
        "pick up your",
        "last date",
        "last day",
        "today only",
        "only today",
    ]
    _HIGH_WORDS = [
        "immediately",
        "critical",
        "deadline",
        "failing",
        "broken",
        "error",
        "failure",
        "crash",
        "alert",
        "warning",
        "expired",
        "expiring",
        "last chance",
        "required",
        "must",
        # College action words
        "collect",
        "distributing",
        "distribution",
        "issued",
        "issuing",
        "marks card",
        "marksheet",
        "hall ticket",
        "admit card",
        "dues",
        "pending",
        "verify",
        "verification",
    ]
    _MEDIUM_WORDS = [
        "reminder",
        "follow up",
        "update",
        "schedule",
        "meeting",
        "tomorrow",
        "next week",
        "please",
        "request",
        "submission",
        "assignment",
        "interview",
        "project",
        "discuss",
        "feedback",
        "confirm",
        "invitation",
        "rescheduled",
        "review",
        "proposal",
        "exam",
        "result",
        "semester",
        "class",
        "lecture",
        "attendance",
    ]
    _WEAK_ALONE = {"today", "now", "important", "attention", "notice"}

    # Academic/Gov sender + any of these words in subject → force HIGH
    _ACADEMIC_ACTION_WORDS = [
        "important",
        "collect",
        "distributing",
        "distribution",
        "marks card",
        "marksheet",
        "hall ticket",
        "admit card",
        "fee",
        "dues",
        "pending",
        "come and get",
        "come and collect",
        "last date",
        "last day",
        "issued",
        "issuing",
        "verify",
        "verification",
        "result",
    ]

    _LOW_SUBJECT = [
        "newsletter",
        "unsubscribe",
        "promo",
        "offer",
        "discount",
        "sale",
        "deal",
        "digest",
        "weekly",
        "monthly",
        "coupon",
        "recharge",
        "cashback",
        "win",
        "winner",
        "prize",
        "free",
        "limited time",
        "special offer",
        "buy now",
        "shop now",
        "new arrivals",
        "exclusive",
        "flash sale",
        "% off",
    ]

    def extract(
        self,
        subject: str,
        sender: str,
        waiting_time: int = 0,
        sender_importance: int | None = None,
    ) -> Email:
        """
        Build an Email from raw subject + sender text.

        waiting_time      — minutes the email has been in inbox (0 = just arrived).
                            Caller (web_ui) is responsible for computing this from
                            arrival timestamps; defaults to 0 for backward compat.
        sender_importance — if provided, overrides rule-based classification
                            (used by SenderMemory to inject adapted score).
        """
        priority = self._classify_priority(subject, sender)
        si = (
            sender_importance
            if sender_importance is not None
            else self._classify_sender(sender)
        )
        workload = self._workload()
        return Email(
            subject=subject,
            sender=sender,
            priority=priority,
            sender_importance=si,
            waiting_time=waiting_time,
            workload=workload,
            time_of_day=datetime.now().hour,
        )

    def explain(self, subject: str, sender: str) -> dict:
        priority = self._classify_priority(subject, sender)
        si = self._classify_sender(sender)
        workload = self._workload()
        hour = datetime.now().hour
        return {
            "priority": priority,
            "sender_importance": si,
            "waiting_time": 0,
            "workload": workload,
            "time_of_day": hour,
            "priority_reason": self._priority_reason(subject, sender),
            "sender_reason": {
                3: "Academic/Gov (trusted)",
                2: "Professional domain",
                1: "Promo/newsletter",
            }[si],
            "workload_reason": {
                1: f"Off-hours ({hour}:00) light",
                2: f"Work hours ({hour}:00) moderate",
                3: f"Peak ({hour}:00) heavy",
            }[workload],
        }

    def _classify_priority(self, subject: str, sender: str) -> int:
        text = subject.lower().strip()
        s = sender.lower().strip()

        if any(p in s for p in self._PROMO_SENDER):
            return 1
        if any(k in text for k in self._LOW_SUBJECT):
            return 1

        # Academic/Gov sender + action word in subject → always HIGH
        is_academic = any(d in s for d in self._ACADEMIC_SENDER)
        if is_academic and any(w in text for w in self._ACADEMIC_ACTION_WORDS):
            return 3

        score = 0
        for phrase in self._HIGH_PHRASES:
            if phrase in text:
                score += 4
        for word in self._HIGH_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", text):
                score += 2
        for word in self._WEAK_ALONE:
            if re.search(rf"\b{re.escape(word)}\b", text):
                score += 1
        for word in self._MEDIUM_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", text):
                score += 1

        # Academic/Gov sender gets a +2 trust bonus
        if is_academic:
            score += 2

        # ── Priority thresholds ──────────────────────────────────────────────
        # score >= 4 : strong urgency (e.g. HIGH_PHRASE hit = +4)
        # score >= 2 : moderate signals (2 weak words, or 1 medium word)
        # score <  2 : generic/empty subjects → LOW (1)
        if score >= 4:
            return 3
        if score >= 2:
            return 2
        return 1

    def _priority_reason(self, subject: str, sender: str) -> str:
        text = subject.lower()
        s = sender.lower()

        if any(p in s for p in self._PROMO_SENDER):
            return "promo sender → LOW"
        if any(k in text for k in self._LOW_SUBJECT):
            return "low keyword → LOW"

        is_academic = any(d in s for d in self._ACADEMIC_SENDER)
        if is_academic and any(w in text for w in self._ACADEMIC_ACTION_WORDS):
            matched = [w for w in self._ACADEMIC_ACTION_WORDS if w in text]
            return f"academic sender + action word {matched[:2]} → HIGH"

        score, hits = 0, []
        for phrase in self._HIGH_PHRASES:
            if phrase in text:
                score += 4
                hits.append(phrase)
        for word in self._HIGH_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", text):
                score += 2
                hits.append(word)
        for word in self._WEAK_ALONE:
            if re.search(rf"\b{re.escape(word)}\b", text):
                score += 1
                hits.append(word)
        for word in self._MEDIUM_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", text):
                score += 1
                hits.append(word)
        if is_academic:
            score += 2

        if score >= 4:
            return f"high urgency {hits[:2]} (score={score}) → HIGH"
        if score >= 1:
            return f"moderate signals {hits[:2]} (score={score}) → MEDIUM"
        return "no strong signal → default MEDIUM"

    def _classify_sender(self, sender: str) -> int:
        s = sender.lower()
        if any(p in s for p in self._PROMO_SENDER):
            return 1
        if any(d in s for d in self._ACADEMIC_SENDER):
            return 3
        return 2

    def _workload(self) -> int:
        h = datetime.now().hour
        if 9 <= h <= 11 or 14 <= h <= 16:
            return 3
        if 8 <= h <= 18:
            return 2
        return 1
