import os
import random
import re
from dataclasses import dataclass

import pandas as pd


@dataclass
class RawEmail:
    subject: str
    sender: str


class EnronLoader:
    """
    Loads and preprocesses the Enron email dataset.
    Extracts subject + sender from raw .txt email files.
    Returns a flat list of RawEmail objects ready for feature extraction.
    """

    _SUBJECT_RE = re.compile(r"^Subject:\s*(.+)$", re.MULTILINE | re.IGNORECASE)
    _FROM_RE = re.compile(r"^From:\s*(.+)$", re.MULTILINE | re.IGNORECASE)

    def __init__(self, dataset_path: str, max_emails: int = 50_000):
        self.dataset_path = dataset_path
        self.max_emails = max_emails

    def load(self) -> list[RawEmail]:
        """
        Walk the Enron directory tree, parse every .txt file,
        extract subject + sender, return as RawEmail list.
        Shuffled so training sees all users/styles evenly.
        """
        emails = []
        for root, _, files in os.walk(self.dataset_path):
            for fname in files:
                if fname.startswith("."):
                    continue
                path = os.path.join(root, fname)
                raw = self._parse(path)
                if raw:
                    emails.append(raw)
                if len(emails) >= self.max_emails:
                    break
            if len(emails) >= self.max_emails:
                break

        random.shuffle(emails)
        print(f"  📧 Enron: loaded {len(emails):,} emails from {self.dataset_path}")
        return emails

    def to_dataframe(self) -> pd.DataFrame:
        emails = self.load()
        return pd.DataFrame(
            [{"subject": e.subject, "sender": e.sender} for e in emails]
        )

    def _parse(self, path: str) -> RawEmail | None:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            subject_match = self._SUBJECT_RE.search(text)
            from_match = self._FROM_RE.search(text)
            if not subject_match or not from_match:
                return None
            subject = subject_match.group(1).strip()
            sender = from_match.group(1).strip()
            # skip forwarded / empty / automated
            if not subject or subject.lower() in ("", "fw:", "re:"):
                return None
            if len(subject) < 3:
                return None
            return RawEmail(subject=subject, sender=sender)
        except Exception:
            return None
