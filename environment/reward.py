from data.email_data import Email


class RewardCalculator:
    """
    Reward table for the Decision Agent.

    The agent was trained against these exact base values.

    Contextual waiting-time overrides dynamically scale rewards
    to add temporal awareness and break degenerate policies.
    Thresholds are priority-dependent:
        HIGH   →  5 min
        MEDIUM → 30 min
        LOW    → 120 min
    """

    REPLY_NOW = 0
    DELAY_REPLY = 1
    MARK_IMPORTANT = 2
    ARCHIVE = 3

    # Minutes before a waiting email's action should shift toward reply_now.
    WAIT_THRESHOLDS = {3: 5, 2: 30, 1: 120}

    def calculate(self, email: Email, action: int) -> float:
        if email.priority == 3:
            wait_bonus = min(email.waiting_time / 60.0, 5.0)
            reward = {
                self.REPLY_NOW: +10.0 + wait_bonus,
                self.DELAY_REPLY: -8.0 - wait_bonus,  # was -6: stronger urgency signal
                self.MARK_IMPORTANT: +2.0,
                self.ARCHIVE: -10.0,
            }[action]
        elif email.priority == 2:
            # Tuned to break conservative policies and encourage responsiveness
            reward = {
                self.REPLY_NOW: -1.0,
                self.DELAY_REPLY: +2.0,
                self.MARK_IMPORTANT: +3.0,
                self.ARCHIVE: -4.0,
            }[action]
        else:  # priority == 1 (Low)
            reward = {
                self.REPLY_NOW: -6.0,
                self.DELAY_REPLY: +1.0,
                self.MARK_IMPORTANT: -5.0,
                self.ARCHIVE: +6.0,
            }[action]

        # Context-aware waiting-time override.
        # Breaks degenerate policies (e.g. "always delay medium").
        # Threshold scales with priority — high-urgency emails go stale faster.
        threshold = self.WAIT_THRESHOLDS.get(email.priority, 30)
        if email.waiting_time > threshold:
            if email.priority == 3:
                # HIGH already has wait_bonus baked in → no additional override
                pass
            elif email.priority == 2:
                if action == self.REPLY_NOW:
                    reward = +5.0  # correct: reply to waited medium email
                elif action == self.DELAY_REPLY:
                    reward = -3.0  # wrong: further delay on a waited email
            else:  # priority == 1 (Low)
                # Even low-priority email after 2+ hrs deserves attention,
                # but not a full reply. Mark is better than archive at this point.
                if action == self.MARK_IMPORTANT:
                    reward = +2.0  # bump: better to flag than silently archive
                elif action == self.ARCHIVE:
                    reward = +3.0  # still ok to archive, just slightly less so

        # Contextual penalties.
        if action == self.REPLY_NOW and email.priority < 3 and email.workload == 3:
            reward -= 4.0
        is_offhours = email.time_of_day >= 19 or email.time_of_day <= 7
        if action == self.REPLY_NOW and email.priority < 3 and is_offhours:
            reward -= 3.0

        return round(reward, 2)
