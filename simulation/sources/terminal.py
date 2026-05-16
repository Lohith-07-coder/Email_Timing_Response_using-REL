from data.email_data import Email
from simulation.sources.base import EmailSource


class TerminalEmailSource(EmailSource):

    def get_email(self) -> Email:
        print("\n── Manual Email Entry ──")
        subject = input("Subject                                   : ").strip()
        sender = input("Sender email                              : ").strip()
        priority = self._prompt_int(
            "Priority           (1=low 2=medium 3=high): ", 1, 3
        )
        sender_importance = self._prompt_int(
            "Sender importance  (1=low 2=medium 3=high): ", 1, 3
        )
        waiting_time = self._prompt_int(
            "Waiting time       (minutes, 0–60)        : ", 0, 60
        )
        workload = self._prompt_int(
            "Workload           (1=light 2=moderate 3=heavy): ", 1, 3
        )
        time_of_day = self._prompt_int(
            "Time of day        (0–23)                 : ", 0, 23
        )
        return Email(
            subject,
            sender,
            priority,
            sender_importance,
            waiting_time,
            workload,
            time_of_day,
        )

    @staticmethod
    def _prompt_int(prompt: str, lo: int, hi: int) -> int:
        while True:
            try:
                val = int(input(prompt))
                if lo <= val <= hi:
                    return val
                print(f"  ⚠  Enter a value between {lo} and {hi}.")
            except ValueError:
                print("  ⚠  Numbers only.")
