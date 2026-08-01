"""Time/severity escalation overlay for Live mode.

Live conversation has fuzzy turns, so escalation is driven by elapsed time
rather than turn counts. Outcome classification mirrors the turn-based engine
(Escalated / Left / Unknown).
"""


class EscalationOverlay:
    def __init__(self, *, escalate_after, final_after):
        self.escalate_after = escalate_after
        self.final_after = final_after

    def tick(self, elapsed):
        if elapsed >= self.final_after:
            return "FINAL"
        if elapsed >= self.escalate_after:
            return "ESCALATING"
        return "WARNING"

    def outcome(self, had_person_turn, state):
        if state == "FINAL":
            return "Escalated"
        if had_person_turn:
            return "Left"
        return "Unknown"
