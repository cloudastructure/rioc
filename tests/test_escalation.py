from live.escalation import EscalationOverlay


def test_state_advances_by_elapsed_time():
    e = EscalationOverlay(escalate_after=10.0, final_after=25.0)
    assert e.tick(0) == "WARNING"
    assert e.tick(12) == "ESCALATING"
    assert e.tick(30) == "FINAL"


def test_outcome_classification():
    e = EscalationOverlay(escalate_after=10.0, final_after=25.0)
    assert e.outcome(had_person_turn=True, state="FINAL") == "Escalated"
    assert e.outcome(had_person_turn=True, state="WARNING") == "Left"
    assert e.outcome(had_person_turn=False, state="WARNING") == "Unknown"
