"""Mode routing + live-config validation for the shared person-detection trigger.

Kept dependency-free (no fastapi / sounddevice / cv2 imports) so it is unit-testable
without pulling in main.py's appliance dependencies.
"""

# Speaker types that can only play from a URL and cannot be interrupted mid-utterance,
# so they can't support full-duplex barge-in.
URL_ONLY_SPEAKER_TYPES = {"ipspk_url", "cs20_url"}


def route_conversation(mode, live_start, turn_based_start):
    """Dispatch the shared person-detected trigger to the right engine by mode."""
    (live_start if mode == "live" else turn_based_start)()


def validate_live_config(mode, speaker_type):
    """Raise ValueError if live mode is requested with a speaker that can't do barge-in."""
    if mode == "live" and speaker_type in URL_ONLY_SPEAKER_TYPES:
        raise ValueError(
            "Live mode requires a WebSocket-capable speaker; "
            f"'{speaker_type}' is URL-only and cannot be interrupted for barge-in."
        )
