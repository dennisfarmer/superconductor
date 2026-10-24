"""Releasing harmony or drums: one reset to the model default, then those lanes stay out.

Run: python audio_engine/test_output.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from engine import Settings, controls_body, gate_pattern, merge_control_posts  # noqa: E402


def check(name, ok):
    print(("ok  " if ok else "FAIL") + " " + name)
    return ok


def main():
    notes = {"loop": True, "steps": [{"frames": 4, "default": 0, "pitches": {}}]}
    drums = {"loop": True, "steps": [{"frames": 4, "first": 1, "rest": 0}]}
    pattern = {"id": "atlas idle #3", "notes": notes, "drums": drums}
    passed = True

    both = gate_pattern(pattern, True, True, False, False)
    passed &= check("sending both keeps both lanes", both == pattern)

    releasing = gate_pattern(pattern, False, True, True, False)
    passed &= check(
        "releasing harmony nulls notes and keeps drums",
        releasing is not None and releasing["notes"] is None and releasing["drums"] == drums
        and releasing["id"] == "atlas release harmony",
    )

    after = gate_pattern(pattern, False, True, False, False)
    passed &= check(
        "after the reset, notes are left out",
        after is not None and "notes" not in after and after["drums"] == drums and after["id"] == pattern["id"],
    )

    released = gate_pattern(pattern, False, False, False, False)
    passed &= check("both released and already reset: nothing to send", released is None)

    reset_both = gate_pattern(None, False, False, True, True)
    passed &= check(
        "releasing both with no timeline still sends the nulls",
        reset_both is not None and reset_both["notes"] is None and reset_both["drums"] is None
        and reset_both["id"] == "atlas release harmony+drums",
    )

    settings = Settings()
    settings.cfg_scales = {"musiccoca": 3.0, "notes": 5.0, "drums": 4.0}
    settings.temperature = 1.1
    reset_notes = controls_body(settings, False, True, reset_harmony=True)
    passed &= check(
        "releasing harmony nulls cfg_scales.notes only",
        reset_notes["cfg_scales"]["notes"] is None
        and reset_notes["cfg_scales"]["drums"] == 4.0
        and reset_notes["cfg_scales"]["musiccoca"] == 3.0
        and reset_notes["temperature"] == 1.1,
    )

    quiet = controls_body(settings, False, False)
    passed &= check(
        "a released lane's cfg key is omitted afterwards",
        "notes" not in quiet["cfg_scales"] and "drums" not in quiet["cfg_scales"]
        and quiet["cfg_scales"]["musiccoca"] == 3.0,
    )

    merged = merge_control_posts(
        {"temperature": 1.3, "cfg_scales": {"notes": None, "musiccoca": 3.0}},
        {"top_k": 40, "cfg_scales": {"drums": 2.0}},
    )
    passed &= check(
        "an unsent release null survives a later controls update",
        merged["cfg_scales"]["notes"] is None and merged["cfg_scales"]["drums"] == 2.0
        and merged["temperature"] == 1.3 and merged["top_k"] == 40,
    )

    taken = merge_control_posts(
        {"cfg_scales": {"notes": None}},
        {"cfg_scales": {"notes": 3.0}},
    )
    passed &= check("taking harmony back replaces the unsent null", taken["cfg_scales"]["notes"] == 3.0)

    print("passed" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
