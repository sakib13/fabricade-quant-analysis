"""
parse_logs.py
=============
Parse raw Fabricade session logs (Google-Sheets CSV export with embedded JSON
event arrays) into two tidy, anonymised datasets for quantitative analysis:

    data/sessions.csv   one row per participant x condition (20 rows)
    data/choices.csv    one row per recorded choice (long format, 260 rows)

Design notes
------------
* Each raw row is one gameplay session (one participant playing one condition).
* The behavioural events are stored as a JSON array in a trailing cell. The
  number of leading CSV columns is inconsistent across rows (an optional
  "video prompted recall" flag shifts them), so we locate the JSON cell by
  content ("starts with '[{'") rather than by a fixed index, and we read the
  session start time and duration from the JSON events themselves rather than
  from the unreliable positional columns.
* Participant identities are replaced with codes P1..P10. The mapping is applied
  here so that no real name reaches the derived datasets or the notebook. The
  codes match those used in the associated qualitative thesis.
* Of the 13 recorded choices per session, only 7 are genuine narrative
  decisions; the remaining 6 are "Continue" navigation presses. We flag this
  with `is_decision` so the latency analysis can be restricted to real
  decisions.

Run:  python parse_logs.py
"""

import csv
import json
import os
import re
import datetime as dt

# --- configuration ---------------------------------------------------------

RAW_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Form responses & Session Logs",
    "Fabcricade Session Logs.csv",
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# The mapping from in-game log name to anonymised participant code (P1..P10) is
# the LINKING KEY for this dataset, and it is deliberately NOT stored in this
# repository. Under the study's data management plan the key is held separately
# from the data it de-anonymises. Only the parse step needs it. Everything in
# data/ is already anonymised and the whole analysis runs without it.
#
# To re-run the parse, point FABRICADE_KEY at a JSON file of the form
# {"<log name>": "P1", ...} or place that file at the default path below.
KEY_PATH = os.environ.get(
    "FABRICADE_KEY",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "participant_key.json",
    ),
)


def load_name_to_code():
    """Load the linking key, or explain clearly why the parse step cannot run."""
    if not os.path.exists(KEY_PATH):
        raise SystemExit(
            "Participant key not found at:\n  %s\n\n"
            "This repository ships the analysis, not the linking key. The key maps\n"
            "raw log names to the anonymised codes P1..P10 and is held separately\n"
            "under the study's data management plan, so the parse step cannot run\n"
            "without it.\n\n"
            "The anonymised datasets in data/ are already built, and the notebook\n"
            "runs end to end without this file. Set FABRICADE_KEY if you hold the\n"
            "key and need to regenerate them." % KEY_PATH
        )
    with open(KEY_PATH, encoding="utf-8") as fh:
        return json.load(fh)

# Canonical scene order (1-based position in the narrative).
SCENE_ORDER = ["intake", "dinner", "bench", "corridor", "room", "discharge"]
SCENE_POS = {s: i + 1 for i, s in enumerate(SCENE_ORDER)}

# The six numeric Ink state variables tracked by the game.
INK_VARS = [
    "openness",
    "deflection",
    "resistance",
    "emotional_posture",
    "trust_in_system",
    "mystery_awareness",
]


# --- helpers ---------------------------------------------------------------

def _find_json_cell(row):
    """Return the index of the cell holding the JSON event array."""
    for j, cell in enumerate(row):
        if cell.strip().startswith("[{"):
            return j
    raise ValueError("no JSON event cell found in row")


def _parse_end_details(details):
    """Parse the 'key=value, key=value' summary string on the story_end event."""
    out = {}
    for pair in details.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)
        out[key.strip()] = val.strip()
    return out


def _choice_text(details):
    """Extract the selected option text from a choice_made 'details' string."""
    # format: 'Selected [0]: "some text"'  or  'Selected [0]: Continue'
    m = re.match(r"Selected \[(\d+)\]:\s*(.*)$", details)
    if not m:
        return None, None
    idx = int(m.group(1))
    text = m.group(2).strip().strip('"')
    return idx, text


# --- core parsing ----------------------------------------------------------

def load_sessions(raw_csv=RAW_CSV):
    """Return (sessions, choices) as lists of dicts, anonymised."""
    name_to_code = load_name_to_code()
    with open(raw_csv, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)  # header
        rows = [r for r in reader if any(c.strip() for c in r)]

    sessions, choices = [], []

    for row in rows:
        jc = _find_json_cell(row)
        events = json.loads(row[jc])

        name = row[3].strip()
        code = name_to_code.get(name)
        if code is None:
            raise KeyError("unmapped participant name in the raw log, and the "
                           "name is deliberately not echoed here")
        condition = row[2].strip()
        video_recall = any(c.strip().lower() == "yes" for c in row[:jc])

        # Session start = timestamp of the first event; robust across the
        # inconsistent CSV column layout.
        start_ts = events[0]["timestamp"]
        end_event = next(e for e in events if e["eventType"] == "story_end")
        duration_s = float(end_event["sessionTimeSeconds"])

        # Final Ink state and labelled choice summary.
        end_details = _parse_end_details(end_event["details"])
        ink = {v: int(end_details[v]) for v in INK_VARS if v in end_details}

        sessions.append(
            dict(
                participant=code,
                condition=condition,
                start_ts=start_ts,
                duration_s=duration_s,
                video_recall=video_recall,
                miss_most=end_details.get("miss_most"),
                **ink,
            )
        )

        # Choice-level rows.
        decision_counter = 0
        for e in events:
            if e["eventType"] != "choice_made":
                continue
            idx, text = _choice_text(e.get("details", ""))
            is_decision = (text is not None) and (text != "Continue")
            if is_decision:
                decision_counter += 1
            choices.append(
                dict(
                    participant=code,
                    condition=condition,
                    scene=e["scene"],
                    scene_pos=SCENE_POS.get(e["scene"]),
                    option_index=idx,
                    choice_text=text,
                    is_decision=is_decision,
                    decision_number=decision_counter if is_decision else None,
                    hesitation_s=float(e.get("hesitationSeconds", 0.0)),
                    session_time_s=float(e.get("sessionTimeSeconds", 0.0)),
                )
            )

    return sessions, choices


def add_play_order(sessions):
    """Add play_order (1 = first playthrough, 2 = second) from start_ts,
    and first_condition, per participant."""
    from collections import defaultdict

    by_p = defaultdict(list)
    for s in sessions:
        by_p[s["participant"]].append(s)

    for code, sess in by_p.items():
        sess.sort(key=lambda s: s["start_ts"])
        first_cond = sess[0]["condition"]
        for order, s in enumerate(sess, start=1):
            s["play_order"] = order
            s["first_condition"] = first_cond
    return sessions


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sessions, choices = load_sessions()
    sessions = add_play_order(sessions)

    # Attach play_order to each choice row (join on participant+condition).
    order_lookup = {(s["participant"], s["condition"]): s["play_order"] for s in sessions}
    for c in choices:
        c["play_order"] = order_lookup[(c["participant"], c["condition"])]

    session_fields = [
        "participant", "condition", "first_condition", "play_order",
        "start_ts", "duration_s", "video_recall", "miss_most", *INK_VARS,
    ]
    choice_fields = [
        "participant", "condition", "play_order", "scene", "scene_pos",
        "option_index", "choice_text", "is_decision", "decision_number",
        "hesitation_s", "session_time_s",
    ]

    # Stable ordering for reproducibility.
    pnum = lambda code: int(code[1:])
    sessions.sort(key=lambda s: (pnum(s["participant"]), s["condition"]))
    choices.sort(key=lambda c: (pnum(c["participant"]), c["condition"], c["session_time_s"]))

    write_csv(os.path.join(OUT_DIR, "sessions.csv"), sessions, session_fields)
    write_csv(os.path.join(OUT_DIR, "choices.csv"), choices, choice_fields)

    print(f"Wrote {len(sessions)} sessions -> data/sessions.csv")
    print(f"Wrote {len(choices)} choices  -> data/choices.csv")
    n_dec = sum(c["is_decision"] for c in choices)
    print(f"  decisions: {n_dec}  (expected 140 = 10 x 2 x 7)")
    print(f"  navigation 'Continue' presses: {len(choices) - n_dec}")


if __name__ == "__main__":
    main()
