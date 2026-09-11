"""Runner for eval/cases.md's 15 cases. Executes each case's turns through
answer_conversational() with a fresh ConversationState per case, saves raw,
untouched output to eval/results/raw/{case_id}.json. Scoring/judgment is done
separately (manually, against cases.md's expected answers) - this script
only runs and preserves raw results.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/

import retrieve
from answer import answer_conversational
from conversation_state import ConversationState

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "eval"
RAW_DIR = EVAL_DIR / "results" / "raw"

CASES = [
    ("case_01", ["According to episode 7, why can no machine decide in general whether an arbitrary program halts?"]),
    ("case_02", ["According to episode 7, what is the equivalence principle?"]),
    ("case_03", ["Compare episodes 1 and 6 - how does each one define which observers get special/privileged treatment?"]),
    ("case_04", ["What do episodes 7 and 8 say about biological evolution?"]),
    ("case_05", ["What's in the relativity episode?"]),
    ("case_06", ["What fundamental limits or impossibility results come up across the episodes?"]),
    ("case_07", ["What do the episodes say about how governments regulated nuclear weapons development?"]),
    ("case_08", ["Which episode should I listen to if I want to understand the limits of what computers can decide?"]),
    ("case_09", ["What does episode 99 say about black holes?"]),
    ("case_10", ["What is discussed in episode 7 between minutes 26 and 28?"]),
    ("case_11", [
        "What does episode 6 say about the equivalence principle?",
        "Walk me through that step by step, I didn't quite follow",
        "What does episode 8 say about the Wallace line?",
    ]),
    ("case_12", [
        "What does episode 4 say about Shannon's source-coding theorem?",
        "Explain that in short and concise",
    ]),
    ("case_13", [
        "What's in your podcast collection?",
        "Which episodes do you contain?",
    ]),
    ("case_14", ["Hey, how's it going?"]),
    ("case_15", ["hey real quick, what's entropy again lol"]),
]


def run_case(queries: list[str]) -> list[dict]:
    state = ConversationState()
    turns_raw = []
    for q in queries:
        before = retrieve.QUERY_CALL_COUNT
        result = answer_conversational(q, state)
        after = retrieve.QUERY_CALL_COUNT
        result["query_call_count_delta"] = after - before
        turns_raw.append(result)
    return turns_raw


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for case_id, queries in CASES:
        print(f"[{case_id}] running {len(queries)} turn(s)...")
        turns_raw = run_case(queries)
        with open(RAW_DIR / f"{case_id}.json", "w", encoding="utf-8") as f:
            json.dump(turns_raw, f, indent=2)
        for i, t in enumerate(turns_raw):
            print(f"  turn {i}: type={t['query_type']} mode={t['retrieval_mode']} "
                  f"calls={t['query_call_count_delta']} refused={t['refused']} chunks={len(t['retrieved_chunk_ids'])}")
    print(f"\nAll raw results written to {RAW_DIR}")


if __name__ == "__main__":
    main()
