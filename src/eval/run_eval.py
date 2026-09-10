"""Phase 5 - loads cases.json, runs each through answer_conversational(), saves raw
results untouched, then runs hard (+ optional judge) scoring and an aggregate summary.

Usage:
    python src/eval/run_eval.py --run-id baseline
    python src/eval/run_eval.py --run-id baseline --with-judge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/

import retrieve
from answer import answer_conversational
from conversation_state import ConversationState

import scoring_hard
import scoring_judge

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "eval"
CASES_PATH = EVAL_DIR / "cases.json"


def run_case(case: dict) -> list[dict]:
    state = ConversationState()
    turns_raw = []
    for turn in case["turns"]:
        before = retrieve.QUERY_CALL_COUNT
        result = answer_conversational(turn["query"], state)
        after = retrieve.QUERY_CALL_COUNT
        result["query_call_count_delta"] = after - before
        turns_raw.append(result)
    return turns_raw


def summarize(scored_by_case: dict) -> dict:
    check_names = set()
    for scored in scored_by_case.values():
        check_names.update(scored["hard"].keys())

    summary = {"n_cases": len(scored_by_case), "hard_checks": {}}
    for check in sorted(check_names):
        applicable = 0
        passed = 0
        for scored in scored_by_case.values():
            r = scored["hard"].get(check, {}).get("result")
            if r == "n_a":
                continue
            applicable += 1
            if r == "pass":
                passed += 1
        summary["hard_checks"][check] = {
            "passed": passed,
            "applicable": applicable,
            "pass_rate": round(passed / applicable, 3) if applicable else None,
        }

    judge_cases = [s["judge"] for s in scored_by_case.values() if s.get("judge") and "error" not in s["judge"]]
    if judge_cases:
        summary["judge_checks"] = {
            "claim_support_rate": round(
                sum(1 for j in judge_cases if j["claim_support"]["supported"]) / len(judge_cases), 3
            ),
            "episode_acknowledgment_rate": round(
                sum(1 for j in judge_cases if j["episode_acknowledgment"]["acknowledged"] is True)
                / max(1, sum(1 for j in judge_cases if j["episode_acknowledgment"]["acknowledged"] is not None)),
                3,
            ) if any(j["episode_acknowledgment"]["acknowledged"] is not None for j in judge_cases) else None,
            "avg_helpfulness": round(
                sum(j["helpfulness"]["rating"] for j in judge_cases) / len(judge_cases), 2
            ),
            "n_judged": len(judge_cases),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the Phase 5 evaluation harness.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cases", default=str(CASES_PATH))
    parser.add_argument("--with-judge", action="store_true")
    args = parser.parse_args()

    with open(args.cases, "r", encoding="utf-8") as f:
        cases = json.load(f)

    raw_dir = EVAL_DIR / "results" / "raw" / args.run_id
    scored_dir = EVAL_DIR / "results" / "scored" / args.run_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    scored_dir.mkdir(parents=True, exist_ok=True)

    scored_by_case = {}
    for case in cases:
        print(f"[{case['case_id']}] running...")
        turns_raw = run_case(case)

        with open(raw_dir / f"{case['case_id']}.json", "w", encoding="utf-8") as f:
            json.dump(turns_raw, f, indent=2)

        hard = scoring_hard.score_case(case, turns_raw)
        judge = scoring_judge.score_case(case, turns_raw) if args.with_judge else None
        scored = {"case_id": case["case_id"], "hard": hard, "judge": judge}
        scored_by_case[case["case_id"]] = scored

        with open(scored_dir / f"{case['case_id']}.json", "w", encoding="utf-8") as f:
            json.dump(scored, f, indent=2)

        fails = [name for name, r in hard.items() if r["result"] == "fail"]
        status = "OK" if not fails else f"FAILS: {fails}"
        print(f"  -> {status}")

    summary = summarize(scored_by_case)
    summary_path = EVAL_DIR / "results" / f"{args.run_id}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
