"""Phase 5 - diffs two eval runs: aggregate pass-rate deltas plus any case that
flipped pass<->fail on any hard check.

Usage:
    python src/eval/compare_runs.py --baseline baseline --candidate v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent.parent / "eval"


def load_summary(run_id: str) -> dict:
    with open(EVAL_DIR / "results" / f"{run_id}_summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_scored_dir(run_id: str) -> dict:
    d = EVAL_DIR / "results" / "scored" / run_id
    scored = {}
    for path in sorted(d.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            scored[path.stem] = json.load(f)
    return scored


def main():
    parser = argparse.ArgumentParser(description="Compare two eval runs.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()

    base_summary = load_summary(args.baseline)
    cand_summary = load_summary(args.candidate)
    base_scored = load_scored_dir(args.baseline)
    cand_scored = load_scored_dir(args.candidate)

    print(f"=== Hard check pass rates: {args.baseline} -> {args.candidate} ===")
    checks = sorted(set(base_summary["hard_checks"]) | set(cand_summary["hard_checks"]))
    for check in checks:
        b = base_summary["hard_checks"].get(check, {}).get("pass_rate")
        c = cand_summary["hard_checks"].get(check, {}).get("pass_rate")
        arrow = ""
        if b is not None and c is not None:
            if c > b:
                arrow = "  IMPROVED"
            elif c < b:
                arrow = "  REGRESSED"
        print(f"  {check:28s} {b} -> {c}{arrow}")

    if "judge_checks" in base_summary or "judge_checks" in cand_summary:
        print(f"\n=== Judge checks: {args.baseline} -> {args.candidate} ===")
        bj = base_summary.get("judge_checks", {})
        cj = cand_summary.get("judge_checks", {})
        for key in sorted(set(bj) | set(cj)):
            print(f"  {key:28s} {bj.get(key)} -> {cj.get(key)}")

    print(f"\n=== Per-case flips ({args.baseline} vs {args.candidate}) ===")
    all_cases = sorted(set(base_scored) | set(cand_scored))
    any_flip = False
    for case_id in all_cases:
        b = base_scored.get(case_id, {}).get("hard", {})
        c = cand_scored.get(case_id, {}).get("hard", {})
        for check in sorted(set(b) | set(c)):
            br = b.get(check, {}).get("result")
            cr = c.get(check, {}).get("result")
            if br != cr and "n_a" not in (br, cr):
                any_flip = True
                print(f"  {case_id} :: {check}: {br} -> {cr}")
    if not any_flip:
        print("  (no per-case hard-check flips)")


if __name__ == "__main__":
    main()
