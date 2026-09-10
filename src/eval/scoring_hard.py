"""Phase 5 - deterministic checks, no LLM call. Run these first; they're the ones
that can't be disputed.

Convention used across all hard checks: within a case's `turns`, index 0 is the
primary/substantive query and is checked against the case's declared expectations;
any turn after index 0 exists specifically to test follow-up behavior (per the
Phase 4 spec's use of `turns` for "a short conversation") and must classify as
`follow_up` with zero new retrieval calls, regardless of what index-0 expects.
"""

from __future__ import annotations

import re

# Matches the system prompt's mandated in-text citation format "(Episode N, mm:ss-mm:ss)".
# Deliberately narrow (requires the opening parenthesis) so it doesn't false-match a
# markdown section header like "**Episode 1 - Title**" that isn't attaching a citation
# to any specific claim.
_CITATION_PATTERN = re.compile(r"\(Episode\s+(\d+)")


def _result(passed: bool | None, detail: str) -> dict:
    if passed is None:
        return {"result": "n_a", "detail": detail}
    return {"result": "pass" if passed else "fail", "detail": detail}


def check_citation_grounded(case: dict, turns_raw: list[dict]) -> dict:
    for i, turn in enumerate(turns_raw):
        retrieved = set(turn["retrieved_chunk_ids"])
        for c in turn["citations"]:
            if c.get("chunk_id") not in retrieved:
                return _result(False, f"turn {i}: citation {c.get('chunk_id')} not in retrieved_chunk_ids")
    total_citations = sum(len(t["citations"]) for t in turns_raw)
    return _result(True, f"all {total_citations} citation(s) across {len(turns_raw)} turn(s) trace to a retrieved chunk")


def check_refusal_correct(case: dict, turns_raw: list[dict]) -> dict:
    """Only meaningful for broad_rebalanced retrieval, where the code-level `refused`
    flag is the intended mechanism. single_episode/named_comparison never hard-refuse
    by design (see Phase 4 architecture note in EVAL.md) - marked n/a there, not failed."""
    expected = case["expected_refused"]
    details = []
    applicable = False
    for i, turn in enumerate(turns_raw):
        if turn["retrieval_mode"] != "broad_rebalanced":
            details.append(f"turn {i}: n/a (mode={turn['retrieval_mode']})")
            continue
        applicable = True
        exp = expected[i] if i < len(expected) else expected[-1]
        ok = turn["refused"] == exp
        details.append(f"turn {i}: expected={exp} actual={turn['refused']} -> {'pass' if ok else 'FAIL'}")
        if not ok:
            return _result(False, "; ".join(details))
    if not applicable:
        return _result(None, "no broad_rebalanced turn in this case; " + "; ".join(details))
    return _result(True, "; ".join(details))


def check_intent_match(case: dict, turns_raw: list[dict]) -> dict:
    turn0 = turns_raw[0]
    acceptable = case["acceptable_query_types"]
    if turn0["query_type"] not in acceptable:
        return _result(False, f"turn 0 query_type={turn0['query_type']!r} not in {acceptable}")

    expected_refs = set(case["expected_episode_refs"])
    actual_refs = set(turn0.get("episode_refs", []))
    if expected_refs and actual_refs != expected_refs:
        return _result(False, f"turn 0 episode_refs={sorted(actual_refs)} expected {sorted(expected_refs)}")

    for i, turn in enumerate(turns_raw[1:], start=1):
        if turn["query_type"] != "follow_up":
            return _result(False, f"turn {i} query_type={turn['query_type']!r}, expected follow_up")

    return _result(True, f"turn 0 type={turn0['query_type']!r} refs={sorted(actual_refs)}, "
                          f"{len(turns_raw) - 1} follow-up turn(s) correctly classified")


def check_no_uncovered_citation(case: dict, turns_raw: list[dict]) -> dict:
    """Checks the answer TEXT's in-text citations, not the `citations` metadata list -
    that list intentionally includes every retrieved chunk (so citations can never be
    hallucinated, per the Phase 3 architecture), so it always contains the uncovered
    episode's chunks too whenever that episode was named/retrieved at all. The real
    question is whether the model attached a specific claim to that episode inline -
    i.e. whether "(Episode N, ...)" appears in the prose for the uncovered N."""
    uncovered = {int(k) for k, v in case["coverage_expectation"].items() if v is False}
    if not uncovered:
        return _result(None, "no episode marked uncovered in this case")
    for i, turn in enumerate(turns_raw):
        cited = {int(m) for m in _CITATION_PATTERN.findall(turn["answer"])}
        bad = cited & uncovered
        if bad:
            return _result(False, f"turn {i}: answer text attaches an in-text citation to uncovered episode(s) {sorted(bad)}")
    return _result(True, f"no in-text citation attributes a claim to uncovered episode(s) {sorted(uncovered)}")


def _expected_call_count(case: dict, turn_index: int) -> int | None:
    """Returns expected new-Chroma-call count for this turn, or None if not determinable
    from the case's declared fields (falls back to reading retrieval_mode from the turn)."""
    if turn_index > 0:
        return 0  # every turn after the first is a follow-up by convention
    refs = case["expected_episode_refs"]
    unknown = set(case["expected_unknown_episodes"])
    valid = [r for r in refs if r not in unknown]
    if not refs:
        return 1  # broad/general/recommendation
    if len(valid) == 0:
        return 0  # all named episodes unknown
    if len(valid) <= 6:
        return len(valid)  # scoped_loop, one call per valid named episode
    return 1  # named_comparison with >6 refs falls back to broad


def check_routing_efficiency(case: dict, turns_raw: list[dict]) -> dict:
    details = []
    for i, turn in enumerate(turns_raw):
        expected = _expected_call_count(case, i)
        actual = turn["query_call_count_delta"]
        ok = actual == expected
        details.append(f"turn {i}: expected={expected} actual={actual} -> {'pass' if ok else 'FAIL'}")
        if not ok:
            return _result(False, "; ".join(details))
    return _result(True, "; ".join(details))


CHECKS = {
    "citation_grounded": check_citation_grounded,
    "refusal_correct": check_refusal_correct,
    "intent_match": check_intent_match,
    "no_uncovered_citation": check_no_uncovered_citation,
    "routing_efficiency": check_routing_efficiency,
}


def score_case(case: dict, turns_raw: list[dict]) -> dict:
    return {name: fn(case, turns_raw) for name, fn in CHECKS.items()}
