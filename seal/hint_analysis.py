"""
seal/hint_analysis.py — measures the judge->agent hint channel (E3c).

Answers: when the judge rewrote its rubric, did a hint token actually
reach the agent, and did it match the failure the oracle flagged?

Requires TaskResult.hints_emitted (populated by E3b runner.py patch).
If hints_emitted is None on all results, you're reading pre-patch logs
— rerun the SEAL_FULL condition with the patched runner first.

Usage:
    import json, glob
    from seal.task_result import TaskResult
    from seal.hint_analysis import hint_emission_rate, silent_events, emission_by_failure_type

    results = [TaskResult.from_json_file(p)
               for p in glob.glob("seallogs/trace_SEAL_FULL_*.json")]

    print(hint_emission_rate(results))
    print(emission_by_failure_type(results))
    json.dump(silent_events(results), open("silent_events.json", "w"), indent=2)

Hand silent_events.json to Anagha — that's her E3d input.
She reads rubric_text for each silent event and judges whether
the rewrite targeted the right mode without triggering a keyword.
"""

from __future__ import annotations
from typing import List


def hint_emission_rate(results: List) -> dict:
    """
    Top-level channel health: of all evolution events, how many emitted
    at least one hint, and of those, how many matched the oracle failure?

    Returns a dict with:
        n_evolution_events : int   — iterations where evolve_rubric() ran
        n_emitted_any      : int   — subset that produced ≥1 hint token
        n_matched_oracle   : int   — subset where hint matched oracle label
        emission_rate      : float — n_emitted_any / n_evolution_events
        match_rate         : float — n_matched_oracle / n_evolution_events
    """
    # Only count iterations where evolution actually ran
    # (hints_emitted is None when evolution didn't run that iter)
    ran = [r for r in results if r.hints_emitted is not None]

    if not ran:
        return {
            "n_evolution_events": 0,
            "n_emitted_any": 0,
            "n_matched_oracle": 0,
            "emission_rate": 0.0,
            "match_rate": 0.0,
            "note": (
                "hints_emitted is None on all results. "
                "Rerun with E3b-patched runner.py first."
            ),
        }

    emitted = [r for r in ran if r.hints_emitted]
    matched = [
        r for r in emitted
        if f"{(r.oracle_failure_type or '').upper()}_ADDRESSED"
        in r.hints_emitted
    ]

    return {
        "n_evolution_events": len(ran),
        "n_emitted_any": len(emitted),
        "n_matched_oracle": len(matched),
        "emission_rate": round(len(emitted) / len(ran), 3),
        "match_rate": round(len(matched) / len(ran), 3),
    }


def silent_events(results: List) -> List[dict]:
    """
    Evolution events that emitted no hint tokens (hints_emitted == []).

    These are the interesting cases: evolution ran and produced a new
    rubric, but no keyword diff triggered a hint, so the agent got
    no behavioral signal. Anagha reads rubric_text for each and manually
    checks whether the rewrite actually addressed the right failure mode
    despite producing no keyword. This is the manual step of E3d.

    Output goes to silent_events.json.
    """
    return [
        {
            "task_id": r.task_id,
            "iteration": r.iteration,
            "oracle_failure_type": r.oracle_failure_type,
            "judge_failure_type": r.judge_failure_type,
            "rubric_drift_score": r.rubric_drift_score,
            "rubric_text": r.rubric_text,
        }
        for r in results
        if r.hints_emitted is not None and not r.hints_emitted
    ]


def emission_by_failure_type(results: List) -> dict:
    """
    Break down emission and match rates by oracle failure type.

    Answers: does the hint channel fail more often for CONTEXT_LOSS
    than GOAL_DRIFT? If one failure type has consistently low match_rate,
    the keyword list in judge.py needs broadening for that mode.

    Returns dict keyed by failure type string, each with:
        events   : int — evolution events for this failure type
        emitted  : int — events that produced ≥1 hint
        matched  : int — events where hint matched oracle label
    """
    out: dict = {}
    for r in results:
        if r.hints_emitted is None:
            continue
        k = (r.oracle_failure_type or "UNKNOWN").upper()
        d = out.setdefault(k, {"events": 0, "emitted": 0, "matched": 0})
        d["events"] += 1
        if r.hints_emitted:
            d["emitted"] += 1
            if f"{k}_ADDRESSED" in r.hints_emitted:
                d["matched"] += 1
    return out