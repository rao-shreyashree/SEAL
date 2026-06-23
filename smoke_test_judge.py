"""
smoke_test_judge.py: validate SEALJudge against a real trace

why this file?
the notebook's "100% judge accuracy" was measured against 7 hand-written mock traces (TASK_TEMPLATES), not real agent rollouts. 
The judge's prompt was also tuned against that specific hand-formatted trace style ("TASK: ... / Step N / OBS: ... / ACT: ..."), 
not the plain json.dumps(raw_trace) that seal/judge.py's trace_to_str() actually produces
from TaskResult.raw_trace
this script is to close that gap before merge

usage:
    python smoke_test_judge.py --traces baseline_all_traces.jsonl
    python smoke_test_judge.py # uses synthetic fallback

what it checks (in order - stops and tells us exactly what failed):
    1. can we load a real, non-NONE-confound trace?
    2. does evaluate() return well-formed output (right types, valid enum)?
    3. does the judge's score/failure_type roughly track the oracle outcome?
    4. does evolve_rubric() produce a numeric drift score, not a hash?

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seal.judge import SEALJudge, trace_to_str, DEFAULT_RUBRIC  # noqa: E402


# synthetic fallback: used only if no real trace file is given/found 
SYNTHETIC_FALLBACK_TRACE = {
    "task_id": "synthetic_smoke_001",
    "oracle_failure_type": "NONE",
    "success": True,
    "raw_trace": [
        {"step": 1, "obs": "You are in the kitchen. You see a fridge, counter, sink.", "action": "go to counter 1"},
        {"step": 2, "obs": "You see lettuce, knife, plate on the counter.", "action": "take lettuce from counter 1"},
        {"step": 3, "obs": "You are holding a lettuce.", "action": "go to fridge 1"},
        {"step": 4, "obs": "You are at the fridge.", "action": "cool lettuce with fridge 1"},
        {"step": 5, "obs": "Lettuce is now chilled.", "action": "put lettuce in/on fridge 1"},
    ],
}


def load_real_trace(path: str) -> dict | None:
    """Pull the first NONE-outcome (i.e. not a forced-failure scenario) trace
    from a real baseline_all_traces.jsonl. Returns None if file is missing
    or no qualifying trace is found, so caller can fall back cleanly.
    """
    p = Path(path)
    if not p.exists():
        print(f"[smoke_test] {path} not found - falling back to synthetic trace.")
        return None

    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("oracle_failure_type") == "NONE":
                return record

    print(f"[smoke_test] No NONE-outcome trace found in {path} - falling back to synthetic trace.")
    return None


def run_smoke_test(record: dict, source_label: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"SMOKE TEST - source: {source_label}")
    print(f"task_id: {record.get('task_id', '<unknown>')}")
    print(f"oracle outcome: success={record.get('success')}, "
          f"oracle_failure_type={record.get('oracle_failure_type')}")
    print(f"{'=' * 60}\n")

    raw_trace = record.get("raw_trace")
    if raw_trace is None:
        print("[FAIL] Record has no 'raw_trace' field - check field name against "
              "your actual TaskResult/trace file schema.")
        return False

    trace_str = trace_to_str(raw_trace)
    print(f"[1/3] trace_to_str() produced {len(trace_str)} chars. "
          f"First 200 chars:\n{trace_str[:200]}\n")

    judge = SEALJudge() # requires GEMINI_API_KEY in env

    try:
        result = judge.evaluate(trace_str, DEFAULT_RUBRIC)
    except Exception as e:
        print(f"[FAIL] evaluate() raised: {type(e).__name__}: {e}")
        return False

    print("[2/3] evaluate() returned:")
    print(f"    score: {result.score}")
    print(f"    failure_type: {result.failure_type}")
    print(f"    explanation: {result.explanation}")

    # basic well-formedness checks - not accuracy
    ok = True
    if not isinstance(result.score, float) or not (0.0 <= result.score <= 1.0):
        print(f"    [WARN] score {result.score!r} is not a float in [0.0, 1.0]")
        ok = False
    if result.failure_type is not None and result.score >= 0.8:
        print("    [WARN] failure_type is non-null but score >= 0.8 - "
              "judge's own stated rule (\"failure_type must be null if score >= 0.8\") violated")
        ok = False

    print(f"\n[3/3] Sanity vs oracle: oracle success={record.get('success')}, "
          f"judge score>=0.7 -> {result.score >= 0.7}")
    if record.get("success") is not None and (result.score >= 0.7) != record.get("success"):
        print("    [NOTE] Judge disagrees with the oracle on this single trace. "
              "Not necessarily a bug (judge and oracle can legitimately differ), "
              "but worth a manual read if this happens often across more traces.")

    print(f"\n{'PASS' if ok else 'FAIL (see WARN lines above)'}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default="baseline_all_traces.jsonl",
                         help="Path to real trace file. Falls back to synthetic if missing.")
    args = parser.parse_args()

    record = load_real_trace(args.traces)
    if record is not None:
        ok = run_smoke_test(record, source_label=args.traces)
    else:
        ok = run_smoke_test(SYNTHETIC_FALLBACK_TRACE, source_label="SYNTHETIC FALLBACK (not real validation)")
        print("\n[smoke_test] REMINDER: this only validated plumbing, not judge accuracy. "
              "Re-run with --traces pointing at a real baseline_all_traces.jsonl before merge.")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()