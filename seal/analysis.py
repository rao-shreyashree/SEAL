"""
seal/analysis.py - E1: judge classification accuracy vs oracle labels.
Owner: Shreyashree

Data check before use:
    judge_failure_type is populated on every record in the 50-task version of production_runner_summary.json
    reflexion_baseline_summary.json has judge_failure_type=None everywhere by design (no judge in that condition) - so do not feed it into this module
    mistral_judge_summary.json has judge_failure_type populated but wrong (100% API failure, forced "execution_error")
    Only production_runner_summary.json (SEAL_FULL) is valid input.

Usage:
    python seal/analysis.py seallogs/production_runner_summary.json
"""

import json
from collections import defaultdict
 
LABELS = ["NONE", "CONTEXT_LOSS", "GOAL_DRIFT", "EXECUTION_ERROR", "HALLUCINATION", "UNKNOWN"]
 
 
def _normalize_oracle(label):
    if not label:
        return "UNKNOWN"
    return label.upper()
 
 
def _normalize_judge(label):
    if not label:
        return "UNKNOWN"
    label = label.upper()
    return label if label in LABELS else "UNKNOWN"
 
 
def load_condition_json(path: str) -> list:
    """Flattens {"task_001": [iter1, iter2, ...], ...} into a flat list of
    iteration-level records. One record = one judge.evaluate() call, so
    n = total evaluate() calls across the run, not n = number of tasks."""
    with open(path) as f:
        data = json.load(f)
    flat = []
    for iterations in data.values():
        flat.extend(iterations)
    return flat
 
 
def judge_label_confusion(records: list) -> dict:
    """confusion[oracle_label][judge_label] = count, over every iteration
    with a non-null judge_failure_type."""
    confusion = defaultdict(lambda: defaultdict(int))
    skipped = 0
    for r in records:
        judge = r.get("judge_failure_type")
        if judge is None:
            skipped += 1
            continue
        o = _normalize_oracle(r.get("oracle_failure_type"))
        j = _normalize_judge(judge)
        confusion[o][j] += 1
    if skipped:
        print(f"[judge_label_confusion] skipped {skipped} records with judge_failure_type=None")
    return {k: dict(v) for k, v in confusion.items()}
 
 
def judge_label_scores(confusion: dict) -> dict:
    """Per-class precision/recall/F1 + overall accuracy from the confusion dict."""
    all_labels = sorted(set(confusion.keys()) | {j for row in confusion.values() for j in row})
    scores = {}
    for label in all_labels:
        tp = confusion.get(label, {}).get(label, 0)
        fn = sum(row.get(label, 0) for oracle, row in confusion.items() if oracle != label)
        fp = sum(row.get(label, 0) for oracle, row in confusion.items() if oracle != label)
        # note: fn and fp above are NOT symmetric mistakes - fn sums the
        # oracle=label row's misses across other judge columns; fp sums
        # other oracle rows' hits on this judge column. Written explicitly:
        row_for_label = confusion.get(label, {})
        fn = sum(v for j, v in row_for_label.items() if j != label)
        fp = sum(row.get(label, 0) for oracle, row in confusion.items() if oracle != label)
 
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = sum(row_for_label.values())
        scores[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": support,
        }
    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion.get(l, {}).get(l, 0) for l in all_labels)
    scores["_overall_accuracy"] = round(correct / total, 3) if total else 0.0
    scores["_n"] = total
    return scores
 
 
def latex_confusion_table(confusion: dict) -> str:
    """LaTeX tabular, oracle rows x judge columns, for §V."""
    all_labels = sorted(set(confusion.keys()) | {j for row in confusion.values() for j in row})
    short = {l: l.replace("_", "\\_") for l in all_labels}
 
    header = " & ".join(["Oracle \\textbackslash\\ Judge"] + [short[l] for l in all_labels])
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{l" + "c" * len(all_labels) + "}",
        "\\toprule",
        header + " \\\\",
        "\\midrule",
    ]
    for oracle in all_labels:
        row = confusion.get(oracle, {})
        cells = [str(row.get(j, 0)) for j in all_labels]
        lines.append(short[oracle] + " & " + " & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Judge failure-type classification vs.\\ oracle ground truth "
        "(n = TOTAL, SEAL\\_FULL condition, all iterations).}",
        "\\label{tab:judge-confusion}",
        "\\end{table}",
    ]
    return "\n".join(lines)
 
 
def rubric_drift_distribution(records: list) -> dict:
    """E2a. rubric_drift_score is only set by a real evolve_rubric() call.
    In the current runner.py, that's ONLY iteration==2 on a failing task
    (iteration >= 2 and iteration < 3 -> iteration==2 exclusively). Every
    other iteration - 1, or 3 in a task that fails all three - keeps the
    structural default 0.0 that's set before the loop even checks whether
    to evolve. That default is NOT a measurement; filtering only on
    iteration==1 (the old version of this function) let iteration-3
    defaults leak in as if they were real near-zero similarity scores,
    which silently doubled n and fabricated a floor-rejection rate that
    wasn't real - confirmed on the actual data (10 events at ~0.0 that
    turned out to be all iteration-3 structural defaults, 0 genuine
    rejections among the 10 real evolve() calls).
 
    Prefer hints_emitted (Tanisha's field) when present - it's an explicit
    None for "evolution didn't run" vs [] for "ran, no hint", so it doesn't
    need this iteration-number heuristic at all. Falls back to the
    iteration==2 heuristic for logs predating that field, with a loud
    warning since the fallback is inherently runner-version-specific.
    """
    scores = []
    skipped_no_evolution = 0
 
    has_hints_field = any("hints_emitted" in r for r in records)
    if not has_hints_field:
        print("[rubric_drift_distribution] WARNING: no record has 'hints_emitted' - "
              "falling back to the iteration==2 heuristic, which is tied to the "
              "current evolve-on-iteration-2-only runner.py logic and will silently "
              "break if that changes. Re-run with the patched runner once possible.")
 
    for r in records:
        s = r.get("rubric_drift_score")
        it = r.get("iteration", 1)
        if s is None:
            skipped_no_evolution += 1
            continue
        if has_hints_field:
            if r.get("hints_emitted") is None:
                continue  # evolution didn't run this iteration
        else:
            if it != 2:
                continue  # legacy heuristic: only iter 2 ever calls evolve_rubric()
        scores.append(s)
 
 
    if not scores:
        return {"n": 0, "scores": [], "note": "no evolve_rubric() calls found in this file"}
 
    n = len(scores)
    mean = sum(scores) / n
    variance = sum((x - mean) ** 2 for x in scores) / n if n > 1 else 0.0
    std = variance ** 0.5
    below_floor = sum(1 for s in scores if s < 0.45)  # floor owned by Anagha, E2b
 
    return {
        "n": n,
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "below_floor_0.45": below_floor,
        "below_floor_pct": round(100 * below_floor / n, 1),
        "scores": scores,  # raw list for figures.py histogram
        "skipped_iteration1_or_null": skipped_no_evolution,
    }
 
 
def rubric_drift_histogram_series(records: list, bins: int = 10):
    """Feeds figures.py's chart tooling (or chart_display_v0) directly.
    Returns (bin_edges, counts) - a plain histogram, no plotting here since
    figures.py owns matplotlib calls per the project's file-ownership rule."""
    dist = rubric_drift_distribution(records)
    scores = dist["scores"]
    if not scores:
        return [], []
 
    lo, hi = 0.0, 1.0  # cosine similarity range
    width = (hi - lo) / bins
    edges = [round(lo + i * width, 3) for i in range(bins + 1)]
    counts = [0] * bins
    for s in scores:
        idx = min(int((s - lo) / width), bins - 1)
        counts[idx] += 1
    return edges, counts
 
 
def hint_emission_rate(records: list) -> dict:
    """
    E3c. Measures how often an evolve_rubric() call emitted the hint token
    the environment gates recovery on, vs. rewrote the rubric but stayed
    silent (the review's §VI-A claim, currently asserted not measured).
 
    CONFIRMED token scheme (judge.py on main, seen 2026-06-2x):
      evolve_rubric() does keyword-diff between old/new rubric text and
      appends to new_rubric["_seal_hints"]: a list containing zero or more
      of "CONTEXT_LOSS_ADDRESSED", "GOAL_DRIFT_ADDRESSED",
      "EXECUTION_ERROR_ADDRESSED". scenarios.py's step() reads
      rubric["_seal_hints"] directly (rubric is now a dict at that call
      site, not a string). This matches Tanisha's description exactly.
 
    BLOCKED as of this version: judge.py's own comment states _seal_hints
    is "stripped before logging" so it never reaches TaskResult.rubric_text.
    That means rubric_text - the only rubric-shaped field currently on
    TaskResult - cannot answer this question; the data is deliberately
    removed before storage. TaskResult needs a `hints_emitted` field
    (Tanisha's E3a) populated in runner.py (E3b) before this function has
    anything to read. Until then this raises rather than silently
    returning 0% and being mistaken for a real result.
    """
    if not records:
        return {"error": "no records"}
 
    if not any("hints_emitted" in r for r in records):
        raise RuntimeError(
            "No record has a 'hints_emitted' field. E3a/E3b (Tanisha: add "
            "hints_emitted to TaskResult, populate it in runner.py after "
            "evolve_rubric()) haven't landed yet. rubric_text can't be used "
            "as a substitute - judge.py strips _seal_hints before it's "
            "logged there, by design. Don't compute this from rubric_text."
        )
 
    by_task = defaultdict(list)
    for r in records:
        by_task[r["task_id"]].append(r)
 
    expected_token = {
        "CONTEXT_LOSS": "CONTEXT_LOSS_ADDRESSED",
        "GOAL_DRIFT": "GOAL_DRIFT_ADDRESSED",
        "EXECUTION_ERROR": "EXECUTION_ERROR_ADDRESSED",
    }
 
    total_events = 0
    emitted = 0
    by_failure_type = defaultdict(lambda: {"total": 0, "emitted": 0})
    silent_events = []  # for Anagha's E3d manual read
 
    for task_id, iters in by_task.items():
        iters = sorted(iters, key=lambda x: x["iteration"])
        for i in range(len(iters) - 1):
            cur, nxt = iters[i], iters[i + 1]
            if cur.get("success"):
                continue
            oracle = _normalize_oracle(cur.get("oracle_failure_type"))
            token = expected_token.get(oracle)
            if token is None:
                continue
            total_events += 1
            by_failure_type[oracle]["total"] += 1
            hints_next = nxt.get("hints_emitted") or []
            if token in hints_next:
                emitted += 1
                by_failure_type[oracle]["emitted"] += 1
            else:
                silent_events.append({
                    "task_id": task_id,
                    "iteration": cur["iteration"],
                    "oracle_failure_type": oracle,
                    "expected_token": token,
                    "hints_emitted_next": hints_next,
                    "rubric_text_next": nxt.get("rubric_text"),  # for manual read
                })
 
    return {
        "total_evolution_events": total_events,
        "emitted": emitted,
        "emission_rate": round(emitted / total_events, 3) if total_events else None,
        "by_failure_type": dict(by_failure_type),
        "silent_events": silent_events,
    }
 
 
if __name__ == "__main__":
    import sys
 
    path = sys.argv[1] if len(sys.argv) > 1 else "seallogs/production_runner_summary.json"
    records = load_condition_json(path)
    print(f"Loaded {len(records)} iteration-level records from {path}")
 
    confusion = judge_label_confusion(records)
    scores = judge_label_scores(confusion)
 
    print("\nConfusion matrix:")
    print(json.dumps(confusion, indent=2))
    print("\nPer-class scores:")
    print(json.dumps(scores, indent=2))
 
    table = latex_confusion_table(confusion)
    table = table.replace("TOTAL", str(scores["_n"]))
    print("\nLaTeX table:\n")
    print(table)
 
    with open("judge_confusion_table.tex", "w") as f:
        f.write(table)
    print("\nWritten to judge_confusion_table.tex")
 
    # E2a: rubric drift distribution
    print("\n" + "=" * 60)
    drift = rubric_drift_distribution(records)
    print("\nRubric drift distribution (evolve_rubric() calls only):")
    print(json.dumps({k: v for k, v in drift.items() if k != "scores"}, indent=2))
 
    edges, counts = rubric_drift_histogram_series(records)
    if edges:
        print("\nHistogram (10 bins, 0.0-1.0 cosine similarity):")
        for i in range(len(counts)):
            print(f"  [{edges[i]:.2f}, {edges[i+1]:.2f}): {'#' * counts[i]} ({counts[i]})")
 
        with open("rubric_drift_distribution.json", "w") as f:
            json.dump(drift, f, indent=2)
        print("\nWritten to rubric_drift_distribution.json (raw scores for figures.py)")
    else:
        print("\nNo drift scores found - check you're pointing at production_runner_summary.json")
 
    # E3c: hint emission rate
    print("\n" + "=" * 60)
    try:
        hints = hint_emission_rate(records)
        print("\nHint emission rate:")
        print(json.dumps({k: v for k, v in hints.items() if k != "silent_events"}, indent=2))
        if hints["silent_events"]:
            with open("silent_evolution_events.json", "w") as f:
                json.dump(hints["silent_events"], f, indent=2)
            print(f"\n{len(hints['silent_events'])} silent events written to "
                  f"silent_evolution_events.json for Anagha's manual read (E3d)")
    except RuntimeError as e:
        print(f"\n[E3c blocked] {e}")
 