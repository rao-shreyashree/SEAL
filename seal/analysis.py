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
    """rubric_drift_score is only set on iterations where evolve_rubric() actually ran (iteration >= 2, on failure) 
    default 0.0 elsewhere isn't a real drift measurement, it's "no evolution attempted", so exclude it
    Distinguish from the floor-rejection case (evolve ran, similarity computed, but mutation discarded)
    those have a real score too and belong in the distribution; only true nulls/no-evolution get dropped.
    """
    scores = []
    skipped_no_evolution = 0
    for r in records:
        s = r.get("rubric_drift_score")
        it = r.get("iteration", 1)
        if s is None:
            skipped_no_evolution += 1
            continue
        if it == 1:
            # iteration 1 is always 0.0 by construction (task_result.py comment: "no evolution has happened yet on the first attempt") 
            # not a real drift measurement, exclude regardless of value
            continue
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
 
    # rubric drift distribution
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
        print("\nNo drift scores found - check if production_runner_summary.json is correctly pointed to")
 