from typing import List
from collections import defaultdict

def task_success_rate(results: List) -> float:
    """% of tasks that succeeded."""
    if not results:
        return 0.0
    return sum(r.success for r in results) / len(results)


def failure_analysis_precision(results: List) -> dict:
    """Per failure-type breakdown of how often each type occurs among failures."""
    counts = defaultdict(int)
    total_failures = 0

    for r in results:
        if not r.success:
            counts[r.failure_type] += 1
            total_failures += 1

    if total_failures == 0:
        return {}

    return {ftype: count / total_failures for ftype, count in counts.items()}


def judge_alignment(results: List) -> float:
    """
    Avg score on successful tasks vs failed tasks.
    Higher gap = judge/agent scoring is well-aligned with actual outcomes.
    """
    success_scores = [r.score for r in results if r.success]
    failure_scores = [r.score for r in results if not r.success]

    avg_success = sum(success_scores) / len(success_scores) if success_scores else 0.0
    avg_failure = sum(failure_scores) / len(failure_scores) if failure_scores else 0.0

    return round(avg_success - avg_failure, 4)


def convergence_speed(results: List) -> dict:
    """Per task: how many iterations until first success."""
    task_iterations = defaultdict(list)
    for r in results:
        task_iterations[r.task_id].append(r)

    iterations_to_success = []
    for task_id, task_results in task_iterations.items():
        task_results.sort(key=lambda x: x.iteration)
        for r in task_results:
            if r.success:
                iterations_to_success.append(r.iteration)
                break

    if not iterations_to_success:
        return {"avg_iterations_to_success": None, "tasks_never_solved": len(task_iterations)}

    return {
        "avg_iterations_to_success": round(sum(iterations_to_success) / len(iterations_to_success), 2),
        "tasks_never_solved": len(task_iterations) - len(iterations_to_success)
    }

# new additions: 
# recovery rate after first failure
# per-failure-type
# success rate
# judge call cost tracking
# all read from existing TaskResult fields,so no schema changes needed


def recovery_rate_after_first_failure(results: List) -> float:
    """
    % of tasks that failed on iteration 1 but succeeded on a later iteration.
    measures whether the agent actually recovers from failures, not just whether it eventually succeeds on easy tasks.
    'recovery rate after the first failure'
    """
    task_iterations = defaultdict(list)
    for r in results:
        task_iterations[r.task_id].append(r)

    recovered = 0
    failed_iter1 = 0

    for task_id, task_results in task_iterations.items():
        task_results.sort(key=lambda x: x.iteration)
        if not task_results[0].success:
            failed_iter1 += 1
            if any(r.success for r in task_results[1:]):
                recovered += 1

    if failed_iter1 == 0:
        return 0.0
    return round(recovered / failed_iter1, 4)


def success_rate_per_failure_type(results: List) -> dict:
    """
    Per oracle failure type: 
    what % of tasks with that failure type eventually succeeded (across all iterations for that task_id).
    distinct from failure_analysis_precision which measures distribution of failures, not whether each failure type is recoverable.
    'success rate per failure type (Goal Drift, Context Loss, Execution Error)'
    """
    # group by task_id to get the oracle failure type + whether it ever succeeded
    task_oracle = {} # task_id -> oracle_failure_type (from iter 1, the ground truth)
    task_success = {} # task_id -> bool (did it ever succeed across any iteration?)

    for r in results:
        if r.task_id not in task_oracle:
            task_oracle[r.task_id] = r.oracle_failure_type
        if r.success:
            task_success[r.task_id] = True
        elif r.task_id not in task_success:
            task_success[r.task_id] = False

    counts = defaultdict(lambda: {"total": 0, "succeeded": 0})
    for task_id, oracle_type in task_oracle.items():
        counts[oracle_type]["total"] += 1
        if task_success.get(task_id, False):
            counts[oracle_type]["succeeded"] += 1

    return {
        ftype: round(v["succeeded"] / v["total"], 4) if v["total"] > 0 else 0.0
        for ftype, v in counts.items()
    }


def judge_call_cost_per_task(per_task_calls: dict) -> dict:
    """
    Summary stats over the per_task_calls dict written by runner.py
    to keep track of number of judge calls or total tokens per successful task
    per_task_calls format: {"task_001": 3, "task_002": 1, ...} pass in the dict loaded from production_call_counts.json
    """
    if not per_task_calls:
        return {}

    calls = list(per_task_calls.values())
    total = sum(calls)
    return {
        "total_calls": total,
        "avg_calls_per_task": round(total / len(calls), 2),
        "min_calls": min(calls),
        "max_calls": max(calls),
        "tasks_at_min": [t for t, c in per_task_calls.items() if c == min(calls)],
        "tasks_at_max": [t for t, c in per_task_calls.items() if c == max(calls)],
    }


# E4b: mean +- std across seeded reruns
#
# run_baselines.py's --seeds N writes base_seed0.json ... base_seed{N-1}.json (or the unsuffixed base name when N==1 - see run_baselines.py's _seeded_filename)
# everything below works on top of the metric functions already defined in this file, unmodified - it just runs one of them once per seed and reduces the per-seed values to mean/std


def _mean_std(values: List[float]) -> dict:
    """Pure-python mean/std, no numpy dependency (matches the rest of this
    file's style). Population std (n, not n-1) - with 5 seeds this is a
    deliberate choice to report; don't silently switch to sample std
    without flagging it, the two differ meaningfully at n=5."""
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "n": 0, "values": []}
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
    return {
        "mean": round(mean, 4),
        "std": round(variance ** 0.5, 4),
        "n": n,
        "values": [round(v, 4) for v in values],
    }


def load_seeded_results(base_filename: str, seeds: List[int], output_dir: str = "./seallogs") -> List[list]:
    """
    Loads one condition's seeded output files into List[List[TaskResult]] -
    one inner list per seed, task iterations flattened (matches the flat
    list shape every function above expects).

    base_filename: the UNSUFFIXED name, e.g. "reflexion_baseline_summary.json"
        - this function reconstructs the _seedN suffix itself, matching
        run_baselines.py's naming exactly. Don't pass an already-suffixed
        name in.
    seeds: which seed indices to load, e.g. list(range(5)) for seeds 0-4.
        A single-element list loads the unsuffixed file directly (same
        run_baselines.py convention: total_seeds<=1 -> no suffix).
    """
    import json
    import os
    from seal.task_result import TaskResult

    all_seeds = []
    missing = []
    for seed in seeds:
        if len(seeds) <= 1:
            fname = base_filename
        else:
            name, ext = os.path.splitext(base_filename)
            fname = f"{name}_seed{seed}{ext}"
        path = os.path.join(output_dir, fname)

        if not os.path.exists(path):
            missing.append(path)
            continue

        with open(path) as f:
            data = json.load(f)

        flat = [TaskResult.from_dict(d) for iterations in data.values() for d in iterations]
        all_seeds.append(flat)

    if missing:
        print(f"[load_seeded_results] {len(missing)}/{len(seeds)} seed file(s) missing, "
              f"skipped: {missing}")
    if not all_seeds:
        raise FileNotFoundError(
            f"No seed files found for '{base_filename}' in {output_dir}. "
            f"Run run_baselines.py --seeds {len(seeds)} for this condition first."
        )
    return all_seeds


def mean_std_across_seeds(metric_fn, seeded_results: List[list]) -> dict:
    """
    Runs metric_fn once per seed's results and reduces to mean/std.

    metric_fn: any function above taking (results: List[TaskResult]) and
        returning either a float (task_success_rate, judge_alignment,
        recovery_rate_after_first_failure) or a dict of floats
        (failure_analysis_precision, success_rate_per_failure_type,
        convergence_speed - note convergence_speed's tasks_never_solved
        is a count, not a rate, still gets mean/std'd the same way).
    seeded_results: List[List[TaskResult]] from load_seeded_results().

    Returns:
      - float-valued metric_fn -> {"mean", "std", "n", "values"}
      - dict-valued metric_fn  -> {key: {"mean","std","n","values"}, ...}
        Keys that don't appear in every seed's output (e.g. a failure type
        that happened to not occur in one seed) get filled with None rather
        than silently dropped - a key present in 3/5 seeds is itself a
        finding, not a gap to paper over. Those None entries are excluded
        from the mean/std calculation for that key but counted in "n".
    """
    per_seed_values = [metric_fn(r) for r in seeded_results]

    if not per_seed_values:
        return {}

    if isinstance(per_seed_values[0], dict):
        all_keys = set()
        for v in per_seed_values:
            all_keys.update(v.keys())

        out = {}
        for key in sorted(all_keys):
            present = [v[key] for v in per_seed_values if key in v and isinstance(v[key], (int, float))]
            missing_count = len(per_seed_values) - len(present)
            stats = _mean_std(present)
            stats["n"] = len(per_seed_values)  # report actual seed count, not just present count
            stats["missing_in_seeds"] = missing_count
            out[key] = stats
        return out

    # float-valued metric_fn
    return _mean_std(per_seed_values)