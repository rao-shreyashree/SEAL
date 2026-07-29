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