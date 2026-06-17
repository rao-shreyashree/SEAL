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