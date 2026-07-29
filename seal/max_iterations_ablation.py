"""max_iterations_ablation.py - for each condition, computes final success
rate if max_iterations had been capped at 1, 2, or 3. 
Answers: how much does each extra iteration actually buy us, per condition.

Zero-Shot excluded - always exactly 1 iteration, capping is meaningless for it.
"""
import json
from collections import defaultdict
from seal.task_result import TaskResult

SEALLOGS = "./seallogs"

def load_condition(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    results = []
    for task_id, iterations in data.items():
        for d in iterations:
            results.append(TaskResult.from_dict(d))
    return results

def success_rate_at_cap(results: list, cap: int) -> float:
    """% of tasks that succeeded within the first `cap` iterations."""
    by_task = defaultdict(list)
    for r in results:
        by_task[r.task_id].append(r)

    successes = 0
    for task_id, rs in by_task.items():
        capped = [r for r in rs if r.iteration <= cap]
        if any(r.success for r in capped):
            successes += 1
    return round(successes / len(by_task), 4) if by_task else 0.0


results_by_condition = {
    "SEAL": load_condition(f"{SEALLOGS}/production_runner_summary.json"),
    "No-Rubric-Evolution": load_condition(f"{SEALLOGS}/no_rubric_evolution_summary.json"),
    "Reflexion": load_condition(f"{SEALLOGS}/reflexion_baseline_summary.json"),
}

print(f"{'Condition':<22}{'iter=1':<10}{'iter<=2':<10}{'iter<=3':<10}")
for name, results in results_by_condition.items():
    r1 = success_rate_at_cap(results, 1)
    r2 = success_rate_at_cap(results, 2)
    r3 = success_rate_at_cap(results, 3)
    print(f"{name:<22}{r1:<10}{r2:<10}{r3:<10}")