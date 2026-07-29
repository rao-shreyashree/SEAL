"""build_figures.py — load all condition outputs, build results_by_condition,
generate all 5 paper figures.
"""
import json
from seal.task_result import TaskResult
from seal.figures import generate_all_figures

SEALLOGS = "./seallogs"

def load_condition(path: str) -> list:
    """Flattens {task_id: [dict, dict, ...]} -> List[TaskResult]"""
    with open(path) as f:
        data = json.load(f)
    results = []
    for task_id, iterations in data.items():
        for d in iterations:
            results.append(TaskResult.from_dict(d))
    return results

results_by_condition = {
    "SEAL": load_condition(f"{SEALLOGS}/production_runner_summary.json"),
    "No-Rubric-Evolution": load_condition(f"{SEALLOGS}/no_rubric_evolution_summary.json"),
    "Reflexion": load_condition(f"{SEALLOGS}/reflexion_baseline_summary.json"),
    "Zero-Shot": load_condition(f"{SEALLOGS}/zeroshot_baseline_summary.json"),
}

for name, results in results_by_condition.items():
    print(f"{name}: {len(results)} rows across {len(set(r.task_id for r in results))} tasks")

paths = generate_all_figures(results_by_condition)
print("\nFigures generated:")
for fig_name, path in paths.items():
    print(f"  {fig_name}: {path}")

for name, results in results_by_condition.items():
    print(name, len(results), len(set(r.task_id for r in results)))