"""
seal/build_figures.py — loads all condition summary JSONs, flattens into
List[TaskResult] per condition, generates all figures.

Run from project root: python -m seal.build_figures
"""

import json
import os

from seal.task_result import TaskResult
from seal.figures import generate_all_figures

SEALLOGS = "./seallogs"

# condition name -> summary filename
CONDITION_FILES = {
    "SEAL": "production_runner_summary.json",
    "No-Rubric-Evolution": "no_rubric_evolution_summary.json",
    "Reflexion": "reflexion_baseline_summary.json",
    "Zero-Shot": "zeroshot_baseline_summary.json",
}


def _load_condition(filename: str) -> list:
    path = os.path.join(SEALLOGS, filename)
    if not os.path.exists(path):
        print(f"  [SKIP] {filename} not found at {path}")
        return []
    with open(path) as f:
        raw = json.load(f)
    results = []
    for task_id, iterations in raw.items():
        for d in iterations:
            results.append(TaskResult.from_dict(d))
    return results


def main():
    results_by_condition = {}
    for condition, filename in CONDITION_FILES.items():
        results = _load_condition(filename)
        results_by_condition[condition] = results
        print(f"  {condition}: {len(results)} rows loaded from {filename}")

    missing = [c for c, r in results_by_condition.items() if not r]
    if missing:
        print(f"\n[WARNING] Empty conditions: {missing} — figures using these will be wrong/empty.")

    print("\nGenerating figures...")
    paths = generate_all_figures(results_by_condition)
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()