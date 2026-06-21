"""
Paper figure generation. 
Every function takes condition-labeled TaskResult lists - condition name is supplied by ME at call time, NOT stored on TaskResult itself. 

Expected input shape for most functions:
    results_by_condition = {
        "SEAL": [...],
        "No-Rubric-Evolution": [...],
        "Reflexion": [...],
        "Zero-Shot": [...],
    }
"""

import matplotlib.pyplot as plt
from collections import defaultdict


def fig1_success_rate_by_iteration(results_by_condition: dict, save_path="fig1_success_by_iteration.png"):
    """Headline result. One line per condition, success rate at each iteration number."""
    plt.figure(figsize=(7, 5))
    for condition, results in results_by_condition.items():
        by_iter = defaultdict(list)
        for r in results:
            by_iter[r.iteration].append(r.success)
        iterations = sorted(by_iter.keys())
        rates = [sum(by_iter[i]) / len(by_iter[i]) for i in iterations]
        plt.plot(iterations, rates, marker="o", label=condition)
    plt.xlabel("Iteration")
    plt.ylabel("Success Rate")
    plt.title("Success Rate by Iteration Across Conditions")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def fig2_recovery_rate_by_failure_type(results_by_condition: dict, save_path="fig2_recovery_by_failure_type.png"):
    """Of tasks that failed with type X on iteration 1, what fraction eventually succeeded."""
    failure_types = ["CONTEXT_LOSS", "GOAL_DRIFT", "EXECUTION_ERROR", "UNKNOWN"]
    conditions = list(results_by_condition.keys())
    width = 0.8 / max(len(conditions), 1)
    x = range(len(failure_types))

    plt.figure(figsize=(8, 5))
    for idx, condition in enumerate(conditions):
        by_task = defaultdict(list)
        for r in results_by_condition[condition]:
            by_task[r.task_id].append(r)

        recovery_rates = []
        for ft in failure_types:
            flagged = [tid for tid, rs in by_task.items()
                       if any(r.iteration == 1 and r.failure_type == ft for r in rs)]
            if not flagged:
                recovery_rates.append(0.0)
                continue
            recovered = sum(1 for tid in flagged if any(r.success for r in by_task[tid]))
            recovery_rates.append(recovered / len(flagged))

        offsets = [xi + idx * width for xi in x]
        plt.bar(offsets, recovery_rates, width=width, label=condition)

    plt.xticks([xi + width * (len(conditions) - 1) / 2 for xi in x], failure_types)
    plt.ylabel("Recovery Rate")
    plt.title("Per-Failure-Type Recovery Rate")
    plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def fig3_strategy_selection_frequency(seal_results: list, save_path="fig3_strategy_frequency.png"):
    """
    BLOCKED - not building this until Tanisha confirms something.
    Need to know which TaskResult field her runner.py actually writes that label into before this
    can be written correctly.
    """
    raise NotImplementedError(
        "Ask Tanisha which field holds the strategy NAME (meta_reflection / "
        "iterative_prompting), not the plan text - strategy_used isn't it."
    )


def fig4_rubric_drift_curve(results_by_condition: dict, save_path="fig4_rubric_drift.png"):
    """Unique to SEAL - everything else should show flat ~0 drift by construction."""
    plt.figure(figsize=(7, 5))
    for condition, results in results_by_condition.items():
        by_iter = defaultdict(list)
        for r in results:
            drift = r.rubric_drift_score if r.rubric_drift_score is not None else 0.0
            by_iter[r.iteration].append(drift)
        iterations = sorted(by_iter.keys())
        avg_drift = [sum(by_iter[i]) / len(by_iter[i]) for i in iterations]
        plt.plot(iterations, avg_drift, marker="o", label=condition)
    plt.xlabel("Iteration")
    plt.ylabel("Avg Rubric Drift Score")
    plt.title("Rubric Drift Across Iterations")
    plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def fig5_ablation_final_success_rate(results_by_condition: dict, save_path="fig5_ablation_bar.png"):
    """SEAL vs No-Rubric-Evolution vs Reflexion vs Zero-Shot, final success rate."""
    conditions = list(results_by_condition.keys())
    final_rates = []

    for condition in conditions:
        by_task = defaultdict(list)
        for r in results_by_condition[condition]:
            by_task[r.task_id].append(r)
        finals = [max(rs, key=lambda r: r.iteration) for rs in by_task.values()]
        rate = sum(r.success for r in finals) / len(finals) if finals else 0.0
        final_rates.append(rate)

    plt.figure(figsize=(7, 5))
    plt.bar(conditions, final_rates)
    plt.ylabel("Final Success Rate")
    plt.title("Ablation: Final Success Rate by Condition")
    plt.ylim(0, 1.05)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path