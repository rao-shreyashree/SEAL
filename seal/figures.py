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

# critical section
# do not change the input contract (dict[str, list[TaskResult]]) without telling
# Tanisha and Anagha - both runner.py and the notebook driver build this dict
# and rely on it staying this shape.
"""

import matplotlib.pyplot as plt
from collections import defaultdict, Counter


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


def fig2_recovery_rate_by_failure_type(results_by_condition: dict, save_path="fig2_recovery_by_failure_type.png",
                                        use_oracle: bool = True):
    """Of tasks that failed with type X on iteration 1, what fraction eventually succeeded.

    use_oracle=True (default): groups by oracle_failure_type (ground truth from scenarios.py).
        Safe default - agent.py's own _detect_failure_type() can disagree with the oracle,
        and we don't want that disagreement silently baked into a published figure.
    use_oracle=False: groups by the agent's own detected failure_type instead. Useful as a
        diagnostic to compare against the oracle version - the gap between the two is itself
        evidence of how well the intrinsic diagnostic tracks ground truth.
    """
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
            flagged = []
            for tid, rs in by_task.items():
                iter1 = next((r for r in rs if r.iteration == 1), None)
                if iter1 is None:
                    continue
                label = iter1.oracle_failure_type if use_oracle else iter1.failure_type
                if label == ft:
                    flagged.append(tid)
            if not flagged:
                recovery_rates.append(0.0)
                continue
            recovered = sum(1 for tid in flagged if any(r.success for r in by_task[tid]))
            recovery_rates.append(recovered / len(flagged))

        offsets = [xi + idx * width for xi in x]
        plt.bar(offsets, recovery_rates, width=width, label=condition)

    plt.xticks([xi + width * (len(conditions) - 1) / 2 for xi in x], failure_types)
    plt.ylabel("Recovery Rate")
    label_source = "Oracle" if use_oracle else "Agent-Detected"
    plt.title(f"Per-Failure-Type Recovery Rate ({label_source} Labels)")
    plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def fig3_strategy_selection_frequency(results_by_condition: dict, save_path="fig3_strategy_frequency.png",
                                       condition: str = "SEAL"):
    """Strategy selection frequency for one condition (default: SEAL).

    Reads strategy_label - confirmed real field on TaskResult as of 2026-06-25
    (strategy_used is plan TEXT, not a label - do not use that field here).
    Values seen in practice: "none", "meta_reflection", "iterative_prompting".
    "none" rows (successful tasks, no strategy needed) are excluded from the
    chart since they're not a recovery strategy choice.
    """
    if condition not in results_by_condition:
        raise ValueError(f"Condition '{condition}' not found in results_by_condition. "
                          f"Available: {list(results_by_condition.keys())}")

    labels = [r.strategy_label for r in results_by_condition[condition] if r.strategy_label != "none"]
    if not labels:
        raise ValueError(f"No non-'none' strategy_label values found for condition '{condition}'. "
                          f"Check that the run actually produced failures requiring a strategy.")

    counts = Counter(labels)
    strategies = list(counts.keys())
    frequencies = [counts[s] for s in strategies]

    plt.figure(figsize=(6, 6))
    plt.pie(frequencies, labels=strategies, autopct="%1.1f%%", startangle=90)
    plt.title(f"Strategy Selection Frequency ({condition})")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def fig4_rubric_drift_curve(results_by_condition: dict, save_path="fig4_rubric_drift.png"):
    """Unique to SEAL - everything else should show flat ~0 drift by construction.

    Note: iteration=1 drift is 0.0 by definition for every condition (no evolution
    has happened yet on the first attempt) - this is expected, not a bug. If SEAL's
    curve doesn't separate from the flat baselines by iteration 2-3, that's a real
    finding worth flagging, not a plotting error.
    """
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
    """SEAL vs No-Rubric-Evolution vs Reflexion vs Zero-Shot, final success rate.

    # do not change without discussing with the team
    # Assumes "highest iteration attempted for a task" == "final outcome for that task".
    # Only valid because every runner/baseline in the current codebase breaks the
    # iteration loop immediately on success. If that break behavior changes, this
    # max(iteration) logic silently produces wrong final rates.
    """
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


def generate_all_figures(results_by_condition: dict) -> dict:
    """Generates all 5 figures. Returns dict of fig_name -> save_path."""
    return {
        "fig1": fig1_success_rate_by_iteration(results_by_condition),
        "fig2": fig2_recovery_rate_by_failure_type(results_by_condition, use_oracle=True),
        "fig3": fig3_strategy_selection_frequency(results_by_condition, condition="SEAL"),
        "fig4": fig4_rubric_drift_curve(results_by_condition),
        "fig5": fig5_ablation_final_success_rate(results_by_condition),
    }