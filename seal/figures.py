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
import numpy as np

from seal.metrics import (
    task_success_rate,
    recovery_rate_after_first_failure,
    convergence_speed,
)

CONDITION_COLORS = {
    "SEAL": "#2b5c8f", # blue
    "No-Rubric-Evolution": "#7b4b94", # purple
    "Reflexion": "#2a9d8f", # teal
    "Zero-Shot": "#e76f51" # orange
}

def fig1_success_rate_by_iteration(results_by_condition: dict, save_path="fig1_success_by_iteration.png"):
    """Headline result. One line per condition, success rate at each iteration number."""
    plt.figure(figsize=(7, 5))
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)
    
    for condition, results in results_by_condition.items():
        by_iter = defaultdict(list)
        for r in results:
            by_iter[r.iteration].append(r.success)
        iterations = sorted(by_iter.keys())
        rates = [sum(by_iter[i]) / len(by_iter[i]) for i in iterations]
        plt.plot(iterations, rates, marker="o", linewidth=2, label=condition, color=CONDITION_COLORS.get(condition), zorder=3)
        
    plt.xlabel("Iteration", fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel("Success Rate", fontsize=12, fontweight='bold', labelpad=10)
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11)
    plt.title("Success Rate by Iteration Across Conditions", fontsize=13, fontweight='bold', pad=15)
    plt.ylim(0, 1.10)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def fig2_recovery_rate_by_failure_type(results_by_condition: dict, save_path="fig2_recovery_by_failure_type.png",
                                        use_oracle: bool = True):
    """Of tasks that failed with type X on iteration 1, what fraction eventually succeeded."""
    failure_types = ["CONTEXT_LOSS", "GOAL_DRIFT", "EXECUTION_ERROR", "UNKNOWN"]
    conditions = list(results_by_condition.keys())
    width = 0.8 / max(len(conditions), 1)
    x = range(len(failure_types))

    plt.figure(figsize=(9, 5))
    plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    bars_collection = []
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
        bars = plt.bar(offsets, recovery_rates, width=width, label=condition, color=CONDITION_COLORS.get(condition), edgecolor='black', linewidth=0.5, zorder=3)
        bars_collection.append((bars, recovery_rates))

    # Annotate bar values
    for bars, rates in bars_collection:
        for bar, rate in zip(bars, rates):
            if rate > 0:
                plt.annotate(
                    f"{rate:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, rate),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color='#222222'
                )

    plt.xticks([xi + width * (len(conditions) - 1) / 2 for xi in x], failure_types, fontsize=10, fontweight='bold')
    plt.yticks(fontsize=11)
    plt.ylabel("Recovery Rate", fontsize=12, fontweight='bold', labelpad=10)
    label_source = "Oracle" if use_oracle else "Agent-Detected"
    plt.title(f"Per-Failure-Type Recovery Rate ({label_source} Labels)", fontsize=13, fontweight='bold', pad=15)
    plt.ylim(0, 1.10)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def fig3_strategy_selection_frequency(results_by_condition: dict, save_path="fig3_strategy_frequency.png",
                                   condition: str = "SEAL"):
    """Strategy selection frequency for one condition (default: SEAL)."""
    if condition not in results_by_condition:
        raise ValueError(f"Condition '{condition}' not found in results_by_condition.")

    labels = [r.strategy_label for r in results_by_condition[condition] if r.strategy_label != "none"]
    if not labels:
        raise ValueError(f"No non-'none' strategy_label values found for condition '{condition}'.")

    counts = Counter(labels)
    strategies = list(counts.keys())
    frequencies = [counts[s] for s in strategies]

    plt.figure(figsize=(6, 6))
    plt.pie(frequencies, labels=strategies, autopct="%1.1f%%", startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
    plt.title(f"Strategy Selection Frequency ({condition})", fontsize=13, fontweight='bold', pad=15)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def fig4_rubric_drift_curve(results_by_condition: dict, save_path="fig4_rubric_drift.png"):
    """Unique to SEAL - rubric drift curve."""
    plt.figure(figsize=(7, 5))
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)

    for condition, results in results_by_condition.items():
        by_iter = defaultdict(list)
        for r in results:
            drift = r.rubric_drift_score if r.rubric_drift_score is not None else 0.0
            by_iter[r.iteration].append(drift)
        iterations = sorted(by_iter.keys())
        avg_drift = [sum(by_iter[i]) / len(by_iter[i]) for i in iterations]
        plt.plot(iterations, avg_drift, marker="o", linewidth=2, label=condition, color=CONDITION_COLORS.get(condition), zorder=3)

    plt.xlabel("Iteration", fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel("Avg Rubric Drift Score", fontsize=12, fontweight='bold', labelpad=10)
    plt.xticks(fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11)
    plt.title("Rubric Drift Across Iterations", fontsize=13, fontweight='bold', pad=15)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
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

    bar_colors = [CONDITION_COLORS.get(cond, "#333333") for cond in conditions]
    
    plt.figure(figsize=(8, 5))
    plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
    
    bars = plt.bar(conditions, final_rates, color=bar_colors, edgecolor='black', linewidth=0.6, zorder=3)
    
    # Add exact numeric values on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=10, fontweight='bold', color='#222222'
        )

    plt.xticks(fontsize=10, fontweight='bold')
    plt.yticks(fontsize=11)
    plt.ylabel("Final Success Rate", fontsize=12, fontweight='bold', labelpad=10)
    plt.title("Ablation: Final Success Rate by Condition", fontsize=13, fontweight='bold', pad=15)
    plt.ylim(0, 1.10)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def fig6_max_iterations_ablation(results_by_condition: dict, save_path="fig6_max_iterations_ablation.png"):
    """Final success rate if max_iterations had been capped at 1, 2, or 3."""
    caps = [1, 2, 3]
    conditions = [c for c in results_by_condition if c != "Zero-Shot"]
    width = 0.8 / max(len(conditions), 1)
    x = range(len(caps))

    plt.figure(figsize=(8, 5))
    plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    bars_collection = []
    for idx, condition in enumerate(conditions):
        by_task = defaultdict(list)
        for r in results_by_condition[condition]:
            by_task[r.task_id].append(r)

        rates = []
        for cap in caps:
            successes = 0
            for task_id, rs in by_task.items():
                capped = [r for r in rs if r.iteration <= cap]
                if any(r.success for r in capped):
                    successes += 1
            rates.append(successes / len(by_task) if by_task else 0.0)

        offsets = [xi + idx * width for xi in x]
        bars = plt.bar(offsets, rates, width=width, label=condition, color=CONDITION_COLORS.get(condition), edgecolor='black', linewidth=0.5, zorder=3)
        bars_collection.append((bars, rates))

    # Annotate numeric values for clustered bars
    for bars, rates in bars_collection:
        for bar, rate in zip(bars, rates):
            plt.annotate(
                f"{rate:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, rate),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='#222222'
            )

    plt.xticks([xi + width * (len(conditions) - 1) / 2 for xi in x], [f"iter<={c}" for c in caps], fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11)
    plt.ylabel("Success Rate", fontsize=12, fontweight='bold', labelpad=10)
    plt.title("Max Iterations Ablation: Success Rate by Iteration Cap", fontsize=13, fontweight='bold', pad=15)
    plt.ylim(0, 1.10)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path

def fig7_multi_metric_comparison(
    results_by_condition: dict,
    save_path="fig7_multi_metric_comparison.png"
):
    """Multi-metric comparison across evaluation conditions.

    Reports:
    - Task Success Rate
    - Recovery After 1st Failure
    - Average Iterations to Success, normalized by 3
    """

    metrics_data = {}

    for condition, results in results_by_condition.items():
        success_rate = task_success_rate(results)
        recovery_rate = recovery_rate_after_first_failure(results)

        conv = convergence_speed(results)
        avg_iters = conv["avg_iterations_to_success"] or 0.0
        avg_iters_norm = avg_iters / 3.0

        metrics_data[condition] = {
            "Task Success Rate": success_rate,
            "Recovery After 1st Failure": recovery_rate,
            "Avg Iters to Success (÷3)": avg_iters_norm,
        }

    labels = list(results_by_condition.keys())
    metric_names = [
        "Task Success Rate",
        "Recovery After 1st Failure",
        "Avg Iters to Success (÷3)",
    ]

    metric_colors = ["#2b5c8f", "#e76f51", "#2a9d8f"]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

    bars_list = []

    for i, metric in enumerate(metric_names):
        values = [metrics_data[label][metric] for label in labels]

        bars = plt.bar(
            x + i * width,
            values,
            width,
            label=metric,
            color=metric_colors[i],
            edgecolor='black',
            linewidth=0.6,
            zorder=3
        )
        bars_list.append(bars)

    for bars in bars_list:
        for bar in bars:
            height = bar.get_height()

            plt.annotate(
                f"{height:.2f}",
                xy=(
                    bar.get_x() + bar.get_width() / 2,
                    height
                ),
                xytext=(0, 4),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='bold',
                color='#222222'
            )

    plt.xticks(x + width, labels, fontsize=11, fontweight='bold')
    plt.yticks(fontsize=11)
    plt.xlabel("Evaluation Condition", fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel("Score (0-1, normalized)", fontsize=12, fontweight='bold', labelpad=10)
    plt.title("Multi-Metric Comparison Across Conditions", fontsize=13, fontweight='bold', pad=15)
    plt.ylim(0, 1.10)
    plt.legend(fontsize=10, framealpha=0.9)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return save_path

def generate_all_figures(results_by_condition: dict) -> dict:
    """Generates all figures. Returns dict of fig_name -> save_path."""
    return {
        "fig1": fig1_success_rate_by_iteration(results_by_condition),
        "fig2": fig2_recovery_rate_by_failure_type(results_by_condition, use_oracle=True),
        "fig3": fig3_strategy_selection_frequency(results_by_condition, condition="SEAL"),
        "fig4": fig4_rubric_drift_curve(results_by_condition),
        "fig5": fig5_ablation_final_success_rate(results_by_condition),
        "fig6": fig6_max_iterations_ablation(results_by_condition),
        "fig7": fig7_multi_metric_comparison(results_by_condition),
    }