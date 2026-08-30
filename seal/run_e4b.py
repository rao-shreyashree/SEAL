import sys
sys.path.insert(0, ".")
from seal.metrics import load_seeded_results, mean_std_across_seeds, task_success_rate, success_rate_per_failure_type

# to be swapped in one by one
# reflexion_baseline_summary.json
# zeroshot_baseline_summary.json
# no_rubric_evolution_summary.json
seeded = load_seeded_results("reflexion_baseline_summary.json", seeds=list(range(5)))

# to be swapped in for the metric that we want to evaluate
# task_success_rate
# success_rate_per_failure_type
print(mean_std_across_seeds(task_success_rate, seeded))
print(mean_std_across_seeds(success_rate_per_failure_type, seeded))