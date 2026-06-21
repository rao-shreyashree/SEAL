"""
Stress test for ReflexionBaseline - runs it across all 20 scenarios in
scenarios.py and dumps everything to JSONL so I can work on it before
it gets tangled up with Tanisha's runner + Anagha's judge

This is a debug/validation script, not part of the actual pipeline.
Makes real HF Inference API calls (1 plan call per iteration + 1 reflect
call per failure)
"""

import json
from agent.scenarios import MultiScenarioALFWorldEnv
from seal.reflexion_baseline import ReflexionBaseline

NUM_SCENARIOS = 20
OUTPUT_PATH = "reflexion_baseline_stress_test.jsonl"


def run_stress_test():
    baseline = ReflexionBaseline()
    env = MultiScenarioALFWorldEnv()
    all_results = []

    for scenario_id in range(NUM_SCENARIOS):
        env.set_scenario(scenario_id)
        task_id = f"task_{str(scenario_id + 1).zfill(3)}"
        oracle = env.data["forced_outcome"]

        results = baseline.run(env, task_id=task_id)
        all_results.extend(results)

        eventually_succeeded = any(r.success for r in results)
        final = results[-1]

        print(
            f"{task_id} | oracle={oracle:14s} | iters_used={len(results)} | "
            f"succeeded={eventually_succeeded} | final_failure_type={final.failure_type}"
        )

    with open(OUTPUT_PATH, "w") as f:
        for r in all_results:
            f.write(json.dumps(r.to_dict()) + "\n")

    print(f"\n{len(all_results)} TaskResults across {NUM_SCENARIOS} scenarios -> {OUTPUT_PATH}")
    return all_results


if __name__ == "__main__":
    run_stress_test()