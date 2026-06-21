"""
run_and_check.py — 20-scenario baseline generation suite.
Exports one TaskResult JSON per scenario + a combined JSONL for batch loading.
"""
import json
import os
from agent import SEALAgent
from scenarios import MultiScenarioALFWorldEnv
from task_result import TaskResult, make_rubric_hash

NUM_SCENARIOS = 20


def run_baseline_suite(output_dir: str = ".") -> list:
    print("=== Launching Automated Multi-Trace Baseline Generation Suite ===")
    os.makedirs(output_dir, exist_ok=True)

    agent = SEALAgent()
    env = MultiScenarioALFWorldEnv()
    rubric = "Approach structural components sequentially. Avoid state duplication loops."
    rubric_version = 1
    rubric_hash = make_rubric_hash(rubric)

    all_results = []

    for idx in range(NUM_SCENARIOS):
        env.set_scenario(idx)
        goal, _ = env.reset()
        print(f"Generating Trace {idx + 1}/{NUM_SCENARIOS} -> Goal: {goal}")

        action_plan = agent.plan(task=goal, rubric=rubric)
        trace_output = agent.execute(plan=action_plan, env=env, rubric=rubric)

        is_success = trace_output["final_outcome"] == "SUCCESS"
        trajectory = trace_output["trajectory"]
        total_steps = len(trajectory)

        stagnant_steps = sum(
            1 for s in trajectory if s["internal_loop_alert"] is not None
        )
        unique_actions = len(set(s["action_executed"] for s in trajectory))
        stagnation_rate = round(stagnant_steps / total_steps, 2) if total_steps > 0 else 0.0

        result = TaskResult(
            task_id=f"task_{str(idx + 1).zfill(3)}",
            iteration=1,
            strategy_used=trace_output["macro_plan"],
            failure_type=trace_output["detected_failure_type"],
            score=1.0 if is_success else 0.0,
            success=is_success,
            rubric_version=rubric_version,
            rubric_hash=rubric_hash,
            raw_trace=trajectory,
            task_description=goal,
            oracle_failure_type="NONE" if is_success else env.data["forced_outcome"],
            agent_confidence=trace_output["agent_intrinsic_confidence"],
            plan_coherence=trace_output["plan_coherence"],
            total_steps=total_steps,
            stagnation_step_count=stagnant_steps,
            trajectory_stagnation_rate=stagnation_rate,
            unique_action_count=unique_actions,
            action_density_index=round(unique_actions / total_steps, 2) if total_steps > 0 else 0.0,
        )

        filename = os.path.join(output_dir, f"sample_execution_trace_{idx + 1}.json")
        with open(filename, "w") as f:
            f.write(result.to_json())

        all_results.append(result)

    combined_path = os.path.join(output_dir, "baseline_all_traces.jsonl")
    with open(combined_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(r.to_dict()) + "\n")

    print(f"\n=== Done. {NUM_SCENARIOS} traces exported. Combined file: {combined_path} ===")
    return all_results


if __name__ == "__main__":
    run_baseline_suite()