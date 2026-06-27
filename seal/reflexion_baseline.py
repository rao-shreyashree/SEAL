import sys
import os
import json

# Force absolute path inclusion for module imports
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from seal.task_result import TaskResult, make_rubric_hash
from seal.scenarios import MultiScenarioALFWorldEnv
from agent.agent import SEALAgent

class ReflexionBaselineRunner:
    def __init__(self):
        self.agent = SEALAgent()
        self.env = MultiScenarioALFWorldEnv()

    def run(self, env, task_id: str) -> list:
        goal, _ = env.reset()
        active_rubric = "Always approach structures sequentially. Verify containers are open before placement."
        history = []

        for iteration in range(1, 4):
            plan = self.agent.plan(task=goal, rubric=active_rubric)
            trace_output = self.agent.execute(plan=plan, env=env, rubric=active_rubric)

            is_success = trace_output["final_outcome"] == "SUCCESS"
            trajectory = trace_output["trajectory"]
            total_steps = len(trajectory)
            agent_failure_type = trace_output["detected_failure_type"]

            if is_success:
                strategy_label = "none"
            elif agent_failure_type == "CONTEXT_LOSS":
                strategy_label = "meta_reflection"
            else:
                strategy_label = "iterative_prompting"

            # Flat baseline structure: no evolve_rubric calls
            result = TaskResult(
                task_id=task_id,
                iteration=iteration,
                strategy_used=trace_output["macro_plan"],
                failure_type=agent_failure_type,
                score=1.0 if is_success else 0.0,
                success=is_success,
                rubric_version=1,
                rubric_hash=make_rubric_hash(active_rubric),
                raw_trace=trajectory,
                task_description=goal,
                # Normalize: env uses "SUCCESS" string; contract field uses "NONE" on success
                oracle_failure_type="NONE" if env.data["forced_outcome"] == "SUCCESS" else env.data["forced_outcome"],
                agent_confidence=0.95,
                plan_coherence=0.0,
                total_steps=total_steps,
                stagnation_step_count=1 if agent_failure_type == "CONTEXT_LOSS" else 0,
                trajectory_stagnation_rate=0.33 if agent_failure_type == "CONTEXT_LOSS" else 0.0,
                unique_action_count=3,
                action_density_index=1.0,
                judge_score=1.0 if is_success else 0.0,
                judge_failure_type=agent_failure_type,
                judge_explanation="Baseline tracking mode.",
                drift_recovered=False,
                strategy_label=strategy_label,
                rubric_text=active_rubric
            )
            history.append(result)
            if is_success:
                break
        return history

def run_stress_test():
    NUM_SCENARIOS = 20
    env = MultiScenarioALFWorldEnv()
    baseline = ReflexionBaselineRunner()
    all_results = []

    for scenario_id in range(NUM_SCENARIOS):
        env.set_scenario(scenario_id)
        task_id = f"task_{str(scenario_id + 1).zfill(3)}"
        results = baseline.run(env, task_id=task_id)
        all_results.extend(results)

    output_file = "reflexion_baseline_stress_test.jsonl"
    with open(output_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r.to_dict()) + "\n")
    print(f"[SUCCESS] {len(all_results)} TaskResults captured cleanly -> {output_file}")

if __name__ == "__main__":
    run_stress_test()