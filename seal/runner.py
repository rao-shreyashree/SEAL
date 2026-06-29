import json
import os
import time
import sys

# Force absolute path inclusion for sub-module relative namespace mapping
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from agent.agent import SEALAgent  
from seal.scenarios import MultiScenarioALFWorldEnv  
from seal.task_result import TaskResult, make_rubric_hash
from seal.judge import JudgeFixed, SEALJudge, DEFAULT_RUBRIC, trace_to_str

class SEALRunner:
    def __init__(self, condition: str = "SEAL_FULL", output_dir: str = "./seallogs"):
        self.agent = SEALAgent()
        self.env = MultiScenarioALFWorldEnv()
        self.condition = condition
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        if condition == "NO_RUBRIC_EVOLUTION":
            self.judge = JudgeFixed() 
        else:
            self.judge = SEALJudge()   

    def run_task_lifecycle(self, scenario_id: int) -> tuple[list, int]:
        self.env.set_scenario(scenario_id)
        goal, _ = self.env.reset()
        
        active_rubric = DEFAULT_RUBRIC.copy()
        task_id = f"task_{str(scenario_id + 1).zfill(3)}"
        task_iteration_history = []
        failure_history_buffer = []
        calls_made = 0

        print(f"Running task {task_id} under [{self.condition}] condition...")

        for iteration in range(1, 4):
            self.env.set_scenario(scenario_id)
            
            rubric_string_representation = json.dumps(active_rubric, indent=2)
            action_plan = self.agent.plan(task=goal, rubric=rubric_string_representation)
            trace_output = self.agent.execute(plan=action_plan, env=self.env, rubric=rubric_string_representation)

            is_success = trace_output["final_outcome"] == "SUCCESS"
            trajectory = trace_output["trajectory"]
            total_steps = len(trajectory)

            stagnant_steps = sum(1 for s in trajectory if s["internal_loop_alert"] is not None)
            unique_actions = len(set(s["action_executed"] for s in trajectory))
            stagnation_rate = round(stagnant_steps / total_steps, 2) if total_steps > 0 else 0.0
            agent_failure_type = trace_output["detected_failure_type"]
            
            formatted_trace_str = trace_to_str(trajectory)
            evaluation_report = self.judge.evaluate(trace=formatted_trace_str, rubric=active_rubric)
            calls_made += 1 # Tracking active evaluate calls

            if is_success:
                strategy_label = "none"
            elif agent_failure_type == "CONTEXT_LOSS":
                strategy_label = "meta_reflection"
            else:
                strategy_label = "iterative_prompting"

            raw_failure_enum = getattr(evaluation_report, "failure_type", None)
            extracted_failure_str = raw_failure_enum.value if raw_failure_enum else "none"

            result = TaskResult(
                task_id=task_id,
                iteration=iteration,
                strategy_used=trace_output["macro_plan"],
                failure_type=agent_failure_type,
                score=1.0 if is_success else 0.0,
                success=is_success,
                rubric_version=iteration,
                rubric_hash=make_rubric_hash(rubric_string_representation),
                raw_trace=trajectory,
                task_description=goal,
                # Normalize: env uses "SUCCESS" string; contract field uses "NONE" on success
                # "SUCCESS" as oracle_failure_type breaks Fig 2 grouping for successful tasks
                oracle_failure_type="NONE" if self.env.data["forced_outcome"] == "SUCCESS" else self.env.data["forced_outcome"],
                agent_confidence=trace_output["agent_intrinsic_confidence"],
                plan_coherence=trace_output["plan_coherence"],
                total_steps=total_steps,
                stagnation_step_count=stagnant_steps,
                trajectory_stagnation_rate=stagnation_rate,
                unique_action_count=unique_actions,
                action_density_index=round(unique_actions / total_steps, 2) if total_steps > 0 else 0.0,
                judge_score=getattr(evaluation_report, "score", 0.0),
                judge_failure_type=extracted_failure_str,
                judge_explanation=getattr(evaluation_report, "explanation", ""),
                drift_recovered=trace_output.get("drift_recovered", False),
                strategy_label=strategy_label,
                rubric_text=rubric_string_representation,
            )

            log_filename = os.path.join(self.output_dir, f"trace_{self.condition}_{task_id}_iter_{iteration}.json")
            with open(log_filename, "w") as f:
                f.write(result.to_json())

            task_iteration_history.append(result)
            failure_history_buffer.append(evaluation_report)

            if is_success:
                break

            # evolve only every 2nd failure (skip iteration 1's evolve call)
            # cuts evolve calls 2->1 per failing task
            # evaluate() is NEVER skipped
            # NOTE: this changes what iteration-2 TaskResult.rubric_text reflects 
            # (no evolve happened before iter 2 anymore)
            if not is_success and iteration < 3 and iteration >= 2:
                try:
                    new_rubric, similarity_score, was_updated = self.judge.evolve_rubric(
                        rubric=active_rubric, 
                        failure_history=failure_history_buffer
                    )
                    calls_made += 1 # Tracking active evolve calls
                    if was_updated and isinstance(new_rubric, dict):
                        active_rubric = new_rubric
                except ImportError as ie:
                    print(f"[{task_id} Iteration {iteration} Rubric Drift Bypass]: {ie}")
                    break

        return task_iteration_history, calls_made

def run_comprehensive_suite(max_calls: int = 200):
    # 50-task full run, no scope reduction
    # max_calls is a quota guard, not a scope cut 
    # stops the run before
    # burning through Gemini key rotation budget
    
    total_scenarios = 50  # 5 will produce a partial run. our benchmark is 50 tasks
    runner = SEALRunner(condition="SEAL_FULL")
    all_results = {}
    per_task_calls = {}  # visibility into calls/task, not just one final total

    request_count = 0
    print("=== Launching SEAL Runner Production Benchmark ===")
    for sid in range(total_scenarios):
        if request_count >= max_calls:
            print(f"[STOPPED] Hit max_calls budget ({max_calls}) at scenario index {sid}. "
                  f"Remaining scenarios not run - rotate keys or raise max_calls to continue.")
            break
        try:
            results, calls = runner.run_task_lifecycle(scenario_id=sid)
            request_count += calls
            task_key = f"task_{str(sid+1).zfill(3)}"
            all_results[task_key] = [r.to_dict() for r in results]
            per_task_calls[task_key] = calls
            time.sleep(3)
        except Exception as e:
            print(f"Skipping scenario index {sid}: {e}")

    with open(os.path.join(runner.output_dir, "production_runner_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(runner.output_dir, "production_call_counts.json"), "w") as f:
        json.dump(per_task_calls, f, indent=2)
    print(f"\nEvaluation summary written cleanly to {runner.output_dir}/production_runner_summary.json")
    print(f"Per-task call counts written to {runner.output_dir}/production_call_counts.json")
    print(f"\nTotal judge calls: {request_count}")

if __name__ == "__main__":
    run_comprehensive_suite()