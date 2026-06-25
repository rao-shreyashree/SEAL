import json
import os
import time  
from agent.agent import SEALAgent  
from seal.scenarios import MultiScenarioALFWorldEnv  
from seal.task_result import TaskResult, make_rubric_hash
from seal.judge import JudgeFixed, SEALJudge

class SEALRunner:
    def __init__(self, condition: str = "SEAL_FULL", output_dir: str = "./seallogs"):
        self.agent = SEALAgent()
        self.env = MultiScenarioALFWorldEnv()
        self.condition = condition
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Select evaluator backend cleanly per Shreya's Figure 5 specification
        if condition == "NO_RUBRIC_EVOLUTION":
            self.judge = JudgeFixed() # Gemini-backed no-op rubric evolution
        else:
            self.judge = SEALJudge()   # Standard evolution path

    def run_task_lifecycle(self, scenario_id: int) -> list:
        self.env.set_scenario(scenario_id)
        goal, _ = self.env.reset()
        active_rubric = "Approach structural components sequentially. Avoid state duplication loops."
        task_id = f"task_{str(scenario_id + 1).zfill(3)}"
        task_iteration_history = []

        print(f"Running task {task_id} under [{self.condition}] condition...")

        for iteration in range(1, 4):
            self.env.set_scenario(scenario_id)
            
            action_plan = self.agent.plan(task=goal, rubric=active_rubric)
            trace_output = self.agent.execute(plan=action_plan, env=self.env, rubric=active_rubric)

            is_success = trace_output["final_outcome"] == "SUCCESS"
            trajectory = trace_output["trajectory"]
            total_steps = len(trajectory)

            stagnant_steps = sum(1 for s in trajectory if s["internal_loop_alert"] is not None)
            unique_actions = len(set(s["action_executed"] for s in trajectory))
            stagnation_rate = round(stagnant_steps / total_steps, 2) if total_steps > 0 else 0.0
            agent_failure_type = trace_output["detected_failure_type"]
            
            # Execute evaluator which returns an EvalResult object rather than a dictionary
            evaluation_report = self.judge.evaluate(trajectory, active_rubric)

            if is_success:
                strategy_label = "none"
            elif agent_failure_type == "CONTEXT_LOSS":
                strategy_label = "meta_reflection"
            else:
                strategy_label = "iterative_prompting"

            # Safe object attribute lookup via getattr to circumvent EvalResult interface mismatch
            result = TaskResult(
                task_id=task_id,
                iteration=iteration,
                strategy_used=trace_output["macro_plan"],
                failure_type=agent_failure_type,
                score=1.0 if is_success else 0.0,
                success=is_success,
                rubric_version=iteration,
                rubric_hash=make_rubric_hash(active_rubric),
                raw_trace=trajectory,
                task_description=goal,
                oracle_failure_type=self.env.data["forced_outcome"],
                agent_confidence=trace_output["agent_intrinsic_confidence"],
                plan_coherence=trace_output["plan_coherence"],
                total_steps=total_steps,
                stagnation_step_count=stagnant_steps,
                trajectory_stagnation_rate=stagnation_rate,
                unique_action_count=unique_actions,
                action_density_index=round(unique_actions / total_steps, 2) if total_steps > 0 else 0.0,
                judge_score=getattr(evaluation_report, "score", 0.0),
                judge_failure_type=getattr(evaluation_report, "judge_failure_type", "PENDING"),
                judge_explanation=getattr(evaluation_report, "judge_explanation", ""),
                drift_recovered=trace_output.get("drift_recovered", False),
                strategy_label=strategy_label,
                rubric_text=active_rubric,
            )

            log_filename = os.path.join(self.output_dir, f"trace_{self.condition}_{task_id}_iter_{iteration}.json")
            with open(log_filename, "w") as f:
                f.write(result.to_json())

            task_iteration_history.append(result)

            if is_success:
                break

            active_rubric = self.judge.evolve_rubric(active_rubric, agent_failure_type)

        return task_iteration_history


def run_comprehensive_suite():
    total_scenarios = 50 
    runner = SEALRunner(condition="NO_RUBRIC_EVOLUTION")
    all_results = {}
    
    print("=== Launching SEAL Runner 50-Task Production Benchmark ===")
    for sid in range(total_scenarios):
        try:
            results = runner.run_task_lifecycle(scenario_id=sid)
            all_results[f"task_{str(sid+1).zfill(3)}"] = [r.to_dict() for r in results]
            time.sleep(3)
        except Exception as e:
            print(f"Skipping scenario index {sid}: {e}")
            
    with open(os.path.join(runner.output_dir, "production_runner_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nEvaluation summary successfully written to {runner.output_dir}/production_runner_summary.json")


if __name__ == "__main__":
    run_comprehensive_suite()