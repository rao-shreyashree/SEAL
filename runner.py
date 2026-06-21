import json
import os
from agent import SEALAgent
from scenarios import MultiScenarioALFWorldEnv
from task_result import TaskResult, make_rubric_hash


class MockSEALJudge:
    """
    Placeholder for Anagha's real SEALJudge. Same method signatures as the
    real judge is expected to expose, so swapping this out is a one-line
    change in SEALRunner.__init__ once her class is ready:
        self.judge = SEALJudge()   instead of   self.judge = MockSEALJudge()
    """
    def evaluate(self, raw_trace: list, rubric: str) -> dict:
        return {
            "score": 0.0,
            "judge_failure_type": "PENDING_VERIFICATION",
            "judge_explanation": "Evaluated by baseline mock evaluator."
        }

    def evolve_rubric(self, current_rubric: str, failure_type: str) -> str:
        if failure_type == "CONTEXT_LOSS":
            return current_rubric + " [META-REFLECTION: Stop using 'look' consecutively. Force navigation shift.]"
        else:
            return current_rubric + " [ITERATIVE-PROMPTING: Double check target items before execution.]"


class SEALRunner:
    def __init__(self, output_dir: str = "./seallogs"):
        self.agent = SEALAgent()
        self.env = MultiScenarioALFWorldEnv()
        self.judge = MockSEALJudge()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run_task_lifecycle(self, scenario_id: int) -> list:
        self.env.set_scenario(scenario_id)
        goal, _ = self.env.reset()
        active_rubric = "Approach structural components sequentially. Avoid state duplication loops."
        task_id = f"task_{str(scenario_id + 1).zfill(3)}"
        task_iteration_history = []

        print(f"\nStarting Outer Loop Lifecycle for {task_id} -- Goal: {goal}")

        for iteration in range(1, 4):
            print(f"Iteration {iteration}/3 (Active Rubric Hash: {make_rubric_hash(active_rubric)})")

            self.env.set_scenario(scenario_id)
            goal, _ = self.env.reset()

            action_plan = self.agent.plan(task=goal, rubric=active_rubric)
            trace_output = self.agent.execute(plan=action_plan, env=self.env, rubric=active_rubric)

            is_success = trace_output["final_outcome"] == "SUCCESS"
            trajectory = trace_output["trajectory"]
            total_steps = len(trajectory)

            stagnant_steps = sum(
                1 for s in trajectory if s["internal_loop_alert"] is not None
            )
            unique_actions = len(set(s["action_executed"] for s in trajectory))
            stagnation_rate = round(stagnant_steps / total_steps, 2) if total_steps > 0 else 0.0

            agent_failure_type = trace_output["detected_failure_type"]
            evaluation_report = self.judge.evaluate(trajectory, active_rubric)

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
                oracle_failure_type="NONE" if is_success else self.env.data["forced_outcome"],
                agent_confidence=trace_output["agent_intrinsic_confidence"],
                plan_coherence=trace_output["plan_coherence"],
                total_steps=total_steps,
                stagnation_step_count=stagnant_steps,
                trajectory_stagnation_rate=stagnation_rate,
                unique_action_count=unique_actions,
                action_density_index=round(unique_actions / total_steps, 2) if total_steps > 0 else 0.0,
                judge_score=evaluation_report["score"],
                judge_failure_type=evaluation_report["judge_failure_type"],
                judge_explanation=evaluation_report["judge_explanation"],
                drift_recovered=trace_output.get("drift_recovered", False),
                rubric_text=active_rubric,
            )

            log_filename = os.path.join(self.output_dir, f"trace_{task_id}_iter_{iteration}.json")
            with open(log_filename, "w") as f:
                f.write(result.to_json())

            task_iteration_history.append(result)

            if is_success:
                print(f"Success achieved in iteration {iteration}")
                break

            strategy_type = "meta_reflection" if agent_failure_type == "CONTEXT_LOSS" else "iterative_prompting"
            print(f"Task Failed via [{agent_failure_type}]. Updating rubric using strategy: {strategy_type}")
            active_rubric = self.judge.evolve_rubric(active_rubric, agent_failure_type)

        return task_iteration_history


def run_all_failure_scenarios():
    unique_ids = [4, 9, 14, 19]
    runner = SEALRunner()
    all_results = {}

    print("=== SEAL Runner -- Week 2 Forced-Failure Lifecycle Test ===")
    for sid in unique_ids:
        results = runner.run_task_lifecycle(scenario_id=sid)
        all_results[f"task_{str(sid+1).zfill(3)}"] = [r.to_dict() for r in results]
        print(f"Logged iterations for scenario {sid+1}")

    summary_path = os.path.join(runner.output_dir, "runner_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    run_all_failure_scenarios()