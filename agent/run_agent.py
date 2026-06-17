"""
run_agent.py — Single-task execution runner.
Runs one scenario, exports one TaskResult JSON.
Use this for debugging / manual trace inspection.
"""

import json
import os
from agent import SEALAgent
from scenarios import MultiScenarioALFWorldEnv
from task_result import TaskResult, make_rubric_hash


def run_single_task(scenario_id: int = 0, output_path: str = "TaskResult.json") -> TaskResult:
    agent = SEALAgent()
    env = MultiScenarioALFWorldEnv(scenario_id=scenario_id)

    rubric = "Always approach structures sequentially. Verify containers are open before placement."
    rubric_version = 1
    rubric_hash = make_rubric_hash(rubric)

    goal, initial_obs = env.reset()
    print(f"Goal: {goal}\nInitial State: {initial_obs}\n" + "-" * 50)

    action_plan = agent.plan(task=goal, rubric=rubric)
    trace_output = agent.execute(plan=action_plan, env=env, rubric=rubric)

    is_success = trace_output["final_outcome"] == "SUCCESS"
    trajectory = trace_output["trajectory"]

    stagnant_steps = sum(
        1 for s in trajectory if s["internal_loop_alert"] is not None
    )
    unique_actions = len(set(s["action_executed"] for s in trajectory))
    total_steps = len(trajectory)
    stagnation_rate = round(stagnant_steps / total_steps, 2) if total_steps > 0 else 0.0

    result = TaskResult(
        task_id=f"task_{str(scenario_id + 1).zfill(3)}",
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

    with open(output_path, "w") as f:
        f.write(result.to_json())

    print(f"TaskResult exported to: {output_path}")
    return result


if __name__ == "__main__":
    run_single_task(scenario_id=0)