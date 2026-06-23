import json
import os
from seal.task_result import TaskResult, make_rubric_hash
from seal.scenarios import MultiScenarioALFWorldEnv
from seal.agent import SEALAgent


def run_single_task(scenario_id: int = 0, output_path: str = "TaskResult.json") -> TaskResult:
    agent = SEALAgent()
    env = MultiScenarioALFWorldEnv(scenario_id=scenario_id)
    rubric = "Always approach structures sequentially. Verify containers are open before placement."
    rubric_version = 1
    rubric_hash = make_rubric_hash(rubric)
    goal, initial_obs = env.reset()

    action_plan = agent.plan(task=goal, rubric=rubric)
    trace_output = agent.execute(plan=action_plan, env=env, rubric=rubric)

    is_success = trace_output["final_outcome"] == "SUCCESS"
    trajectory = trace_output["trajectory"]
    stagnant_steps = sum(1 for s in trajectory if s["internal_loop_alert"] is not None)
    unique_actions = len(set(s["action_executed"] for s in trajectory))
    total_steps = len(trajectory)
    stagnation_rate = round(stagnant_steps / total_steps, 2) if total_steps > 0 else 0.0
    agent_failure_type = trace_output["detected_failure_type"]

    if is_success:
        strategy_label = "none"
    elif agent_failure_type == "CONTEXT_LOSS":
        strategy_label = "meta_reflection"
    else:
        strategy_label = "iterative_prompting"

    result = TaskResult(
        task_id=f"task_{str(scenario_id + 1).zfill(3)}",
        iteration=1,
        strategy_used=trace_output["macro_plan"],
        failure_type=agent_failure_type,
        score=1.0 if is_success else 0.0,
        success=is_success,
        rubric_version=rubric_version,
        rubric_hash=rubric_hash,
        raw_trace=trajectory,
        task_description=goal,
        oracle_failure_type=env.data["forced_outcome"],
        agent_confidence=trace_output["agent_intrinsic_confidence"],
        plan_coherence=trace_output["plan_coherence"],
        total_steps=total_steps,
        stagnation_step_count=stagnant_steps,
        trajectory_stagnation_rate=stagnation_rate,
        unique_action_count=unique_actions,
        action_density_index=round(unique_actions / total_steps, 2) if total_steps > 0 else 0.0,
        strategy_label=strategy_label,
        rubric_text=rubric
    )

    assert not (result.success and result.oracle_failure_type not in ("SUCCESS", "NONE")), \
        f"Contradiction at task_{str(scenario_id+1).zfill(3)}: success=True but oracle={result.oracle_failure_type}"

    with open(output_path, "w") as f:
        f.write(result.to_json())
    return result


if __name__ == "__main__":
    run_single_task(scenario_id=0)