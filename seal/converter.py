import json
from seal.dataclass import TaskResult, FailureType
from seal.database import insert_task_result

def trace_to_task_result(trace: dict, task_id: str, iteration: int = 1, rubric_version: int = 1) -> TaskResult:
    success = trace["final_outcome"] == "SUCCESS"
    
    failure_type = FailureType.NONE
    if not success:
        alerts = [s["internal_loop_alert"] for s in trace["trajectory"] if s["internal_loop_alert"] != "None"]
        if alerts:
            failure_type = FailureType.CONTEXT_LOSS  # looping = context loss
        else:
            failure_type = FailureType.EXECUTION_ERROR

    return TaskResult(
        task_id=task_id,
        task_description=trace["task_goal"],
        success=success,
        failure_type=failure_type,
        score=1.0 if success else 0.0,
        explanation=f"Completed in {trace['total_steps']} steps. Outcome: {trace['final_outcome']}",
        iteration=iteration,
        strategy_used=None,
        rubric_version=rubric_version
    )

def load_and_store_trace(trace_path: str, task_id: str):
    with open(trace_path) as f:
        trace = json.load(f)
    result = trace_to_task_result(trace, task_id)
    insert_task_result(result)
    return result