import json
import hashlib
from dataclasses import dataclass, asdict

@dataclass
class TaskResult:
    task_id: str
    iteration: int
    strategy_used: str
    failure_type: str
    score: float
    success: bool
    rubric_version: int
    rubric_hash: str
    raw_trace: list
    task_description: str
    oracle_failure_type: str
    agent_confidence: float
    plan_coherence: float
    total_steps: int
    stagnation_step_count: int
    trajectory_stagnation_rate: float
    unique_action_count: int
    action_density_index: float
    judge_score: float
    judge_failure_type: str
    judge_explanation: str
    drift_recovered: bool = False
    strategy_label: str = "none"
    rubric_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

def make_rubric_hash(rubric_text: str) -> str:
    return hashlib.md5(rubric_text.encode('utf-8')).hexdigest()[:8]