from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import hashlib


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
    raw_trace: List[dict]       

    task_description: str = ""
    oracle_failure_type: str = "NONE"   
    agent_confidence: float = 0.50
    plan_coherence: float = 0.0         
    total_steps: int = 0

    judge_score: Optional[float] = None
    judge_failure_type: Optional[str] = None
    judge_explanation: Optional[str] = None
    rubric_drift_score: Optional[float] = None

    stagnation_step_count: int = 0
    trajectory_stagnation_rate: float = 0.0
    unique_action_count: int = 0
    action_density_index: float = 0.0

    # NEW: additive field. Defaults to False so existing JSON files / callers
    # that don't pass this kwarg are unaffected (from_dict() filters unknown
    # keys, so old files without this field load fine too).
    drift_recovered: bool = False

    # NEW: additive field, optional. The full rubric TEXT active for this
    # specific iteration (not just its hash). Lets Shreyashree compute
    # rubric_drift_score by diffing actual rubric content across iterations
    # 1->3 for a task_id, instead of reconstructing it indirectly from
    # rubric_hash alone. Defaults to None - fully backward compatible with
    # existing trace files that don't have this key.
    rubric_text: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "TaskResult":
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json_file(cls, path: str) -> "TaskResult":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def make_rubric_hash(rubric: str) -> str:
    return hashlib.md5(rubric.encode()).hexdigest()[:8]