from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import hashlib


@dataclass
class TaskResult:
    task_id: str
    iteration: int
    strategy_used: str          # raw plan text from Mistral/Ollama — do NOT use for Fig 3
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

    # Additive. Defaults False so old JSON files without this key load fine.
    drift_recovered: bool = False

    # NEW: strategy label written by SEALRunner — one of:
    #   "meta_reflection"      (fired when failure_type == CONTEXT_LOSS)
    #   "iterative_prompting"  (all other failure types)
    #   "none"                 (task succeeded on iteration 1, no strategy applied)
    # SEPARATE from strategy_used which holds raw plan text.
    # Shreyashree's figures.py reads THIS for Fig 3 (strategy selection frequency)
    # and Fig 4 (strategy-outcome correlation).
    strategy_label: str = "none"

    # Additive. Full rubric TEXT active for this iteration (not just hash).
    # Lets Shreyashree compute rubric_drift_score by diffing rubric content
    # across iterations 1->3 per task_id. Defaults None — backward compatible.
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