"""
seal/task_result.py - Shared integration contract for SEAL project.
everyone import TaskResult from here.


Tanisha  → produces TaskResult objects in run_agent.py / run_and_check.py
Anagha   → reads TaskResult.raw_trace + rubric; writes back score/failure_type
Shreyashree → receives List[TaskResult], computes all four metrics as pure functions
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import hashlib


@dataclass
class TaskResult:
    # ── Core contract fields (all required) ──────────────────────────────────
    task_id: str
    iteration: int
    strategy_used: str          # Mistral-generated action plan text - raw plan, NOT a strategy label (use strategy_label below for Fig 3/4)
    failure_type: str           # Agent-detected: NONE|CONTEXT_LOSS|GOAL_DRIFT|EXECUTION_ERROR|UNKNOWN
    score: float                # 0.0 or 1.0 (binary for now; judge can write fractional here)
    success: bool
    rubric_version: int
    rubric_hash: str            # MD5 first 8 chars of rubric string
    raw_trace: List[dict]       # List of step dicts from SEALAgent.execute()

    # ── Agent-side extras (Tanisha produces; others may read) ─────────────
    task_description: str = ""
    oracle_failure_type: str = "NONE"   # Ground truth from env.data["forced_outcome"]
    agent_confidence: float = 0.50
    plan_coherence: float = 0.0         # 0.0–1.0; computed by compute_plan_coherence()
    total_steps: int = 0

    # ── Judge-side extras (Anagha writes after evaluate()) ───────────────
    judge_score: Optional[float] = None
    judge_failure_type: Optional[str] = None
    judge_explanation: Optional[str] = None
    rubric_drift_score: Optional[float] = None

    # ── Metrics (Shreyashree computes; stored here for SQLite logging) ────
    stagnation_step_count: int = 0
    trajectory_stagnation_rate: float = 0.0
    unique_action_count: int = 0
    action_density_index: float = 0.0

    # New additive fields (taken from Tanisha's file)
    # All default-valued so old JSON files / existing call sites without
    # these keys still construct fine via from_dict().

    # Behavioral drift-recovery flag. Confirm with Tanisha exactly how this
    # is computed in whichever runner/baseline sets it - must reflect real
    # agent behavior, not just oracle/iteration bookkeeping.
    drift_recovered: bool = False

    # one of "meta_reflection" | "iterative_prompting" | "none"
    # Written by SEALRunner / ReflexionEvolving. SEPARATE from strategy_used
    # (raw plan text). Shreyashree's figures.py reads THIS for Fig 3/Fig 4.
    strategy_label: str = "none"

    # Full rubric TEXT active for this iteration (not just hash). Lets
    # Shreyashree compute rubric_drift_score by diffing across iter 1->3
    # for conditions that don't go through SEALJudge.evolve_rubric().
    rubric_text: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "TaskResult":
        """Load a TaskResult from a dict (e.g. from JSON file)."""
        # Only pass fields that exist in the dataclass to avoid TypeError
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json_file(cls, path: str) -> "TaskResult":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def make_rubric_hash(rubric: str) -> str:
    return hashlib.md5(rubric.encode()).hexdigest()[:8]