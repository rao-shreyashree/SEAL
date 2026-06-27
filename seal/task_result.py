"""
task_result.py — Shared integration contract for SEAL project.
Everyone imports TaskResult from here.

Tanisha       → produces TaskResult objects in runner.py / run_agent.py / reflexion_baseline.py
Anagha        → reads TaskResult.raw_trace + rubric; writes back judge_score/judge_failure_type/judge_explanation
Shreyashree   → receives List[TaskResult], computes all four metrics as pure functions

FIELD CONTRACT (do not change types/defaults without team agreement):
  oracle_failure_type : str  = "NONE"   ← ground truth from env; "NONE" on success, NOT "SUCCESS"
  judge_score         : Optional[float] = None  ← None until Anagha's judge runs
  judge_failure_type  : Optional[str]   = None  ← None until Anagha's judge runs
  judge_explanation   : Optional[str]   = None  ← None until Anagha's judge runs
  rubric_drift_score  : Optional[float] = None  ← optional judge output for Fig 3
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class TaskResult:
    # ── Core contract fields (required — no defaults) ─────────────────────
    task_id: str
    iteration: int
    strategy_used: str          # Mistral-generated action plan text
    failure_type: str           # Agent-detected: NONE|CONTEXT_LOSS|GOAL_DRIFT|EXECUTION_ERROR|UNKNOWN
    score: float                # 0.0 or 1.0 (binary; judge may write fractional)
    success: bool
    rubric_version: int
    rubric_hash: str            # MD5 first 8 chars of rubric string
    raw_trace: List[dict]       # List of step dicts from SEALAgent.execute()

    # ── Agent-side extras (Tanisha produces; others may read) ─────────────
    task_description: str = ""
    # IMPORTANT: "NONE" on success, not "SUCCESS" — "SUCCESS" breaks Fig 2 grouping
    oracle_failure_type: str = "NONE"
    agent_confidence: float = 0.50
    plan_coherence: float = 0.0         # 0.0–1.0; computed by compute_plan_coherence()
    total_steps: int = 0

    # ── Judge-side extras (Anagha writes after evaluate()) ───────────────
    # All Optional with None defaults so reflexion_baseline clean-run doesn't break
    judge_score: Optional[float] = None
    judge_failure_type: Optional[str] = None
    judge_explanation: Optional[str] = None
    rubric_drift_score: Optional[float] = None  # Used in Fig 3 strategy selection

    # ── Metrics (Shreyashree computes; stored here for SQLite logging) ────
    stagnation_step_count: int = 0
    trajectory_stagnation_rate: float = 0.0
    unique_action_count: int = 0
    action_density_index: float = 0.0

    # ── Strategy tracking ─────────────────────────────────────────────────
    drift_recovered: bool = False
    strategy_label: str = "none"
    rubric_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "TaskResult":
        """Load a TaskResult from a dict (e.g. from JSON file). Ignores unknown keys."""
        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_json_file(cls, path: str) -> "TaskResult":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def make_rubric_hash(rubric: str) -> str:
    return hashlib.md5(rubric.encode()).hexdigest()[:8]