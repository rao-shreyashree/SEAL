from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime

class FailureType(Enum):
    HALLUCINATION = "hallucination"
    CONTEXT_LOSS = "context_loss"
    GOAL_DRIFT = "goal_drift"
    EXECUTION_ERROR = "execution_error"
    NONE = "none"

@dataclass
class TaskResult:
    task_id: str
    task_description: str
    success: bool
    failure_type: FailureType
    score: float # 0.0 to 1.0
    explanation: str # judge's explanation
    iteration: int 
    strategy_used: Optional[str] # recovery strategy applied
    rubric_version: int # version of rubric used
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())