"""
seal/judge.py - SEALJudge: LLM-based evaluator with rubric self-evolution.

Extracted from notebook/SEALjudge.ipynb (Anagha, origin/judge @ 28273a3)
Ported, not copy-pasted: model name parameterized, mock-trace generation
and Colab scaffolding stripped, the two near-duplicate retry loops merged
into one helper.

Integration contract: see task_result.py (Tanisha). This module never
imports TaskResult directly. The callers are responsible for adapting
TaskResult.raw_trace (List[dict]) into the `trace: str` this class expects,
and for writing EvalResult fields back onto a TaskResult instance.

Field mapping used by callers:
    judge_score <- EvalResult.score
    judge_failure_type <- EvalResult.failure_type.value (or None)
    judge_explanation <- EvalResult.explanation
    rubric_drift_score <- evolve_rubric()'s returned similarity float

NOTE: EvalResult.dimension_scores is intentionally NOT persisted onto
TaskResult. There's no field for it in the current schema and it isnt
needed by any of the 5 paper figures. Its still returned here in case a
caller wants it for ad-hoc inspection; just don't store it.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError


# failure taxonomy 
class FailureType(Enum):
    HALLUCINATION = "hallucination" # asserts false facts about env state
    CONTEXT_LOSS = "context_loss" # forgets earlier task constraints
    GOAL_DRIFT = "goal_drift" # pursues wrong sub-goal
    EXECUTION_ERROR = "execution_error"  # invalid action / wrong object


# result type 
@dataclass
class EvalResult:
    score: float
    failure_type: Optional[FailureType]
    explanation: str
    dimension_scores: dict = field(default_factory=dict) # not persisted to TaskResult


# default seed rubric 
DEFAULT_RUBRIC = {
    "goal_completion": {
        "description": "Did the agent fully achieve the stated household task goal?",
        "weight": 0.35,
        "rules": [
            "Verify the final state matches the explicit task objective instructions."
        ],
    },
    "action_validity": {
        "description": "Were all actions syntactically valid and applied to correct objects?",
        "weight": 0.25,
        "rules": [
            "Ensure the agent does not interact with items it hasn't picked up.",
            "Verify actions use legal ALFWorld environment commands.",
        ],
    },
    "context_retention": {
        "description": "Did the agent remember all task constraints across all steps?",
        "weight": 0.20,
        "rules": [
            "Check that state changes (e.g., heating, cooling) are executed before final placement."
        ],
    },
    "efficiency": {
        "description": "Did the agent avoid unnecessary steps or backtracking?",
        "weight": 0.20,
        "rules": [
            "Flag repetitive actions or loops moving between the same locations sequentially."
        ],
    },
}


def trace_to_str(raw_trace: list[dict]) -> str:
    """Adapter: TaskResult.raw_trace (List[dict]) -> the string evaluate() expects.

    The notebook's evaluate() was built and tested against hand-formatted
    "TASK: ... / Step N / OBS / ACT" strings, not raw step dicts. We don't
    know the exact step dict keys Tanisha's agent.py produces, so this does
    a generic, readable dump rather than assuming field names. Revisit if
    the judge's accuracy suffers from a worse trace format than the
    notebook's hand-built mock traces used.
    """
    return json.dumps(raw_trace, indent=2)


# judge 
class SEALJudge:
    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        max_retries: int = 5,
        backoff_time: float = 12.0,
    ):
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.model_name = model_name
        self.max_retries = max_retries
        self.backoff_time = backoff_time

    def _call_with_retry(self, prompt: str, temperature: float, log_label: str) -> str:
        """Shared retry/backoff wrapper for both evaluate() and evolve_rubric().

        Resets backoff_time per call rather than mutating self.backoff_time,
        so repeated evaluate()/evolve_rubric() calls on a long-lived judge
        instance don't permanently inflate the wait time after one transient
        rate-limit blip.
        """
        backoff = self.backoff_time
        response = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=temperature,
                    ),
                )
                break
            except (ClientError, ServerError) as e:
                if attempt < self.max_retries - 1:
                    print(
                        f"\n[RETRY:{log_label}] {type(e).__name__} "
                        f"({getattr(e, 'status_code', 'Error')}). "
                        f"Retrying in {backoff}s... (Attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    print(f"\n[FAILED:{log_label}] Exhausted all retries.")
                    raise
        return response.text.strip()

    def evaluate(self, trace: str, rubric: dict) -> EvalResult:
        failure_values = [f.value for f in FailureType]

        prompt = f"""You are a strict AI task evaluator for household robot agents.

Given an agent execution trace and a scoring rubric, evaluate the agent's performance.

RUBRIC CRITERIA:
{json.dumps(rubric, indent=2)}

AGENT TRACE:
{trace}

Return a JSON object with exactly these keys:
{{
  "score": <float 0.0 to 1.0>,
  "failure_type": <one of {failure_values} or null if successful>,
  "explanation": "<1-2 sentences explaining the score>",
  "dimension_scores": {{
    "<criterion_name>": <float 0.0 to 1.0>,
    ...
  }}
}}

Rules:
- score 1.0 = perfect task completion
- score 0.0 = total failure
- failure_type must be null if score >= 0.8
- dimension_scores must have one entry per rubric criterion
"""
        raw = self._call_with_retry(prompt, temperature=0.1, log_label="evaluate")
        data = json.loads(raw)

        ft_str = data.get("failure_type")
        ft = FailureType(ft_str) if ft_str and ft_str in failure_values else None

        return EvalResult(
            score=float(data["score"]),
            failure_type=ft,
            explanation=data["explanation"],
            dimension_scores=data.get("dimension_scores", {}),
        )

    def evolve_rubric(
        self,
        rubric: dict,
        failure_history: list[EvalResult],
        drift_floor: float = 0.45,
    ) -> tuple[dict, float, bool]:
        """Core novelty: judge rewrites its own rubric based on failure distributions.

        Returns (new_rubric, similarity_score, was_updated).
        Requires sentence-transformers + scikit-learn for compute_drift_score();
        see optional import guard below.
        """
        failure_summary = [
            {
                "failure_type": r.failure_type.value if r.failure_type else "none",
                "score": round(r.score, 2),
                "explanation": r.explanation,
            }
            for r in failure_history
        ]
        ft_counts = Counter(
            r.failure_type.value for r in failure_history if r.failure_type
        )

        prompt = f"""You are a meta-evaluator for autonomous AI agent architectures.

Your job is to optimize an evaluation rubric by appending or refining concrete execution checks based on observed task failure patterns.

CURRENT RUBRIC CONFIGURATION:
{json.dumps(rubric, indent=2)}

RECENT TRACE FAILURE LOGS ({len(failure_history)} iterations):
{json.dumps(failure_summary, indent=2)}

DIAGNOSED FAILURE DISTRIBUTION PROFILE:
{json.dumps(dict(ft_counts), indent=2)}

Instructions:
1. Identify which criterion's 'rules' failed to deter or capture these errors.
2. Update existing definitions or append strict strings to the 'rules' arrays to specifically alert the agent against making these errors again.
3. Keep total weights summing exactly to 1.0.
4. Maintain a structured collection containing between 3 and 6 criteria categories.
5. All rules must remain concrete, actionable, and specific to ALFWorld household command states.

Return a JSON object following this exact schema structure:
{{
  "criterion_name": {{
    "description": "...",
    "weight": <float>,
    "rules": ["rule 1", "rule 2"]
  }}
}}
"""
        raw = self._call_with_retry(prompt, temperature=0.2, log_label="evolve_rubric")
        new_rubric = json.loads(raw)

        total_w = sum(v.get("weight", 0.0) for v in new_rubric.values())
        if abs(total_w - 1.0) > 0.02:
            for k in new_rubric:
                new_rubric[k]["weight"] /= total_w

        similarity = compute_drift_score(rubric, new_rubric)

        if similarity < drift_floor:
            print(
                f"[BLOCKED] Semantic similarity {similarity:.3f} below floor "
                f"{drift_floor}. Rubric mutation discarded."
            )
            return rubric, similarity, False

        return new_rubric, similarity, True


# rubric drift scoring 
# fills TaskResult.rubric_drift_score. Requires sentence-transformers + sklearn,
# which aren't in the base SEAL deps yet - guarded import so this module is
# still importable (and SEALJudge.evaluate() still usable) without them

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    _embedder: Optional[SentenceTransformer] = None

    def _get_embedder() -> SentenceTransformer:
        global _embedder
        if _embedder is None:
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return _embedder

    def rubric_to_text(rubric: dict) -> str:
        chunks = []
        for k, v in rubric.items():
            rules_str = " ".join(v.get("rules", []))
            chunks.append(f"{k}: {v.get('description', '')} Rules: {rules_str}")
        return " | ".join(chunks)

    def compute_drift_score(rubric_old: dict, rubric_new: dict) -> float:
        """1.0 = identical rubrics, 0.0 = completely different."""
        embedder = _get_embedder()
        old_vec = embedder.encode([rubric_to_text(rubric_old)])
        new_vec = embedder.encode([rubric_to_text(rubric_new)])
        return float(cosine_similarity(old_vec, new_vec)[0][0])

except ImportError:

    def compute_drift_score(rubric_old: dict, rubric_new: dict) -> float:
        raise ImportError(
            "compute_drift_score() requires sentence-transformers and scikit-learn. "
            "Install both, or avoid calling evolve_rubric() without them."
        )