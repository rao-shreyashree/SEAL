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
import re

from groq import Groq
from groq import APIError, APIStatusError, RateLimitError


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
    Tolerant of key naming: checks action_executed/observation_received first,
    falls back to action/observation if present, and falls back to the loop
    index if 'step' is missing. 
    internal_loop_alert is intentionally dropped here -- noisy, degrades judge context window, not needed for evaluation

    Format matches the notebook's original hand-built "Step N / Action / Obs"
    style the judge prompt was tuned against, replacing the earlier generic json.dumps() placeholder
    """
    lines = []

    for i, s in enumerate(raw_trace, start=1):
        step = s.get("step", i)
        action = s.get("action_executed", s.get("action", ""))
        obs = s.get("observation_received", s.get("observation", ""))

        lines.append(
            f"Step {step}: Action -> '{action}' | Obs -> {obs}"
        )

    return "\n".join(lines)

def _extract_json(raw: str) -> dict:
    """Groq's llama-3.3-70b-versatile often prepends reasoning prose before
    the JSON block (unlike gemini, which returned near-raw JSON). 
    Pull the JSON object out from wherever it sits in the response instead of
    assuming the whole string is JSON."""
    clean = raw.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1)
    else:
        # fallback: grab from the first '{' to the last '}' in the response
        start = clean.find("{")
        end = clean.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError(f"[evaluate] No JSON object found in Groq response: {raw!r}")
        candidate = clean[start:end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"[evaluate] Failed to parse extracted JSON: {candidate!r}") from e

# judge 
class SEALJudge:
    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        max_retries: int = 5,
        backoff_time: float = 2.5,  # Groq free tier: 30 RPM -> 2s+ between calls
        rotator=None,  # optional seal.runner.KeyRotator - enables force-rotate on 429s
    ):
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.model_name = model_name
        self.max_retries = max_retries
        self.backoff_time = backoff_time
        self.rotator = rotator

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
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                break
            except (APIError, APIStatusError, RateLimitError) as e:
                is_quota_error = getattr(e, "status_code", None) == 429 or "rate_limit" in str(e).lower()
                if is_quota_error and self.rotator:
                    # real quota/token limit hit - rotate key immediately instead of
                    # sleeping and retrying on the same exhausted key
                    self.rotator.force_rotate(reason=f"[{log_label}] {type(e).__name__}: {e}")
                    self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))  # rebuild client with new key
                    continue
                if attempt < self.max_retries - 1:
                    print(
                        f"\n[RETRY:{log_label}] {type(e).__name__} "
                        f"({getattr(e, 'status_code', 'Error')}). "
                        f"Retrying in {backoff}s... (Attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(backoff)
                else:
                    print(f"\n[FAILED:{log_label}] Exhausted all retries.")
                    raise
        return response.choices[0].message.content.strip()

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
        data = _extract_json(raw)

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
        new_rubric = _extract_json(raw)

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

        # Inject structured behavioral hints derived from the rubric's actual content.
        # The agent reads _seal_hints to change action-selection — NOT literal marker strings.
        # This is option (a) from the design discussion: agent reacts to rubric substance.
        # Hints are computed from what actually changed between old and new rubric text,
        # so if context_retention rules didn't change, CONTEXT_LOSS hint won't fire.
        #
        # _seal_hints is stripped before logging (runner strips it from rubric_string_representation)
        # so it never leaks into TaskResult.rubric_text or judge's own evaluate() prompt.
        context_kw = [
            "repeat", "stagnation", "revisit", "redundant", "no progress", "identical observation", "unproductive", "same location",
            "does not advance", "no new information", "prolonged", "static", "unchanged", "escape", "break the loop", "vary the action",
        ]
        drift_kw   = ["target", "substitut", "correct item", "wrong item", "drift", "goal object"]
        exec_kw    = ["block", "jam", "cannot open", "locked", "obstacle", "stuck"]

        def _rules_text(r: dict) -> str:
            parts = []
            for v in r.values():
                if isinstance(v, dict):
                    parts.extend(v.get("rules", []))
                    parts.append(v.get("description", ""))
            return " ".join(parts).lower()

        new_txt = _rules_text(new_rubric)
        old_txt = _rules_text(rubric)

        # Hint fires only when the relevant keyword appears in the NEW rubric
        # but was absent from the old one — guards against spurious signals
        # on criteria that didn't actually change.
        hints = []
        if any(k in new_txt and k not in old_txt for k in context_kw):
            hints.append("CONTEXT_LOSS_ADDRESSED")
        if any(k in new_txt and k not in old_txt for k in drift_kw):
            hints.append("GOAL_DRIFT_ADDRESSED")
        if any(k in new_txt and k not in old_txt for k in exec_kw):
            hints.append("EXECUTION_ERROR_ADDRESSED")

        if not hints and any(
            h.get("failure_type") == "context_loss" for h in failure_summary
        ):
            print(f"[HINT-MISS] CONTEXT_LOSS failure present but no keyword matched.\n"
                  f"  new_txt sample: {new_txt[:300]}")
            
        new_rubric["_seal_hints"] = hints  # list[str], may be [] if nothing meaningfully changed

        return new_rubric, similarity, True


class JudgeFixed(SEALJudge):
    """Ablation baseline: No-Rubric-Evolution condition for Fig 5.

    # critical section
    # do not change without discussing with the team
    # This subclasses SEALJudge (Groq) on purpose, NOT judge_mistral.SEALJudge (Mistral/HF)
    # The ablation in Fig 5 is only valid if SEAL and No-Rubric-Evolution differ in exactly one variable: 
    # whether evolve_rubric() actually mutates the rubric
    # Using a different judge model here would confound "rubric evolution on/off" with
    # "different LLM backend" and invalidate the ablation bar chart

    evaluate() is inherited unchanged (still Groq)
    evolve_rubric() is overridden to be a true no-op: returns the input rubric, similarity 1.0
    (identical) and was_updated=False
    so rubric_hash and rubric_drift_score stay flat across iterations
    the same way reflexion's BASELINE_RUBRIC does
    """

    def evolve_rubric(
        self,
        rubric: dict,
        failure_history: list[EvalResult],
        drift_floor: float = 0.45,
    ) -> tuple[dict, float, bool]:
        return rubric, 1.0, False


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