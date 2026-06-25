"""
seal/judge_mistral.py: SEALJudge (Mistral/HF backend) + JudgeFixed

extracted from notebook/Week2.ipynb (Anagha, origin/judge @ a771baa)

STATUS UPDATE: evolve_rubric() now uses real cosine similarity
(imported from seal.judge, not duplicated - one embedder/compute_drift_score
implementation, shared) instead of returning a bare MD5 fingerprint
Return shape now matches seal.judge.SEALJudge.evolve_rubric() exactly:
(new_rubric, similarity_score, was_updated). 
Old (rubric, hash) callers need to be updated to unpack 3 values (not 2)

Relationship to seal/judge.py (Gemini backend), i have now resolved:
    
    seal.judge.SEALJudge -> primary judge, SEAL condition (Gemini)
    
    seal.judge.JudgeFixed -> No-Rubric-Evolution ablation, Fig 5 (Gemini)
    
    seal.judge_mistral.SEALJudge / JudgeFixed -> Mistral/HF backend, kept as an OPTIONAL extra comparison point, 
        NOT wired into Fig 5's ablation
        
    reason: Fig 5 must isolate exactly one variable (rubric evolution on/off).
            a Mistral-backed JudgeFixed differs from SEAL's Gemini judge in two variables at once 
            (model + evolution), which would confound the ablation
            if we want to have a Mistral comparison, it will be a separate, explicitly-labeled 5th condition 
            but not a substitute for the No-Rubric-Evolution bar

Known integration gaps vs TaskResult (task_result.py, Tanisha):

    1. evaluate() returns a raw dict here, not EvalResult. Different shape
        than seal.judge.SEALJudge.evaluate(). Use evaluate_to_result() below
        to normalize before handing to any call site that treats both
        judges uniformly.

    2. evaluate()'s exception path returns a dict with score/failure_type/
        explanation but no "dimension_scores" key at all (unlike seal.judge's
        EvalResult, which defaults it to {}). evaluate_to_result() papers
        over this with .get(..., {}).

    3. Requires HF_TOKEN env var instead of GEMINI_API_KEY
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import InferenceClient

# we reuse the one real drift-scoring implementation instead of duplicating it
# compute_drift_score() requires sentence-transformers + scikit-learn
# refer to seal/judge.py's import guard for the no-deps fallback behavior
from seal.judge import compute_drift_score


class FailureType(Enum):
    HALLUCINATION = "hallucination"
    CONTEXT_LOSS = "context_loss"
    GOAL_DRIFT = "goal_drift"
    EXECUTION_ERROR = "execution_error"


@dataclass
class EvalResult:
    """Kept identical in shape to seal.judge.EvalResult so callers can treat
    both judges uniformly after normalization. Not what evaluate() returns
    natively here - see evaluate_to_result()."""
    score: float
    failure_type: Optional[FailureType]
    explanation: str
    dimension_scores: dict = field(default_factory=dict)


MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

EVALUATOR_SYSTEM_PROMPT = """You are an independent, objective external Auditor and Task Evaluator for autonomous robotic environments.
Your objective is to rigidly inspect an execution trace against a criteria ruleset.

CRITICAL PROTOCOLS:
1. You are NOT the agent. You are NOT planning or executing steps.
2. Evaluate what happened with zero bias or leniency.
3. You must output a valid JSON string containing exactly three keys: "score", "failure_type", and "explanation".

Allowed failure_type values:
- "hallucination"
- "context_loss"
- "goal_drift"
- "execution_error"
- null (only if score >= 0.8)
"""

META_EVOLVER_SYSTEM_PROMPT = """You are a Meta-Optimization Director. Your job is to modify an evaluation rubric's rule list to catch historical vulnerabilities earlier.
You add explicit, defensive testing checks based on a pattern of observed agent failures.
"""


class SEALJudge:
    def __init__(self, model_id: str = MISTRAL_MODEL, hf_token: Optional[str] = None):
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN must be set (env var or hf_token param) to run Mistral inference.")
        self.client = InferenceClient(api_key=token)
        self.model_id = model_id

    def evaluate(self, trace: str, rubric: Dict[str, Any]) -> Dict[str, Any]:
        user_message = f"""RUBRIC SPECIFICATION:
{json.dumps(rubric, indent=2)}

AGENT EXECUTION TRAJECTORY TRACE:
{trace}

Generate evaluation results adhering to this exact schema layout structure:
{{
  "score": <float between 0.0 and 1.0>,
  "failure_type": <string or null>,
  "explanation": "<1-2 sentence rationale details>"
}}
"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.1,
                    max_tokens=300,
                )
                raw_response = completion.choices[0].message.content.strip()
                return json.loads(raw_response)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    return {
                        "score": 0.0,
                        "failure_type": "execution_error",
                        "explanation": f"Evaluation pipeline error: {str(e)}",
                    }

    def evolve_rubric(
        self,
        rubric: Dict[str, Any],
        failure_history: List[Dict[str, Any]],
        drift_floor: float = 0.45,
    ) -> Tuple[Dict[str, Any], float, bool]:
        """Returns (new_rubric, similarity_score, was_updated) - SAME SHAPE as
        seal.judge.SEALJudge.evolve_rubric(). 
        similarity_score is now cosine similarity (via seal.judge.compute_drift_score), not a hash

        # critical section
        # we do not change this return shape without updating seal.judge's
        # equivalent too - callers (Tanisha's runner) rely on both judges
        # being interchangeable here
        """
        user_message = f"""CURRENT RUBRIC STRUCTURE:
{json.dumps(rubric, indent=2)}

HISTORICAL FAILURES OBSERVED IN ACTIVE TESTING WINDOW:
{json.dumps(failure_history, indent=2)}

INSTRUCTIONS:
Given these past failures, rewrite the rubric configuration rules array blocks to catch this failure type earlier!
Maintain the exact same JSON criteria keys, but alter or append specific 'rules' strings defensively.
Keep total cumulative weights totaling exactly 1.0.

Return the modified rubric as a clean JSON dictionary matching the original root structural schema.
"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": META_EVOLVER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
                max_tokens=1000,
            )
            raw_response = completion.choices[0].message.content.strip()
            new_rubric = json.loads(raw_response)
        except Exception:
            # Fallback guardrail if the call or json parsing breaks - mirrors
            # seal.judge's behavior of not silently advancing on a bad response
            return rubric, 1.0, False

        total_w = sum(v.get("weight", 0.0) for v in new_rubric.values())
        if total_w and abs(total_w - 1.0) > 0.02:
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


class JudgeFixed(SEALJudge):
    """Mistral-backed no-evolution judge.

    NOT wired into Fig 5's ablation bar chart
    Kept here only in case we later want an explicit, separately labeled Mistral-vs-Gemini comparison condition 
    For the actual No-Rubric-Evolution bar, we use seal.judge.JudgeFixed instead
    """

    def evolve_rubric(
        self,
        rubric: Dict[str, Any],
        failure_history: List[Dict[str, Any]],
        drift_floor: float = 0.45,
    ) -> Tuple[Dict[str, Any], float, bool]:
        return rubric, 1.0, False


def evaluate_to_result(raw: Dict[str, Any]) -> EvalResult:
    """Normalizer: converts this module's raw evaluate() dict into the same
    EvalResult shape seal.judge.SEALJudge.evaluate() returns natively, so
    call sites can treat both judges uniformly. Added here - not present in
    the original notebook.
    """
    ft_str = raw.get("failure_type")
    try:
        ft = FailureType(ft_str) if ft_str else None
    except ValueError:
        ft = None  # unrecognized string from the model; don't crash on it
    return EvalResult(
        score=float(raw.get("score", 0.0)),
        failure_type=ft,
        explanation=raw.get("explanation", ""),
        dimension_scores=raw.get("dimension_scores", {}),
    )