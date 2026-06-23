"""
seal/judge_mistral.py: SEALJudge (Mistral/HF backend) + JudgeFixed ablation baseline

extracted from notebook/Week2.ipynb (Anagha, origin/judge @ a771baa)

STATUS: NOT SURE. 
This is a second, independent SEALJudge implementation alongside seal/judge.py (Gemini backend, 
from SEALjudge.ipynb @ 28273a3). Week2.ipynb does not modify or reference SEALjudge.ipynb 
in any way - git diff 28273a3 a771baa confirms its a pure addition
No commit message explains the relationship between the two

Working theory (NOT CONFIRMED - Anagha should tell me):
    seal.judge.SEALJudge -> primary judge, SEAL condition
    seal.judge_mistral.JudgeFixed -> ablation condition ("no-rubric-evolution" in the Day 5 ablation bar chart)

If thats wrong and this was meant to replace seal/judge.py outright, the
two files should be reconciled (most likely: i will delete this one, or merge the
JudgeFixed ablation class into judge.py and drop the rest)

Known integration gaps vs TaskResult (task_result.py, Tanisha):

    1. evaluate() returns a raw dict here, not EvalResult. Different shape
        than seal.judge.SEALJudge.evaluate(). Any call site that handles both
        judges needs to normalize this - see evaluate_to_result() below, added
        here (not in the notebook) to paper over that gap

    2. evolve_rubric() returns (rubric, rubric_hash: str) - a fingerprint,
        not a similarity float. This does NOT fill TaskResult.rubric_drift_score
        (Optional[float]) directly. The hash tells whether the rubric
        changed, not how much. A comment in the original notebook ("facilitate
        Shreyashree's drift calculations") suggests the hash was meant as an
        input to a downstream drift computation, not the final score - but
        that downstream step doesnt exist yet. so i wont write this hash into
        rubric_drift_score as-is; its the wrong type and wrong semantics

    3. evaluate()'s exception path returns a dict with score/failure_type/
        explanation but no "dimension_scores" key at all (unlike seal.judge's
        EvalResult, which defaults it to {})

    4. Requires HF_TOKEN env var (Colab secrets in the original) instead of
        GEMINI_API_KEY. Different credential, different rate limits/quotas
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from huggingface_hub import InferenceClient


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

    def _get_rubric_hash(self, rubric: Dict[str, Any]) -> str:
        """Fingerprint, not a drift score. See module docstring point 2."""
        rubric_string = json.dumps(rubric, sort_keys=True)
        return hashlib.md5(rubric_string.encode("utf-8")).hexdigest()[:8]

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
        self, rubric: Dict[str, Any], failure_history: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], str]:
        """Returns (new_rubric, rubric_hash). rubric_hash is a fingerprint,
        NOT a drift similarity score - see module docstring point 2 before
        wiring this into TaskResult.rubric_drift_score."""
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
            return new_rubric, self._get_rubric_hash(new_rubric)
        except Exception:
            return rubric, self._get_rubric_hash(rubric)


class JudgeFixed(SEALJudge):
    """Ablation baseline: inherits evaluate(), but evolve_rubric() is a no-op.
    Strong candidate for the "no-rubric-evolution" condition in the Day 5
    ablation bar chart (SEAL vs no-rubric-evolution vs Reflexion vs zero-shot)
    - unconfirmed, see module docstring."""

    def evolve_rubric(
        self, rubric: Dict[str, Any], failure_history: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], str]:
        return rubric, self._get_rubric_hash(rubric)


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