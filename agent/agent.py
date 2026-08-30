import os
import re
from groq import Groq

# Valid ALFWorld action verbs for plan coherence scoring
VALID_ACTION_VERBS = ["go to", "open", "put", "place", "examine", "pick up", "take", "close"]


def _read_rubric_hints(rubric) -> list:
    """Extract _seal_hints from the rubric if the judge evolved it.

    rubric can be a dict (from runner.py, which passes active_rubric directly
    to execute()) or a JSON string (legacy path). Both are handled.

    Returns a list of hint strings e.g. ["CONTEXT_LOSS_ADDRESSED", "GOAL_DRIFT_ADDRESSED"].
    Returns [] if no hints present (seed rubric, JudgeFixed, or evolution produced no diff).

    This is the ONLY place agent.py reads rubric structure - it never inspects
    individual criteria keys, so Anagha can rename rubric fields freely without
    breaking agent behavior.
    """
    if isinstance(rubric, str):
        import json as _json
        try:
            rubric = _json.loads(rubric)
        except Exception:
            return []
    if not isinstance(rubric, dict):
        return []
    return rubric.get("_seal_hints", [])


def compute_plan_coherence(plan: str) -> float:
    """
    Parses the planner's output and returns a 0.0–1.0 coherence score.
    Criteria: numbered steps, valid action verbs, no empty lines mid-plan.
    Exported in TaskResult so Shreyashree can use it as a metric directly.
    """
    if not plan or "[FALLBACK" in plan:
        return 0.0

    lines = [l.strip() for l in plan.strip().split("\n") if l.strip()]
    numbered = [l for l in lines if re.match(r"^\d+[\.\)]\s+", l)]
    if not numbered:
        return 0.1  # has content but not structured

    valid_steps = sum(
        1 for step in numbered
        if any(verb in step.lower() for verb in VALID_ACTION_VERBS)
    )
    coherence = valid_steps / len(numbered)
    return round(coherence, 2)


class SEALAgent:

    def __init__(self, api_key=None, rotator=None):
        # Migrated off HF Inference Providers (provider="auto" + Qwen2.5-7B)
        # after persistent 402s across accounts, including brand-new tokens
        # with $0 usage - root cause: provider="auto" routing this model
        # through a paid-only backend, not genuine per-account depletion.
        # Groq's openai/gpt-oss-20b is smaller/faster and sufficient for planning.
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.model_name = "openai/gpt-oss-20b"
        self.rotator = rotator
        self.steps_history = []
        self.consecutive_failures = 0

    def plan(self, task: str, rubric: str, max_retries: int = 3, retry_delay: float = 5.0) -> str:
        """Calls openai/gpt-oss-20b via Groq to generate a structured action plan.

        Retries transient failures before falling back - a single flaky call
        used to permanently corrupt strategy_used/plan_coherence for that
        iteration. On 401/429/invalid_api_key, rebuilds self.client from the
        current GROQ_API_KEY env value, in case KeyRotator rotated keys on
        the judge side mid-run.
        """
        system_prompt = "You are a household task planning agent."
        user_message = (
            f"Rubric: {rubric}\n"
            f"Task: {task}\n\n"
            f"Produce a numbered step-by-step action plan to complete this task. "
            f"Each step must be a single executable action such as "
            f"'go to <object>', 'open <object>', 'put <item> in <container>', or 'examine <item> using <object>'. "
            f"Output ONLY the numbered plan, no preamble."
        )

        last_err = None
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0.3,
                    max_tokens=256,
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                is_quota_error = "429" in err_str or "rate_limit" in err_str
                if is_quota_error and self.rotator:
                    self.rotator.force_rotate(reason=f"[agent.plan] {type(e).__name__}: {e}")
                    self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                    continue  
                if "401" in err_str or "429" in err_str or "invalid_api_key" in err_str:
                    self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                if attempt < max_retries - 1:
                    import time
                    time.sleep(retry_delay)

        return (
            f"1. Go to container\n2. Open container\n3. Place item\n"
            f"[FALLBACK - Groq planner unavailable: {last_err}]"
        )

    def _detect_failure_type(self, success: bool, trajectory: list) -> str:
        """Intrinsic failure diagnostic engine (independent from environmental oracle).

        # critical section
        # do not change this priority order without discussing with the team
        # Decided priority when signals overlap: GOAL_DRIFT > EXECUTION_ERROR > CONTEXT_LOSS
        # GOAL_DRIFT and EXECUTION_ERROR are both explicit keyword signals
        # and outrank CONTEXT_LOSS's inferred stagnation-rate heuristic
        # since a drifted or blocked trajectory can ALSO look stagnant
        """
        if success:
            return "NONE"

        total = len(trajectory)
        if total == 0:
            return "EXECUTION_ERROR"

        # 1. GOAL_DRIFT: target-substitution is a stronger, more specific signal
        # than stagnation rate. so we check it first
        wrong_object_steps = [
            s for s in trajectory
            if "wrong item" in s["observation_received"].lower()
            or "task drift" in s["observation_received"].lower()
        ]
        if wrong_object_steps:
            return "GOAL_DRIFT"

        # 2. EXECUTION_ERROR: explicit blocked-keyword signal, checked BEFORE
        # the stagnation-rate check so a trajectory that is both stagnant and
        # contains a blocked-keyword observation returns EXECUTION_ERROR
        blocked_keywords = ["jammed", "mechanical failure", "cannot open", "blocked"]
        for step in trajectory:
            obs = step["observation_received"].lower()
            if any(kw in obs for kw in blocked_keywords):
                return "EXECUTION_ERROR"

        # 3. CONTEXT_LOSS: inferred rate-based heuristic, checked last
        # internal_loop_alert is now None or a string warning
        stagnant = sum(
            1 for s in trajectory if s["internal_loop_alert"] is not None
        )
        stagnation_rate = stagnant / total

        if stagnation_rate >= 0.6:
            return "CONTEXT_LOSS"

        return "UNKNOWN"

    def execute(self, plan: str, env, rubric: str) -> dict:
        """Executes the step trajectory while monitoring for loop anomalies."""
        self.steps_history = []
        self.consecutive_failures = 0
        goal, current_obs = env.reset()
        done = False
        step_count = 0
        max_steps = 10
        sequence_state = 0

        target_match = re.search(r"see a (\b\w+\b) 1", current_obs)
        target = target_match.group(1) if target_match else "container"

        item_match = re.search(
            r"Put a (\b\w+\b)|Place a (\b\w+\b)|Examine a (\b\w+\b)", goal
        )
        item = "item"
        if item_match:
            item = [g for g in item_match.groups() if g is not None][0]

        # Resolve GOAL_DRIFT wrong-item token from scenario config if available
        # Falls back to hardcoded "key ring" only as last resort
        drift_item = getattr(env, "drift_item", None) or env.data.get("drift_item", "key ring")

        while not done and step_count < max_steps:
            step_count += 1

            forced_outcome = env.data["forced_outcome"]

            # Strategy selection - ordered by priority.
            # Rubric hints from judge.evolve_rubric() tell us which failure type
            # the judge addressed in its latest rewrite. We check hints (substance)
            # not literal marker strings (option-a from design discussion).
            # "CONTEXT_LOSS_ADDRESSED" in hints means the judge added rules targeting
            # loop/stagnation - agent should attempt escape actions, not just "look".
            # "GOAL_DRIFT_ADDRESSED" means the judge flagged wrong-item substitution -
            # agent should stay on the correct item instead of drifting.
            hints = _read_rubric_hints(rubric)
            context_loss_rubric_updated = "CONTEXT_LOSS_ADDRESSED" in hints
            goal_drift_rubric_updated   = "GOAL_DRIFT_ADDRESSED" in hints

            if forced_outcome == "CONTEXT_LOSS" and not context_loss_rubric_updated:
                # Judge hasn't addressed context loss yet - agent stays stuck (stagnates)
                action = "look"
            elif self.consecutive_failures >= 2:
                # Recovery: skip ahead to placement attempt
                action = f"put {item} in {target} 1"
            elif forced_outcome == "GOAL_DRIFT" and step_count >= 3 and not goal_drift_rubric_updated:
                # Judge hasn't addressed goal drift yet - agent drifts to wrong item
                action = f"put {drift_item} in {target} 1"
            elif sequence_state == 0:
                action = f"go to {target} 1"
                sequence_state = 1
            elif sequence_state == 1:
                action = f"open {target} 1"
                if forced_outcome != "EXECUTION_ERROR":
                    sequence_state = 2
            else:
                if "examine" in goal.lower():
                    action = f"examine {item} using {target} 1"
                else:
                    action = f"put {item} in {target} 1"

            next_obs, success = env.step(action, rubric)

            # internal_loop_alert is None (Python None) or a warning string
            # IMPORTANT: use None not the string "None"
            internal_warning = None
            if next_obs == current_obs:
                self.consecutive_failures += 1
                internal_warning = f"WARNING: Loop detected. Stagnation count: {self.consecutive_failures}."
            else:
                self.consecutive_failures = 0

            self.steps_history.append({
                "step": step_count,
                "action_executed": action,
                "observation_received": next_obs,
                "internal_loop_alert": internal_warning,
            })

            current_obs = next_obs
            done = success
            if done:
                break

        final_outcome = "SUCCESS" if done else "FAILED"
        detected_failure_type = self._detect_failure_type(done, self.steps_history)

        confidence_map = {
            "NONE": 0.95,
            "GOAL_DRIFT": 0.85,
            "CONTEXT_LOSS": 0.35,
            "EXECUTION_ERROR": 0.35,
            "UNKNOWN": 0.50,
        }
        confidence_score = confidence_map.get(detected_failure_type, 0.50)
        plan_coherence = compute_plan_coherence(plan)

        # Behavioral drift recovery:
        # did the agent initially drift to the wrong item, then later place the correct item in the same trajectory?
        drifted_at_some_step = any(
            "wrong item" in s["observation_received"].lower()
            or "task drift" in s["observation_received"].lower()
            for s in self.steps_history
        )
        drift_recovered = bool(drifted_at_some_step and done)

        return {
            "task_goal": goal,
            "macro_plan": plan,
            "plan_coherence": plan_coherence,       # NEW: metric per architecture diagram
            "total_steps": step_count,
            "final_outcome": final_outcome,
            "detected_failure_type": detected_failure_type,
            "agent_intrinsic_confidence": confidence_score,
            "trajectory": self.steps_history,
            "drift_recovered": drift_recovered,
        }