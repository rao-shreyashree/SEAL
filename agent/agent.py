import os
import re
from huggingface_hub import InferenceClient

# Valid ALFWorld action verbs for plan coherence scoring
VALID_ACTION_VERBS = ["go to", "open", "put", "place", "examine", "pick up", "take", "close"]


def _read_rubric_hints(rubric) -> list:
    """Extract _seal_hints from the rubric if the judge evolved it.

    rubric can be a dict (from runner.py, which passes active_rubric directly
    to execute()) or a JSON string (legacy path). Both are handled.

    Returns a list of hint strings e.g. ["CONTEXT_LOSS_ADDRESSED", "GOAL_DRIFT_ADDRESSED"].
    Returns [] if no hints present (seed rubric, JudgeFixed, or evolution produced no diff).

    This is the ONLY place agent.py reads rubric structure — it never inspects
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
    Parses Mistral's plan output and returns a 0.0–1.0 coherence score.
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

    def __init__(self, hf_token=None):
        token = hf_token or os.environ.get("HF_TOKEN")
        # provider="auto" routes through HF Inference Providers (nebius, sambanova, etc.)
        # instead of hf-inference, which as of mid-2025 only serves CPU tasks like
        # embeddings/classification and no longer serves LLMs.
        # This is what broke Mistral-7B text_generation — it was routing through hf-inference.
        self.client = InferenceClient(
            provider="auto",
            api_key=token,
        )
        self.steps_history = []
        self.consecutive_failures = 0

    def plan(self, task: str, rubric: str) -> str:
        """Calls Qwen2.5-7B-Instruct via HF Inference Providers to generate a structured action plan.

        Model history (for reference):
          Mistral-Nemo-Instruct-2407  — chat.completions, "not a chat model" error, FALLBACK every task
          Mistral-7B-Instruct-v0.3    — text_generation, routed through hf-inference (CPU only), broken
          HuggingFaceH4/zephyr-7b-beta — chat.completions workaround, unstable
          Qwen/Qwen2.5-7B-Instruct    — chat.completions + provider=auto, stable on free HF token ✓
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
        try:
            completion = self.client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=256,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            return (
                f"1. Go to container\n2. Open container\n3. Place item\n"
                f"[FALLBACK - Qwen unavailable: {e}]"
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

    def _parse_plan_steps(self, plan: str) -> list:
        """Parse LLM plan output into a list of action strings.

        Handles both numbered formats:
          "1. go to fridge 1"
          "1) go to fridge 1"
        Strips the number prefix and returns clean action strings.
        Empty or FALLBACK plans return an empty list.
        """
        if not plan or "[FALLBACK" in plan:
            return []

        steps = []
        for line in plan.strip().split("\n"):
            line = line.strip()
            m = re.match(r"^\d+[\.\)\s]\s*(.*)", line)
            if m:
                action = m.group(1).strip()
                if action:
                    steps.append(action)
        return steps

    def execute(self, plan: str, env, rubric) -> dict:
        """Execute the LLM plan against the environment step by step.

        E5 change: execute() now CONSUMES the plan produced by plan().
        It parses the numbered steps and dispatches them in order.
        The forced_outcome branch ladder is gone — success comes from
        env.step() returning True, not from reading env.data["forced_outcome"].

        For real ALFWorldEnv (E5): env._match_admissible() normalizes each
        plan step to the closest admissible command before dispatch, so
        ALFWorld does not silently reject unrecognized action strings.

        For MultiScenarioALFWorldEnv (E1-E4): plan steps are dispatched
        directly. The scripted env handles forced_outcome internally in
        env.step() — agent.execute() no longer reads that flag at all.
        This is the paper's main claim: agent reacts to the plan and rubric
        hints, not to an oracle.

        Rubric hints (from judge.evolve_rubric()) now condition the PLANNER
        PROMPT in runner.py rather than flipping a branch in execute().
        The causal chain becomes: rubric evolution → richer prompt → better
        plan → better execution, which is what the paper actually measures.

        Fallback: if the plan is empty or unparseable, a minimal hardcoded
        sequence is used so the run does not crash. Logged via loop alert.
        """
        self.steps_history = []
        self.consecutive_failures = 0
        goal, current_obs = env.reset()
        done = False
        step_count = 0
        max_steps = 10

        # Parse LLM plan into ordered action list
        plan_steps = self._parse_plan_steps(plan)

        # Fallback if plan is empty or unparseable
        if not plan_steps:
            target_match = re.search(r"see a (\b\w+\b) 1", current_obs)
            target = target_match.group(1) if target_match else "container"
            item_match = re.search(
                r"Put a (\b\w+\b)|Place a (\b\w+\b)|Examine a (\b\w+\b)", goal
            )
            item = "item"
            if item_match:
                item = [g for g in item_match.groups() if g is not None][0]
            plan_steps = [
                f"go to {target} 1",
                f"open {target} 1",
                f"put {item} in {target} 1",
            ]

        # Rubric hints read once for confidence scoring only.
        # execute() no longer branches on hints — that job moved to the
        # planner prompt in runner.py (E5 step 4 architectural change).
        hints = _read_rubric_hints(rubric)

        plan_idx = 0

        while not done and step_count < max_steps:
            step_count += 1

            # Dispatch next plan step; cycle last step if plan exhausted
            if plan_idx < len(plan_steps):
                action = plan_steps[plan_idx]
                plan_idx += 1
            else:
                action = plan_steps[-1] if plan_steps else "look"

            # Pass rubric as string for scripted env compatibility;
            # ALFWorldEnv.step() ignores it
            rubric_str = rubric if isinstance(rubric, str) else ""
            next_obs, success = env.step(action, rubric_str)

            internal_warning = None
            if next_obs == current_obs:
                self.consecutive_failures += 1
                internal_warning = (
                    f"WARNING: Loop detected. Stagnation count: {self.consecutive_failures}."
                )
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

        drifted_at_some_step = any(
            "wrong item" in s["observation_received"].lower()
            or "task drift" in s["observation_received"].lower()
            for s in self.steps_history
        )
        drift_recovered = bool(drifted_at_some_step and done)

        return {
            "task_goal": goal,
            "macro_plan": plan,
            "plan_coherence": plan_coherence,
            "total_steps": step_count,
            "final_outcome": final_outcome,
            "detected_failure_type": detected_failure_type,
            "agent_intrinsic_confidence": confidence_score,
            "trajectory": self.steps_history,
            "drift_recovered": drift_recovered,
        }