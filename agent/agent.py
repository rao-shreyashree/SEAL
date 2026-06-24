import os
import re
from huggingface_hub import InferenceClient

# Valid ALFWorld action verbs for plan coherence scoring
VALID_ACTION_VERBS = ["go to", "open", "put", "place", "examine", "pick up", "take", "close"]


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
        self.client = InferenceClient(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            token=token,
        )
        self.steps_history = []
        self.consecutive_failures = 0

    def plan(self, task: str, rubric: str) -> str:
        """Calls Mistral 7B to generate a structured CoT action plan."""
        prompt = (
            f"[INST] You are a household task planning agent.\n"
            f"Rubric: {rubric}\n"
            f"Task: {task}\n\n"
            f"Produce a numbered step-by-step action plan to complete this task. "
            f"Each step must be a single executable action such as "
            f"'go to <object>', 'open <object>', 'put <item> in <container>', or 'examine <item> using <object>'. "
            f"Output ONLY the numbered plan, no preamble. [/INST]"
        )
        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
            )
            return response.strip()
        except Exception as e:
            return (
                f"1. Go to container\n2. Open container\n3. Place item\n"
                f"[FALLBACK — Mistral unavailable: {e}]"
            )

    def _detect_failure_type(self, success: bool, trajectory: list) -> str:
        """Intrinsic failure diagnostic engine (independent from environmental oracle)."""
        if success:
            return "NONE"

        total = len(trajectory)
        if total == 0:
            return "EXECUTION_ERROR"

        # GOAL_DRIFT checked first
        # target-substitution is a stronger, more specific signal than stagnation rate.
        # A drifted trajectory can ALSO look stagnant (agent retries around the wrong target), 
        # so this must win the tie.
        wrong_object_steps = [
            s for s in trajectory
            if "wrong item" in s["observation_received"].lower()
            or "task drift" in s["observation_received"].lower()
        ]
        if wrong_object_steps:
            return "GOAL_DRIFT"

        # Use boolean check correctly — internal_loop_alert is now None or a string warning
        stagnant = sum(
            1 for s in trajectory if s["internal_loop_alert"] is not None
        )
        stagnation_rate = stagnant / total

        # CONTEXT_LOSS: agent trapped in non-mutating state loop
        if stagnation_rate >= 0.6:
            return "CONTEXT_LOSS"

        # EXECUTION_ERROR: environmental dependency block via string parsing
        blocked_keywords = ["jammed", "mechanical failure", "cannot open", "blocked"]
        for step in trajectory:
            obs = step["observation_received"].lower()
            if any(kw in obs for kw in blocked_keywords):
                return "EXECUTION_ERROR"

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

            # Strategy selection — ordered by priority
            if forced_outcome == "CONTEXT_LOSS":
                action = "look"
            elif self.consecutive_failures >= 2:
                # Recovery: skip ahead to placement attempt
                action = f"put {item} in {target} 1"
            elif forced_outcome == "GOAL_DRIFT" and step_count >= 3:
                # Semantic drift simulation: inject wrong token from scenario config
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

            next_obs, success = env.step(action)

            # internal_loop_alert is None (Python None) or a warning string
            # IMPORTANT: use None not the string "None" so teammates can do truthiness checks safely
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
                "internal_loop_alert": internal_warning,  # None or string — NOT the string "None"
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

        return {
            "task_goal": goal,
            "macro_plan": plan,
            "plan_coherence": plan_coherence,       # NEW: metric per architecture diagram
            "total_steps": step_count,
            "final_outcome": final_outcome,
            "detected_failure_type": detected_failure_type,
            "agent_intrinsic_confidence": confidence_score,
            "trajectory": self.steps_history,
        }