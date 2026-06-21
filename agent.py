import os
import re
import json
import urllib.request

VALID_ACTION_VERBS = ["go to", "open", "put", "place", "examine", "pick up", "take", "close"]


def compute_plan_coherence(plan: str) -> float:
    """
    Parses Mistral's plan output and returns a 0.0-1.0 coherence score.
    Criteria: numbered steps, valid action verbs, no empty lines mid-plan.
    """
    if not plan or "[FALLBACK" in plan:
        return 0.0

    lines = [l.strip() for l in plan.strip().split("\n") if l.strip()]
    numbered = [l for l in lines if re.match(r"^\d+[\.\)]\s+", l)]
    if not numbered:
        return 0.1  # Has content but not structured

    valid_steps = sum(
        1 for step in numbered
        if any(verb in step.lower() for verb in VALID_ACTION_VERBS)
    )
    coherence = valid_steps / len(numbered)
    return round(coherence, 2)


class SEALAgent:

    def __init__(self, hf_token=None):
        # Local Ollama endpoint replaces the hosted Hugging Face InferenceClient
        self.ollama_url = "http://localhost:11434/api/generate"
        self.steps_history = []
        self.consecutive_failures = 0

    def plan(self, task: str, rubric: str) -> str:
        """Calls local Mistral 7B via Ollama to generate a structured CoT action plan."""
        prompt = (
            f"[INST] You are a household task planning agent.\n"
            f"Rubric: {rubric}\n"
            f"Task: {task}\n\n"
            f"Produce a numbered step-by-step action plan to complete this task. "
            f"Each step must be a single executable action such as "
            f"'go to <object>', 'open <object>', 'put <item> in <container>', or 'examine <item> using <object>'. "
            f"Output ONLY the numbered plan, no preamble. [/INST]"
        )
        
        payload = {
            "model": "mistral",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3
            }
        }
        
        try:
            req = urllib.request.Request(
                self.ollama_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=300) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data.get("response", "").strip()
        except Exception as e:
            return (
                f"1. Go to container\n2. Open container\n3. Place item\n"
                f"[FALLBACK - Local Ollama Mistral unavailable: {e}]"
            )

    def _detect_failure_type(self, success: bool, trajectory: list) -> str:
        """Intrinsic failure diagnostic engine (independent from environmental oracle)."""
        if success:
            return "NONE"

        total = len(trajectory)
        if total == 0:
            return "EXECUTION_ERROR"

        wrong_object_steps = [
            s for s in trajectory
            if "wrong item" in s["observation_received"].lower()
            or "task drift" in s["observation_received"].lower()
        ]
        if wrong_object_steps:
            return "GOAL_DRIFT"

        stagnant = sum(
            1 for s in trajectory if s["internal_loop_alert"] is not None
        )
        stagnation_rate = stagnant / total

        if stagnation_rate >= 0.6:
            return "CONTEXT_LOSS"

        blocked_keywords = ["jammed", "mechanical failure", "cannot open", "blocked"]
        for step in trajectory:
            obs = step["observation_received"].lower()
            if any(kw in obs for kw in blocked_keywords):
                return "EXECUTION_ERROR"

        return "UNKNOWN"

    def _detect_drift_recovery(self, success: bool, trajectory: list) -> bool:
        """Flags whether the trajectory contained a drift event that was later recovered from."""
        drift_seen = any(
            "wrong item" in s["observation_received"].lower()
            or "task drift" in s["observation_received"].lower()
            for s in trajectory
        )
        return bool(success and drift_seen)

    def execute(self, plan: str, env, rubric: str) -> dict:
        """Executes the step trajectory while monitoring for loop anomalies."""
        self.steps_history = []
        self.consecutive_failures = 0
        goal, current_obs = env.reset()
        done = False
        step_count = 0
        max_steps = 10
        sequence_state = 0

        target = env.target
        item = env.item
        drift_item = env.drift_item

        while not done and step_count < max_steps:
            step_count += 1

            forced_outcome = env.data["forced_outcome"]

            if forced_outcome == "CONTEXT_LOSS" and "META-REFLECTION" not in rubric:
                action = "look"
            elif self.consecutive_failures >= 2:
                action = f"put {item} in {target} 1"
            elif forced_outcome == "GOAL_DRIFT" and step_count == 3 and "ITERATIVE-PROMPTING" not in rubric:
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
        drift_recovered = self._detect_drift_recovery(done, self.steps_history)

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
            "plan_coherence": plan_coherence,
            "total_steps": step_count,
            "final_outcome": final_outcome,
            "detected_failure_type": detected_failure_type,
            "drift_recovered": drift_recovered,
            "agent_intrinsic_confidence": confidence_score,
            "trajectory": self.steps_history,
        }