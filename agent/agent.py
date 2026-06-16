import os
import json
import re
from huggingface_hub import InferenceClient

class SEALAgent:
    def __init__(self, hf_token=None):
        # Fallback to environment variable if token isn't passed directly
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            print("WARNING: No HF_TOKEN found. Using mock responses for local testing.")
            self.client = None
        else:
            # Qwen2.5-7B-Instruct natively supports structured serverless chat completions
            self.client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=token)
        
        self.steps_history = []

    def plan(self, task: str, rubric: str) -> str:
        """
        Queries the LLM to build the macro-level CoT strategy based on the 
        current evolving task rubric.
        """
        system_prompt = (
            "You are an advanced autonomous agent loop operating within an ALFWorld text simulator.\n"
            f"CRITICAL COMPLIANCE RUBRIC:\n{rubric}\n\n"
            "Your job is to generate a high-level step-by-step strategy. "
            "You MUST structure your output exactly as follows:\n"
            "THOUGHT: <Your step-by-step reasoning considering the constraints and rubric>\n"
            "PLAN:\n"
            "1. <First action item>\n"
            "2. <Second action item>"
        )
        user_prompt = f"Task Goal: {task}\nGenerate your structured Chain-of-Thought action plan now:"

        if not self.client:
            return self._mock_plan_fallback(task)

        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error in planning phase: {e}")
            return self._mock_plan_fallback(task)

    def execute(self, plan: str, env, rubric: str) -> dict:
        """
        Steps through the environment using live LLM atomic actions, monitors
        internal failure states, and assembles the execution trace.
        """
        self.steps_history = []
        goal, current_obs = env.reset()
        done = False
        step_count = 0
        max_steps = 10
        
        print(f"--- Executing Plan for Goal: {goal} ---")
        
        while not done and step_count < max_steps:
            step_count += 1
            
            # --- INTERNAL FAILURE DETECTOR ---
            # Proactively inspect current trajectory patterns to trap infinite state loops
            internal_warning = None
            if len(self.steps_history) >= 2:
                last_action = self.steps_history[-1]["action_executed"]
                prior_action = self.steps_history[-2]["action_executed"]
                
                # Verify if the agent is stuck dispatching identical commands into static environments
                if last_action == prior_action and current_obs == self.steps_history[-1]["observation_received"]:
                    internal_warning = (
                        f"WARNING: You just attempted '{last_action}' on the previous step and the environment "
                        "state remained unchanged. Do NOT repeat it. Diverge to an alternative valid command."
                    )

            # --- LIVE ATOMIC LLM STEP ACTION ---
            # Dispatch operational states directly into the micro-action LLM processor
            action = self._determine_next_atomic_action_llm(plan, current_obs, rubric, internal_warning)
            
            # Commit the evaluated text step into the simulator environment boundary
            next_obs, success = env.step(action)
            
            self.steps_history.append({
                "step": step_count,
                "action_executed": action,
                "observation_received": next_obs,
                "internal_loop_alert": internal_warning if internal_warning else "None"
            })
            
            print(f"Step {step_count}: Action -> '{action}'")
            print(f"        Obs    -> '{next_obs}'")
            if internal_warning:
                print(f"        ⚠️ Failure Detector Alert -> {internal_warning}")
            
            current_obs = next_obs
            done = success
            if done:
                break

        # Output payload structurally structured to match Shreyashree's database ingestion expectations
        return {
            "task_goal": goal,
            "macro_plan": plan,
            "total_steps": step_count,
            "final_outcome": "SUCCESS" if done else "FAILED",
            "trajectory": self.steps_history
        }

    def _determine_next_atomic_action_llm(self, plan: str, current_obs: str, rubric: str, internal_warning: str = None) -> str:
        """Uses a low-latency LLM inference query to parse the best immediate string command."""
        if not self.client:
            return "look"

        system_prompt = (
            "You are the execution framework unit of an ALFWorld text sandbox agent.\n"
            f"Evolving Active Rubric: {rubric}\n\n"
            "Based on the high-level macro plan and current observation metadata, determine the single best upcoming action command.\n"
            "Standard vocabulary template examples:\n"
            "- go to fridge 1\n- open fridge 1\n- take apple 1 from counter 1\n- put apple 1 in fridge 1\n- look\n\n"
            "CRITICAL: Return ONLY the exact literal command string string. No formatting markdown, no markdown backticks, no quotes, no periods, and no additional conversation."
        )

        user_content = (
            f"Macro Strategy Blueprint: {plan}\n"
            f"Current Environment State Observation: {current_obs}\n"
        )
        if internal_warning:
            user_content += f"\nCRITICAL FAILURE RUNTIME ALERT: {internal_warning}\n"

        user_content += "Dispatched Action Command String:"

        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=25,
                temperature=0.0  # Zero out randomness for deterministic execution adherence
            )
            
            # Sanitize trailing characters or formatting wrappers from response payload
            action_text = response.choices[0].message.content.strip()
            action_text = action_text.replace('"', '').replace("'", "").replace('`', '').rstrip('.')
            return action_text
        except Exception as e:
            print(f"Action Selection LLM Exception: {e}. Defaulting to safety pass.")
            return "look"

    def _mock_plan_fallback(self, task: str) -> str:
        return (
            "THOUGHT: I need to locate the fridge and insert the object.\n"
            "PLAN:\n1. Locate fridge 1\n2. Open fridge 1\n3. Insert object."
        )