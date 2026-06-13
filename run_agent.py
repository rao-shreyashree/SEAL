import json
import os
from agent import TanishaAgentCore

# Abstracted Mock Environment to illustrate standard ALFWorld gym wrapper structures 
# (Replaced by direct 'import alfworld.agents.environment' dependencies in execution testbed)
class MockALFWorldEnv:
    def __init__(self):
        self.steps = 0
        self.goal = "Put a clean apple inside the fridge."
        self.states = {
            0: "You are in the middle of a kitchen. Looking around, you see a counter 1, a sink 1, and a fridge 1.",
            1: "You arrive at fridge 1. The fridge 1 is closed.",
            2: "You open the fridge 1. The fridge 1 is open. Inside it, you see nothing.",
            3: "You put the clean apple in the fridge 1. You won!"
        }
    
    def reset(self):
        self.steps = 0
        return self.goal, self.states[0]
        
    def step(self, action):
        self.steps += 1
        action = action.lower().strip()
        if "go to fridge" in action:
            return self.states[1], False
        elif "open fridge" in action:
            return self.states[2], False
        elif "put" in action and "fridge" in action:
            return self.states[3], True
        else:
            return "Nothing happens.", False

def run_evaluation_trajectory(output_path="TaskResult.json"):
    # Instantiate your core module
    agent = TanishaAgentCore()
    
    # Initialize environment tracking parameters
    env = MockALFWorldEnv()
    goal, initial_obs = env.reset()
    
    current_obs = initial_obs
    done = False
    max_steps = 15
    step_counter = 0
    
    # Inter-component trace schema creation
    trajectory_log = []
    task_success = "FAILED"

    print(f"Starting Task Pipeline Execution Loop...")
    print(f"Goal: {goal}\nInitial State: {current_obs}\n" + "-"*50)

    feedback_alert = None

    while not done and step_counter < max_steps:
        # 1. Step the Agent Framework to get strategic selection and textual actions
        strategy, thought, action = agent.generate_structured_cot_and_action(
            goal, current_obs, system_feedback=feedback_alert
        )
        
        # Track history arrays
        agent.action_history.append(action)
        
        print(f"Step {step_counter + 1}:")
        print(f" -> Strategy: {strategy}")
        print(f" -> Thought: {thought}")
        print(f" -> Dispatched Action: {action}")
        
        # 2. Commit action execution to the environment simulator boundary
        next_obs, success = env.step(action)
        agent.observation_history.append(next_obs)
        print(f" -> Environment Observation: {next_obs}\n")
        
        # 3. Process execution feedback loop through internal failure detector
        feedback_alert = agent.failure_detector(action, next_obs)
        
        # Track parameters inside step trace dictionaries
        trajectory_log.append({
            "step": step_counter + 1,
            "strategy_selected": strategy,
            "thought_trace": thought,
            "action_executed": action,
            "observation_received": next_obs,
            "failure_detected_internally": str(feedback_alert)
        })
        
        current_obs = next_obs
        done = success
        step_counter += 1
        
        if success:
            task_success = "SUCCESS"
            break

    # 4. Construct the unified "TaskResult JSON" (Shared Integration Schema)
    task_result_payload = {
        "task_metadata": {
            "goal": goal,
            "final_outcome": task_success,
            "total_steps_executed": step_counter,
            "max_steps_boundary": max_steps
        },
        "evaluation_metrics_export": {
            "plan_coherence_rating": "Pending_Judge_Evaluation",  # Ingested by Anagha
            "steps_per_task": step_counter,
            "task_success_rate_binary": 1 if task_success == "SUCCESS" else 0
        },
        "step_by_step_trajectory": trajectory_log
    }

    # Atomically export payload data to disk for subsequent components
    with open(output_path, 'w') as out_file:
        json.dump(task_result_payload, out_file, indent=2)
        
    print(f"Trajectory cycle finished. Integration schema exported cleanly to: {output_path}")

if __name__ == "__main__":
    run_evaluation_trajectory()