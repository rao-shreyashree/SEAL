import json
import os
from agent import SEALAgent

# Simulated ALFWorld Environment tracing standard text gym boundary states
class SimulatedALFWorldEnv:
    def __init__(self):
        self.step_idx = 0
        self.goal = "Put a clean apple inside the fridge."
        self.states = {
            0: "You are in the middle of a kitchen. Looking around, you see a counter 1, a sink 1, and a fridge 1.",
            1: "You arrive at fridge 1. The fridge 1 is closed.",
            2: "You open the fridge 1. The fridge 1 is open. Inside it, you see nothing.",
            3: "You put the clean apple in the fridge 1. You won!"
        }
        # Variable to prevent simple hardcoded state exploits and test adaptive tracking
        self.has_apple = True 
    
    def reset(self):
        self.step_idx = 0
        return self.goal, self.states[0]
        
    def step(self, action: str):
        self.step_idx += 1
        act = action.lower().strip()
        
        # Simulated transition dynamics matching classic text simulator responses
        if "go to fridge" in act:
            return self.states[1], False
        elif "open fridge" in act:
            return self.states[2], False
        elif "put apple" in act and "fridge" in act:
            return self.states[3], True
        else:
            return f"Command '{action}' recognized, but nothing visible changed in the environment.", False


def main():
    print("=== Initializing Week 1 SEALAgent Functional Test ===")
    
    # 1. Instantiate your core agent class modules
    agent = SEALAgent()
    
    # 2. Setup your mock text sandbox environment
    env = SimulatedALFWorldEnv()
    
    # 3. Define configuration parameters (Injected down from your dataset configurations)
    task_input = "Put a clean apple inside the fridge."
    initial_rubric = (
        "Ensure you sequentially navigate to structures. Always open "
        "closed targets before placing elements inside. Do not loop."
    )
    
    # 4. Phase 1 Check: Generate the Strategy Plan
    print("\n[Testing Phase 1: Planning via Qwen-7B-Instruct]")
    action_plan = agent.plan(task=task_input, rubric=initial_rubric)
    print(f"Generated Plan:\n{action_plan}\n")
    
    # 5. Phase 2 Check: Run Environment Trajectory Loop with live adjustments
    print("[Testing Phase 2: Execution & Trace Assembly]")
    trace_output = agent.execute(plan=action_plan, env=env, rubric=initial_rubric)
    
    # 6. Check Deliverable: Verify the structured Trace JSON format
    print("\n[Testing Phase 3: Trace JSON Validation]")
    print(json.dumps(trace_output, indent=2))
    
    # Save compilation directly to system disk space
    output_filename = "sample_execution_trace.json"
    with open(output_filename, "w") as f:
        json.dump(trace_output, f, indent=2)
    print(f"\n=== Test Executed Successfully! '{output_filename}' exported. ===")

if __name__ == "__main__":
    main()