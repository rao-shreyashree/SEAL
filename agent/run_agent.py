import sys
import os

# Clean up sys.path to remove the local folder priority trap
current_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_dir)

if current_dir in sys.path:
    sys.path.remove(current_dir)
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Try loading as a package first; fallback to direct import if Python is stubborn
try:
    from agent.agent import SEALAgent
except ModuleNotFoundError:
    import agent
    SEALAgent = getattr(agent, 'SEALAgent', None)

if __name__ == "__main__":
    print("Testing SEALAgent initialization...")
    if SEALAgent is not None:
        agent_instance = SEALAgent()
        print("SEALAgent initialized successfully!")
    else:
        print("Error: Could not resolve SEALAgent class.")