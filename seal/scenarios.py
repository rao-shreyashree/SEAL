import json

class MultiScenarioALFWorldEnv:
    def __init__(self):
        self.current_scenario = 0
        self.data = {"forced_outcome": "SUCCESS"}

    def set_scenario(self, scenario_id: int):
        self.current_scenario = scenario_id
        # Standard configuration array mappings for evaluation profiling
        if scenario_id == 4 or scenario_id == 19:
            self.data["forced_outcome"] = "CONTEXT_LOSS"
        elif scenario_id == 9:
            self.data["forced_outcome"] = "GOAL_DRIFT"
        elif scenario_id == 14:
            self.data["forced_outcome"] = "EXECUTION_ERROR"
        else:
            self.data["forced_outcome"] = "SUCCESS"

    def reset(self):
        return f"Task objective for scenario template configuration {self.current_scenario}", {}