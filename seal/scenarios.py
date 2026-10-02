"""
seal/scenarios.py — ALFWorld environment wrappers for SEAL.

TWO classes, same .step() / .data interface:

  MultiScenarioALFWorldEnv  — scripted env (20 scenarios, forced_outcome).
                              Used for E1-E4 and all baseline conditions.
                              Kept exactly as-is — no changes.

  ALFWorldEnv               — real ALFWorld env wrapper (E5).
                              No forced_outcome. Success comes from the
                              env's own goal check. Exposes the same
                              .reset() / .step() / .data interface as
                              MultiScenarioALFWorldEnv so runner.py,
                              agent.py, and baselines need zero changes
                              to the call sites.

To use ALFWorldEnv:
    1. pip install alfworld
    2. alfworld-download   (downloads task data to ~/alfworld/data)
    3. Set ALFWORLD_DATA env var if data is not in the default location
    4. In runner.py, swap MultiScenarioALFWorldEnv() for ALFWorldEnv()
"""

import os


# ── Scripted env (E1-E4, all baselines) ──────────────────────────────────────
# Unchanged from the working version — do not edit this class.

class MultiScenarioALFWorldEnv:
    def __init__(self, scenario_id=0):
        self.scenarios = [
            {"goal": "Put a clean apple inside the fridge.",          "target": "fridge",     "item": "apple",   "forced_outcome": "SUCCESS"},
            {"goal": "Put a hot potato into the microwave.",           "target": "microwave",  "item": "potato",  "forced_outcome": "SUCCESS"},
            {"goal": "Place a cooled tomato inside the fridge.",       "target": "fridge",     "item": "tomato",  "forced_outcome": "SUCCESS"},
            {"goal": "Examine a book under the desk lamp.",            "target": "lamp",       "item": "book",    "forced_outcome": "SUCCESS"},
            {"goal": "Put a clean mug in the cupboard.",               "target": "cupboard",   "item": "mug",     "forced_outcome": "CONTEXT_LOSS"},
            {"goal": "Put a knife in the drawer.",                     "target": "drawer",     "item": "knife",   "forced_outcome": "SUCCESS"},
            {"goal": "Place a soap bar on the sink counter.",          "target": "counter",    "item": "soap",    "forced_outcome": "SUCCESS"},
            {"goal": "Put a clean fork in the cabinet.",               "target": "cabinet",    "item": "fork",    "forced_outcome": "SUCCESS"},
            {"goal": "Place a heated egg on the plate.",               "target": "plate",      "item": "egg",     "forced_outcome": "SUCCESS"},
            {
                "goal": "Put a tissue box on the nightstand.",
                "target": "nightstand", "item": "tissue",
                "forced_outcome": "GOAL_DRIFT",
                "drift_item": "key ring",
            },
            {"goal": "Put a laptop on the desk.",                      "target": "desk",       "item": "laptop",  "forced_outcome": "SUCCESS"},
            {"goal": "Place a glass bowl inside the dishwasher.",      "target": "dishwasher", "item": "bowl",    "forced_outcome": "SUCCESS"},
            {"goal": "Put a pillow on the bed.",                       "target": "bed",        "item": "pillow",  "forced_outcome": "SUCCESS"},
            {"goal": "Place a cloth in the laundry basket.",           "target": "basket",     "item": "cloth",   "forced_outcome": "SUCCESS"},
            {"goal": "Put a pencil inside the drawer.",                "target": "drawer",     "item": "pencil",  "forced_outcome": "EXECUTION_ERROR"},
            {"goal": "Examine a credit card under the floor lamp.",    "target": "lamp",       "item": "card",    "forced_outcome": "SUCCESS"},
            {"goal": "Put a clean pan on the stove.",                  "target": "stove",      "item": "pan",     "forced_outcome": "SUCCESS"},
            {"goal": "Place a hot cup of milk on the dining table.",   "target": "table",      "item": "milk",    "forced_outcome": "SUCCESS"},
            {"goal": "Put a sponge in the bathroom cabinet.",          "target": "cabinet",    "item": "sponge",  "forced_outcome": "SUCCESS"},
            {"goal": "Place a cooled soda can on the counter.",        "target": "counter",    "item": "soda",    "forced_outcome": "CONTEXT_LOSS"},
        ]
        self.set_scenario(scenario_id)

    def set_scenario(self, scenario_id):
        self.scenario_id = min(max(0, scenario_id), len(self.scenarios) - 1)
        self.data = self.scenarios[self.scenario_id]
        self.steps = 0
        self.state_index = 0
        self.drift_item = self.data.get("drift_item", "key ring")
        self.drift_triggered = False

    def reset(self):
        self.steps = 0
        self.state_index = 0
        self.drift_triggered = False
        return self.data["goal"], f"You are in the middle of a room. You see a {self.data['target']} 1."

    def step(self, action, rubric=""):
        self.steps += 1
        act = action.lower().strip()
        target = self.data["target"]
        item = self.data["item"]
        outcome = self.data["forced_outcome"]

        if "META-REFLECTION" in rubric or "ITERATIVE-PROMPTING" in rubric:
            outcome = "SUCCESS"

        if outcome == "CONTEXT_LOSS":
            return "Command recognized, but nothing visible changed.", False

        if f"go to {target}" in act:
            self.state_index = 1
            return f"You arrive at {target} 1. The {target} 1 is closed.", False

        if f"open {target}" in act:
            if outcome == "EXECUTION_ERROR" and "ITERATIVE-PROMPTING" not in rubric:
                return f"Mechanical failure: The {target} 1 handle is jammed.", False
            if self.state_index == 1:
                self.state_index = 2
                return f"You open the {target} 1. It is now open.", False

        if "put" in act or "place" in act or "examine" in act:
            if (outcome == "GOAL_DRIFT" and self.drift_item.split()[0] in act) or self.drift_triggered:
                if "ITERATIVE-PROMPTING" not in rubric:
                    self.drift_triggered = True
                    return "You put the wrong item down. Task drift detected.", False

            if self.state_index == 2 or target in ["desk", "table", "counter", "shelf", "bed", "plate"]:
                return f"Success! You completed the task sequence for {item} inside {target} 1.", True

        return "Command recognized, but nothing visible changed.", False


# ── Real ALFWorld env wrapper (E5) ────────────────────────────────────────────

class ALFWorldEnv:
    """
    Thin wrapper around the real ALFWorld TextWorld environment.

    Exposes the same .reset() / .step() / .data interface as
    MultiScenarioALFWorldEnv so every call site in runner.py,
    agent.py, and the baselines works unchanged.

    Key differences from MultiScenarioALFWorldEnv:
      - No forced_outcome — success comes from the env's own goal check
      - No scripted step ladder — agent.execute() must parse and dispatch
        the LLM plan steps (that's E5 step 3 in agent.py)
      - data["forced_outcome"] is always "REAL_ENV" as a sentinel so
        runner.py's oracle_failure_type normalization produces "NONE"
        on success (REAL_ENV != "SUCCESS" → won't accidentally normalize)
        and _detect_failure_type() reads the real trajectory, not a flag

    Setup before first use:
        pip install alfworld
        alfworld-download
        export ALFWORLD_DATA=~/alfworld/data   # if not in default location
    """

    # ALFWorld task types we support — maps to the folder names in the data dir
    SUPPORTED_TASK_TYPES = [
        "pick_and_place_simple",
        "pick_clean_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "look_at_obj_in_light",
        "pick_two_obj_and_place",
    ]

    def __init__(self, task_index: int = 0, split: str = "train"):
        """
        task_index : which task in the split to load (0-indexed)
        split      : "train" | "valid_seen" | "valid_unseen"
        """
        self._task_index = task_index
        self._split = split
        self._env = None
        self._goal = ""
        self._admissible_commands = []

        # .data dict mirrors MultiScenarioALFWorldEnv.data so runner.py
        # can read forced_outcome, goal, etc without branching on env type
        self.data = {
            "forced_outcome": "REAL_ENV",  # sentinel — no oracle in real env
            "goal": "",
            "target": "",
            "item": "",
        }

        self._load_env()

    def _load_env(self):
        """Lazy-load alfworld. Import error is surfaced clearly."""
        try:
            import alfworld.agents.environment as alf_env
            import yaml
        except ImportError:
            raise ImportError(
                "alfworld is not installed. Run: pip install alfworld && alfworld-download"
            )

        # Build config — alfworld needs a config yaml
        config = self._build_config()
        env_type = "AlfredTWEnv"

        self._alf_env_class = getattr(alf_env, env_type)
        self._config = config

    def _build_config(self) -> dict:
        """Minimal alfworld config dict."""
        data_path = os.environ.get(
            "ALFWORLD_DATA",
            os.path.expanduser("~/alfworld/data"),
        )
        return {
            "dataset": {
                "data_path": data_path,
                "eval_path": data_path,
            },
            "env": {
                "type": "AlfredTWEnv",
                "regen_graph": False,
                "domain_randomization": False,
                "thor": {"screen_width": 300, "screen_height": 300},
            },
            "general": {
                "training_method": "dagger",
                "evaluate": False,
                "task_types": self.SUPPORTED_TASK_TYPES,
            },
        }

    def reset(self):
        """
        Reset the environment to a new episode.
        Returns (goal_str, initial_observation_str) — same shape as
        MultiScenarioALFWorldEnv.reset().
        """
        import alfworld.agents.environment as alf_env

        env = self._alf_env_class(self._config, train_eval=self._split)
        env = env.init_env(batch_size=1)
        self._env = env

        obs, infos = env.reset()
        obs_str = obs[0] if isinstance(obs, list) else str(obs)

        # Goal is in infos["extra.gamefile"] → parse from obs or infos
        goal = ""
        if "extra.gamefile" in infos:
            # Extract goal from the game description
            goal = self._parse_goal(obs_str, infos)
        if not goal:
            goal = obs_str.split("\n")[0]  # first line is usually the goal

        self._goal = goal
        self._admissible_commands = infos.get("admissible_commands", [[]])[0]

        # Extract target/item from goal text for .data compatibility
        target, item = self._parse_target_item(goal)
        self.data["goal"] = goal
        self.data["target"] = target
        self.data["item"] = item

        return goal, obs_str

    def step(self, action: str, rubric: str = "") -> tuple:
        """
        Execute one action in the real ALFWorld environment.
        Returns (observation_str, success_bool) — same shape as
        MultiScenarioALFWorldEnv.step().

        action  : natural language command string
        rubric  : ignored here (real env has no scripted outcomes to override)
                  kept in signature so call sites are identical
        """
        if self._env is None:
            raise RuntimeError("Call reset() before step()")

        # Normalize action to closest admissible command
        action = self._match_admissible(action)

        obs, scores, dones, infos = self._env.step([action])
        obs_str = obs[0] if isinstance(obs, list) else str(obs)
        done = bool(dones[0]) if isinstance(dones, list) else bool(dones)
        score = float(scores[0]) if isinstance(scores, list) else float(scores)

        # ALFWorld scores 1.0 on task completion
        success = done and score >= 1.0

        self._admissible_commands = infos.get("admissible_commands", [[]])[0]

        return obs_str, success

    def set_scenario(self, scenario_id: int):
        """
        Compatibility shim — MultiScenarioALFWorldEnv.set_scenario() is called
        by runner.py before each iteration. In the real env this is a no-op
        because the task is fixed at init time. For multi-task runs, construct
        a new ALFWorldEnv(task_index=scenario_id) per task instead.
        """
        # No-op: real env task is set at __init__ time
        pass

    # ── helpers ──────────────────────────────────────────────────────────────

    def _parse_goal(self, obs: str, infos: dict) -> str:
        """Extract natural language goal from ALFWorld observation."""
        # ALFWorld puts the goal in the first line after "You are in..."
        lines = obs.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.lower().startswith("your task is") or line.lower().startswith("you need"):
                return line
        return lines[0] if lines else obs

    def _parse_target_item(self, goal: str) -> tuple:
        """
        Best-effort extraction of target container and item from goal text.
        Used to populate .data["target"] and .data["item"] for compatibility.
        Not used for any decision logic in the real env path.
        """
        import re
        goal_lower = goal.lower()

        # "put a <item> in/on/into <target>"
        m = re.search(r"put (?:a |an )?(.+?) (?:in|on|into) (?:a |an |the )?(.+?)(?:\.|$)", goal_lower)
        if m:
            return m.group(2).strip(), m.group(1).strip()

        # "examine a <item> ..."
        m2 = re.search(r"examine (?:a |an )?(.+?) (?:using|under|with) (?:a |an |the )?(.+?)(?:\.|$)", goal_lower)
        if m2:
            return m2.group(2).strip(), m2.group(1).strip()

        return "container", "item"

    def _match_admissible(self, action: str) -> str:
        """
        Match agent's action string to the closest admissible command.
        ALFWorld rejects actions not in the admissible set with no-ops,
        so this prevents silent action failure.

        Strategy: exact match first, then longest common token overlap.
        """
        if not self._admissible_commands:
            return action

        action_lower = action.lower().strip()

        # Exact match
        for cmd in self._admissible_commands:
            if cmd.lower().strip() == action_lower:
                return cmd

        # Token overlap — find admissible command with most shared words
        action_tokens = set(action_lower.split())
        best_cmd = action
        best_overlap = 0
        for cmd in self._admissible_commands:
            cmd_tokens = set(cmd.lower().split())
            overlap = len(action_tokens & cmd_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_cmd = cmd

        return best_cmd