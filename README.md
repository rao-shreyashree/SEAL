## 📂 File Registry and Specifications

### 1. `task_result.py`
* **Role:** **The Shared Integration Contract **
* **Technical Specification:** Implements the `TaskResult` Python dataclass wrapper. It defines the exact JSON structure and data types shared by all three teammates to eliminate integration breaking changes.
* **Data Flow Mechanics:**
  * **Tanisha (Agent Loop):** Creates and instantiates this object at the end of a trajectory simulation, populating core fields, action history, and basic diagnostic metrics.
  * **Anagha (Judge Loop):** Imports this class to read the compiled `raw_trace` and `strategy_used`, evaluates the rubric, and writes back the `judge_score` and `judge_failure_type`.
  * **Shreyashree (Metrics/Database):** Ingests batches of these records via file parsing arrays to update the centralized performance metrics dashboard.

### 2. `agent.py`
* **Role:** **Autonomous Agent Core**
* **Technical Specification:** Houses the primary `SEALAgent` class module integrated with the Hugging Face `InferenceClient` utilizing the `mistralai/Mistral-7B-Instruct-v0.3` model core.
* **Key Components:**
  * `plan()`: Utilizes instruction-tuned text generation parameters to output a structured, multi-step Chain-of-Thought blueprint.
  * `execute()`: Runs a state-machine driver that steps sequentially through the text sandbox, monitoring consecutive unmutated state signatures.
  * `_detect_failure_type()`: An **agent-intrinsic** diagnostics component. It performs algorithmic post-mortem trace parsing completely blind from the simulator ground truth to self-classify errors (`CONTEXT_LOSS`, `GOAL_DRIFT`, `EXECUTION_ERROR`) based on stagnation mathematical properties and textual keyword matches.
  * `compute_plan_coherence()`: A structural evaluation utility that parses the plan text string to rank syntactic formatting and verb density before environment boundary commitment.

### 3. `scenarios.py`
* **Role:** **ALFWorld Simulation World Matrix (20 Task Variations)**
* **Technical Specification:** Implements `MultiScenarioALFWorldEnv`, a deterministic, mock text-gym simulator that maps classic ALFWorld task paradigms (Pick and Place, Heat, Cool, Clean, Examine) to explicitly controlled baseline environments.
* **Mechanics:**
  * Programmatically isolates variables into distinct operational modes: standard success spaces, non-mutating feedback traps (simulating context degradation blocks), semantic object mutation drift spaces, and rigid handle constraints (jammed barriers) to rigorously evaluate the robustness of our self-reflective tracking loops.

### 4. `run_agent.py`
* **Role:** **Single-Task Trajectory Debugger**
* **Technical Specification:** An operational entry script that instantiates a single agent-environment intersection (Defaulting to `Scenario 001`).
* **Output:** Compiles step outputs and saves a singular standalone payload tracking trace called `TaskResult.json` directly to local storage for quick structural inspection and unit testing.

### 5. `run_and_check.py`
* **Role:** **20-Scenario Baseline Evaluation Matrix Runner**
* **Technical Specification:** The main orchestration batch execution engine that loops through all twenty separate environment configurations consecutively.
* **Output:**
  * Exports twenty standalone structured files named `sample_execution_trace_1.json` through `sample_execution_trace_20.json`.
  * Automatically serializes and concatenates all records into a unified line-delimited schema file named `baseline_all_traces.jsonl` for instantaneous batch database parsing.

---

## 📈 Baseline Anomaly Classification Reference

The baseline dataset tracks and separates three distinct error modes, mapping out an empirical classification grid for our framework analysis:

1. **Context Loss (`CONTEXT_LOSS`):** Identified when the trajectory loop hits a $\ge 60\%$ stagnation threshold (the environment text payload locks up completely while the agent endlessly resubmits `look`).
2. **Goal Drift (`GOAL_DRIFT`):** Triggered when action execution momentum stays high, but the agent's target item reference undergoes semantic drift away from the assignment baseline (e.g., swapping a tissue box out for a key ring).
3. **Execution Error (`EXECUTION_ERROR`):** Triggered when the agent's pathing, token logic, and verbs match perfectly, but the trajectory drops to $0$ progress due to physical physical blocks in the sandbox (e.g., a jammed handle notification string).
