# SEAL - Self-Evaluating Agent Loop

SEAL is a modular, open-source framework for self-improving AI agents. 
The core idea: an agent attempts a task, a separate LLM-based judge evaluates the outcome, and both the agent and judge iteratively improve through a Plan → Act → Reflect → Judge feedback loop.

The judge doesn't just score - it evolves its own rubric based on observed failure patterns (hallucination, context loss, goal drift, execution errors), making evaluation smarter over iterations.

Built and tested on ALFWorld, WebArena, and SWE-Bench. Uses open-source models (Mistral 7B, LLaMA 3 8B) via HuggingFace.

## Team
- Tanisha: Agent loop (SEALAgent)
- Anagha: Judge (SEALJudge)
- Shreyashree: Schema, metrics, database (TaskResult contract, SQLite, baseline runs)

## Structure
- `agent/` - agent core: `agent.py` (SEALAgent), `scenarios.py` (20 mock ALFWorld environments), `run_agent.py` (single-task debug runner), `run_and_check.py` (20-scenario baseline suite), `config.json` (rubric config)
- `seal/` - shared core modules: `task_result.py` (shared `TaskResult` dataclass - the integration contract all three of us read/write), `database.py` (SQLite schema + insert logic), `metrics.py` (4 metric functions)
- `notebooks/` - Colab notebooks (`week1_baseline.ipynb` - clones repo, runs agent, stores results, computes metrics)
- `data/` - ALFWorld tasks (placeholder for real ALFWorld data in later weeks)
- `results/` - SQLite DB output (`seal_results.db`)

## Week 1 - What Happened
Tanisha built `SEALAgent` with Mistral 7B via HuggingFace, including an intrinsic failure detector that self-classifies failures (`CONTEXT_LOSS`, `GOAL_DRIFT`, `EXECUTION_ERROR`) purely from trajectory patterns. She also built `scenarios.py`: 20 controlled mock ALFWorld tasks covering success cases and each forced failure mode, so we could test the pipeline deterministically before plugging in real ALFWorld.

Shreyashree built the shared `TaskResult` dataclass (now the single integration contract everyone imports from), the SQLite schema, and all 4 metric functions (success rate, failure precision, judge alignment, convergence speed) as pure functions over `List[TaskResult]`. Also built the Colab notebook that runs the 20-scenario suite, loads the resulting `baseline_all_traces.jsonl` into SQLite, and computes the baseline metrics.

### Zero-Shot Baseline Results (20 tasks, no SEAL logic - pure agent, no retries)
| Metric | Value |
|---|---|
| Task Success Rate | 0.85 (17/20) |
| Failure Breakdown | CONTEXT_LOSS: 66.7%, EXECUTION_ERROR: 33.3% |
| Judge Alignment | 1.0 |
| Avg Iterations to Success | 1.0 |
| Tasks Never Solved | 3 |

This is the pre-SEAL baseline - every success happened in a single shot since no retry/judge-evolution loop exists yet. Week 2 (SEALRunner integration loop) should improve on the 3 unsolved tasks via the Plan → Act → Judge → StrategyUpdate cycle.

## Week 1 Status
- TaskResult dataclass (shared contract, adopted from Tanisha's `task_result.py`)
- SQLite schema
- Metric functions
- Colab notebook
- Zero-shot baseline - 20/20 tasks run, numbers above