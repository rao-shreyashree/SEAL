# SEAL - Self-Evolving Agent Loop

SEAL is a modular, open-source framework for self-improving AI agents. The core idea: an agent attempts a task, a separate LLM-based judge evaluates the outcome, and the judge evolves its own evaluation rubric based on observed failure patterns - not just scoring, but changing *what it checks for* over iterations.

The agent then reacts to the judge's rubric changes to select recovery strategies, closing a **Plan → Act → Judge → Evolve → Retry** loop.

Built and tested on ALFWorld (20 controlled scenarios covering success cases and each forced failure mode: `CONTEXT_LOSS`, `GOAL_DRIFT`, `EXECUTION_ERROR`), benchmarked against a Reflexion-style baseline, a no-rubric-evolution ablation, and a zero-shot baseline.

## Architecture

<!-- TODO: add pipeline / architecture diagram here -->

## Team
- **Tanisha** - agent loop and action-selection logic (`agent/agent.py`, `seal/reflexion_baseline.py`, `seal/run_baselines.py`)
- **Anagha** - judge and rubric-evolution logic (`seal/judge.py`, `seal/judge_mistral.py`)
- **Shreyashree** - integration contract, metrics, database, figures, benchmark orchestration (`task_result.py`, `metrics.py`, `figures.py`, `database.py`, `runner.py`)

## How it works

1. **Agent plans and executes** a household task in ALFWorld against the current rubric (Groq, `llama-3.1-8b-instant`).
2. **Judge evaluates** the trajectory against the rubric (Groq, `llama-3.3-70b-versatile`) and scores it.
3. On repeated failure, the **judge rewrites its own rubric**, targeting whichever failure category the recent trace exposed.
4. The judge diffs the old and new rubric text and emits structured **behavioral hints** (e.g. `CONTEXT_LOSS_ADDRESSED`) describing what actually changed - not a hardcoded marker string.
5. The **agent reads these hints** to decide whether it should attempt a different recovery action on retry, rather than reacting to literal text markers.

This keeps "does rubric evolution actually change agent behavior" an empirical question the pipeline can fail, rather than something true by construction.

## Requirements
```bash
pip install groq huggingface_hub sentence-transformers scikit-learn matplotlib
```

### API keys

**Groq** (required - powers the agent planner and the primary judge):

Create `keys.json` in the project root:
```json
{
  "keys": ["gsk_your_key_1", "gsk_your_key_2"]
}
```
Multiple keys are optional but recommended (have 2 keys) - `KeyRotator` (in `runner.py`) rotates automatically on rate limits. `keys.json` is gitignored and must never be committed.

Alternatively, set a single key via environment variable (the quota might get exhausted after a few runs):
```bash
set GROQ_API_KEY=gsk_your_key_here
```
**Hugging Face** (only required for the Mistral-judge comparison condition):
```bash
set HF_TOKEN=hf_your_token_here
```

## Structure

- `agent/agent.py` - `SEALAgent`: planning (Groq), execution, intrinsic failure classification, and hint-driven recovery action selection
- `seal/judge.py` - `SEALJudge` / `JudgeFixed`: Groq-backed evaluation and rubric evolution (primary judge)
- `seal/judge_mistral.py` - Mistral/HF-backed judge, kept as a separate labeled comparison condition
- `seal/scenarios.py` - `MultiScenarioALFWorldEnv`: 20 controlled ALFWorld scenarios covering success and each forced failure mode
- `seal/task_result.py` - `TaskResult`: the shared integration contract every module reads/writes
- `seal/database.py` - SQLite schema + insert logic
- `seal/metrics.py` - success rate, failure precision, judge alignment, convergence speed, recovery rate, per-failure-type success
- `seal/figures.py` - paper figure generation (6 figures: success-by-iteration, recovery-by-failure-type, strategy frequency, rubric drift, ablation bar, max-iterations ablation)
- `seal/build_figures.py` - loads all condition summary JSONs and generates the full figure set
- `seal/reflexion_baseline.py` - Reflexion-style baseline (verbal self-reflection, no judge, no rubric evolution)
- `seal/run_baselines.py` - CLI runner for the Reflexion, Zero-Shot, No-Rubric-Evolution, and Mistral-judge conditions
- `runner.py` - `SEALRunner`: the full SEAL condition, 50-task production benchmark orchestrator
- `run_agent.py` / `run_and_check.py` - single-task and small-batch debug runners

## Running a full benchmark
```bash 
python runner.py # SEAL_FULL condition
python -m seal.run_baselines --no_rubric # No-Rubric-Evolution ablation
python -m seal.run_baselines --reflexion # Reflexion baseline
python -m seal.run_baselines --zeroshot # Zero-Shot baseline
python -m seal.run_baselines --mistral_judge # Mistral-judge comparison
python -m seal.build_figures # regenerate all figures
```

All run output (traces, summary JSONs, generated `.png` figures) is written to `seallogs/`, which is gitignored - regenerable from a fresh run, not committed. Curated figures selected for the paper live in `paper/figures/`.

## Current results

Across the same 20 ALFWorld scenarios cycled to 50 tasks, SEAL is the only condition that shows recovery on later iterations - Reflexion and No-Rubric-Evolution stay flat regardless of how many iterations are allowed, confirming that the lift comes from rubric evolution specifically, not from retrying alone.