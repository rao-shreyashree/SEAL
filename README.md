# SEAL - Self-Evaluating Agent Loop

SEAL is a modular, open-source framework for self-improving AI agents. 
The core idea: an agent attempts a task, a separate LLM-based judge evaluates the outcome, and both the agent and judge iteratively improve through a Plan → Act → Reflect → Judge feedback loop.

The judge doesn't just score - it evolves its own rubric based on observed failure patterns hallucination, context loss, goal drift, execution errors), making evaluation smarter over terations.

Built and tested on ALFWorld, WebArena, and SWE-Bench. Uses open-source models (Mistral 7B, LLaMA 3 8B) via HuggingFace.

## Team
- Tanisha: Agent loop (SEALAgent)
- Anagha: Judge (SEALJudge)
- Shreyashree: Schema, metrics, database (TaskResult, SQLite)

## Structure
- `seal/` - core modules
- `notebooks/` - Colab notebooks
- `data/` - ALFWorld tasks
- `results/` - SQLite DB output

## Week 1 Status
- TaskResult dataclass
- SQLite schema
- Metric functions
- Colab notebook skeleton
- Zero-shot baseline (waiting for Tanisha's agent)