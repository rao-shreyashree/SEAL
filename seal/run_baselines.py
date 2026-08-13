"""

Output files (all written to ./seallogs/):
    reflexion_baseline_summary.json       ← Fig 1 comparison condition
    zeroshot_baseline_summary.json        ← Fig 1 third condition
    no_rubric_evolution_summary.json      ← Fig 5 ablation bar
    mistral_judge_summary.json            ← Groq vs Mistral supplementary


    Every condition writes List[TaskResult.to_dict()] under task keys,
    identical shape to production_runner_summary.json from seal/runner.py.
    Shreyashree's metrics functions work on all of them without modification.

Quota notes:
    - Reflexion + ZeroShot: HF token only (Qwen via agent.py). No Groq calls.
    - NO_RUBRIC_EVOLUTION: Groq calls (evaluate only, no evolve). ~50-100 calls.
    - Mistral judge: HF token only for judge. No Groq calls.

"""

import argparse
import json
import os
import sys
import time

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from seal.scenarios import MultiScenarioALFWorldEnv
from seal.task_result import TaskResult, make_rubric_hash
from seal.reflexion_baseline import ReflexionBaseline
from agent.agent import SEALAgent, compute_plan_coherence

OUTPUT_DIR = "./seallogs"
TOTAL_SCENARIOS = 50


# ── helpers ───────────────────────────────────────────────────────────────────

def _write(filename: str, data: dict) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Written → {path}")


def _task_key(scenario_id: int) -> str:
    return f"task_{str(scenario_id + 1).zfill(3)}"


# ── Condition 1: Reflexion baseline ──────────────────────────────────────────

def run_reflexion(total: int = TOTAL_SCENARIOS) -> None:
    """
    ReflexionBaseline already exists in seal/reflexion_baseline.py.
    This just runs it at scale (50 tasks) and writes the summary JSON.
    No judge calls - Groq quota untouched.
    """
    print(f"\n=== Reflexion Baseline ({total} tasks) ===")
    baseline = ReflexionBaseline()
    all_results = {}

    for sid in range(total):
        task_key = _task_key(sid)
        env = MultiScenarioALFWorldEnv(scenario_id=sid % 20)  # 20 defined scenarios, cycle
        try:
            results = baseline.run(env=env, task_id=task_key)
            all_results[task_key] = [r.to_dict() for r in results]
            status = "✓" if any(r.success for r in results) else "✗"
            print(f"  {task_key}: {status} ({len(results)} iter)")
        except Exception as e:
            print(f"  {task_key}: SKIP — {e}")
        time.sleep(0.5)  # light throttle for HF rate limits

    _write("reflexion_baseline_summary.json", all_results)
    print(f"Reflexion baseline complete. {len(all_results)}/{total} tasks logged.")


# ── Condition 2: Zero-shot baseline ──────────────────────────────────────────

# Fixed rubric passed to agent.plan() so it gets a coherent prompt —
# identical to ReflexionBaseline.BASELINE_RUBRIC so the plan prompt
# is exactly the same. The ONLY difference vs reflexion: no reflection,
# no retry. Single attempt, done.
_ZEROSHOT_RUBRIC = (
    "Always approach structures sequentially. "
    "Verify containers are open before placement."
)

def _run_zeroshot_task(agent: SEALAgent, env, task_id: str) -> TaskResult:
    """Single-attempt, no reflection, no rubric evolution."""
    goal, _ = env.reset()
    plan = agent.plan(task=goal, rubric=_ZEROSHOT_RUBRIC)
    trace_output = agent.execute(plan=plan, env=env, rubric=_ZEROSHOT_RUBRIC)

    is_success = trace_output["final_outcome"] == "SUCCESS"
    trajectory = trace_output["trajectory"]
    total_steps = len(trajectory)
    stagnant = sum(1 for s in trajectory if s["internal_loop_alert"] is not None)
    unique_actions = len(set(s["action_executed"] for s in trajectory))

    return TaskResult(
        task_id=task_id,
        iteration=1,                          # always 1 — zero-shot means one shot
        strategy_used=trace_output["macro_plan"],
        failure_type=trace_output["detected_failure_type"],
        score=1.0 if is_success else 0.0,
        success=is_success,
        rubric_version=1,
        rubric_hash=make_rubric_hash(_ZEROSHOT_RUBRIC),
        raw_trace=trajectory,
        task_description=goal,
        # ground truth from env — normalize SUCCESS→NONE same as runner.py
        oracle_failure_type=(
            "NONE" if env.data["forced_outcome"] == "SUCCESS"
            else env.data["forced_outcome"]
        ),
        agent_confidence=trace_output["agent_intrinsic_confidence"],
        plan_coherence=trace_output["plan_coherence"],
        total_steps=total_steps,
        judge_score=None,           # no judge in zero-shot
        judge_failure_type=None,    # no judge in zero-shot
        judge_explanation=None,     # no judge in zero-shot
        rubric_drift_score=0.0,     # nothing evolves
        stagnation_step_count=stagnant,
        trajectory_stagnation_rate=(
            round(stagnant / total_steps, 2) if total_steps > 0 else 0.0
        ),
        unique_action_count=unique_actions,
        action_density_index=(
            round(unique_actions / total_steps, 2) if total_steps > 0 else 0.0
        ),
        drift_recovered=False,
        strategy_label="none",      # no strategy — that's the whole point
        rubric_text=_ZEROSHOT_RUBRIC,
    )


def run_zeroshot(total: int = TOTAL_SCENARIOS) -> None:
    """
    Zero-shot: one plan, one execute, no retry, no reflection, no judge.
    Bare agent performance — lower bound for Fig 1.
    No Groq calls.
    """
    print(f"\n=== Zero-Shot Baseline ({total} tasks) ===")
    agent = SEALAgent()
    all_results = {}

    for sid in range(total):
        task_key = _task_key(sid)
        env = MultiScenarioALFWorldEnv(scenario_id=sid % 20)
        try:
            result = _run_zeroshot_task(agent=agent, env=env, task_id=task_key)
            all_results[task_key] = [result.to_dict()]   # list for schema consistency
            status = "✓" if result.success else "✗"
            print(f"  {task_key}: {status}")
        except Exception as e:
            print(f"  {task_key}: SKIP — {e}")
        time.sleep(0.5)

    _write("zeroshot_baseline_summary.json", all_results)
    print(f"Zero-shot baseline complete. {len(all_results)}/{total} tasks logged.")


# ── Condition 3: NO_RUBRIC_EVOLUTION ablation ────────────────────────────────

def run_no_rubric_evolution(total: int = TOTAL_SCENARIOS) -> None:
    """
    Uses SEALRunner with condition='NO_RUBRIC_EVOLUTION' (JudgeFixed).
    Runner already supports this — just needs to be invoked and logged.
    Uses Groq (evaluate calls only, no evolve). ~50-150 Groq calls total.
    """
    print(f"\n=== NO_RUBRIC_EVOLUTION Ablation ({total} tasks) ===")

    # Import here so missing Groq key doesn't crash other conditions
    from seal.runner import SEALRunner

    runner = SEALRunner(
        condition="NO_RUBRIC_EVOLUTION",
        output_dir=os.path.join(OUTPUT_DIR, "no_rubric_traces"),
    )
    all_results = {}
    request_count = 0

    for sid in range(total):
        task_key = _task_key(sid)
        try:
            results, calls = runner.run_task_lifecycle(scenario_id=sid % 20, task_id=task_key)
            request_count += calls
            all_results[task_key] = [r.to_dict() for r in results]
            status = "✓" if any(r.success for r in results) else "✗"
            print(f"  {task_key}: {status} ({calls} judge calls)")
        except Exception as e:
            print(f"  {task_key}: SKIP — {e}")
        time.sleep(3)  # Groq rate limit buffer

    _write("no_rubric_evolution_summary.json", all_results)
    print(f"NO_RUBRIC_EVOLUTION complete. {len(all_results)}/{total} tasks. "
          f"Total Groq calls: {request_count}")


# ── Condition 4: Groq vs Mistral judge ablation ────────────────────────────

def run_mistral_judge(total: int = TOTAL_SCENARIOS) -> None:
    """
    Full SEAL run but with judge_mistral.SEALJudge instead of seal.judge.SEALJudge.
    Same agent, same env, same rubric evolution logic — only the judge model changes.
    No Groq calls. Uses HF_TOKEN for both agent (Qwen) and judge (Mistral).

    NOTE: judge_mistral.evaluate() returns a raw dict, not EvalResult.
    Use evaluate_to_result() to normalize before building TaskResult.
    """
    print(f"\n=== Groq vs Mistral Judge Ablation ({total} tasks) ===")

    from seal.judge import DEFAULT_RUBRIC, trace_to_str
    from seal.judge_mistral import SEALJudge as MistralJudge, evaluate_to_result

    agent = SEALAgent()
    env = MultiScenarioALFWorldEnv()
    judge = MistralJudge()
    all_results = {}

    for sid in range(total):
        task_key = _task_key(sid)
        env.set_scenario(sid % 20)
        goal, _ = env.reset()

        active_rubric = DEFAULT_RUBRIC.copy()
        task_results = []
        failure_history_buffer = []

        try:
            for iteration in range(1, 4):
                env.set_scenario(sid % 20)
                rubric_str = json.dumps(active_rubric, indent=2)
                plan = agent.plan(task=goal, rubric=rubric_str)
                trace_output = agent.execute(plan=plan, env=env, rubric=rubric_str)

                is_success = trace_output["final_outcome"] == "SUCCESS"
                trajectory = trace_output["trajectory"]
                total_steps = len(trajectory)
                stagnant = sum(1 for s in trajectory if s["internal_loop_alert"] is not None)
                unique_actions = len(set(s["action_executed"] for s in trajectory))
                agent_failure_type = trace_output["detected_failure_type"]

                formatted_trace = trace_to_str(trajectory)
                raw_eval = judge.evaluate(trace=formatted_trace, rubric=active_rubric)
                eval_result = evaluate_to_result(raw_eval)   # normalize to EvalResult

                failure_history_buffer.append(raw_eval)

                if is_success:
                    strategy_label = "none"
                elif agent_failure_type == "CONTEXT_LOSS":
                    strategy_label = "meta_reflection"
                else:
                    strategy_label = "iterative_prompting"

                raw_ft = getattr(eval_result, "failure_type", None)
                ft_str = raw_ft.value if raw_ft else "none"

                result = TaskResult(
                    task_id=task_key,
                    iteration=iteration,
                    strategy_used=trace_output["macro_plan"],
                    failure_type=agent_failure_type,
                    score=1.0 if is_success else 0.0,
                    success=is_success,
                    rubric_version=iteration,
                    rubric_hash=make_rubric_hash(rubric_str),
                    raw_trace=trajectory,
                    task_description=goal,
                    oracle_failure_type=(
                        "NONE" if env.data["forced_outcome"] == "SUCCESS"
                        else env.data["forced_outcome"]
                    ),
                    agent_confidence=trace_output["agent_intrinsic_confidence"],
                    plan_coherence=trace_output["plan_coherence"],
                    total_steps=total_steps,
                    stagnation_step_count=stagnant,
                    trajectory_stagnation_rate=(
                        round(stagnant / total_steps, 2) if total_steps > 0 else 0.0
                    ),
                    unique_action_count=unique_actions,
                    action_density_index=(
                        round(unique_actions / total_steps, 2) if total_steps > 0 else 0.0
                    ),
                    judge_score=eval_result.score,
                    judge_failure_type=ft_str,
                    judge_explanation=eval_result.explanation,
                    drift_recovered=trace_output.get("drift_recovered", False),
                    strategy_label=strategy_label,
                    rubric_text=rubric_str,
                )
                task_results.append(result)

                if is_success:
                    break

                if not is_success and iteration < 3 and iteration >= 2:
                    try:
                        new_rubric, _, was_updated = judge.evolve_rubric(
                            rubric=active_rubric,
                            failure_history=failure_history_buffer,
                        )
                        if was_updated and isinstance(new_rubric, dict):
                            active_rubric = new_rubric
                    except Exception as e:
                        print(f"  [{task_key} iter {iteration}] evolve skip: {e}")
                        break

            all_results[task_key] = [r.to_dict() for r in task_results]
            status = "✓" if any(r.success for r in task_results) else "✗"
            print(f"  {task_key}: {status} ({len(task_results)} iter)")

        except Exception as e:
            print(f"  {task_key}: SKIP — {e}")

        time.sleep(0.5)

    _write("mistral_judge_summary.json", all_results)
    print(f"Mistral judge ablation complete. {len(all_results)}/{total} tasks logged.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SEAL comparison conditions for paper figures."
    )
    parser.add_argument("--reflexion",    action="store_true", help="Reflexion baseline (50 tasks)")
    parser.add_argument("--zeroshot",     action="store_true", help="Zero-shot baseline (50 tasks)")
    parser.add_argument("--no_rubric",    action="store_true", help="NO_RUBRIC_EVOLUTION ablation")
    parser.add_argument("--mistral_judge",action="store_true", help="Groq vs Mistral judge ablation")
    parser.add_argument("--all",          action="store_true", help="Run all four conditions")
    parser.add_argument("--tasks",        type=int, default=50, help="Number of tasks (default 50)")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    if args.all or args.reflexion:
        run_reflexion(total=args.tasks)
    if args.all or args.zeroshot:
        run_zeroshot(total=args.tasks)
    if args.all or args.no_rubric:
        run_no_rubric_evolution(total=args.tasks)
    if args.all or args.mistral_judge:
        run_mistral_judge(total=args.tasks)