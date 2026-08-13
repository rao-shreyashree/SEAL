import json
import os
import time
import sys

# Force absolute path inclusion for sub-module relative namespace mapping
root_path = os.path.dirname(os.path.abspath(__file__))

if root_path not in sys.path:
    sys.path.insert(0, root_path)

from agent.agent import SEALAgent  
from seal.scenarios import MultiScenarioALFWorldEnv  
from seal.task_result import TaskResult, make_rubric_hash
from seal.judge import JudgeFixed, SEALJudge, DEFAULT_RUBRIC, trace_to_str


# Groq key rotation
# load from keys.json
# format: {"keys": ["gsk_...", "gsk_...", ...]}
# each key = one Groq account's free-tier quota (1000 RPD, 30 RPM)
# auto-switches when a key hits the per-key call ceiling (KEY_CALL_CEILING)
# SEALJudge/JudgeFixed read GROQ_API_KEY from env - _rotate_key() 
# patches os.environ so the judge picks up the new key without reinit

KEY_CALL_CEILING = 900  # Groq free tier: 1000 RPD/key

def _load_keys(path: str = "keys.json") -> list:
    if not os.path.exists(path):
        key = os.environ.get("GROQ_API_KEY", "")
        return [key] if key else []
    with open(path) as f:
        return json.load(f)["keys"]

class KeyRotator:
    """
    Patching os.environ is the only way to hot-swap the key without
    reinitializing SEALJudge - judge reads GROQ_API_KEY at call time via
    os.environ.get() in judge.py, not at __init__ time.
    """
    def __init__(self, keys: list):
        if not keys:
            raise ValueError("No Groq API keys found. Add keys.json or set GROQ_API_KEY env var.")
        self.keys = keys
        self.index = 0
        self.calls_on_current_key = 0
        self._total_calls_made = 0
        self._apply_current()

    def _apply_current(self):
        os.environ["GROQ_API_KEY"] = self.keys[self.index]

    def tick(self):
        """Call once per judge API call. Rotates key if ceiling hit."""
        self.calls_on_current_key += 1
        self._total_calls_made += 1
        if self.calls_on_current_key >= KEY_CALL_CEILING:
            if self.index + 1 >= len(self.keys):
                raise RuntimeError(
                    f"[KEY EXHAUSTED] All {len(self.keys)} keys have hit their ceiling "
                    f"({KEY_CALL_CEILING} calls each). Rotate in new keys or wait 24h."
                )
            self.index += 1
            self.calls_on_current_key = 0
            self._apply_current()
            print(f"[KEY ROTATED] Switched to key index {self.index} "
                  f"(key ...{self.keys[self.index][-6:]})")

    def force_rotate(self, reason: str = ""):
        """Immediate rotation on a real API error (e.g. 429 TPD/RPD limit),
        instead of waiting for the proactive call-count ceiling in tick().
        tick() alone doesn't catch this - this counts calls, not tokens, so a
        token-per-day limit can be hit well before KEY_CALL_CEILING trips."""
        if self.index + 1 >= len(self.keys):
            raise RuntimeError(
                f"[KEY EXHAUSTED] All {len(self.keys)} keys exhausted. {reason}"
            )
        self._total_calls_made += self.calls_on_current_key 
        self.index += 1
        self.calls_on_current_key = 0
        self._apply_current()
        print(f"[KEY FORCE-ROTATED] {reason} -> switched to key index {self.index} "
              f"(key ...{self.keys[self.index][-6:]})")

    @property
    def total_calls(self) -> int:
        return self._total_calls_made + self.calls_on_current_key

    def status(self) -> dict:
        return {
            "active_key_index": self.index,
            "calls_on_current_key": self.calls_on_current_key,
            "total_keys": len(self.keys),
            "total_calls": self.total_calls,
        }


class SEALRunner:
    def __init__(self, condition: str = "SEAL_FULL", output_dir: str = "./seallogs",
                 rotator: KeyRotator = None):
        self.agent = SEALAgent()
        self.env = MultiScenarioALFWorldEnv()
        self.condition = condition
        self.output_dir = output_dir
        self.rotator = rotator  # shared across the full run, not per-task
        os.makedirs(self.output_dir, exist_ok=True)
        
        if condition == "NO_RUBRIC_EVOLUTION":
            self.judge = JudgeFixed(rotator=self.rotator) 
        else:
            self.judge = SEALJudge(rotator=self.rotator)

    def run_task_lifecycle(self, scenario_id: int, task_id: str = None) -> tuple[list, int]:
        self.env.set_scenario(scenario_id)
        goal, _ = self.env.reset()
        
        active_rubric = DEFAULT_RUBRIC.copy()
        if task_id is None:
            task_id = f"task_{str(scenario_id + 1).zfill(3)}"
        task_iteration_history = []
        failure_history_buffer = []
        calls_made = 0

        print(f"Running task {task_id} under [{self.condition}] condition...")

        for iteration in range(1, 4):
            self.env.set_scenario(scenario_id)

            # Strip _seal_hints before serializing — hints are agent-internal
            # signals, must not leak into TaskResult.rubric_text,
            # judge.evaluate()'s prompt, or rubric_hash.
            #
            # The live active_rubric dict keeps the hints so agent.execute()
            # can read them via _read_rubric_hints(rubric).

            rubric_for_logging = {
                k: v
                for k, v in active_rubric.items()
                if k != "_seal_hints"
            }

            rubric_string_representation = json.dumps(
                rubric_for_logging,
                indent=2,
            )

            action_plan = self.agent.plan(
                task=goal,
                rubric=rubric_string_representation,
            )

            # IMPORTANT:
            # Agent receives active_rubric, including _seal_hints,
            # because Scenario A requires the agent to react to the
            # substantive rubric evolution signal.
            trace_output = self.agent.execute(
                plan=action_plan,
                env=self.env,
                rubric=active_rubric,
            )

            is_success = (
                trace_output["final_outcome"] == "SUCCESS"
            )

            trajectory = trace_output["trajectory"]
            total_steps = len(trajectory)

            stagnant_steps = sum(
                1
                for s in trajectory
                if s["internal_loop_alert"] is not None
            )

            unique_actions = len(
                set(
                    s["action_executed"]
                    for s in trajectory
                )
            )

            stagnation_rate = (
                round(stagnant_steps / total_steps, 2)
                if total_steps > 0
                else 0.0
            )

            agent_failure_type = trace_output[
                "detected_failure_type"
            ]

            formatted_trace_str = trace_to_str(
                trajectory
            )

            # IMPORTANT:
            # Judge receives rubric_for_logging, NOT active_rubric.
            #
            # This keeps _seal_hints internal to the agent. The judge
            # evaluates against the actual rubric criteria and does not
            # see the behavioral control signal that was derived from it.
            evaluation_report = self.judge.evaluate(
                trace=formatted_trace_str,
                rubric=rubric_for_logging,
            )

            calls_made += 1

            if self.rotator:
                self.rotator.tick()

            if is_success:
                strategy_label = "none"
            elif agent_failure_type == "CONTEXT_LOSS":
                strategy_label = "meta_reflection"
            else:
                strategy_label = "iterative_prompting"

            raw_failure_enum = getattr(evaluation_report, "failure_type", None)
            extracted_failure_str = raw_failure_enum.value if raw_failure_enum else "none"

            result = TaskResult(
                task_id=task_id,
                iteration=iteration,
                strategy_used=trace_output["macro_plan"],
                failure_type=agent_failure_type,
                score=1.0 if is_success else 0.0,
                success=is_success,
                rubric_version=iteration,
                rubric_hash=make_rubric_hash(rubric_string_representation),
                raw_trace=trajectory,
                task_description=goal,

                # Normalize: env uses "SUCCESS" string;
                # contract field uses "NONE" on success.
                #
                # "SUCCESS" as oracle_failure_type breaks Fig 2
                # grouping for successful tasks.
                oracle_failure_type=(
                    "NONE"
                    if self.env.data["forced_outcome"] == "SUCCESS"
                    else self.env.data["forced_outcome"]
                ),

                agent_confidence=trace_output[
                    "agent_intrinsic_confidence"
                ],

                plan_coherence=trace_output[
                    "plan_coherence"
                ],

                total_steps=total_steps,

                stagnation_step_count=stagnant_steps,

                trajectory_stagnation_rate=stagnation_rate,

                unique_action_count=unique_actions,

                action_density_index=(
                    round(
                        unique_actions / total_steps,
                        2,
                    )
                    if total_steps > 0
                    else 0.0
                ),

                judge_score=getattr(
                    evaluation_report,
                    "score",
                    0.0,
                ),

                judge_failure_type=extracted_failure_str,

                judge_explanation=getattr(
                    evaluation_report,
                    "explanation",
                    "",
                ),

                drift_recovered=trace_output.get(
                    "drift_recovered",
                    False,
                ),

                strategy_label=strategy_label,
                rubric_text=rubric_string_representation,
            )
            result.rubric_drift_score = 0.0  # default; overwritten below if evolve ran

            log_filename = os.path.join(self.output_dir, f"trace_{self.condition}_{task_id}_iter_{iteration}.json")
            with open(log_filename, "w") as f:
                f.write(result.to_json())

            task_iteration_history.append(result)
            failure_history_buffer.append(evaluation_report)

            if is_success:
                break

            # evolve only after iteration 2 failure, not iteration 1
            # this cuts evolve calls 2->1 per failing task
            # evaluate() is never skipped
            if not is_success and iteration < 3 and iteration >= 2:
                try:
                    new_rubric, similarity_score, was_updated = (
                        self.judge.evolve_rubric(
                            rubric=active_rubric,
                            failure_history=failure_history_buffer,
                        )
                    )

                    calls_made += 1

                    if self.rotator:
                        self.rotator.tick()

                    result.rubric_drift_score = (
                        similarity_score
                    )

                    if (
                        was_updated
                        and isinstance(new_rubric, dict)
                    ):
                        active_rubric = new_rubric

                except ImportError as ie:
                    print(
                        f"[{task_id} Iteration {iteration} "
                        f"Rubric Drift Bypass]: {ie}"
                    )
                    break

        return task_iteration_history, calls_made


def run_comprehensive_suite(max_calls: int = 200):
    # 50-task full run, no scope reduction
    # max_calls is a quota guard - stops before burning through
    # key rotation budget

    total_scenarios = 50  # 5 will produce a partial run. our benchmark is 50 tasks

    rotator = KeyRotator(
        _load_keys()
    )

    runner = SEALRunner(
        condition="SEAL_FULL",
        rotator=rotator,
    )

    all_results = {}
    per_task_calls = {}  # per-task call visibility for quota planning

    print("=== Launching SEAL Runner Production Benchmark ===")
    print(f"Keys loaded: {len(rotator.keys)} | Ceiling per key: {KEY_CALL_CEILING} | Max calls: {max_calls}")

    for sid in range(total_scenarios):
        if rotator.total_calls >= max_calls:
            print(f"[STOPPED] Hit max_calls budget ({max_calls}) at scenario {sid}. "
                  f"Rotate keys or raise max_calls to continue.")
            break
        try:
            task_key = (
                f"task_{str(sid + 1).zfill(3)}"
            )

            results, calls = runner.run_task_lifecycle(
                scenario_id=sid % 20,
                task_id=task_key,
            )

            all_results[task_key] = [
                r.to_dict()
                for r in results
            ]

            per_task_calls[task_key] = calls
            print(f"  {task_key}: {calls} calls | key status: {rotator.status()}")
            time.sleep(2)  # Groq free tier: 30 RPM -> 2s

        except RuntimeError as e:
            # KeyRotator raises RuntimeError when all keys exhausted
            print(f"[KEY EXHAUSTION] {e}")
            break

        except Exception as e:
            print(f"Skipping scenario {sid}: {e}")

    with open(
        os.path.join(
            runner.output_dir,
            "production_runner_summary.json",
        ),
        "w",
    ) as f:
        json.dump(
            all_results,
            f,
            indent=2,
        )

    with open(
        os.path.join(
            runner.output_dir,
            "production_call_counts.json",
        ),
        "w",
    ) as f:
        json.dump(
            per_task_calls,
            f,
            indent=2,
        )

    print(
        f"\nEvaluation summary: "
        f"{runner.output_dir}/production_runner_summary.json"
    )

    print(
        f"Per-task call counts: "
        f"{runner.output_dir}/production_call_counts.json"
    )

    print(
        f"Final key status: "
        f"{rotator.status()}"
    )


if __name__ == "__main__":
    run_comprehensive_suite(max_calls=1000)