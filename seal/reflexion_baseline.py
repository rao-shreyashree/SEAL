"""
Reflexion baseline - the non-SEAL comparison condition.

No judge here. Agent attempts the task, reflects on its own trajectory in
plain language, and retries using that reflection as memory. Keeping the
output schema-identical to what Tanisha's SEALRunner produces (same
TaskResult, NO new fields) so I can run my metric functions and figures
across all 3 conditions without separate code paths.

How I'm reusing existing TaskResult fields here (since there's no judge,
the judge-only fields would otherwise just sit empty):
  - judge_score / judge_failure_type -> always None. No judge in this condition.
  - judge_explanation -> repurposing this to hold the agent's OWN self-reflection text per iteration. Not a judge writing here, it's the agent talking to itself.
  - rubric_version / rubric_hash -> DO NOT TOUCH / let these change across iterations. 
        No rubric evolution happens in this condition by design - that's the whole point of the ablation. 
        Relying on this staying flat to show ~0 drift here vs SEAL's nonzero drift. 
  - rubric_drift_score -> always 0.0, nothing evolves so there's nothing to measure.

IMPORTANT
do NOT route any TaskResult from this file through SEALJudge.evaluate(). 
It'll overwrite judge_explanation with real judge output and quietly break the repurposing above. 
These rows are baseline-only. 
Anagha: do NOT use this in the judge pipeline.

Execution reuses Tanisha's SEALAgent.execute() as-is - same env stepping, same action policy, untouched. 
Only the adaptation signal changes (verbal reflection vs judge + rubric evolution), which is what keeps this a fair ablation instead of comparing two different agents.
"""

from agent.agent import SEALAgent
from seal.task_result import TaskResult, make_rubric_hash

# Fixed, never-evolved rubric - same text Tanisha's run_agent.py uses, kept as a constant here
# so rubric_hash never changes across iterations
BASELINE_RUBRIC = (
    "Always approach structures sequentially. "
    "Verify containers are open before placement."
)


class ReflexionBaseline:
    """Attempt -> verbal self-reflection -> retry. No judge, no rubric evolution."""

    def __init__(self, hf_token=None, max_iterations: int = 3):
        self.agent = SEALAgent(hf_token)
        self.max_iterations = max_iterations
        self.rubric = BASELINE_RUBRIC
        self.rubric_version = 1
        self.rubric_hash = make_rubric_hash(self.rubric)

    def _plan_with_memory(self, task: str, memory: str) -> str:
        """Same Mistral call as SEALAgent.plan(), but injects reflection memory
        instead of an evolving rubric. The rubric text itself is never touched."""
        if not memory:
            return self.agent.plan(task=task, rubric=self.rubric)

        prompt = (
            f"[INST] You are a household task planning agent.\n"
            f"Rubric: {self.rubric}\n"
            f"Reflection from your previous attempt: {memory}\n"
            f"Task: {task}\n\n"
            f"Produce a numbered step-by-step action plan to complete this task, "
            f"taking your previous reflection into account. "
            f"Each step must be a single executable action such as "
            f"'go to <object>', 'open <object>', 'put <item> in <container>', or "
            f"'examine <item> using <object>'. Output ONLY the numbered plan, no preamble. [/INST]"
        )
        try:
            response = self.agent.client.text_generation(
                prompt, max_new_tokens=256, temperature=0.3, do_sample=True,
            )
            return response.strip()
        except Exception as e:
            return (
                f"1. Go to container\n2. Open container\n3. Place item\n"
                f"[FALLBACK - Mistral unavailable: {e}]"
            )

    def _reflect(self, task: str, trace_output: dict) -> str:
        """Agent reflects verbally on its own failed trajectory. Same model, same
        voice as the agent itself - no separate evaluator persona, no judge."""
        trajectory_summary = "\n".join(
            f"Step {s['step']}: action='{s['action_executed']}' -> '{s['observation_received']}'"
            for s in trace_output["trajectory"]
        )
        prompt = (
            f"[INST] You just attempted the following task and failed:\n"
            f"Task: {task}\n"
            f"Your plan was:\n{trace_output['macro_plan']}\n\n"
            f"What you actually did:\n{trajectory_summary}\n\n"
            f"In 2-3 sentences, reflect on what went wrong and what you should do "
            f"differently next time. Be specific and actionable. [/INST]"
        )
        try:
            response = self.agent.client.text_generation(
                prompt, max_new_tokens=128, temperature=0.3, do_sample=True,
            )
            return response.strip()
        except Exception as e:
            return f"[FALLBACK reflection - Mistral unavailable: {e}]"

    def run(self, env, task_id: str) -> list:
        """Runs up to self.max_iterations attempts on one task. Returns
        List[TaskResult] - one per iteration, same shape SEALRunner produces,
        so Shreyashree's metrics functions and convergence curves work unmodified."""
        results = []
        memory = ""

        for iteration in range(1, self.max_iterations + 1):
            goal, _ = env.reset()
            plan = self._plan_with_memory(task=goal, memory=memory)
            trace_output = self.agent.execute(plan=plan, env=env, rubric=self.rubric)

            is_success = trace_output["final_outcome"] == "SUCCESS"
            trajectory = trace_output["trajectory"]
            total_steps = len(trajectory)
            stagnant_steps = sum(
                1 for s in trajectory if s["internal_loop_alert"] is not None
            )
            unique_actions = len(set(s["action_executed"] for s in trajectory))
            stagnation_rate = (
                round(stagnant_steps / total_steps, 2) if total_steps > 0 else 0.0
            )

            reflection_text = "" if is_success else self._reflect(goal, trace_output)

            result = TaskResult(
                task_id=task_id,
                iteration=iteration,
                strategy_used=trace_output["macro_plan"],
                failure_type=trace_output["detected_failure_type"],
                score=1.0 if is_success else 0.0,
                success=is_success,
                rubric_version=self.rubric_version,
                rubric_hash=self.rubric_hash, # constant - no evolution
                raw_trace=trajectory,
                task_description=goal,
                oracle_failure_type="NONE" if is_success else env.data["forced_outcome"],
                agent_confidence=trace_output["agent_intrinsic_confidence"],
                plan_coherence=trace_output["plan_coherence"],
                total_steps=total_steps,
                judge_score=None, # no judge in this condition
                judge_failure_type=None, # no judge in this condition
                judge_explanation=reflection_text or None,  # repurposed: self-reflection text
                rubric_drift_score=0.0, # nothing evolves to drift
                stagnation_step_count=stagnant_steps,
                trajectory_stagnation_rate=stagnation_rate,
                unique_action_count=unique_actions,
                action_density_index=(
                    round(unique_actions / total_steps, 2) if total_steps > 0 else 0.0
                ),
            )
            results.append(result)

            if is_success:
                break
            memory = reflection_text

        return results


if __name__ == "__main__":
    # Quick manual smoke test - mirrors agent/run_agent.py's pattern
    from agent.scenarios import MultiScenarioALFWorldEnv

    env = MultiScenarioALFWorldEnv(scenario_id=4)  # CONTEXT_LOSS scenario
    baseline = ReflexionBaseline()
    task_results = baseline.run(env, task_id="task_baseline_001")

    for r in task_results:
        print(
            f"Iteration {r.iteration}: success={r.success}, "
            f"failure_type={r.failure_type}, reflection={r.judge_explanation}"
        )