"""
Task Generation Environment for SkyRL.

Single-turn BaseTextEnv where the LLM generates (prompt, verifier) for a Fleet
environment. Reward is computed from rollout outcomes:

    R(task) = validity * (variance + alpha * separation)

The evaluator (inner loop) runs on Fleet infrastructure via OpenEnv's
task_evaluator module.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)
from skyrl_gym.envs.task_gen.verifier_sandbox import (
    VerifierSandbox,
    parse_task_output,
)

logger = logging.getLogger(__name__)


class TaskGenEnv(BaseTextEnv):
    """Environment for RL-based task generation.

    The LLM (task generator policy) receives environment context (tool schemas,
    priors on verifier/task quality) and must output a task specification
    consisting of a prompt and verifier function.

    This is a single-turn environment: one generation = one task = one episode.
    The reward comes from evaluating the generated task via Fleet rollouts.

    Constructor args (via env_config):
        evaluator_url: URL for the task evaluator service (OpenEnv)
        alpha: Weight for separation term in reward (default: 0.5)
        k_rollouts: Number of rollouts per model (default: 4)
        models: List of model IDs for rollout evaluation
        api_key: Fleet API key for the evaluator

    Constructor args (via extras, from dataset):
        env_key: Fleet environment key (e.g., "github", "booking-com")
        env_version: Fleet environment version
        env_tools: JSON string of tool name list
        env_tools_schema: JSON string of full OpenAI-format tool schemas
    """

    def __init__(
        self,
        env_config: DictConfig,
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        # Single-turn: one generation per episode
        self.max_turns = 1

        # Reward weights
        self.alpha = env_config.get("alpha", 0.5)

        # Evaluator configuration
        self.evaluator_url = env_config.get("evaluator_url")
        self.k_rollouts = env_config.get("k_rollouts", 4)
        self.models = env_config.get("models", ["weak"])
        self.api_key = env_config.get("api_key") or os.environ.get("FLEET_API_KEY")

        # Environment context from dataset (extras)
        self.env_key = extras.get("env_key", "unknown")
        self.env_version = extras.get("env_version", "")

        # Parse env_tools_schema (full tool schemas for prompt building)
        env_tools_schema_raw = extras.get("env_tools_schema", "[]")
        if isinstance(env_tools_schema_raw, str):
            try:
                self.env_tools_schema: List[Dict[str, Any]] = json.loads(env_tools_schema_raw)
            except json.JSONDecodeError:
                self.env_tools_schema: List[Dict[str, Any]] = []
        else:
            self.env_tools_schema: List[Dict[str, Any]] = env_tools_schema_raw or []

        # Parse env_tools (tool name list for sandbox validation)
        env_tools_raw = extras.get("env_tools", [])
        if isinstance(env_tools_raw, str):
            try:
                self.env_tools: List[str] = json.loads(env_tools_raw)
            except json.JSONDecodeError:
                self.env_tools: List[str] = []
        else:
            self.env_tools: List[str] = env_tools_raw or []

        # If env_tools is empty but we have schemas, extract names from schemas
        if not self.env_tools and self.env_tools_schema:
            self.env_tools = [
                t["function"]["name"] for t in self.env_tools_schema if "function" in t and "name" in t["function"]
            ]

        # Verifier sandbox
        self.sandbox = VerifierSandbox(available_tools=set(self.env_tools) if self.env_tools else None)

        # Evaluator instance (lazy import to avoid circular deps)
        self._evaluator = None

    def _get_evaluator(self):
        """Lazy-load the evaluator to avoid import issues at module level."""
        if self._evaluator is None:
            try:
                from envs.fleet_env.task_evaluator import TaskEvaluator

                self._evaluator = TaskEvaluator(
                    api_key=self.api_key,
                    k_rollouts=self.k_rollouts,
                    models=self.models,
                )
            except ImportError:
                logger.warning("TaskEvaluator not available. Install OpenEnv or set evaluator_url.")
        return self._evaluator

    def _format_tool_schema(self, tool: Dict[str, Any]) -> str:
        """Format a single tool schema for the system prompt."""
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        params = func.get("parameters", {})
        properties = params.get("properties", {})
        required = set(params.get("required", []))

        lines = [f"**{name}**: {desc}"]
        if properties:
            lines.append("  Parameters:")
            for pname, pschema in properties.items():
                ptype = pschema.get("type", "any")
                pdesc = pschema.get("description", "")
                req_marker = " (required)" if pname in required else ""
                lines.append(f"  - {pname} ({ptype}{req_marker}): {pdesc}")

        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        """Build the system prompt with environment context and priors."""
        parts = []

        parts.append(f'You are a task designer for the "{self.env_key}" environment.')

        # --- A. Environment context (from tool discovery) ---
        parts.append(f"\n## Environment: {self.env_key}")
        parts.append("\n### Available Tools")

        if self.env_tools_schema:
            for tool in self.env_tools_schema:
                parts.append(self._format_tool_schema(tool))
        elif self.env_tools:
            parts.append("\n".join(f"- {t}" for t in self.env_tools))
        else:
            parts.append("No tools discovered for this environment.")

        # --- B. Priors (concise, static, same for all envs) ---
        parts.append(
            """
## Verifier Guidelines

- Signature: `async def verify(env, final_answer=None) -> float` returning 1.0 (pass) or 0.0 (fail)
- Use `env` to call tools and check state — don't hardcode expected values
- Must return 0.0 on a fresh environment (before agent acts)
- Avoid underspecification: if the prompt says "find the designer in Mexico" but multiple designers exist, the verifier must accept all valid answers (or make the prompt specific enough to have one answer)
- Avoid overspecification: don't prescribe exact tool call sequences in the prompt — specify WHAT, not HOW

## Task Guidelines

- Write as a realistic user request with concrete parameters
- Vary difficulty: some tasks need 1-2 tool calls, others 5-15+
- Don't leak the answer in the prompt
- Use the actual tool names and data entities from this environment
- Prioritize structural complexity — tasks requiring genuine multi-step reasoning, not tasks exploiting a specific model's blind spot
- Prioritize environment fidelity — tasks that reflect real software workflows so skills transfer to real-world use
- Prioritize diverse coverage — broad tool/workflow coverage rather than deep exploitation of a few failure modes"""
        )

        # --- C. Output format ---
        parts.append(
            """
## Output Format

Generate exactly ONE task. Output it in this format:

<task>
<prompt>
[Natural language task instruction for the agent. Be specific about what needs to be done.]
</prompt>
<verifier>
[Python async function: async def verify(env, final_answer=None) -> float]
</verifier>
</task>"""
        )

        return "\n".join(parts)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Process the generated task and compute reward.

        Args:
            action: LLM output containing <task><prompt>...</prompt><verifier>...</verifier></task>

        Returns:
            BaseTextEnvStepOutput with reward from rollout evaluation.
        """
        return asyncio.run(self.step_async(action))

    async def step_async(self, action: str) -> BaseTextEnvStepOutput:
        """Async version of step."""
        self.turns += 1
        metadata: Dict[str, Any] = {"env_key": self.env_key}

        # 1. Parse the generated task
        parsed = parse_task_output(action)
        if parsed is None:
            metadata["error"] = "parse_failed"
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata=metadata,
            )

        prompt = parsed["prompt"]
        verifier = parsed["verifier"]
        metadata["generated_prompt"] = prompt
        metadata["generated_verifier"] = verifier

        # 2. Validate via sandbox (validity gate)
        validation = self.sandbox.validate(verifier, prompt)
        metadata["validation"] = {
            "valid": validation.valid,
            "passed": validation.checks_passed,
            "failed": validation.checks_failed,
            "error": validation.error,
        }

        if not validation.valid:
            metadata["reward_breakdown"] = {
                "validity": 0.0,
                "variance": 0.0,
                "separation": 0.0,
                "total": 0.0,
            }
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata=metadata,
            )

        # 3. Run evaluation (inner loop) — k rollouts x m models
        evaluator = self._get_evaluator()
        if evaluator is None:
            # No evaluator available — return validity-only reward
            metadata["reward_breakdown"] = {
                "validity": 1.0,
                "variance": 0.0,
                "separation": 0.0,
                "total": 0.0,
            }
            metadata["error"] = "no_evaluator"
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata=metadata,
            )

        try:
            eval_results = await evaluator.evaluate(
                prompt=prompt,
                verifier_code=verifier,
                env_key=self.env_key,
                env_version=self.env_version,
            )
        except Exception as e:
            logger.error(f"Evaluation failed for env={self.env_key}: {e}")
            metadata["error"] = f"eval_failed: {e}"
            metadata["reward_breakdown"] = {
                "validity": 1.0,
                "variance": 0.0,
                "separation": 0.0,
                "total": 0.0,
            }
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata=metadata,
            )

        # 4. Compute reward from evaluation results
        reward, breakdown = self._compute_reward(eval_results)
        metadata["eval_results"] = eval_results
        metadata["reward_breakdown"] = breakdown

        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata=metadata,
        )

    def _compute_reward(self, eval_results: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Compute composite reward from evaluation results.

        R(task) = validity * (variance + alpha * separation)

        Args:
            eval_results: Dict with 'results_per_model' mapping model_id -> list[float]

        Returns:
            (total_reward, breakdown_dict)
        """
        # Import here to avoid circular dependency
        from integrations.fleet.task_gen_reward import (
            compute_learnability,
            compute_separation,
        )

        results_per_model = eval_results.get("results_per_model", {})

        if not results_per_model:
            return 0.0, {
                "validity": 1.0,
                "variance": 0.0,
                "separation": 0.0,
                "total": 0.0,
            }

        variance = compute_learnability(results_per_model)
        separation = compute_separation(results_per_model)

        # Validity already passed (we're past the gate)
        validity = 1.0
        total = validity * (variance + self.alpha * separation)

        breakdown = {
            "validity": validity,
            "variance": variance,
            "separation": separation,
            "alpha": self.alpha,
            "total": total,
        }

        return total, breakdown

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Initialize the environment with env context as the prompt.

        The dataset provides env_key/env_tools/etc via extras. We build
        the system prompt from those and return it as the initial conversation.
        """
        system_prompt = self._build_system_prompt()

        # Build the initial conversation
        conversation = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Generate a task for the {self.env_key} environment.",
            },
        ]

        metadata = {
            "env_key": self.env_key,
            "env_version": self.env_version,
            "num_tools": len(self.env_tools),
        }

        return conversation, metadata

    def get_metrics(self) -> Dict[str, Any]:
        """Return per-episode metrics."""
        return {
            "env_key": self.env_key,
        }

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across episodes."""
        if not metrics:
            return {}

        # Group by env_key
        env_counts: Dict[str, int] = {}
        for m in metrics:
            env_key = m.get("env_key", "unknown")
            env_counts[env_key] = env_counts.get(env_key, 0) + 1

        result = {"total_episodes": len(metrics)}
        for env_key, count in env_counts.items():
            result[f"{env_key}/episodes"] = count

        return result
