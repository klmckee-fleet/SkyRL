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

    The LLM (task generator policy) receives environment context (tool list,
    schema, example tasks) and must output a task specification consisting of
    a prompt and verifier function.

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
        env_tools: List of available tool names in the environment
        env_schema: Schema description for the environment
        example_tasks: Example tasks for few-shot context
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
        # env_tools may be a JSON string from parquet deserialization
        env_tools_raw = extras.get("env_tools", [])
        if isinstance(env_tools_raw, str):
            try:
                self.env_tools: List[str] = json.loads(env_tools_raw)
            except json.JSONDecodeError:
                self.env_tools: List[str] = []
        else:
            self.env_tools: List[str] = env_tools_raw or []
        self.env_schema: str = extras.get("env_schema", "")
        self.example_tasks: List[Dict[str, str]] = extras.get("example_tasks", [])

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

    def _build_system_prompt(self) -> str:
        """Build the system prompt with environment context."""
        tools_str = "\n".join(f"- {t}" for t in self.env_tools) if self.env_tools else "No tools listed."

        examples_str = ""
        if self.example_tasks:
            for i, ex in enumerate(self.example_tasks[:3], 1):
                examples_str += f"\n### Example {i}\n"
                examples_str += f"Prompt: {ex.get('prompt', '')}\n"
                if ex.get("verifier"):
                    examples_str += f"Verifier:\n```python\n{ex['verifier']}\n```\n"

        schema_str = self.env_schema if self.env_schema else "No schema available."

        return f"""You are a task designer for the "{self.env_key}" environment. Your job is to create a task that an AI agent would need to solve using the available tools.

## Environment: {self.env_key}

### Available Tools
{tools_str}

### Data Schema
{schema_str}

### Example Tasks
{examples_str}

## Your Output Format

Generate exactly ONE task. Output it in this format:

<task>
<prompt>
[Natural language task instruction for the agent. Be specific about what needs to be done, but don't prescribe exact tool calls. Write as a realistic user request.]
</prompt>
<verifier>
[Python async function that checks if the task was completed correctly. Must be `async def verify(env, final_answer=None):` and return 1.0 for success, 0.0 for failure. Use env to query the environment state after the agent acts.]
</verifier>
</task>

## Guidelines
- Tasks should be realistic — something a real user would ask
- Verifiers must check observable state changes, not just echo the prompt
- The verifier must return 0.0 on a fresh environment (before any agent actions)
- Use the actual tool names and data entities from this environment
- Vary difficulty: some tasks should need 1-2 tool calls, others 5-15+
- Don't hardcode expected values that mirror the prompt
- Verifiers should check end-state thoroughly enough that a model can't appear to succeed without actually completing the task"""

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

        # 3. Run evaluation (inner loop) — k rollouts × m models
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
            "num_examples": len(self.example_tasks),
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
