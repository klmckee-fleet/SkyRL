"""
Task Generation Environment for SkyRL.

Single-turn BaseTextEnv where the LLM generates (prompt, verifier) for a Fleet
environment. Reward:

    R(task) = judge_gate * (variance + alpha * separation)

    judge_gate:  Binary 0/1 from LLM-as-a-judge (is the task valid and coherent?)
    variance:    Variance of verifier scores across k rollouts (difficulty calibration)
    separation:  Performance gap between strong and weak models on the task
    alpha:       Weight for separation term (default 0.5)

The evaluator runs generated tasks through Fleet harness with multiple models
to compute variance and separation.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
from typing import Any, Dict, List, Tuple

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


def _run_async(coro):
    """Run an async coroutine from sync code, even if an event loop is running."""
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


class TaskGenEnv(BaseTextEnv):
    """Environment for RL-based task generation.

    The LLM generates (prompt, verifier) pairs for Fleet environments.
    Single-turn: one generation = one task = one episode.

    Reward = judge_gate * (variance + alpha * separation)

    Constructor args (via extras, from dataset):
        env_key, env_version, data_key, data_version
        env_tools, env_tools_schema, env_variable_keys

    Constructor args (via env_config, from Hydra):
        judge_model: Model ID for LLM-as-a-judge gate
        evaluator_models: List of Fleet model IDs for rollout evaluation
        k_rollouts: Number of rollouts per model (default 4)
        alpha: Weight for separation term (default 0.5)
        max_eval_steps: Max agent steps per evaluation session (default 30)
        evaluator_timeout: Max seconds to wait for evaluation job (default 600)
    """

    def __init__(
        self,
        env_config: DictConfig,
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        # Single-turn: one generation per episode
        self.max_turns = 1

        # Environment context from dataset (extras)
        self.env_key = extras.get("env_key", "unknown")
        self.env_version = extras.get("env_version", "")
        self.data_key = extras.get("data_key", "")
        self.data_version = extras.get("data_version", "")

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

        # Parse env_variable_keys (available context variables for this env)
        env_var_keys_raw = extras.get("env_variable_keys", "[]")
        if isinstance(env_var_keys_raw, str):
            try:
                self.env_variable_keys: List[str] = json.loads(env_var_keys_raw)
            except json.JSONDecodeError:
                self.env_variable_keys: List[str] = []
        else:
            self.env_variable_keys: List[str] = env_var_keys_raw or []

        # Verifier sandbox — filters out CUA-only tool "computer" from available tools
        api_tools = set(self.env_tools) - {"computer"} if self.env_tools else None
        self.sandbox = VerifierSandbox(available_tools=api_tools if api_tools else None)

        # --- Evaluator and judge config (from env_config) ---
        self.judge_model = str(env_config.get("judge_model", "")) if env_config else ""
        self.alpha = float(env_config.get("alpha", 0.5)) if env_config else 0.5
        self.k_rollouts = int(env_config.get("k_rollouts", 4)) if env_config else 4
        self.max_eval_steps = int(env_config.get("max_eval_steps", 30)) if env_config else 30
        self.evaluator_timeout = int(env_config.get("evaluator_timeout", 600)) if env_config else 600

        # Parse evaluator_models (list of Fleet model IDs)
        eval_models_raw = env_config.get("evaluator_models", []) if env_config else []
        if isinstance(eval_models_raw, str):
            try:
                self.evaluator_models: List[str] = json.loads(eval_models_raw)
            except (json.JSONDecodeError, TypeError):
                self.evaluator_models: List[str] = [eval_models_raw] if eval_models_raw else []
        else:
            self.evaluator_models: List[str] = list(eval_models_raw) if eval_models_raw else []

        # API keys from environment
        self.fleet_api_key = os.environ.get("FLEET_API_KEY", "")
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")

        # Initialize evaluator if configured
        self.evaluator = None
        if self.evaluator_models and self.fleet_api_key:
            try:
                from envs.fleet_env.task_evaluator import TaskEvaluator

                self.evaluator = TaskEvaluator(
                    api_key=self.fleet_api_key,
                    k_rollouts=self.k_rollouts,
                    models=self.evaluator_models,
                    max_steps=self.max_eval_steps,
                    max_poll_time_s=self.evaluator_timeout,
                )
                logger.info(
                    f"TaskEvaluator initialized: models={self.evaluator_models}, "
                    f"k={self.k_rollouts}, max_steps={self.max_eval_steps}"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize TaskEvaluator: {e}")

        logger.info(
            f"TaskGenEnv: env={self.env_key}, judge={self.judge_model or 'none'}, "
            f"evaluator={'enabled' if self.evaluator else 'disabled'}, alpha={self.alpha}"
        )

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

        # Filter out CUA-only "computer" tool — task-gen is for tool-use APIs
        api_schemas = [t for t in self.env_tools_schema if t.get("function", {}).get("name") != "computer"]
        api_tool_names = [t for t in self.env_tools if t != "computer"]

        if api_schemas:
            for tool in api_schemas:
                parts.append(self._format_tool_schema(tool))
        elif api_tool_names:
            parts.append("\n".join(f"- {t}" for t in api_tool_names))
        else:
            parts.append("No tools discovered for this environment.")

        # Environment variables (user context available at task runtime)
        if self.env_variable_keys:
            parts.append("\n### Environment Variables")
            parts.append("These context variables are set at task runtime and can be referenced in prompts/verifiers:")
            for var_key in self.env_variable_keys:
                parts.append(f"- {var_key}")

        # --- B. Priors (concise, static, same for all envs) ---
        parts.append(
            """
## Verifier Guidelines

The verifier checks whether the agent completed the task by inspecting database state changes.

Signature: `def verify(env, final_answer=None) -> float` returning 1.0 (pass) or 0.0 (fail).

### Verifier API
```python
env.instance.load()              # Load current state (call first)
seed = env.db("seed")            # Original DB before agent acted
current = env.db("current")      # Current DB after agent acted

# Query tables:
rows = current.table("table_name").eq("column", value).all()
rows = current.table("table_name").neq("column", value).all()
count = current.table("table_name").eq("column", value).count()

# Compare seed vs current to detect state changes:
seed_rows = seed.table("orders").all()
current_rows = current.table("orders").all()
new_orders = [r for r in current_rows if r not in seed_rows]
```

### Rules
- Compare `seed` (before) vs `current` (after) to detect what the agent did
- Must return 0.0 on a fresh environment (before agent acts)
- Use `final_answer` for tasks that require the agent to report a value
- Don't hardcode expected values — query the DB to find them
- Reference actual tool names from this environment

## Task Guidelines

- Write as a realistic user request with concrete parameters
- Vary difficulty: some tasks need 1-2 tool calls, others 5-15+
- Don't leak the answer in the prompt
- Use the actual tool names and data entities from this environment
- Avoid underspecification: if the prompt says "find the designer in Mexico" but multiple exist, the verifier must accept all valid answers (or make the prompt specific enough)
- Avoid overspecification: specify WHAT to do, not HOW"""
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
[Python function: def verify(env, final_answer=None) -> float]
</verifier>
</task>"""
        )

        return "\n".join(parts)

    def _judge_task(self, prompt: str, verifier: str) -> float:
        """LLM-as-a-judge gate: returns 0.0 (invalid) or 1.0 (valid).

        Uses a fast model to check if the generated (prompt, verifier) pair
        is valid and coherent. This is the binary gate in the reward formula.
        """
        if not self.judge_model or not self.openrouter_api_key:
            return 1.0  # No judge configured, pass through

        # Build concise tool list for context
        tool_names = [t for t in self.env_tools if t != "computer"]
        tools_str = ", ".join(tool_names[:20]) if tool_names else "none discovered"

        judge_prompt = (
            f'Evaluate this task for the "{self.env_key}" environment.\n\n'
            f"Available tools: {tools_str}\n\n"
            f"Task prompt:\n{prompt}\n\n"
            f"Verifier code:\n```python\n{verifier}\n```\n\n"
            "A valid task must:\n"
            "1. Have a clear, specific prompt describing what an agent should do\n"
            "2. Have a verifier that checks the correct outcome via the DB API "
            '(env.db("seed"), env.db("current"), .table().eq().all())\n'
            "3. The verifier must check what the prompt actually asks\n"
            "4. The prompt must not leak the answer or expected values\n"
            "5. The verifier must return 0.0 on a fresh env (before agent acts)\n\n"
            "Answer with exactly one word: VALID or INVALID"
        )

        try:
            import litellm

            response = litellm.completion(
                model=f"openrouter/{self.judge_model}",
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
                max_tokens=10,
                api_key=self.openrouter_api_key,
            )
            answer = response.choices[0].message.content.strip().upper()
            is_valid = "VALID" in answer and "INVALID" not in answer
            logger.info(f"LLM judge: {answer} -> {'VALID' if is_valid else 'INVALID'}")
            return 1.0 if is_valid else 0.0
        except Exception as e:
            logger.warning(f"LLM judge failed, defaulting to valid: {e}")
            return 1.0

    def _evaluate_task(self, prompt: str, verifier: str) -> Dict[str, Any]:
        """Run Fleet harness evaluation to compute variance and separation.

        Submits the generated task to Fleet for k rollouts across m models.
        Returns variance (within-model score variance) and separation
        (performance gap between strongest and weakest model).
        """
        if not self.evaluator:
            return {"variance": 0.0, "separation": 0.0, "scores": {}}

        try:
            result = _run_async(
                self.evaluator.evaluate(
                    prompt=prompt,
                    verifier_code=verifier,
                    env_key=self.env_key,
                    env_version=self.env_version,
                    data_key=self.data_key or None,
                    data_version=self.data_version or None,
                )
            )

            results_per_model = result.get("results_per_model", {})
            all_scores = []
            model_means = {}

            for model_id, scores in results_per_model.items():
                if scores:
                    all_scores.extend(scores)
                    model_means[model_id] = sum(scores) / len(scores)

            # Variance across all rollouts (measures difficulty calibration)
            if len(all_scores) > 1:
                mean = sum(all_scores) / len(all_scores)
                variance = sum((s - mean) ** 2 for s in all_scores) / len(all_scores)
            else:
                variance = 0.0

            # Separation: max mean - min mean across models
            if len(model_means) > 1:
                separation = max(model_means.values()) - min(model_means.values())
            else:
                separation = 0.0

            logger.info(
                f"Evaluation: variance={variance:.4f}, separation={separation:.4f}, "
                f"scores={results_per_model}, job={result.get('job_id')}"
            )

            return {
                "variance": variance,
                "separation": separation,
                "scores": results_per_model,
                "model_means": model_means,
                "job_id": result.get("job_id"),
                "num_sessions": result.get("num_rollouts", 0),
                "num_errors": result.get("num_errors", 0),
            }
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"variance": 0.0, "separation": 0.0, "scores": {}, "error": str(e)}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Process the generated task and compute reward.

        Reward = judge_gate * (variance + alpha * separation)

        Pipeline:
            1. Parse output → fail = reward 0
            2. Sandbox validation → fail = reward 0
            3. LLM-as-a-judge → gate (0/1)
            4. Fleet evaluator → variance + separation
            5. Reward = gate * (variance + alpha * separation)
        """
        self.turns += 1
        metadata: Dict[str, Any] = {"env_key": self.env_key}

        # 1. Parse the generated task
        parsed = parse_task_output(action)
        if parsed is None:
            metadata["error"] = "parse_failed"
            metadata["reward_breakdown"] = {"total": 0.0}
            return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata=metadata)

        prompt = parsed["prompt"]
        verifier = parsed["verifier"]
        metadata["generated_prompt"] = prompt
        metadata["generated_verifier"] = verifier

        # 2. Sandbox validation (fast pre-filter)
        validation = self.sandbox.validate(verifier, prompt)
        metadata["validation"] = {
            "valid": validation.valid,
            "passed": validation.checks_passed,
            "failed": validation.checks_failed,
            "error": validation.error,
        }
        if not validation.valid:
            metadata["reward_breakdown"] = {"sandbox": 0.0, "total": 0.0}
            return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata=metadata)

        # 3. LLM-as-a-judge gate (binary 0/1)
        judge_gate = self._judge_task(prompt, verifier)
        metadata["judge_gate"] = judge_gate
        if judge_gate == 0.0:
            metadata["reward_breakdown"] = {"sandbox": 1.0, "judge": 0.0, "total": 0.0}
            return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata=metadata)

        # 4. Fleet evaluator (variance + separation)
        eval_result = self._evaluate_task(prompt, verifier)
        metadata["evaluation"] = eval_result

        variance = eval_result["variance"]
        separation = eval_result["separation"]
        reward = variance + self.alpha * separation

        metadata["reward_breakdown"] = {
            "sandbox": 1.0,
            "judge": judge_gate,
            "variance": variance,
            "separation": separation,
            "alpha": self.alpha,
            "total": reward,
        }

        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata=metadata)

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Initialize the environment with env context as the prompt.

        The dataset provides env_key/env_tools/etc via extras. We build
        the system prompt from those and return it as the initial conversation.
        """
        system_prompt = self._build_system_prompt()

        logger.info(
            f"[{self.env_key}] System prompt ({len(system_prompt)} chars, "
            f"{len(self.env_tools)} tools):\n{system_prompt}"
        )

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
