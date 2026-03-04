"""
Task Generation Environment for SkyRL.

Single-turn BaseTextEnv where the LLM generates (prompt, verifier) for a Fleet
environment. Reward:

    R(task) = validity_gate * (base_reward + variance + alpha * separation)

    validity_gate: Binary 0/1 from LLM-as-a-judge (is the task well-formed?)
    base_reward:   Small positive value (default 0.1) for cold-start bootstrap
    variance:      Score variance across k rollouts on Fleet harness
    separation:    Performance gap between strong and weak models
    alpha:         Weight for separation term (default 0.5)
"""

import asyncio
import json
import logging
import os
import time
import uuid
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


class TaskGenEnv(BaseTextEnv):
    """Environment for RL-based task generation.

    The LLM generates (prompt, verifier) pairs for Fleet environments.
    Single-turn: one generation = one task = one episode.

    Reward = validity_gate * (base_reward + variance + alpha * separation)

    Constructor args (via extras, from dataset):
        env_key, env_version, data_key, data_version
        env_tools, env_tools_schema, env_variable_keys

    Constructor args (via env_config, from Hydra):
        judge_model: Model ID for LLM-as-a-judge gate
        base_reward: Reward for passing the judge gate (default 0.1)
        evaluator_models: JSON list of Fleet model IDs for rollouts
        k_rollouts: Number of rollouts per model (default 4)
        alpha: Weight for separation term (default 0.5)
        max_eval_steps: Max agent steps per rollout (default 30)
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

        # Parse env_variables (actual values for harness evaluation)
        env_vars_raw = extras.get("env_variables", "{}")
        if isinstance(env_vars_raw, str):
            try:
                self.env_variables: Dict[str, Any] = json.loads(env_vars_raw)
            except json.JSONDecodeError:
                self.env_variables: Dict[str, Any] = {}
        else:
            self.env_variables: Dict[str, Any] = env_vars_raw or {}

        # Parse env_schema (compact DB schema: table→columns)
        self.env_schema: str = extras.get("env_schema", "") or ""

        # Verifier sandbox — filters out CUA-only tool "computer" from available tools
        api_tools = set(self.env_tools) - {"computer"} if self.env_tools else None
        self.sandbox = VerifierSandbox(available_tools=api_tools if api_tools else None)

        # Judge config (from Hydra env_config)
        self.judge_model = str(env_config.get("judge_model", "")) if env_config else ""
        self.base_reward = float(env_config.get("base_reward", 0.1)) if env_config else 0.1

        # Evaluator config (from Hydra env_config)
        # Hydra may pass evaluator_models as a ListConfig (from YAML list syntax)
        # or as a JSON string. Handle both.
        evaluator_models_raw = env_config.get("evaluator_models", []) if env_config else []
        if isinstance(evaluator_models_raw, str):
            try:
                self.evaluator_models: List[str] = json.loads(evaluator_models_raw)
            except (json.JSONDecodeError, TypeError):
                self.evaluator_models: List[str] = []
        elif isinstance(evaluator_models_raw, (list,)):
            self.evaluator_models: List[str] = list(evaluator_models_raw)
        else:
            # OmegaConf ListConfig or other iterable
            try:
                self.evaluator_models: List[str] = list(evaluator_models_raw)
            except TypeError:
                self.evaluator_models: List[str] = []
        self.k_rollouts = int(env_config.get("k_rollouts", 4)) if env_config else 4
        self.alpha = float(env_config.get("alpha", 0.5)) if env_config else 0.5
        self.max_eval_steps = int(env_config.get("max_eval_steps", 30)) if env_config else 30

        # API keys from environment variables (set by SkyPilot YAML)
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.fleet_api_key = os.environ.get("FLEET_API_KEY", "")

        logger.info(
            f"TaskGenEnv: env={self.env_key}, judge={self.judge_model or 'none'}, "
            f"base_reward={self.base_reward}, tools={len(self.env_tools)}, "
            f"evaluator_models={self.evaluator_models}, k={self.k_rollouts}, alpha={self.alpha}"
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
        if self.env_variables:
            parts.append("\n### Environment Variables")
            parts.append(
                "These variables parameterize each environment instance. "
                'Access them in verifiers via `env.env_variables["KEY"]`:'
            )
            for var_key, var_val in self.env_variables.items():
                parts.append(f"- `{var_key}` = `{var_val}`")
        elif self.env_variable_keys:
            parts.append("\n### Environment Variables")
            parts.append(
                "These variables parameterize each environment instance. "
                'Access them in verifiers via `env.env_variables["KEY"]`:'
            )
            for var_key in self.env_variable_keys:
                parts.append(f"- `{var_key}`")

        # Database schema (table names and columns)
        if self.env_schema:
            parts.append("\n### Database Schema")
            parts.append(
                "Use these exact table and column names in verifiers "
                '(e.g., `current.table("bookings").eq("guest_email", val).all()`):'
            )
            parts.append(f"```\n{self.env_schema}\n```")

        # --- B. Priors (concise, static, same for all envs) ---
        env_var_api = ""
        if self.env_variable_keys:
            example_key = self.env_variable_keys[0]
            env_var_api = f"""
# Access environment variables:
val = env.env_variables["{example_key}"]"""

        parts.append(
            f"""
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
seed_rows = seed.table("table_name").all()
current_rows = current.table("table_name").all()
new_rows = [r for r in current_rows if r not in seed_rows]{env_var_api}
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

        Uses a model to check if the generated (prompt, verifier) pair
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
            logger.info(f"LLM judge [{self.env_key}]: {answer} -> {'VALID' if is_valid else 'INVALID'}")
            return 1.0 if is_valid else 0.0
        except Exception as e:
            logger.warning(f"LLM judge failed, defaulting to valid: {e}")
            return 1.0

    async def _evaluate_task(self, prompt: str, verifier: str) -> Tuple[float, float]:
        """Run Fleet harness evaluation and compute variance + separation.

        Imports task to Fleet, creates a harness job with k rollouts per model,
        polls for completion, then computes:
        - variance: score variance across ALL rollouts (well-calibrated difficulty)
        - separation: max model mean - min model mean (discriminates capabilities)

        Returns (variance, separation). Returns (0.0, 0.0) on any failure.
        """
        if not self.evaluator_models or not self.fleet_api_key:
            return 0.0, 0.0

        task_key = f"taskgen_{uuid.uuid4().hex[:12]}"
        start = time.time()

        try:
            from fleet import Fleet
            from fleet.tasks import Task

            fleet = Fleet(api_key=self.fleet_api_key)

            # 1. Create and import task
            task = Task(
                key=task_key,
                prompt=prompt,
                env_id=self.env_key,
                version=self.env_version or None,
                verifier_func=verifier,
                data_id=self.data_key or None,
                data_version=self.data_version or None,
                env_variables=self.env_variables,
            )

            import_response = fleet.import_single_task(task)
            if import_response is None:
                logger.error(f"[{task_key}] Failed to import task to Fleet")
                return 0.0, 0.0

            logger.info(f"[{task_key}] Task imported to Fleet")

            # 2. Create harness job
            job_response = fleet.create_job(
                models=self.evaluator_models,
                task_keys=[task_key],
                pass_k=self.k_rollouts,
                max_steps=self.max_eval_steps,
                mode="tool-use",
                name=f"taskgen-eval-{task_key}",
            )
            job_id = job_response.job_id
            logger.info(
                f"[{task_key}] Harness job {job_id}: " f"models={self.evaluator_models}, pass_k={self.k_rollouts}"
            )

            # 3. Poll for completion (async — allows trajectory timeout to cancel)
            max_poll_time = 1800  # 30 min
            poll_start = time.time()
            final_status = "timeout"
            while time.time() - poll_start < max_poll_time:
                try:
                    job = fleet.get_job(job_id)
                    if job.status in ("completed", "cancelled", "errored"):
                        final_status = job.status
                        break
                except Exception as e:
                    logger.warning(f"[{task_key}] Poll error: {e}")
                await asyncio.sleep(10)

            if final_status != "completed":
                logger.warning(f"[{task_key}] Job {job_id} ended: {final_status}")
                return 0.0, 0.0

            # 4. Extract per-model scores
            results_per_model: Dict[str, List[float]] = {m: [] for m in self.evaluator_models}
            sessions_response = fleet.list_job_sessions(job_id)
            for task_group in sessions_response.tasks:
                for session in task_group.sessions:
                    # Match model ID (Fleet may strip provider prefix)
                    matched = None
                    session_bare = session.model.split("/")[-1] if "/" in session.model else session.model
                    for configured in self.evaluator_models:
                        configured_bare = configured.split("/")[-1] if "/" in configured else configured
                        if session.model == configured or session_bare == configured_bare:
                            matched = configured
                            break

                    score = 0.0
                    if session.verifier_execution and session.verifier_execution.score is not None:
                        score = float(session.verifier_execution.score)
                    elif session.verifier_execution and getattr(session.verifier_execution, "success", False):
                        score = 1.0

                    if matched and matched in results_per_model:
                        results_per_model[matched].append(score)
                    else:
                        key = matched or session.model
                        if key not in results_per_model:
                            results_per_model[key] = []
                        results_per_model[key].append(score)

            # 5. Compute variance (across ALL rollouts)
            all_scores = [s for scores in results_per_model.values() for s in scores]
            if len(all_scores) > 1:
                mean = sum(all_scores) / len(all_scores)
                variance = sum((s - mean) ** 2 for s in all_scores) / len(all_scores)
            else:
                variance = 0.0

            # 6. Compute separation (max model mean - min model mean)
            model_means = []
            for scores in results_per_model.values():
                if scores:
                    model_means.append(sum(scores) / len(scores))

            if len(model_means) >= 2:
                separation = max(model_means) - min(model_means)
            else:
                separation = 0.0

            duration = time.time() - start
            logger.info(
                f"[{task_key}] Eval done in {duration:.0f}s: "
                f"{len(all_scores)} rollouts, variance={variance:.4f}, "
                f"separation={separation:.4f}, scores={results_per_model}"
            )

            return variance, separation

        except Exception as e:
            logger.error(f"[{task_key}] Fleet evaluation failed: {e}")
            return 0.0, 0.0

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Sync step — judge gate only (no Fleet harness).

        Used as fallback when generator doesn't support async.
        R(task) = judge_gate * base_reward
        """
        self.turns += 1
        metadata: Dict[str, Any] = {"env_key": self.env_key}

        parsed = parse_task_output(action)
        if parsed is None:
            metadata["error"] = "parse_failed"
            metadata["reward_breakdown"] = {"total": 0.0}
            return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata=metadata)

        prompt = parsed["prompt"]
        verifier = parsed["verifier"]
        metadata["generated_prompt"] = prompt
        metadata["generated_verifier"] = verifier

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

        judge_gate = self._judge_task(prompt, verifier)
        metadata["judge_gate"] = judge_gate

        reward = judge_gate * self.base_reward
        metadata["reward_breakdown"] = {
            "sandbox": 1.0,
            "judge": judge_gate,
            "base_reward": self.base_reward,
            "total": reward,
        }

        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata=metadata)

    async def step_async(self, action: str) -> BaseTextEnvStepOutput:
        """Async step with Fleet harness evaluation.

        R(task) = validity_gate * (base_reward + variance + alpha * separation)

        Pipeline:
            1. Parse output -> fail = reward 0
            2. Sandbox validation -> fail = reward 0
            3. LLM-as-a-judge -> gate (0/1), fail = reward 0
            4. Fleet harness -> k rollouts per model -> variance, separation
            5. Reward = gate * (base_reward + variance + alpha * separation)
        """
        self.turns += 1
        metadata: Dict[str, Any] = {"env_key": self.env_key}

        # 1. Parse
        parsed = parse_task_output(action)
        if parsed is None:
            metadata["error"] = "parse_failed"
            metadata["reward_breakdown"] = {"total": 0.0}
            return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata=metadata)

        prompt = parsed["prompt"]
        verifier = parsed["verifier"]
        metadata["generated_prompt"] = prompt
        metadata["generated_verifier"] = verifier

        # 2. Sandbox validation
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

        # 3. LLM-as-a-judge gate
        judge_gate = self._judge_task(prompt, verifier)
        metadata["judge_gate"] = judge_gate

        if judge_gate == 0.0:
            metadata["reward_breakdown"] = {"sandbox": 1.0, "judge": 0.0, "total": 0.0}
            return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata=metadata)

        # 4. Fleet harness evaluation (async)
        variance, separation = await self._evaluate_task(prompt, verifier)
        metadata["variance"] = variance
        metadata["separation"] = separation

        # 5. R = gate * (base_reward + variance + alpha * separation)
        reward = judge_gate * (self.base_reward + variance + self.alpha * separation)

        metadata["reward_breakdown"] = {
            "sandbox": 1.0,
            "judge": judge_gate,
            "base_reward": self.base_reward,
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
