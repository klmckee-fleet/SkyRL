"""
Task Generation Environment for SkyRL.

Multi-turn BaseTextEnv where the LLM can explore the seed database via
``describe_db`` / ``query_db`` meta-tools before generating a task.

When ``max_turns > 1`` (the default), the model explores the DB first
and then produces a ``<task>`` block.  When ``max_turns == 1`` it
behaves identically to the original single-turn variant.

Reward:

    R(task) = llm_validity * (alpha * var(raw_scores) + (p_hint - p_raw))

    llm_validity:     Binary 0/1 from LLM-as-a-judge (is the task well-formed?)
    var(raw_scores):  Variance of k raw evaluator rollouts (difficulty calibration)
    p_hint - p_raw:   Hint gap — solvable with hints but not without (learnability)
    alpha:            Weight balancing variance vs hint gap (default 0.5)
"""

import ast
import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)
from skyrl_gym.envs.task_gen.tool_call_parser import parse_tool_call
from skyrl_gym.envs.task_gen.verifier_sandbox import (
    VerifierSandbox,
    parse_task_output,
)

logger = logging.getLogger(__name__)

# Meta-tools the model can call to explore the seed database.
_META_TOOLS = {"describe_db", "query_db"}

# All callable tools = meta-tools + any MCP env tools discovered at init time.
# Populated per-instance in init_async().


class TaskGenEnv(BaseTextEnv):
    """Environment for RL-based task generation.

    The LLM generates (prompt, verifier) pairs for Fleet environments.
    Supports multi-turn: the model can explore the seed DB via ``describe_db``
    and ``query_db`` meta-tools before outputting a ``<task>`` block.

    Reward = llm_validity * (alpha * var(raw_scores) + (p_hint - p_raw))

    Constructor args (via extras, from dataset):
        env_key, env_version, data_key, data_version
        env_tools, env_tools_schema, env_variable_keys

    Constructor args (via env_config, from Hydra):
        max_turns: Max turns before forced termination (default 10)
        judge_model: Model ID for LLM-as-a-judge gate
        k_rollouts: Number of rollouts per condition (raw/hinted, default 4)
        alpha: Weight for variance term (default 0.5)
        max_eval_steps: Max agent steps per evaluator rollout (default 30)
    """

    def __init__(
        self,
        env_config: DictConfig,
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        # Configurable multi-turn (default 10; set to 1 for single-turn)
        self.max_turns = int(env_config.get("max_turns", 10)) if env_config else 10

        # Fleet orchestrator for DB exploration (set in init_async)
        self.orch = None
        # MCP tools client for calling env tools (set in init_async)
        self.mcp_tools = None
        # Set of all callable tool names (meta-tools + MCP tools)
        self.callable_tools = set(_META_TOOLS)

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

        # Evaluator config (from Hydra env_config)
        self.k_rollouts = int(env_config.get("k_rollouts", 4)) if env_config else 4
        self.alpha = float(env_config.get("alpha", 0.5)) if env_config else 0.5
        self.max_eval_steps = int(env_config.get("max_eval_steps", 30)) if env_config else 30

        # API keys from environment variables (set by SkyPilot YAML)
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.fleet_api_key = os.environ.get("FLEET_API_KEY", "")

        logger.info(
            f"TaskGenEnv: env={self.env_key}, max_turns={self.max_turns}, "
            f"judge={self.judge_model or 'none'}, "
            f"tools={len(self.env_tools)}, k={self.k_rollouts}, alpha={self.alpha}"
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

## Task Design Guidelines

Design tasks that maximize learnability: an ideal task is one that a capable agent can solve with effort, but not trivially. Tasks that are too easy (always solved) or too hard (never solved) produce no learning signal.

### Realism
Write prompts as a real user would — natural language, concrete parameters, plausible intent. The task should sound like something a person would actually ask, not a test case.

BAD:  "Call get_user with id=5, then call update_user to set email to test@example.com"
GOOD: "Update the email address for Jamie Chen to jamie.chen@newdomain.com"

### Avoiding Underspecification
A prompt is underspecified when multiple valid solutions exist but the verifier only accepts one. This creates false negatives — the agent solves the task correctly but gets reward 0.

BAD prompt:  "Find a designer in Mexico" (3 designers exist, verifier checks for one specific one)
FIX option 1: Make the prompt specific: "Find the designer in Mexico City who joined after 2023"
FIX option 2: Make the verifier accept all valid answers: check that ANY designer in Mexico is returned

Use `describe_db`/`query_db` to check the actual data before writing the prompt. If a query returns multiple rows, either narrow the prompt or widen the verifier.

### Avoiding Overspecification
A prompt is overspecified when it dictates HOW to accomplish the task rather than WHAT outcome is needed. This makes the task trivially easy (no learning signal) and doesn't test real problem-solving.

BAD:  "First call list_tables, then call get_bookings with check_in_date='2024-03-15', then count the results and call submit_answer with the count"
GOOD: "How many bookings have a check-in date of March 15, 2024?"

The prompt should specify the desired outcome. The agent should figure out which tools to use and in what order.

### Diversity
Vary tasks across multiple dimensions:
- Operations: reads (lookup, search, aggregate) AND writes (create, update, delete)
- Complexity: simple (1-2 tool calls) through complex (5-15+ tool calls with dependencies)
- Reasoning: some tasks need multi-step logic (find X, use X to look up Y, modify Y based on Z)
- Data entities: use different tables, columns, and relationships in the schema

### Verifier-Prompt Consistency
The verifier must check exactly what the prompt asks — no more, no less. Before writing, verify:
1. Is there exactly one correct outcome for this prompt? (If not, widen the verifier or narrow the prompt)
2. Does the verifier return 0.0 on a fresh environment? (It must — the agent hasn't acted yet)
3. Does the verifier avoid hardcoded values? (Query the DB instead)
4. Could a different valid approach fool the verifier? (If so, fix the verifier to accept it)"""
        )

        # --- C. Exploration tools (multi-turn only) ---
        if self.max_turns > 1:
            parts.append(
                """
## Exploration Tools

Before generating a task, explore the environment to understand the actual data and API behavior.

### Database Tools
<tool_call>{"name": "describe_db", "arguments": {}}</tool_call>
Returns the full schema: table names, columns, types.

<tool_call>{"name": "query_db", "arguments": {"sql": "SELECT * FROM table_name LIMIT 5"}}</tool_call>
Runs a read-only SQL query against the seed database.

### Environment Tools
You can also call any of the environment's API tools listed above to see how they work:

<tool_call>{"name": "tool_name", "arguments": {"param": "value"}}</tool_call>
Calls the tool and returns its result. Use this to understand input/output formats.

### Workflow
1. Call `describe_db` to see all tables and columns.
2. Call `query_db` with SELECT queries to inspect real data (values, ranges, patterns).
3. Optionally call environment tools to understand their behavior and edge cases.
4. Use what you learned to design a realistic, data-grounded task.
5. Output the task in the format below."""
            )

        # --- D. Output format ---
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

    @staticmethod
    def _build_hint_text(
        verifier_stdout: Optional[str],
        verifier_error: Optional[str],
        tool_error_messages: Optional[List[str]],
    ) -> str:
        """Build hint text from verifier feedback. No LLM call.

        Parses ERROR_ACCUMULATOR / SUCCESS_ACCUMULATOR from verifier stdout
        and formats tool errors into structured feedback for hinted rollouts.
        """
        parts: List[str] = []

        if verifier_stdout:
            err_match = re.search(
                r">>> ERROR_ACCUMULATOR >>>\n(.+?)\n<<< ERROR_ACCUMULATOR <<<",
                verifier_stdout,
                re.DOTALL,
            )
            suc_match = re.search(
                r">>> SUCCESS_ACCUMULATOR >>>\n(.+?)\n<<< SUCCESS_ACCUMULATOR <<<",
                verifier_stdout,
                re.DOTALL,
            )
            if err_match or suc_match:
                try:
                    errors = ast.literal_eval(err_match.group(1)) if err_match else []
                    successes = ast.literal_eval(suc_match.group(1)) if suc_match else []
                except Exception:
                    errors, successes = [], []
                if successes:
                    parts.append(f"Checks passed ({len(successes)}): " + ", ".join(str(s)[:100] for s in successes[:5]))
                if errors:
                    parts.append(f"Checks failed ({len(errors)}): " + ", ".join(str(e)[:100] for e in errors[:5]))

        if verifier_error:
            parts.append(f"Verifier: {verifier_error}")

        if tool_error_messages:
            unique = list(dict.fromkeys(tool_error_messages))[:5]
            parts.append("Tool errors: " + "; ".join(e[:200] for e in unique))

        return "\n".join(parts) if parts else "The previous attempt failed. Try a different approach."

    async def _run_evaluator_rollout(
        self, prompt: str, verifier: str, hint: Optional[str] = None
    ) -> Tuple[float, Optional[str], Optional[str], Optional[List[str]]]:
        """Run a single evaluator rollout using FleetTaskEnv.

        Args:
            prompt: Task prompt.
            verifier: Verifier code.
            hint: Optional hint text to append to prompt.

        Returns:
            (score, verifier_stdout, verifier_error, tool_errors_list)
        """
        from envs.fleet_env import FleetTaskEnv

        task_prompt = prompt
        if hint:
            task_prompt += f"\n\nHere is feedback from a previous attempt to help you:\n{hint}"

        task_config = {
            "task_key": f"taskgen_{uuid.uuid4().hex[:8]}",
            "prompt": task_prompt,
            "env_key": self.env_key,
            "env_version": self.env_version or "",
            "data_key": self.data_key or "",
            "data_version": self.data_version or "",
            "verifier_code": verifier,
            "task_modality": "tool_use",
        }

        env = FleetTaskEnv(
            task_config,
            api_key=self.fleet_api_key,
            max_steps=self.max_eval_steps,
            ttl_seconds=900,
        )

        try:
            obs = await env.reset_async()
            # Immediately signal done to trigger verifier execution.
            # Without a real agent driving tool calls, we just test if the
            # verifier runs and what its baseline score is.
            obs, reward, done, info = await env.step_async({"done": True})
            score = float(reward) if reward else 0.0
            return (
                score,
                getattr(env, "verifier_stdout", None),
                getattr(env, "verifier_error", None),
                getattr(env, "tool_errors_list", None),
            )
        except Exception as e:
            logger.warning(f"Evaluator rollout failed: {e}")
            return 0.0, None, None, None
        finally:
            try:
                await env.close()
            except Exception:
                pass

    async def _evaluate_task(self, prompt: str, verifier: str) -> Dict[str, float]:
        """Run hint-based evaluation: k raw rollouts + k hinted rollouts.

        1. Run k raw rollouts via FleetTaskEnv
        2. Build hints from raw rollout feedback (verifier stdout + errors)
        3. Run k hinted rollouts via FleetTaskEnv (with hint in prompt)
        4. Compute R = alpha * var(raw) + (p_hint - p_raw)

        Returns reward breakdown dict. Returns zeros on failure.
        """
        if not self.fleet_api_key:
            return {"var_raw": 0.0, "hint_gap": 0.0, "p_raw": 0.0, "p_hint": 0.0}

        task_key = f"taskgen_{uuid.uuid4().hex[:12]}"
        start = time.time()

        try:
            # 1. Run k raw rollouts in parallel
            raw_tasks = [self._run_evaluator_rollout(prompt, verifier) for _ in range(self.k_rollouts)]
            raw_results = await asyncio.gather(*raw_tasks, return_exceptions=True)

            raw_scores: List[float] = []
            # Collect feedback from the first failing rollout for hint building
            hint_stdout: Optional[str] = None
            hint_error: Optional[str] = None
            hint_tool_errors: Optional[List[str]] = None

            for r in raw_results:
                if isinstance(r, Exception):
                    logger.warning(f"[{task_key}] Raw rollout exception: {r}")
                    raw_scores.append(0.0)
                    continue
                score, v_stdout, v_error, t_errors = r
                raw_scores.append(score)
                # Use feedback from the first failing rollout
                if score < 1.0 and hint_stdout is None:
                    hint_stdout = v_stdout
                    hint_error = v_error
                    hint_tool_errors = t_errors

            p_raw = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0

            # 2. Build hint from raw rollout feedback
            hint_text = self._build_hint_text(hint_stdout, hint_error, hint_tool_errors)

            # 3. Run k hinted rollouts in parallel
            hinted_tasks = [
                self._run_evaluator_rollout(prompt, verifier, hint=hint_text) for _ in range(self.k_rollouts)
            ]
            hinted_results = await asyncio.gather(*hinted_tasks, return_exceptions=True)

            hinted_scores: List[float] = []
            for r in hinted_results:
                if isinstance(r, Exception):
                    logger.warning(f"[{task_key}] Hinted rollout exception: {r}")
                    hinted_scores.append(0.0)
                    continue
                score, _, _, _ = r
                hinted_scores.append(score)

            p_hint = sum(hinted_scores) / len(hinted_scores) if hinted_scores else 0.0

            # 4. Compute reward components
            from integrations.fleet.task_gen_reward import compute_variance

            var_raw = compute_variance(raw_scores)
            hint_gap = p_hint - p_raw

            duration = time.time() - start
            logger.info(
                f"[{task_key}] Eval done in {duration:.0f}s: "
                f"raw={raw_scores} (p={p_raw:.2f}), hinted={hinted_scores} (p={p_hint:.2f}), "
                f"var_raw={var_raw:.4f}, hint_gap={hint_gap:.4f}"
            )

            return {
                "var_raw": var_raw,
                "hint_gap": hint_gap,
                "p_raw": p_raw,
                "p_hint": p_hint,
                "raw_scores": raw_scores,
                "hinted_scores": hinted_scores,
            }

        except Exception as e:
            logger.error(f"[{task_key}] Evaluation failed: {e}")
            return {"var_raw": 0.0, "hint_gap": 0.0, "p_raw": 0.0, "p_hint": 0.0}

    async def _handle_task_generation(self, action: str) -> BaseTextEnvStepOutput:
        """Evaluate a generated task through the full pipeline.

        Pipeline:
            1. Parse <task> output -> fail = reward 0
            2. Sandbox validation -> fail = reward 0
            3. LLM-as-a-judge -> gate (0/1), fail = reward 0
            4. Hint-based evaluation: k raw + k hinted rollouts via FleetTaskEnv
            5. R = validity * (alpha * var(raw) + (p_hint - p_raw))
        """
        metadata: Dict[str, Any] = {"env_key": self.env_key, "turn": self.turns}

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

        # 4. Reward: sandbox + judge are the primary signals for now.
        # Evaluator rollouts (k raw + k hinted via FleetTaskEnv) are skipped
        # because they require a real agent to drive tool calls — without one,
        # they always return 0 and waste ~35s per eval on Fleet instance creation.
        # TODO: re-enable evaluator once harness-based rollouts are implemented.
        base_reward = 0.3
        reward = judge_gate * base_reward

        metadata["reward_breakdown"] = {
            "sandbox": 1.0,
            "judge": judge_gate,
            "base_reward": base_reward,
            "total": reward,
        }

        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata=metadata)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Sync wrapper for step_async."""
        return asyncio.run(self.step_async(action))

    async def step_async(self, action: str) -> BaseTextEnvStepOutput:
        """Execute one step — tool call, task generation, or nudge.

        Multi-turn flow:
            1. <task> block detected  → evaluation pipeline (done=True)
            2. <tool_call> detected   → execute describe_db/query_db (done=False)
            3. Neither                → nudge observation (done=False)
            4. max_turns reached      → done=True, reward=0
        """
        self.turns += 1
        max_turns_reached = self.turns >= self.max_turns

        # 1. Check for <task> block → evaluation pipeline
        if "<task>" in action:
            return await self._handle_task_generation(action)

        # 2. Check for tool call → execute via Fleet orchestrator or MCP
        tool_call = parse_tool_call(action)
        if tool_call and tool_call["name"] in self.callable_tools:
            if tool_call["name"] in _META_TOOLS:
                obs_content = await self._execute_meta_tool(tool_call)
            else:
                obs_content = await self._execute_mcp_tool(tool_call)

            if max_turns_reached:
                return BaseTextEnvStepOutput(
                    observations=[],
                    reward=0.0,
                    done=True,
                    metadata={"env_key": self.env_key, "turn": self.turns, "done_reason": "max_turns"},
                )

            observation = {"role": "user", "content": obs_content}
            return BaseTextEnvStepOutput(
                observations=[observation],
                reward=0.0,
                done=False,
                metadata={"env_key": self.env_key, "turn": self.turns, "tool_call": tool_call},
            )

        # 3. Neither task nor tool call → nudge
        if max_turns_reached:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata={"env_key": self.env_key, "turn": self.turns, "done_reason": "max_turns"},
            )

        nudge = (
            "Use <tool_call> to explore the database or call environment tools, then generate a <task> block."
            if self.max_turns > 1
            else "No <task> block found. Output your task in <task>...</task> format."
        )
        observation = {"role": "user", "content": nudge}
        return BaseTextEnvStepOutput(
            observations=[observation],
            reward=0.0,
            done=False,
            metadata={"env_key": self.env_key, "turn": self.turns},
        )

    async def _execute_meta_tool(self, tool_call: Dict[str, Any]) -> str:
        """Execute a describe_db or query_db meta-tool call via the Fleet orchestrator."""
        name = tool_call["name"]
        args = tool_call.get("arguments", {})

        if self.orch is None:
            return "Error: Fleet environment not provisioned. Generate a <task> directly."

        try:
            if name == "describe_db":
                result = await self.orch.describe_db_async(db_name=args.get("db_name", "seed"))
            elif name == "query_db":
                sql = args.get("sql", "")
                if not sql:
                    return "Error: query_db requires a 'sql' argument."
                result = await self.orch.query_db_async(sql=sql, db_name=args.get("db_name", "seed"))
            else:
                return f"Error: Unknown meta-tool '{name}'."

            if isinstance(result, dict):
                return f"Tool result:\n{json.dumps(result, indent=2, default=str)}"
            return f"Tool result:\n{result}"
        except Exception as e:
            return f"Error: {e}"

    async def _execute_mcp_tool(self, tool_call: Dict[str, Any]) -> str:
        """Execute an MCP tool call via FleetMCPTools."""
        name = tool_call["name"]
        args = tool_call.get("arguments", {})

        if self.mcp_tools is None:
            return f"Error: MCP tools not available. Use describe_db/query_db or generate a <task>."

        try:
            result = await self.mcp_tools.call_tool(name, args)
            if isinstance(result, dict):
                return f"Tool result:\n{json.dumps(result, indent=2, default=str)}"
            return f"Tool result:\n{result}"
        except Exception as e:
            return f"Error calling {name}: {e}"

    async def init_async(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Initialize the environment, optionally provisioning a Fleet env for DB exploration.

        When ``max_turns > 1``, provisions a Fleet environment via
        ``FleetEnvClient.from_fleet_async`` so the model can call
        ``describe_db`` / ``query_db`` during exploration turns.
        Falls back to single-turn if provisioning fails.
        """
        self.turns = 0
        self.orch = None
        self.mcp_tools = None
        self.callable_tools = set(_META_TOOLS)

        # Provision Fleet env for multi-turn exploration (DB + MCP tools)
        if self.max_turns > 1 and self.fleet_api_key and self.data_key:
            try:
                from envs.fleet_env import FleetEnvClient

                self.orch, self.mcp_tools = await FleetEnvClient.from_fleet_async(
                    api_key=self.fleet_api_key,
                    env_key=self.env_key,
                    data_key=self.data_key,
                    data_version=self.data_version,
                    image_type="standard",
                    ttl_seconds=900,
                )
                # Load instance resources so db("seed") works
                await asyncio.to_thread(self.orch._fleet_env.instance.load)
                logger.info(f"TaskGenEnv [{self.env_key}]: Fleet env provisioned for DB + tool exploration")

                # Discover MCP tools so the model can call them
                if self.mcp_tools:
                    try:
                        tools_action = await self.mcp_tools.list_tools()
                        mcp_tool_names = {t["function"]["name"] for t in tools_action.tools if "function" in t}
                        # Exclude "computer" (CUA-only) from callable tools
                        mcp_tool_names.discard("computer")
                        self.callable_tools = set(_META_TOOLS) | mcp_tool_names
                        logger.info(f"TaskGenEnv [{self.env_key}]: {len(mcp_tool_names)} MCP tools available")
                    except Exception as e:
                        logger.warning(f"TaskGenEnv [{self.env_key}]: Failed to list MCP tools: {e}")
            except Exception as e:
                logger.warning(
                    f"TaskGenEnv [{self.env_key}]: Fleet provisioning failed, " f"falling back to single-turn: {e}"
                )
                self.max_turns = 1

        system_prompt = self._build_system_prompt()

        user_content = (
            f"Explore the database and then generate a task for the {self.env_key} environment."
            if self.max_turns > 1
            else f"Generate a task for the {self.env_key} environment."
        )

        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        metadata = {
            "env_key": self.env_key,
            "env_version": self.env_version,
            "num_tools": len(self.env_tools),
            "multi_turn": self.max_turns > 1,
        }

        return conversation, metadata

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """Sync wrapper for init_async."""
        return asyncio.run(self.init_async(prompt))

    def close(self):
        """Close the Fleet orchestrator if provisioned."""
        if self.orch is not None:
            try:
                self.orch.close()
            except Exception:
                pass
            self.orch = None

    async def close_async(self):
        """Async close — release Fleet orchestrator resources."""
        if self.orch is not None:
            try:
                await self.orch.close_async()
            except Exception:
                pass
            self.orch = None

    def get_metrics(self) -> Dict[str, Any]:
        """Return per-episode metrics."""
        return {
            "env_key": self.env_key,
            "turns": self.turns,
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
