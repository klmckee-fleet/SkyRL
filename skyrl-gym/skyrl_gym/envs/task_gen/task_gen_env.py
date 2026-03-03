"""
Task Generation Environment for SkyRL.

Single-turn BaseTextEnv where the LLM generates (prompt, verifier) for a Fleet
environment. Reward:

    R(task) = judge_gate * base_reward

    judge_gate:  Binary 0/1 from LLM-as-a-judge (is the task valid and coherent?)
    base_reward: Positive value for passing (default 0.1). Provides GRPO signal
                 via variance across samples (some pass judge, some don't).
"""

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


class TaskGenEnv(BaseTextEnv):
    """Environment for RL-based task generation.

    The LLM generates (prompt, verifier) pairs for Fleet environments.
    Single-turn: one generation = one task = one episode.

    Reward = judge_gate * base_reward

    Constructor args (via extras, from dataset):
        env_key, env_version, data_key, data_version
        env_tools, env_tools_schema, env_variable_keys

    Constructor args (via env_config, from Hydra):
        judge_model: Model ID for LLM-as-a-judge gate (e.g. "anthropic/claude-sonnet-4-6")
        base_reward: Reward for passing the judge gate (default 0.1)
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

        # Judge config (from Hydra env_config)
        self.judge_model = str(env_config.get("judge_model", "")) if env_config else ""
        self.base_reward = float(env_config.get("base_reward", 0.1)) if env_config else 0.1

        # API key from environment variable (set by SkyPilot YAML)
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")

        logger.info(
            f"TaskGenEnv: env={self.env_key}, judge={self.judge_model or 'none'}, "
            f"base_reward={self.base_reward}, tools={len(self.env_tools)}"
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

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Process the generated task and compute reward.

        Reward = judge_gate * base_reward

        Pipeline:
            1. Parse output -> fail = reward 0
            2. Sandbox validation -> fail = reward 0
            3. LLM-as-a-judge -> gate (0/1)
            4. Reward = gate * base_reward
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

        # 4. Reward = gate * base_reward
        reward = judge_gate * self.base_reward

        metadata["reward_breakdown"] = {
            "sandbox": 1.0,
            "judge": judge_gate,
            "base_reward": self.base_reward,
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
