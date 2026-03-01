"""
Task Generation Environment for SkyRL.

Single-turn BaseTextEnv where the LLM generates (prompt, verifier) for a Fleet
environment. Reward is computed from validation quality:

    R(task) = validation_score  (graduated: 0.0 to 1.0 based on checks passed)

Phase 1: Validity-based reward (no inner-loop evaluation).
Phase 2 (future): Add evaluator-based reward via Fleet harness rollouts.
"""

import json
import logging
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

    The LLM (task generator policy) receives environment context (tool schemas,
    priors on verifier/task quality) and must output a task specification
    consisting of a prompt and verifier function.

    This is a single-turn environment: one generation = one task = one episode.

    Phase 1: Reward comes from graduated validity scoring (sandbox checks).
    Phase 2 (future): Add evaluator-based reward via Fleet harness rollouts.

    Constructor args (via extras, from dataset):
        env_key: Fleet environment key (e.g., "github", "booking-com")
        env_version: Fleet environment version
        env_tools: JSON string of tool name list
        env_tools_schema: JSON string of full OpenAI-format tool schemas
        env_variable_keys: JSON string of available context variable names
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

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Process the generated task and compute reward.

        Reward is a graduated validity score based on how many sandbox
        checks pass. This gives GRPO signal even without inner-loop
        evaluation (Phase 1).

        Args:
            action: LLM output containing <task><prompt>...</prompt><verifier>...</verifier></task>

        Returns:
            BaseTextEnvStepOutput with graduated validity reward.
        """
        self.turns += 1
        metadata: Dict[str, Any] = {"env_key": self.env_key}

        # 1. Parse the generated task
        parsed = parse_task_output(action)
        if parsed is None:
            metadata["error"] = "parse_failed"
            metadata["reward_breakdown"] = {"parse": 0.0, "total": 0.0}
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

        # 2. Validate via sandbox — graduated score
        validation = self.sandbox.validate(verifier, prompt)
        metadata["validation"] = {
            "valid": validation.valid,
            "passed": validation.checks_passed,
            "failed": validation.checks_failed,
            "error": validation.error,
        }

        # Graduated reward: fraction of checks passed
        # Parse success = 1 bonus check (we got past parse_task_output)
        total_checks = len(validation.checks_passed) + len(validation.checks_failed) + 1
        passed_checks = len(validation.checks_passed) + 1  # +1 for parse success
        reward = passed_checks / total_checks

        metadata["reward_breakdown"] = {
            "parse": 1.0,
            "checks_passed": len(validation.checks_passed),
            "checks_failed": len(validation.checks_failed),
            "total_checks": total_checks,
            "total": reward,
        }

        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata=metadata,
        )

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
