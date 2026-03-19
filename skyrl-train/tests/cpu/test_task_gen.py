"""Unit tests for task generation dataset prep and environment.

Tests the full pipeline: dataset prep -> TaskGenEnv -> system prompt.
Catches issues like empty tool schemas, missing env_variables, and
prompt construction failures before they waste GPU hours.
"""

import asyncio
import importlib.util
import json
import os
import tempfile
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

SKYRL_GYM_AVAILABLE = importlib.util.find_spec("skyrl_gym") is not None

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products by keyword",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "Add a product to the shopping cart",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Product ID to add",
                    },
                },
                "required": ["product_id"],
            },
        },
    },
]


def make_sample_tasks(
    env_key: str = "testenv",
    count: int = 5,
    data_key: str = "kinesis",
    data_version: str = "v0.0.1",
    env_variables: Dict[str, str] = None,
) -> Dict[str, Any]:
    """Create a sample tasks JSON structure."""
    if env_variables is None:
        env_variables = {"LOGGED_IN_USER": "testuser", "CURRENT_DATE": "2026-01-15"}

    tasks = []
    for i in range(count):
        tasks.append(
            {
                "key": f"{env_key}_task_{i}",
                "prompt": f"Do something #{i} in {env_key}",
                "env_key": env_key,
                "data_key": data_key,
                "data_version": data_version,
                "env_variables": env_variables,
                "verifier_code": f"async def verify(env, final_answer=None):\n    result = await env.call_tool('check', {{}})\n    return 1.0 if result else 0.0\n# padding to meet min length requirement for task {i}",
            }
        )
    return {"tasks": tasks}


# ---------------------------------------------------------------------------
# Tests: prepare_task_gen_dataset.py
# ---------------------------------------------------------------------------


class TestCollectEnvMetadata:
    """Test _collect_env_metadata extracts correct per-env info."""

    def test_extracts_data_key_and_version(self):
        from integrations.fleet.prepare_task_gen_dataset import _collect_env_metadata

        tasks_by_env = {
            "github": [
                {"data_key": "kinesis", "data_version": "v0.0.7", "env_variables": {}},
            ],
        }
        meta = _collect_env_metadata(tasks_by_env)
        assert meta["github"]["data_key"] == "kinesis"
        assert meta["github"]["data_version"] == "v0.0.7"

    def test_collects_env_variable_keys_across_tasks(self):
        from integrations.fleet.prepare_task_gen_dataset import _collect_env_metadata

        tasks_by_env = {
            "booking": [
                {"data_key": "k", "data_version": "v1", "env_variables": {"LOGGED_IN_NAME": "Alice"}},
                {
                    "data_key": "k",
                    "data_version": "v1",
                    "env_variables": {"LOGGED_IN_NAME": "Bob", "CURRENT_DATE": "2026-01-01"},
                },
                {"data_key": "k", "data_version": "v1", "env_variables": {}},
            ],
        }
        meta = _collect_env_metadata(tasks_by_env)
        assert sorted(meta["booking"]["env_variable_keys"]) == ["CURRENT_DATE", "LOGGED_IN_NAME"]

    def test_handles_missing_env_variables(self):
        from integrations.fleet.prepare_task_gen_dataset import _collect_env_metadata

        tasks_by_env = {
            "wallst": [
                {"data_key": "k", "data_version": "v1"},
            ],
        }
        meta = _collect_env_metadata(tasks_by_env)
        assert meta["wallst"]["env_variable_keys"] == []

    def test_handles_json_string_env_variables(self):
        from integrations.fleet.prepare_task_gen_dataset import _collect_env_metadata

        tasks_by_env = {
            "test": [
                {"data_key": "k", "data_version": "v1", "env_variables": '{"FOO": "bar"}'},
            ],
        }
        meta = _collect_env_metadata(tasks_by_env)
        assert meta["test"]["env_variable_keys"] == ["FOO"]


class TestBuildGRPODataset:
    """Test build_task_gen_dataset_grpo output records."""

    def test_records_have_required_fields(self):
        from integrations.fleet.prepare_task_gen_dataset import build_task_gen_dataset_grpo

        with tempfile.TemporaryDirectory() as tmpdir:
            tasks_path = os.path.join(tmpdir, "tasks.json")
            with open(tasks_path, "w") as f:
                json.dump(make_sample_tasks(), f)

            out_dir = os.path.join(tmpdir, "output")
            build_task_gen_dataset_grpo(
                tasks_json=tasks_path,
                output_dir=out_dir,
                discover_tools=False,
                max_tasks=5,
            )

            # Read back parquet
            from datasets import Dataset

            train_path = os.path.join(out_dir, "train.parquet")
            assert os.path.exists(train_path), "train.parquet not created"

            ds = Dataset.from_parquet(train_path)
            record = ds[0]

            required_fields = [
                "prompt",
                "env_class",
                "data_source",
                "task_key",
                "env_key",
                "env_version",
                "env_tools",
                "env_tools_schema",
                "env_variable_keys",
            ]
            for field in required_fields:
                assert field in record, f"Missing field: {field}"

    def test_env_variable_keys_populated(self):
        from integrations.fleet.prepare_task_gen_dataset import build_task_gen_dataset_grpo

        with tempfile.TemporaryDirectory() as tmpdir:
            tasks_path = os.path.join(tmpdir, "tasks.json")
            with open(tasks_path, "w") as f:
                json.dump(
                    make_sample_tasks(env_variables={"LOGGED_IN_USER": "alice", "CURRENT_DATE": "2026-01-01"}),
                    f,
                )

            out_dir = os.path.join(tmpdir, "output")
            build_task_gen_dataset_grpo(
                tasks_json=tasks_path,
                output_dir=out_dir,
                discover_tools=False,
            )

            from datasets import Dataset

            ds = Dataset.from_parquet(os.path.join(out_dir, "train.parquet"))
            record = ds[0]
            var_keys = json.loads(record["env_variable_keys"])
            assert "LOGGED_IN_USER" in var_keys
            assert "CURRENT_DATE" in var_keys

    def test_no_discover_tools_gives_empty_schemas(self):
        from integrations.fleet.prepare_task_gen_dataset import build_task_gen_dataset_grpo

        with tempfile.TemporaryDirectory() as tmpdir:
            tasks_path = os.path.join(tmpdir, "tasks.json")
            with open(tasks_path, "w") as f:
                json.dump(make_sample_tasks(), f)

            out_dir = os.path.join(tmpdir, "output")
            build_task_gen_dataset_grpo(
                tasks_json=tasks_path,
                output_dir=out_dir,
                discover_tools=False,
            )

            from datasets import Dataset

            ds = Dataset.from_parquet(os.path.join(out_dir, "train.parquet"))
            record = ds[0]
            tools = json.loads(record["env_tools"])
            schemas = json.loads(record["env_tools_schema"])
            assert tools == []
            assert schemas == []

    def test_no_example_tasks_in_records(self):
        """GRPO records should NOT contain example_tasks (removed by design)."""
        from integrations.fleet.prepare_task_gen_dataset import build_task_gen_dataset_grpo

        with tempfile.TemporaryDirectory() as tmpdir:
            tasks_path = os.path.join(tmpdir, "tasks.json")
            with open(tasks_path, "w") as f:
                json.dump(make_sample_tasks(count=10), f)

            out_dir = os.path.join(tmpdir, "output")
            build_task_gen_dataset_grpo(
                tasks_json=tasks_path,
                output_dir=out_dir,
                discover_tools=False,
            )

            from datasets import Dataset

            ds = Dataset.from_parquet(os.path.join(out_dir, "train.parquet"))
            for record in ds:
                assert "example_tasks" not in record

    def test_all_tasks_become_records(self):
        """Every task should become a record (no example holdout in GRPO)."""
        from integrations.fleet.prepare_task_gen_dataset import build_task_gen_dataset_grpo

        with tempfile.TemporaryDirectory() as tmpdir:
            tasks_path = os.path.join(tmpdir, "tasks.json")
            with open(tasks_path, "w") as f:
                json.dump(make_sample_tasks(count=10), f)

            out_dir = os.path.join(tmpdir, "output")
            build_task_gen_dataset_grpo(
                tasks_json=tasks_path,
                output_dir=out_dir,
                discover_tools=False,
            )

            from datasets import Dataset

            train_ds = Dataset.from_parquet(os.path.join(out_dir, "train.parquet"))
            val_path = os.path.join(out_dir, "validation.parquet")
            val_count = len(Dataset.from_parquet(val_path)) if os.path.exists(val_path) else 0
            assert len(train_ds) + val_count == 10


# ---------------------------------------------------------------------------
# Tests: Reward computation
# ---------------------------------------------------------------------------


class TestVariance:
    """Test compute_variance for raw rollout scores."""

    def test_max_variance_at_half(self):
        from integrations.fleet.task_gen_reward import compute_variance

        scores = [1.0, 0.0, 1.0, 0.0]
        var = compute_variance(scores)
        assert var == 0.25, f"Expected 0.25 (max Bernoulli), got {var}"

    def test_zero_variance_all_pass(self):
        from integrations.fleet.task_gen_reward import compute_variance

        assert compute_variance([1.0, 1.0, 1.0, 1.0]) == 0.0

    def test_zero_variance_all_fail(self):
        from integrations.fleet.task_gen_reward import compute_variance

        assert compute_variance([0.0, 0.0, 0.0, 0.0]) == 0.0

    def test_single_rollout_returns_zero(self):
        from integrations.fleet.task_gen_reward import compute_variance

        assert compute_variance([1.0]) == 0.0

    def test_empty_returns_zero(self):
        from integrations.fleet.task_gen_reward import compute_variance

        assert compute_variance([]) == 0.0


class TestHintGap:
    """Test compute_hint_gap measures hint improvement."""

    def test_positive_hint_gap(self):
        from integrations.fleet.task_gen_reward import compute_hint_gap

        gap = compute_hint_gap([0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0])
        assert gap == 0.5  # p_hint=0.75 - p_raw=0.25

    def test_zero_gap_same_performance(self):
        from integrations.fleet.task_gen_reward import compute_hint_gap

        gap = compute_hint_gap([1.0, 0.0], [1.0, 0.0])
        assert gap == 0.0

    def test_negative_gap_hints_hurt(self):
        from integrations.fleet.task_gen_reward import compute_hint_gap

        gap = compute_hint_gap([1.0, 1.0], [0.0, 0.0])
        assert gap == -1.0

    def test_empty_raw_returns_zero(self):
        from integrations.fleet.task_gen_reward import compute_hint_gap

        assert compute_hint_gap([], [1.0, 0.0]) == 0.0

    def test_empty_hinted_returns_zero(self):
        from integrations.fleet.task_gen_reward import compute_hint_gap

        assert compute_hint_gap([1.0, 0.0], []) == 0.0


class TestCompositeReward:
    """Test compute_task_reward end-to-end."""

    def test_full_reward(self):
        from integrations.fleet.task_gen_reward import compute_task_reward

        # raw: p=0.25 (1/4), var=0.1875; hinted: p=0.75; gap=0.5
        result = compute_task_reward(
            raw_scores=[0.0, 0.0, 1.0, 0.0],
            hinted_scores=[1.0, 1.0, 1.0, 0.0],
            validity=1.0,
            alpha=0.5,
        )
        expected = 1.0 * (0.5 * 0.1875 + 0.5)
        assert abs(result["total"] - expected) < 1e-6
        assert result["var_raw"] == 0.1875
        assert result["hint_gap"] == 0.5

    def test_validity_gate_kills_reward(self):
        from integrations.fleet.task_gen_reward import compute_task_reward

        result = compute_task_reward(
            raw_scores=[0.0, 0.0, 1.0, 0.0],
            hinted_scores=[1.0, 1.0, 1.0, 0.0],
            validity=0.0,
        )
        assert result["total"] == 0.0

    def test_trivial_task_no_gap(self):
        from integrations.fleet.task_gen_reward import compute_task_reward

        # All pass raw — trivial, no hint gap, no variance
        result = compute_task_reward(
            raw_scores=[1.0, 1.0, 1.0, 1.0],
            hinted_scores=[1.0, 1.0, 1.0, 1.0],
        )
        assert result["total"] == 0.0
        assert result["var_raw"] == 0.0
        assert result["hint_gap"] == 0.0


# ---------------------------------------------------------------------------
# Tests: TaskGenEnv system prompt
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SKYRL_GYM_AVAILABLE, reason="skyrl_gym not installed")
class TestTaskGenEnvPrompt:
    """Test TaskGenEnv builds correct system prompts."""

    def _make_env(self, env_config_overrides=None, **extra_overrides):
        from omegaconf import DictConfig
        from skyrl_gym.envs.task_gen.task_gen_env import TaskGenEnv

        cfg = {"alpha": 0.5, "k_rollouts": 4, "models": ["weak"], "max_turns": 1}
        if env_config_overrides:
            cfg.update(env_config_overrides)
        env_config = DictConfig(cfg)
        extras = {
            "env_key": "testenv",
            "env_version": "v1",
            "env_tools": json.dumps(["search_products", "add_to_cart"]),
            "env_tools_schema": json.dumps(SAMPLE_TOOL_SCHEMAS),
            "env_variable_keys": json.dumps(["LOGGED_IN_USER", "CURRENT_DATE"]),
        }
        extras.update(extra_overrides)
        return TaskGenEnv(env_config=env_config, extras=extras)

    def test_system_prompt_contains_tool_names(self):
        env = self._make_env()
        prompt = env._build_system_prompt()
        assert "search_products" in prompt
        assert "add_to_cart" in prompt

    def test_system_prompt_contains_tool_descriptions(self):
        env = self._make_env()
        prompt = env._build_system_prompt()
        assert "Search for products by keyword" in prompt
        assert "Add a product to the shopping cart" in prompt

    def test_system_prompt_contains_tool_parameters(self):
        env = self._make_env()
        prompt = env._build_system_prompt()
        assert "query" in prompt
        assert "product_id" in prompt
        assert "(required)" in prompt

    def test_system_prompt_contains_env_variables(self):
        env = self._make_env()
        prompt = env._build_system_prompt()
        assert "LOGGED_IN_USER" in prompt
        assert "CURRENT_DATE" in prompt
        assert "Environment Variables" in prompt

    def test_system_prompt_no_env_variables_section_when_empty(self):
        env = self._make_env(env_variable_keys=json.dumps([]))
        prompt = env._build_system_prompt()
        assert "Environment Variables" not in prompt

    def test_system_prompt_contains_priors(self):
        env = self._make_env()
        prompt = env._build_system_prompt()
        assert "Verifier Guidelines" in prompt
        assert "def verify(env" in prompt
        assert "Task Design Guidelines" in prompt

    def test_system_prompt_contains_output_format(self):
        env = self._make_env()
        prompt = env._build_system_prompt()
        assert "<task>" in prompt
        assert "<prompt>" in prompt
        assert "<verifier>" in prompt

    def test_system_prompt_no_empty_tools_message_when_tools_exist(self):
        env = self._make_env()
        prompt = env._build_system_prompt()
        assert "No tools discovered" not in prompt

    def test_system_prompt_shows_no_tools_when_schemas_empty(self):
        env = self._make_env(
            env_tools=json.dumps([]),
            env_tools_schema=json.dumps([]),
        )
        prompt = env._build_system_prompt()
        assert "No tools discovered" in prompt

    def test_env_tools_extracted_from_schemas(self):
        """If env_tools is empty but schemas exist, names should be extracted."""
        env = self._make_env(env_tools=json.dumps([]))
        assert "search_products" in env.env_tools
        assert "add_to_cart" in env.env_tools

    def test_init_returns_non_empty_system_prompt(self):
        """init() must return a conversation with a non-empty system prompt."""
        env = self._make_env()
        conversation, metadata = env.init(prompt=[])
        system_msg = conversation[0]
        assert system_msg["role"] == "system"
        assert len(system_msg["content"]) > 100, (
            f"System prompt too short ({len(system_msg['content'])} chars). " "Tool schemas likely missing."
        )
        assert "search_products" in system_msg["content"]

    def test_init_prompt_not_empty_string(self):
        """Regression: system prompt must never be empty string at init time."""
        env = self._make_env()
        conversation, _ = env.init(prompt=[])
        system_content = conversation[0]["content"]
        assert system_content != "", "System prompt is empty! Model will hallucinate tools."
        assert system_content.strip() != "", "System prompt is whitespace-only!"


# ---------------------------------------------------------------------------
# Tests: parse_tool_call (from tool_call_parser.py)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SKYRL_GYM_AVAILABLE, reason="skyrl_gym not installed")
class TestParseToolCall:
    """Tests for tool_call_parser.parse_tool_call."""

    def test_valid_tool_call(self):
        from skyrl_gym.envs.task_gen.tool_call_parser import parse_tool_call

        action = '<tool_call>{"name": "describe_db", "arguments": {}}</tool_call>'
        result = parse_tool_call(action)
        assert result is not None
        assert result["name"] == "describe_db"
        assert result["arguments"] == {}

    def test_tool_call_with_arguments(self):
        from skyrl_gym.envs.task_gen.tool_call_parser import parse_tool_call

        action = '<tool_call>{"name": "query_db", "arguments": {"sql": "SELECT * FROM users LIMIT 5"}}</tool_call>'
        result = parse_tool_call(action)
        assert result is not None
        assert result["name"] == "query_db"
        assert result["arguments"]["sql"] == "SELECT * FROM users LIMIT 5"

    def test_missing_closing_tag(self):
        """Stop string </tool_call> may not be in output."""
        from skyrl_gym.envs.task_gen.tool_call_parser import parse_tool_call

        action = '<tool_call>{"name": "describe_db", "arguments": {}}'
        result = parse_tool_call(action)
        assert result is not None
        assert result["name"] == "describe_db"

    def test_invalid_json(self):
        from skyrl_gym.envs.task_gen.tool_call_parser import parse_tool_call

        action = "<tool_call>{not valid json}</tool_call>"
        result = parse_tool_call(action)
        assert result is None

    def test_no_match(self):
        from skyrl_gym.envs.task_gen.tool_call_parser import parse_tool_call

        action = "I will now generate a task."
        result = parse_tool_call(action)
        assert result is None

    def test_json_with_missing_braces(self):
        """Models sometimes drop trailing braces."""
        from skyrl_gym.envs.task_gen.tool_call_parser import parse_tool_call

        action = '<tool_call>{"name": "query_db", "arguments": {"sql": "SELECT 1"}'
        result = parse_tool_call(action)
        assert result is not None
        assert result["name"] == "query_db"

    def test_function_call_tag(self):
        from skyrl_gym.envs.task_gen.tool_call_parser import parse_tool_call

        action = '<function_call>{"name": "describe_db", "arguments": {}}</function_call>'
        result = parse_tool_call(action)
        assert result is not None
        assert result["name"] == "describe_db"

    def test_tool_key_normalization(self):
        """'tool' key should be normalized to 'name'."""
        from skyrl_gym.envs.task_gen.tool_call_parser import parse_tool_call

        action = '<tool_call>{"tool": "query_db", "params": {"sql": "SELECT 1"}}</tool_call>'
        result = parse_tool_call(action)
        assert result is not None
        assert result["name"] == "query_db"
        assert result["arguments"] == {"sql": "SELECT 1"}


# ---------------------------------------------------------------------------
# Tests: Multi-turn TaskGenEnv
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SKYRL_GYM_AVAILABLE, reason="skyrl_gym not installed")
class TestTaskGenEnvMultiTurn:
    """Tests for multi-turn task generation (DB exploration + task output)."""

    def _make_env(self, max_turns=5, **extra_overrides):
        from omegaconf import DictConfig
        from skyrl_gym.envs.task_gen.task_gen_env import TaskGenEnv

        env_config = DictConfig({"alpha": 0.5, "k_rollouts": 4, "models": ["weak"], "max_turns": max_turns})
        extras = {
            "env_key": "testenv",
            "env_version": "v1",
            "data_key": "kinesis",
            "data_version": "v0.0.1",
            "env_tools": json.dumps(["search_products", "add_to_cart"]),
            "env_tools_schema": json.dumps(SAMPLE_TOOL_SCHEMAS),
            "env_variable_keys": json.dumps(["LOGGED_IN_USER"]),
        }
        extras.update(extra_overrides)
        return TaskGenEnv(env_config=env_config, extras=extras)

    def test_max_turns_from_env_config(self):
        env = self._make_env(max_turns=7)
        assert env.max_turns == 7

    def test_default_max_turns(self):
        from omegaconf import DictConfig
        from skyrl_gym.envs.task_gen.task_gen_env import TaskGenEnv

        env_config = DictConfig({})
        extras = {"env_key": "test"}
        env = TaskGenEnv(env_config=env_config, extras=extras)
        assert env.max_turns == 10

    def test_system_prompt_includes_meta_tools_when_multi_turn(self):
        env = self._make_env(max_turns=5)
        prompt = env._build_system_prompt()
        assert "Database Exploration Tools" in prompt
        assert "describe_db" in prompt
        assert "query_db" in prompt

    def test_system_prompt_excludes_meta_tools_when_single_turn(self):
        env = self._make_env(max_turns=1)
        prompt = env._build_system_prompt()
        assert "Database Exploration Tools" not in prompt

    def test_step_async_tool_call_returns_observation(self):
        """Tool call should return observation with done=False."""
        env = self._make_env(max_turns=5)

        # Mock the orchestrator
        mock_orch = MagicMock()
        mock_orch.describe_db_async = AsyncMock(
            return_value={
                "success": True,
                "tables": [{"name": "users", "columns": [{"name": "id", "type": "INTEGER"}]}],
            }
        )
        env.orch = mock_orch

        action = '<tool_call>{"name": "describe_db", "arguments": {}}</tool_call>'
        result = asyncio.run(env.step_async(action))

        assert result["done"] is False
        assert result["reward"] == 0.0
        assert len(result["observations"]) == 1
        assert "Tool result:" in result["observations"][0]["content"]
        assert "users" in result["observations"][0]["content"]

    def test_step_async_query_db(self):
        """query_db tool call should execute and return results."""
        env = self._make_env(max_turns=5)

        mock_orch = MagicMock()
        mock_orch.query_db_async = AsyncMock(
            return_value={
                "success": True,
                "columns": ["id", "name"],
                "rows": [[1, "Alice"], [2, "Bob"]],
            }
        )
        env.orch = mock_orch

        action = '<tool_call>{"name": "query_db", "arguments": {"sql": "SELECT * FROM users"}}</tool_call>'
        result = asyncio.run(env.step_async(action))

        assert result["done"] is False
        assert "Alice" in result["observations"][0]["content"]
        assert "Bob" in result["observations"][0]["content"]

    def test_step_async_task_after_exploration(self):
        """After tool calls, a <task> block should trigger evaluation (done=True)."""
        env = self._make_env(max_turns=5)

        # Mock orch for exploration turn
        mock_orch = MagicMock()
        mock_orch.describe_db_async = AsyncMock(return_value={"success": True, "tables": []})
        env.orch = mock_orch

        # Turn 1: explore
        result1 = asyncio.run(env.step_async('<tool_call>{"name": "describe_db", "arguments": {}}</tool_call>'))
        assert result1["done"] is False
        assert env.turns == 1

        # Turn 2: generate task
        task_action = """<task>
<prompt>Search for a product called "Widget" and add it to the cart.</prompt>
<verifier>
def verify(env, final_answer=None):
    env.instance.load()
    current = env.db("current")
    cart_items = current.table("cart_items").all()
    if not cart_items:
        return 0.0
    for item in cart_items:
        if "widget" in str(item.get("name", "")).lower():
            return 1.0
    return 0.0
</verifier>
</task>"""

        result2 = asyncio.run(env.step_async(task_action))
        assert result2["done"] is True
        assert env.turns == 2

    def test_step_async_max_turns_on_tool_call(self):
        """When max_turns is reached during a tool call, done=True."""
        env = self._make_env(max_turns=1)

        mock_orch = MagicMock()
        mock_orch.describe_db_async = AsyncMock(return_value={"success": True, "tables": []})
        env.orch = mock_orch

        action = '<tool_call>{"name": "describe_db", "arguments": {}}</tool_call>'
        result = asyncio.run(env.step_async(action))

        assert result["done"] is True
        assert result["reward"] == 0.0
        assert result["metadata"]["done_reason"] == "max_turns"

    def test_step_async_max_turns_on_nudge(self):
        """When max_turns is reached with no tool call/task, done=True."""
        env = self._make_env(max_turns=1)

        action = "I'm thinking about what task to generate..."
        result = asyncio.run(env.step_async(action))

        assert result["done"] is True
        assert result["reward"] == 0.0
        assert result["metadata"]["done_reason"] == "max_turns"

    def test_step_async_nudge_message(self):
        """Without tool call or task, env should nudge the model."""
        env = self._make_env(max_turns=5)

        action = "Let me think about this..."
        result = asyncio.run(env.step_async(action))

        assert result["done"] is False
        assert (
            "tool_call" in result["observations"][0]["content"].lower()
            or "<task>" in result["observations"][0]["content"]
        )

    def test_step_async_no_orch_returns_error(self):
        """Tool call without provisioned orch should return error message."""
        env = self._make_env(max_turns=5)
        assert env.orch is None

        action = '<tool_call>{"name": "describe_db", "arguments": {}}</tool_call>'
        result = asyncio.run(env.step_async(action))

        assert result["done"] is False
        assert "not provisioned" in result["observations"][0]["content"]

    def test_step_async_query_db_missing_sql(self):
        """query_db without sql argument should return error."""
        env = self._make_env(max_turns=5)
        mock_orch = MagicMock()
        env.orch = mock_orch

        action = '<tool_call>{"name": "query_db", "arguments": {}}</tool_call>'
        result = asyncio.run(env.step_async(action))

        assert result["done"] is False
        assert "requires" in result["observations"][0]["content"].lower()

    def test_close_async(self):
        """close_async should call orch.close_async and set orch to None."""
        env = self._make_env(max_turns=5)
        mock_orch = MagicMock()
        mock_orch.close_async = AsyncMock()
        env.orch = mock_orch

        asyncio.run(env.close_async())

        mock_orch.close_async.assert_called_once()
        assert env.orch is None

    def test_close_async_handles_error(self):
        """close_async should not raise on orch.close_async failure."""
        env = self._make_env(max_turns=5)
        mock_orch = MagicMock()
        mock_orch.close_async = AsyncMock(side_effect=Exception("conn error"))
        env.orch = mock_orch

        asyncio.run(env.close_async())
        assert env.orch is None

    def test_close_async_noop_without_orch(self):
        """close_async should be a noop when orch is None."""
        env = self._make_env(max_turns=5)
        assert env.orch is None
        asyncio.run(env.close_async())
        assert env.orch is None
