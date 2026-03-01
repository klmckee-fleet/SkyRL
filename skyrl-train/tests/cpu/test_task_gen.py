"""Unit tests for task generation dataset prep and environment.

Tests the full pipeline: dataset prep -> TaskGenEnv -> system prompt.
Catches issues like empty tool schemas, missing env_variables, and
prompt construction failures before they waste GPU hours.
"""

import importlib.util
import json
import os
import tempfile
from typing import Any, Dict, List

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
# Tests: TaskGenEnv system prompt
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not SKYRL_GYM_AVAILABLE, reason="skyrl_gym not installed")
class TestTaskGenEnvPrompt:
    """Test TaskGenEnv builds correct system prompts."""

    def _make_env(self, **extra_overrides):
        from omegaconf import DictConfig
        from skyrl_gym.envs.task_gen.task_gen_env import TaskGenEnv

        env_config = DictConfig({"alpha": 0.5, "k_rollouts": 4, "models": ["weak"]})
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
        assert "async def verify" in prompt
        assert "Task Guidelines" in prompt
        assert "structural complexity" in prompt

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
