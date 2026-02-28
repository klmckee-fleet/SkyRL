"""
Prepare SFT warmup dataset for task generation.

Converts existing Fleet tasks into (env_context, task_spec) pairs for
supervised fine-tuning of the task generator model.

Each training example:
    Input: Environment context (tool list, schema, example tasks)
    Output: Task spec (<task><prompt>...</prompt><verifier>...</verifier></task>)

Usage:
    python -m integrations.fleet.prepare_task_gen_dataset \
        --tasks-json ~/data/fleet/all_tool_use.json \
        --output-dir ./data/task_gen \
        --env-context-dir ./configs/environments
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from datasets import Dataset


def load_tasks(json_path: str) -> List[Dict[str, Any]]:
    """Load tasks from Fleet export JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "tasks" in data:
        return data["tasks"]
    else:
        raise ValueError("Invalid JSON format")


def load_env_context(env_context_dir: str, env_key: str) -> Optional[Dict[str, Any]]:
    """Load environment context from YAML/JSON config.

    Returns dict with 'tools', 'schema', 'description' keys if available.
    """
    for ext in [".yaml", ".yml", ".json"]:
        path = os.path.join(env_context_dir, f"{env_key}{ext}")
        if os.path.exists(path):
            if ext == ".json":
                with open(path) as f:
                    return json.load(f)
            else:
                try:
                    import yaml

                    with open(path) as f:
                        return yaml.safe_load(f)
                except ImportError:
                    pass
    return None


def format_task_spec(task: Dict[str, Any]) -> str:
    """Format a task as the expected output format for SFT.

    Returns the <task><prompt>...</prompt><verifier>...</verifier></task> string.
    """
    prompt = task.get("prompt", "")
    verifier = task.get("verifier_func") or task.get("verifier_code", "")

    return f"""<task>
<prompt>
{prompt}
</prompt>
<verifier>
{verifier}
</verifier>
</task>"""


def format_env_context_prompt(
    env_key: str,
    tools: List[str],
    schema: str = "",
    example_tasks: List[Dict[str, str]] = [],
) -> str:
    """Build the system prompt for task generation from env context."""
    tools_str = "\n".join(f"- {t}" for t in tools) if tools else "No tools listed."

    examples_str = ""
    if example_tasks:
        for i, ex in enumerate(example_tasks[:3], 1):
            examples_str += f"\n### Example {i}\n"
            examples_str += f"Prompt: {ex.get('prompt', '')}\n"

    schema_str = schema if schema else "No schema available."

    return f"""You are a task designer for the "{env_key}" environment.

## Environment: {env_key}

### Available Tools
{tools_str}

### Data Schema
{schema_str}

### Example Tasks
{examples_str}

## Your Output Format

Generate exactly ONE task with a prompt and verifier function.

<task>
<prompt>[Task instruction]</prompt>
<verifier>[Python async verify function]</verifier>
</task>"""


def build_task_gen_dataset(
    tasks_json: str,
    output_dir: str,
    env_context_dir: Optional[str] = None,
    eval_ratio: float = 0.15,
    min_verifier_len: int = 50,
    max_examples_per_env: int = 3,
):
    """Build SFT dataset from existing Fleet tasks.

    Groups tasks by environment, creates (env_context, task_spec) pairs.
    Holds out some tasks per env as few-shot examples in the prompt.

    Args:
        tasks_json: Path to Fleet tasks JSON
        output_dir: Output directory for parquet files
        env_context_dir: Directory with per-env context configs
        eval_ratio: Fraction for evaluation split
        min_verifier_len: Minimum verifier code length to include
        max_examples_per_env: Number of example tasks to include in prompt
    """
    print(f"Loading tasks from {tasks_json}...")
    tasks = load_tasks(tasks_json)
    print(f"Loaded {len(tasks)} tasks")

    # Filter: must have verifier code
    tasks_with_verifier = []
    for t in tasks:
        verifier = t.get("verifier_func") or t.get("verifier_code", "")
        if verifier and len(verifier) >= min_verifier_len:
            tasks_with_verifier.append(t)
    print(f"Tasks with verifier (>= {min_verifier_len} chars): {len(tasks_with_verifier)}")

    # Group by environment
    tasks_by_env: Dict[str, List[Dict]] = defaultdict(list)
    for t in tasks_with_verifier:
        env_key = t.get("env_key") or t.get("env_id") or "unknown"
        tasks_by_env[env_key].append(t)

    print(f"\nEnvironments: {len(tasks_by_env)}")
    for env_key, env_tasks in sorted(tasks_by_env.items()):
        print(f"  {env_key}: {len(env_tasks)} tasks")

    # Build dataset records
    all_records = []

    for env_key, env_tasks in tasks_by_env.items():
        # Load env context if available
        env_ctx = None
        if env_context_dir:
            env_ctx = load_env_context(env_context_dir, env_key)

        # Extract tool names from env context or from task verifiers
        tools = []
        if env_ctx:
            tools = env_ctx.get("tools", [])

        # Use first N tasks as few-shot examples, rest as training targets
        example_tasks = env_tasks[:max_examples_per_env]
        target_tasks = env_tasks[max_examples_per_env:]

        if not target_tasks:
            # If too few tasks, use all as both examples and targets
            target_tasks = env_tasks

        # Build the system prompt for this environment
        example_dicts = [{"prompt": t.get("prompt", "")} for t in example_tasks]

        system_prompt = format_env_context_prompt(
            env_key=env_key,
            tools=tools,
            schema=env_ctx.get("schema", "") if env_ctx else "",
            example_tasks=example_dicts,
        )

        for task in target_tasks:
            task_spec = format_task_spec(task)
            task_key = task.get("key") or task.get("task_key", "unknown")

            record = {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Generate a task for the {env_key} environment.",
                    },
                ],
                "response": task_spec,
                "env_class": "task_gen",
                "data_source": env_key,
                "task_key": task_key,
                # Store env context as extras for the TaskGenEnv
                "env_key": env_key,
                "env_version": task.get("version") or task.get("env_version", ""),
                "env_tools": json.dumps(tools),
            }
            all_records.append(record)

    print(f"\nTotal records: {len(all_records)}")

    # Split into train/eval
    import hashlib

    train_records = []
    eval_records = []

    for record in all_records:
        h = hashlib.md5(record["task_key"].encode()).hexdigest()
        if int(h[:8], 16) / (16**8) < eval_ratio:
            eval_records.append(record)
        else:
            train_records.append(record)

    print(f"Train: {len(train_records)}, Eval: {len(eval_records)}")

    # Save
    os.makedirs(output_dir, exist_ok=True)

    if train_records:
        train_ds = Dataset.from_list(train_records)
        train_ds.to_parquet(os.path.join(output_dir, "train.parquet"))
        print(f"Saved train to {output_dir}/train.parquet")

    if eval_records:
        eval_ds = Dataset.from_list(eval_records)
        eval_ds.to_parquet(os.path.join(output_dir, "validation.parquet"))
        print(f"Saved validation to {output_dir}/validation.parquet")

    # Print per-env breakdown
    env_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"train": 0, "eval": 0})
    for r in train_records:
        env_counts[r["data_source"]]["train"] += 1
    for r in eval_records:
        env_counts[r["data_source"]]["eval"] += 1

    print(f"\n{'Environment':<20} {'Train':>8} {'Eval':>8}")
    print("-" * 40)
    for env_key in sorted(env_counts.keys()):
        c = env_counts[env_key]
        print(f"{env_key:<20} {c['train']:>8} {c['eval']:>8}")


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT dataset for task generation")
    parser.add_argument(
        "--tasks-json",
        type=str,
        required=True,
        help="Path to Fleet tasks JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/task_gen",
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--env-context-dir",
        type=str,
        default=None,
        help="Directory with per-env context configs (YAML/JSON)",
    )
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.15,
        help="Fraction of data for evaluation",
    )
    parser.add_argument(
        "--min-verifier-len",
        type=int,
        default=50,
        help="Minimum verifier code length to include",
    )

    args = parser.parse_args()

    build_task_gen_dataset(
        tasks_json=args.tasks_json,
        output_dir=args.output_dir,
        env_context_dir=args.env_context_dir,
        eval_ratio=args.eval_ratio,
        min_verifier_len=args.min_verifier_len,
    )


if __name__ == "__main__":
    main()
