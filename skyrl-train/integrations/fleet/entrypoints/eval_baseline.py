"""
Fleet Task Baseline Evaluation with vLLM.

Runs inference-only evaluation on Fleet computer_use tasks using a vLLM-served model.
No training — just collects agent rollouts and computes pass@k metrics.

Usage:
    # Start vLLM server first:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3.5-9B --host 0.0.0.0 --port 8000

    # Then run eval:
    python -m integrations.fleet.entrypoints.eval_baseline \
        --tasks-file /path/to/all_computer_use.json \
        --model Qwen/Qwen3.5-9B \
        --num-tasks 12 \
        --rollouts-per-task 10

Environment Variables:
    FLEET_API_KEY: Fleet API key for environment access (required)
    VLLM_BASE_URL: vLLM server URL (default: http://localhost:8000/v1)
"""

import argparse
import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf
from openai import AsyncOpenAI

from integrations.fleet.env import FleetTaskEnv, load_tasks_from_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)

# Thread pool for env operations — isolates MCP connections per thread
_env_executor: Optional[ThreadPoolExecutor] = None


def _get_env_executor(max_workers: int = 16) -> ThreadPoolExecutor:
    global _env_executor
    if _env_executor is None:
        _env_executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="fleet-env-"
        )
    return _env_executor


async def _run_in_executor(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_get_env_executor(), func, *args)


async def collect_rollout(
    task_config: Dict[str, Any],
    tasks_file: str,
    client: AsyncOpenAI,
    model: str,
    rollout_idx: int,
    max_turns: int = 50,
    max_generate_length: int = 2048,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Collect a single agent rollout using Fleet env + vLLM inference."""
    rollout_start = time.time()
    task_key = task_config.get("task_key") or task_config.get("key")

    env_config = OmegaConf.create({"tasks_file": tasks_file, "ttl_seconds": 7200})
    extras = {"task_key": task_key, "max_turns": max_turns}
    env = FleetTaskEnv(env_config=env_config, extras=extras)

    try:
        chat_history, metadata = await _run_in_executor(env.init, [])
        env_key = metadata.get("env_key", "unknown")
        logger.info(
            f"[{task_key}] rollout={rollout_idx} env={env_key} initialized "
            f"({len(env.tools)} tools)"
        )

        done = False
        total_gen_time = 0.0
        total_step_time = 0.0

        while not done and env.turns < max_turns:
            turn_num = env.turns + 1

            # Generate with vLLM
            gen_start = time.time()
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=env.chat_history,
                    max_tokens=max_generate_length,
                    temperature=temperature,
                    top_p=1.0,
                    stop=["</tool_call>"],
                )
                output_text = response.choices[0].message.content or ""
                # Re-append stop string if generation stopped on it
                if response.choices[0].finish_reason == "stop" and "<tool_call>" in output_text:
                    output_text += "</tool_call>"
            except Exception as e:
                logger.error(f"[{task_key}] turn {turn_num}: vLLM error: {e}")
                break
            gen_time = time.time() - gen_start
            total_gen_time += gen_time

            # Step environment
            step_start = time.time()
            step_output = await _run_in_executor(env.step, output_text)
            step_time = time.time() - step_start
            total_step_time += step_time

            done = step_output["done"]
            reward = step_output["reward"]

            logger.debug(
                f"[{task_key}] turn {turn_num}: "
                f"gen={gen_time:.1f}s step={step_time:.1f}s "
                f"done={done} reward={reward}"
            )

        duration = time.time() - rollout_start
        final_reward = step_output["reward"] if done else 0.0

        result = {
            "task_key": task_key,
            "env_key": env_key,
            "rollout_idx": rollout_idx,
            "reward": final_reward,
            "turns": env.turns,
            "tool_calls": env.tool_calls,
            "tool_errors": env.tool_errors,
            "stop_reason": "agent_done" if done else "max_turns",
            "duration": round(duration, 2),
            "total_gen_time": round(total_gen_time, 2),
            "total_step_time": round(total_step_time, 2),
            "conversation": env.chat_history,
        }

        logger.info(
            f"[{task_key}] rollout={rollout_idx} done: "
            f"reward={final_reward} turns={env.turns} "
            f"tool_calls={env.tool_calls} errors={env.tool_errors} "
            f"duration={duration:.1f}s"
        )
        return result

    except Exception as e:
        duration = time.time() - rollout_start
        logger.error(f"[{task_key}] rollout={rollout_idx} failed: {e}")
        return {
            "task_key": task_key,
            "env_key": task_config.get("env_key", "unknown"),
            "rollout_idx": rollout_idx,
            "reward": 0.0,
            "turns": env.turns,
            "tool_calls": env.tool_calls,
            "tool_errors": env.tool_errors,
            "stop_reason": "error",
            "duration": round(duration, 2),
            "error": str(e),
            "conversation": env.chat_history,
        }
    finally:
        env.close()


async def run_eval(
    tasks_file: str,
    model: str,
    num_tasks: int = 12,
    rollouts_per_task: int = 10,
    max_turns: int = 50,
    max_generate_length: int = 2048,
    temperature: float = 1.0,
    max_concurrent: int = 8,
    output_dir: str = "./eval_results",
    vllm_base_url: str = "http://localhost:8000/v1",
):
    """Run baseline evaluation on Fleet tasks."""
    # Load tasks
    with open(os.path.expanduser(tasks_file)) as f:
        data = json.load(f)
    task_list = data["tasks"] if isinstance(data, dict) else data
    tasks = task_list[:num_tasks]
    total_rollouts = num_tasks * rollouts_per_task

    logger.info(
        f"Starting eval: {num_tasks} tasks x {rollouts_per_task} rollouts = "
        f"{total_rollouts} total, max_concurrent={max_concurrent}"
    )

    # Setup vLLM client
    client = AsyncOpenAI(base_url=vllm_base_url, api_key="unused")

    # Verify vLLM is reachable
    try:
        models = await client.models.list()
        logger.info(f"vLLM server ready, models: {[m.id for m in models.data]}")
    except Exception as e:
        raise RuntimeError(f"Cannot connect to vLLM at {vllm_base_url}: {e}")

    # Collect rollouts with concurrency limit
    semaphore = asyncio.Semaphore(max_concurrent)
    all_rollouts: List[Dict[str, Any]] = []

    async def collect_with_semaphore(task_config, rollout_idx):
        async with semaphore:
            return await collect_rollout(
                task_config=task_config,
                tasks_file=tasks_file,
                client=client,
                model=model,
                rollout_idx=rollout_idx,
                max_turns=max_turns,
                max_generate_length=max_generate_length,
                temperature=temperature,
            )

    # Build all rollout coroutines
    coros = []
    for task in tasks:
        for r_idx in range(rollouts_per_task):
            coros.append(collect_with_semaphore(task, r_idx))

    # Run with progress tracking
    completed = 0
    for coro in asyncio.as_completed(coros):
        result = await coro
        all_rollouts.append(result)
        completed += 1
        if completed % 10 == 0 or completed == total_rollouts:
            logger.info(f"Progress: {completed}/{total_rollouts} rollouts complete")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save all trajectories as single JSONL
    trajectories_file = os.path.join(output_dir, f"trajectories_{timestamp}.jsonl")
    with open(trajectories_file, "w") as f:
        for r in sorted(all_rollouts, key=lambda x: (x["task_key"], x["rollout_idx"])):
            f.write(json.dumps(r, default=str) + "\n")
    logger.info(f"Saved {len(all_rollouts)} trajectories to {trajectories_file}")

    # Also save per-env JSONL files
    by_env = defaultdict(list)
    for r in all_rollouts:
        by_env[r["env_key"]].append(r)
    for env_key, rollouts in by_env.items():
        env_file = os.path.join(output_dir, f"{env_key}_{timestamp}.jsonl")
        with open(env_file, "w") as f:
            for r in sorted(rollouts, key=lambda x: (x["task_key"], x["rollout_idx"])):
                f.write(json.dumps(r, default=str) + "\n")

    # Compute and print metrics
    print_metrics(all_rollouts, model, num_tasks, rollouts_per_task)

    # Save summary
    summary = compute_summary(all_rollouts, model, num_tasks, rollouts_per_task)
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")


def compute_pass_at_k(rollouts: List[Dict[str, Any]], k: int) -> float:
    """Compute pass@k: for each task, did any of the first k rollouts succeed?"""
    by_task = defaultdict(list)
    for r in rollouts:
        by_task[r["task_key"]].append(r["reward"])

    passes = 0
    total = 0
    for task_key, rewards in by_task.items():
        # Sort by rollout_idx to use first k
        task_rollouts = [
            r for r in rollouts if r["task_key"] == task_key
        ]
        task_rollouts.sort(key=lambda x: x["rollout_idx"])
        first_k_rewards = [r["reward"] for r in task_rollouts[:k]]
        if any(r > 0 for r in first_k_rewards):
            passes += 1
        total += 1

    return passes / total if total > 0 else 0.0


def compute_summary(
    rollouts: List[Dict[str, Any]], model: str, num_tasks: int, rollouts_per_task: int
) -> Dict[str, Any]:
    """Compute full summary metrics."""
    summary = {
        "model": model,
        "num_tasks": num_tasks,
        "rollouts_per_task": rollouts_per_task,
        "total_rollouts": len(rollouts),
        "timestamp": datetime.now().isoformat(),
    }

    # Overall pass@k
    for k in [1, 5, 10]:
        if k <= rollouts_per_task:
            summary[f"pass_at_{k}"] = compute_pass_at_k(rollouts, k)

    # Overall stats
    rewards = [r["reward"] for r in rollouts]
    summary["mean_reward"] = sum(rewards) / len(rewards) if rewards else 0
    summary["success_rate"] = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0
    summary["avg_turns"] = sum(r["turns"] for r in rollouts) / len(rollouts)
    summary["avg_tool_calls"] = sum(r["tool_calls"] for r in rollouts) / len(rollouts)
    summary["avg_duration"] = sum(r["duration"] for r in rollouts) / len(rollouts)
    summary["error_count"] = sum(1 for r in rollouts if r.get("error"))

    # Per-env breakdown
    by_env = defaultdict(list)
    for r in rollouts:
        by_env[r["env_key"]].append(r)

    summary["per_env"] = {}
    for env_key, env_rollouts in sorted(by_env.items()):
        env_summary = {
            "num_tasks": len(set(r["task_key"] for r in env_rollouts)),
            "num_rollouts": len(env_rollouts),
        }
        for k in [1, 5, 10]:
            if k <= rollouts_per_task:
                env_summary[f"pass_at_{k}"] = compute_pass_at_k(env_rollouts, k)
        env_summary["mean_reward"] = sum(r["reward"] for r in env_rollouts) / len(env_rollouts)
        env_summary["avg_turns"] = sum(r["turns"] for r in env_rollouts) / len(env_rollouts)
        env_summary["avg_tool_calls"] = sum(r["tool_calls"] for r in env_rollouts) / len(env_rollouts)
        summary["per_env"][env_key] = env_summary

    return summary


def print_metrics(
    rollouts: List[Dict[str, Any]], model: str, num_tasks: int, rollouts_per_task: int
):
    """Print eval results summary."""
    print("\n" + "=" * 70)
    print(f"  Eval Results: {model}")
    print(f"  {num_tasks} tasks, {rollouts_per_task} rollouts each, {len(rollouts)} total")
    print("=" * 70)

    # Overall pass@k
    for k in [1, 5, 10]:
        if k <= rollouts_per_task:
            p = compute_pass_at_k(rollouts, k)
            print(f"  pass@{k}  = {p:.3f}")

    rewards = [r["reward"] for r in rollouts]
    print(f"  success  = {sum(1 for r in rewards if r > 0)}/{len(rewards)}")
    print(f"  avg turns     = {sum(r['turns'] for r in rollouts) / len(rollouts):.1f}")
    print(f"  avg tool calls= {sum(r['tool_calls'] for r in rollouts) / len(rollouts):.1f}")
    print(f"  avg duration  = {sum(r['duration'] for r in rollouts) / len(rollouts):.1f}s")
    print(f"  errors        = {sum(1 for r in rollouts if r.get('error'))}")

    # Per-env breakdown
    by_env = defaultdict(list)
    for r in rollouts:
        by_env[r["env_key"]].append(r)

    print("\n  Per-environment:")
    print(f"  {'env':<20} {'tasks':>5} {'pass@1':>8} {'pass@10':>8} {'avg_turns':>10}")
    print("  " + "-" * 55)
    for env_key in sorted(by_env.keys()):
        env_rollouts = by_env[env_key]
        n_tasks = len(set(r["task_key"] for r in env_rollouts))
        p1 = compute_pass_at_k(env_rollouts, 1)
        p10 = compute_pass_at_k(env_rollouts, min(10, rollouts_per_task))
        avg_t = sum(r["turns"] for r in env_rollouts) / len(env_rollouts)
        print(f"  {env_key:<20} {n_tasks:>5} {p1:>8.3f} {p10:>8.3f} {avg_t:>10.1f}")
    print("=" * 70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Fleet baseline evaluation with vLLM")
    parser.add_argument(
        "--tasks-file",
        required=True,
        help="Path to tasks JSON file (e.g., all_computer_use.json)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.5-9B",
        help="Model name (must match vLLM server)",
    )
    parser.add_argument("--num-tasks", type=int, default=12)
    parser.add_argument("--rollouts-per-task", type=int, default=10)
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--max-generate-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--output-dir", default="./eval_results")
    parser.add_argument(
        "--vllm-base-url",
        default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(
        run_eval(
            tasks_file=args.tasks_file,
            model=args.model,
            num_tasks=args.num_tasks,
            rollouts_per_task=args.rollouts_per_task,
            max_turns=args.max_turns,
            max_generate_length=args.max_generate_length,
            temperature=args.temperature,
            max_concurrent=args.max_concurrent,
            output_dir=args.output_dir,
            vllm_base_url=args.vllm_base_url,
        )
    )


if __name__ == "__main__":
    main()
