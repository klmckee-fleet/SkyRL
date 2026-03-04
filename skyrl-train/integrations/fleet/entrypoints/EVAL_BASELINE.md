# Eval Baseline: Qwen3.5-9B on Fleet Computer Use Tasks

Standalone inference-only evaluation using vLLM + FleetTaskEnv. No training — just collects agent rollouts against live Fleet environments and computes pass@k metrics.

## Usage

```bash
# Start vLLM server:
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B --host 0.0.0.0 --port 8000

# Run eval:
python -m integrations.fleet.entrypoints.eval_baseline \
    --tasks-file /path/to/all_computer_use.json \
    --model Qwen/Qwen3.5-9B \
    --num-tasks 12 \
    --rollouts-per-task 10
```

Or launch via SkyPilot (handles vLLM server automatically):

```bash
sky launch tasks/eval-baseline-qwen3.5-9b.yaml --env FLEET_API_KEY=<key>
```

Or via GitHub Actions: trigger the "Fleet Task Training (SkyPilot)" workflow with `task_name=eval-baseline-qwen3.5-9b`.

## Architecture

### Entry Point

`main()` parses CLI args and calls `asyncio.run(run_eval(...))`.

### `run_eval()` — Orchestrator

1. Loads the first N tasks from `all_computer_use.json`
2. Creates an `AsyncOpenAI` client pointed at the local vLLM server (`http://localhost:8000/v1`)
3. Verifies vLLM is reachable by listing models
4. Builds `num_tasks × rollouts_per_task` coroutines and runs them with `asyncio.Semaphore(max_concurrent)` to cap concurrent Fleet environments (default 8) — same pattern as `main_fleet_tinker.py`
5. Saves results: all trajectories as JSONL, per-env JSONL files, and a JSON summary
6. Prints pass@k metrics to stdout

### `collect_rollout()` — Single Agent Trajectory

This is the core loop, modeled after `collect_fleet_rollout()` in `main_fleet_tinker.py`:

```
1. Create FleetTaskEnv (reuses integrations/fleet/env.py)
     │
2. env.init([])
     │  Creates Fleet environment via OpenEnv
     │  Fetches MCP tools
     │  Builds system prompt with tool definitions
     │  Returns chat_history = [system_msg, user_task_prompt]
     │
3. while not done and turns < max_turns:
     │
     ├─ a. Call vLLM via OpenAI chat completions API:
     │       client.chat.completions.create(
     │           messages=env.chat_history,
     │           stop=["</tool_call>"]
     │       )
     │     stop=["</tool_call>"] makes vLLM stop after a tool call.
     │     We re-append "</tool_call>" so env.step() can parse it.
     │
     ├─ b. env.step(output_text)
     │       parse_tool_call() extracts {"name": "...", "arguments": {...}}
     │       Executes tool via MCP on the live Fleet environment
     │       Returns observation, reward, done
     │       Observation appended to chat_history automatically
     │
     └─ c. Loop until agent says <done> or hits max_turns
     │
4. env.close()
5. Return {task_key, env_key, reward, turns, tool_calls, conversation, duration, ...}
```

Key difference from the Tinker version: uses `openai.AsyncOpenAI` instead of `tinker.SamplingClient`, and doesn't track token IDs, logprobs, or loss masks (not needed for eval).

### `_run_in_executor()` — MCP Connection Isolation

`env.init()` and `env.step()` are sync methods that internally run async MCP calls. They run in a `ThreadPoolExecutor` so each Fleet environment gets its own thread with isolated MCP connections — same pattern as `main_fleet_tinker.py` (lines 85-99).

### Metrics

- `compute_pass_at_k(rollouts, k)` — groups rollouts by `task_key`, checks if any of the first k rollouts got reward > 0
- Reports pass@1, pass@5, pass@10 overall and per-environment

### Output Example

```
======================================================================
  Eval Results: Qwen/Qwen3.5-9B
  12 tasks, 10 rollouts each, 120 total
======================================================================
  pass@1  = 0.167
  pass@5  = 0.333
  pass@10 = 0.417
  success  = 15/120
  avg turns     = 18.3
  avg tool calls= 14.7
  avg duration  = 52.1s
  errors        = 2

  Per-environment:
  env                 tasks   pass@1   pass@10  avg_turns
  -------------------------------------------------------
  outlook                 3    0.333    0.667        12.4
  zillow                  2    0.000    0.000        24.1
  ...
======================================================================
```

## Output Files

All saved to `--output-dir` (default `./eval_results/`):

| File | Contents |
|------|----------|
| `trajectories_TIMESTAMP.jsonl` | All rollouts. Each line: `{task_key, env_key, rollout_idx, reward, turns, tool_calls, tool_errors, stop_reason, duration, conversation}` |
| `{env_key}_TIMESTAMP.jsonl` | Same data, split by environment |
| `summary_TIMESTAMP.json` | Aggregated metrics: pass@k overall and per-env, avg turns, success rates |

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--tasks-file` | (required) | Path to tasks JSON |
| `--model` | `Qwen/Qwen3.5-9B` | Model name (must match vLLM server) |
| `--num-tasks` | 12 | Number of tasks to evaluate |
| `--rollouts-per-task` | 10 | Rollouts per task |
| `--max-turns` | 50 | Max agent turns per rollout |
| `--max-generate-length` | 2048 | Max tokens per generation |
| `--temperature` | 1.0 | Sampling temperature |
| `--max-concurrent` | 8 | Max concurrent Fleet environments |
| `--output-dir` | `./eval_results` | Where to save results |
| `--vllm-base-url` | `http://localhost:8000/v1` | vLLM server URL |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FLEET_API_KEY` | Yes | Fleet platform API key |
| `VLLM_BASE_URL` | No | Override vLLM server URL (default: `http://localhost:8000/v1`) |
