# Fleet Integration for SkyRL

This module provides a SkyRL-compatible environment wrapper for Fleet-hosted tasks using OpenEnv as the abstraction layer.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SkyRL FleetTaskEnv (integrations/fleet/env.py)                              │
│                                                                             │
│  __init__(env_config, extras)                                               │
│      └── Stores config, task_key, api_key                                   │
│      └── self.openenv_task_env = None (not created yet)                     │
│                                                                             │
│  init(prompt)  ←── Called by SkyRL trainer to start episode                 │
│      │                                                                      │
│      ├── OpenEnvFleetTaskEnv(task_config, ...)                              │
│      │       ├── fleet.make()      ←── Creates Fleet env (provisions VM)    │
│      │       └── list_tools()      ←── Fetches & caches tools               │
│      │                                                                      │
│      ├── reset_async()                                                      │
│      │       ├── _orch.reset()     ←── Resets episode state                 │
│      │       └── Returns cached tools                                       │
│      │                                                                      │
│      └── Builds system prompt with tools                                    │
│                                                                             │
│  step(action)                                                               │
│      ├── Parse tool call from LLM response                                  │
│      ├── Execute via OpenEnv step_async()                                   │
│      └── Return observation, reward, done                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ OpenEnv FleetTaskEnv (envs.fleet_env.task_env)                              │
│                                                                             │
│  __init__(task_config, api_key, ...)                                        │
│      ├── FleetEnvClient.from_fleet()  ←── HTTP: Creates Fleet env instance  │
│      │       └── fleet.make()         ←── Provisions cloud VM/container     │
│      └── list_tools()                 ←── MCP: Fetches available tools      │
│              └── Cached in _tools_cache                                     │
│                                                                             │
│  reset_async()                                                              │
│      ├── _orch.reset()  ←── HTTP: Resets episode (logs warning if fails)    │
│      └── Returns obs with cached _tools_cache                               │
│                                                                             │
│  step_async(action)                                                         │
│      └── _tools.call_tool()  ←── MCP: Executes tool                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Fleet SDK                                                                   │
│                                                                             │
│  fleet.make(env_key, ...)                                                   │
│      └── HTTP call to Fleet API                                             │
│      └── Provisions environment instance (VM/container)                     │
│      └── Returns env handle with URLs for HTTP + MCP endpoints              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

1. **Environment created in `__init__()`**: Fleet environment is provisioned and tools are fetched when `OpenEnvFleetTaskEnv` is instantiated, not during `reset()`. This ensures tools are available immediately.

2. **Tools cached**: Tools are fetched once during initialization and cached in `_tools_cache`. Every `reset_async()` and `step_async()` returns the cached tools.

3. **Reset failure handling**: If `_orch.reset()` fails (timeout), a warning is logged but the episode continues with empty observation. Tools remain available from cache.

4. **OpenEnv as abstraction layer**: SkyRL does not call Fleet SDK directly. All Fleet interactions go through OpenEnv's `FleetTaskEnv`.

## Tool Flow

```
init()
    └── Tools fetched via MCP list_tools()
    └── Tools injected into system prompt (JSON format)
    └── Tools cached for episode

step()
    └── LLM generates: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    └── SkyRL parses tool call
    └── OpenEnv executes via MCP call_tool()
    └── Result returned to LLM
```

## Configuration

### Environment Config
```yaml
environment:
  env_class: fleet_task
  skyrl_gym:
    fleet_task:
      tasks_file: /path/to/tasks.json  # Exported from Fleet
      api_key: ${FLEET_API_KEY}        # Or set via environment
      ttl_seconds: null                  # Auto: CUA=1800s, tool_use=600s (or override)
```

### Task JSON Format
```json
{
  "tasks": [
    {
      "key": "task-001",
      "prompt": "Search for flights from NYC to LA",
      "env_id": "booking-com",
      "version": "v1.2.3",
      "task_modality": "tool_use",
      "verifier_code": "async def verify(env): ..."
    }
  ]
}
```

## Error Handling

| Failure Point | Behavior |
|--------------|----------|
| `fleet.make()` fails | RuntimeError raised, episode fails |
| `list_tools()` fails | RuntimeError raised, episode fails |
| `_orch.reset()` fails | Warning logged, continues with empty observation |
| `call_tool()` fails | Error returned in observation, episode continues |

## Task Generation (GRPO)

RL-based task generation: trains Qwen3-8B to produce (prompt, verifier) pairs for Fleet environments using GRPO.

**Reward formula**: `R(task) = validity_gate * (base_reward + variance + alpha * separation)`

### Dataset Preparation

`prepare_task_gen_dataset.py` builds GRPO training data by:
1. Loading validated tasks from S3 (`all_tool_use.json`)
2. Discovering tool schemas from live Fleet environments via OpenEnv MCP
3. Fetching DB schemas from Supabase `seed_versions` -> S3 `schema.sql`
4. Storing env context (tools, schema, env_variables) in each parquet record

### Training Runs

#### Run: `task_gen_bf9229d1` (enkfchnh) — 2026-03-04

Config: Qwen3-8B, 4xGPU, batch=4, n_samples=4, lr=1e-6, base_reward=0.1

**Before schema injection.** env_variables injected but no DB schema. github included (wasting compute).

| Env | Steps | pass@4 | GRPO Signal | Avg Variance | Notes |
|-----|-------|--------|-------------|--------------|-------|
| github | 152 | 0% | 0% | 0.0000 | Context overflow (160 tools) — excluded in next run |
| booking | 121 | 70.5% | 79.3% | 0.0018 | Best signal |
| reddit | 66 | 100% | 16.7% | 0.0003 | Valid tasks but low variance |
| ticketmaster | 34 | 100% | 52.9% | 0.0011 | |
| zillow | 21 | 95.2% | 61.9% | 0.0047 | Highest variance |
| amazon | 21 | 100% | 38.1% | 0.0007 | |
| rops | 13 | 0% | 0% | 0.0000 | |
| fira | 8 | 100% | 62.5% | 0.0013 | |
| wallst | 8 | 12.5% | 12.5% | 0.0002 | |
| carlisle | 6 | 100% | 66.7% | 0.0014 | |

- **169 steps, 12.4h runtime**
- Reward: avg=0.0348, max=0.1312 (mostly base_reward from judge pass)
- Reward trend: 0.0295 (first half) -> 0.0326 (second half)
- 151/169 steps (89%) produced non-zero reward
- **Key issue**: github consumed ~90% of steps with 0% signal
- **Root cause of low variance**: model guesses wrong DB table/column names in verifiers

### Changelog

- `a0913bf5` — Exclude github from dataset (context overflow, 0% signal)
- `99fcda49` — Inject DB schema (table/column names) from Supabase/S3
- `18fa5d55` — Pass env_variables to prompt and Fleet harness
- `c51abf78` — Document env_variables access pattern in prompt
- `4b905a0f` — Fix evaluator_models shell quoting
- `2f360e0e` — Add Fleet harness rollouts for full reward formula
- `efbe658d` — Handle prompt-too-long crash (response_end_idx=None)
- `de0efc56` — Add LLM-as-a-judge reward gate

## Dependencies

- **OpenEnv**: `pip install openenv[fleet]` or add to PYTHONPATH
- **Fleet SDK**: Installed as OpenEnv dependency
- **FLEET_API_KEY**: Must be set in environment or config
