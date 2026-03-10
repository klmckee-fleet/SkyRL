"""
Fleet Task Environment for SkyRL.

This module provides a SkyRL-compatible environment wrapper for Fleet-hosted tasks.
It uses OpenEnv's FleetTaskEnv as the abstraction layer for Fleet environments.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)
from envs.fleet_env import FleetTaskEnv as OpenEnvFleetTaskEnv
from envs.fleet_env import ContextManager, configure_fleet_telemetry

# Reduce MCP client log noise
# - loguru: some MCP libs use loguru
# - standard logging: mcp.client.streamable_http uses standard logging
try:
    from loguru import logger as loguru_logger

    loguru_logger.disable("mcp")
except ImportError:
    pass
logging.getLogger("mcp").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Global task cache to avoid reloading JSON for each env instance
_TASK_CACHE: Dict[str, Dict[str, Any]] = {}


def load_tasks_from_json(tasks_file: str) -> Dict[str, Any]:
    """Load tasks from JSON file with caching.

    Returns a dict mapping task_key -> task_config dict.
    """
    if tasks_file not in _TASK_CACHE:
        expanded_path = os.path.expanduser(tasks_file)
        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"Tasks file not found: {expanded_path}")

        with open(expanded_path, "r") as f:
            data = json.load(f)

        # Handle both formats: array or {"tasks": [...]}
        if isinstance(data, list):
            tasks = data
        elif isinstance(data, dict) and "tasks" in data:
            tasks = data["tasks"]
        else:
            raise ValueError(f"Invalid JSON format in {tasks_file}: expected array or object with 'tasks' key")

        if not tasks:
            raise ValueError(f"No tasks found in {tasks_file}")

        # Index by task_key (support both 'key' and 'task_key' fields)
        _TASK_CACHE[tasks_file] = {t.get("key") or t.get("task_key"): t for t in tasks}

    return _TASK_CACHE[tasks_file]


def parse_tool_call(action: str) -> Optional[Dict[str, Any]]:
    """
    Parse tool call from LLM response.

    Supports tag-based formats:
    - <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    - <function_call>{"name": "...", "arguments": {...}}</function_call>

    Also handles cases where the closing tag is missing (e.g., when </tool_call>
    is used as the stop string and not included in the output).

    Returns dict with "name" and "arguments" keys, or None if not found.
    """
    # Try common tag formats
    for tag in ["tool_call", "function_call"]:
        # First try with closing tag
        match = re.search(rf"<{tag}>(.*?)</{tag}>", action, re.DOTALL)
        if not match:
            # Try without closing tag (for when </tool_call> is the stop string)
            # Match from opening tag to end of string or next special token
            match = re.search(rf"<{tag}>(.*?)(?:<\||\Z)", action, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1).strip())
                # json.loads can return any JSON type, we need a dict
                if not isinstance(parsed, dict):
                    continue
                # Normalize keys
                name = parsed.get("name") or parsed.get("tool")
                args = parsed.get("arguments") or parsed.get("params", {})
                if name:
                    return {"name": name, "arguments": args}
            except (json.JSONDecodeError, ValueError):
                # ValueError catches Python's integer string conversion limit
                # (e.g., model generates 8000+ digit numbers)
                pass

    return None


class FleetTaskEnv(BaseTextEnv):
    """
    SkyRL environment for Fleet-hosted tasks.

    Uses OpenEnv's FleetTaskEnv as the abstraction layer for Fleet environments.
    This provides a clean separation between SkyRL's training interface and
    Fleet's environment management.
    """

    _trace_config: Optional[Dict[str, str]] = None

    @classmethod
    def set_trace_config(cls, job_id: str, model: str):
        """Set trace config for uploading eval traces to Fleet."""
        cls._trace_config = {"job_id": job_id, "model": model}

    @classmethod
    def clear_trace_config(cls):
        """Clear trace config after eval is done."""
        cls._trace_config = None

    def __init__(
        self,
        env_config: DictConfig,
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        self.extras = extras
        self.max_turns = extras.get("max_turns", 50)

        # Task configuration from extras (set by dataset)
        self.task_key = extras.get("task_key")
        self.tasks_file = env_config.get("tasks_file") or extras.get("tasks_file")

        if not self.task_key:
            raise ValueError("task_key must be provided in extras (from dataset)")
        if not self.tasks_file:
            raise ValueError("tasks_file must be provided in env_config or extras")

        # Expand path
        self.tasks_file = os.path.expanduser(self.tasks_file)

        # Load task config from JSON
        tasks = load_tasks_from_json(self.tasks_file)
        self.task_config = tasks.get(self.task_key)
        if not self.task_config:
            available_keys = list(tasks.keys())[:5]
            raise ValueError(
                f"Task '{self.task_key}' not found in {self.tasks_file}. " f"Available keys (first 5): {available_keys}"
            )

        # API key
        self.api_key = env_config.get("api_key") or os.environ.get("FLEET_API_KEY")
        if not self.api_key:
            raise ValueError("FLEET_API_KEY must be set in env_config or environment")

        # Logfire telemetry (no-op if LOGFIRE_TOKEN is not set)
        logfire_token = os.environ.get("LOGFIRE_TOKEN")
        if logfire_token:
            configure_fleet_telemetry(token=logfire_token)

        # TTL for Fleet environment instances
        self.ttl_seconds = env_config.get("ttl_seconds", None)  # None = auto (CUA: 1800s, tool_use: 600s)

        # Environment state (initialized on init())
        self.openenv_task_env: Optional[OpenEnvFleetTaskEnv] = None
        self.chat_history: ConversationType = []
        self.turns = 0
        self.tool_calls = 0
        self.tool_errors = 0
        self.tools: List[Dict[str, Any]] = []

        # Context management (uses OpenEnv's ContextManager)
        # Read from env_config (Hydra config) not extras (per-sample dataset config)
        self.enable_context_tools = env_config.get("enable_context_tools", False)
        self.context_manager: Optional[ContextManager] = None
        if self.enable_context_tools:
            logger.info(
                f"Enabling context management tools with max_output_chars={extras.get('max_output_chars', 10000)}"
            )
            self.context_manager = ContextManager(max_output_chars=extras.get("max_output_chars", 10000))

    def _normalize_task_config(self) -> Dict[str, Any]:
        """Normalize task config to OpenEnv's expected format."""
        config = self.task_config.copy()

        # Map field names if needed
        if "key" in config and "task_key" not in config:
            config["task_key"] = config["key"]
        if "env_id" in config and "env_key" not in config:
            config["env_key"] = config["env_id"]
        if "version" in config and "env_version" not in config:
            config["env_version"] = config["version"]

        return config

    def _adapt_computer_tool_for_qwen(self):
        """Adapt the computer tool description for Qwen VL models.

        Qwen3-VL/3.5 models output coordinates in a normalized [0, 1000] grid
        regardless of actual screen resolution. This method:
        1. Parses the actual screen dimensions from the tool description
        2. Rewrites the description to use [0, 1000] coordinate range

        Coordinates are converted back to pixels in step_async() before MCP execution.
        Ref: https://github.com/QwenLM/Qwen3-VL/issues/1521
        """
        for tool in self.tools:
            func = tool.get("function", {})
            if func.get("name") != "computer":
                continue

            desc = func.get("description", "")

            # Parse actual screen dimensions: "Screen resolution: 1366x768 pixels"
            res_match = re.search(r"Screen resolution:\s*(\d+)x(\d+)", desc)
            if res_match:
                self.screen_width = int(res_match.group(1))
                self.screen_height = int(res_match.group(2))
            else:
                self.screen_width = 1366
                self.screen_height = 768

            w, h = self.screen_width, self.screen_height

            # Rewrite description for Qwen's [0, 1000] coordinate space
            desc = re.sub(
                r"Screen resolution:\s*\d+x\d+\s*pixels\s*(\([^)]*\))?",
                "Screen resolution: 1000x1000",
                desc,
            )
            desc = re.sub(
                r"\(0, 0\) is top-left,\s*\(\d+, \d+\) is bottom-right",
                "(0, 0) is top-left, (999, 999) is bottom-right",
                desc,
            )
            desc = re.sub(
                r"valid range: x=0-\d+, y=0-\d+",
                "valid range: x=0-999, y=0-999",
                desc,
            )
            desc = re.sub(
                r"JPEG format at \d+x\d+",
                "JPEG format at 1000x1000",
                desc,
            )
            func["description"] = desc

            logger.info(
                f"Adapted computer tool for Qwen VL: actual_screen={w}x{h}, " f"model coordinate space=[0, 1000]"
            )
            break

    def _convert_qwen_coordinates(self, tool_call: Dict[str, Any]):
        """Convert Qwen's [0, 1000] normalized coordinates to pixel coordinates.

        Modifies tool_call arguments in-place.
        """
        if not self.screen_width or not self.screen_height:
            return
        args = tool_call.get("arguments", {})
        if not args or tool_call.get("name") != "computer":
            return
        for field in ("coordinate", "start_coordinate"):
            coords = args.get(field)
            if coords and isinstance(coords, (list, tuple)) and len(coords) == 2:
                args[field] = [
                    int(coords[0] / 1000 * self.screen_width),
                    int(coords[1] / 1000 * self.screen_height),
                ]

    async def init_async(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """
        Initialize the Fleet environment and return initial observation (async version).

        Creates Fleet environment via OpenEnv's FleetTaskEnv and returns the task prompt.
        OpenEnv's FleetTaskEnv.__init__() creates the Fleet env and fetches tools.
        """
        # Close any existing environment
        self.close()

        # Create OpenEnv's FleetTaskEnv with normalized config
        # This creates the Fleet env (fleet.make) and fetches tools (list_tools)
        task_config = self._normalize_task_config()

        try:
            self.openenv_task_env = OpenEnvFleetTaskEnv(
                task_config=task_config,
                api_key=self.api_key,
                ttl_seconds=self.ttl_seconds,
                max_steps=self.max_turns,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create OpenEnv FleetTaskEnv: {e}") from e

        # Reset episode state (tools are already cached from __init__)
        obs = await self.openenv_task_env.reset_async()

        # Reset state
        self.turns = 0
        self.tool_calls = 0
        self.tool_errors = 0

        # Reset context manager if enabled
        if self.context_manager:
            self.context_manager.reset()

        # Get tools from observation (cached from __init__)
        self.tools = obs.get("tools", [])

        # For computer_use: adapt coordinate system for Qwen VL models
        # Qwen3-VL/3.5 output coordinates in normalized [0, 1000] space (pre-trained convention).
        # The MCP server's computer tool expects pixel coordinates.
        # We rewrite the tool description to [0, 1000] and convert back in step_async().
        # Ref: https://github.com/QwenLM/Qwen3-VL/issues/1521
        modality = self.task_config.get("task_modality", "tool_use")
        self.screen_width = None
        self.screen_height = None
        if modality == "computer_use":
            self._adapt_computer_tool_for_qwen()

        # Add context management tools if enabled
        if self.context_manager:
            self.tools = self.tools + self.context_manager.get_tools()
        if not self.tools:
            raise RuntimeError(f"Task {self.task_key}: no tools found in observation. Fleet env requires tools.")

        # Build initial prompt with task instruction
        task_prompt = self.task_config.get("prompt", "")

        # Build system prompt with tool definitions
        tools_json = json.dumps(self.tools, indent=2)
        # Include current date so model uses correct year for date-related tasks
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Build environment context section from env_variables
        env_context = ""
        env_vars = self.task_config.get("env_variables", {})
        if env_vars:
            env_lines = []
            if "LOGGED_IN_USER" in env_vars:
                env_lines.append(f"- Logged in user ID: {env_vars['LOGGED_IN_USER']}")
            if "LOGGED_IN_NAME" in env_vars:
                env_lines.append(f"- Logged in as: {env_vars['LOGGED_IN_NAME']}")
            # Include other env variables (skip CURRENT_DATE as it's handled separately)
            for key, value in env_vars.items():
                if key not in ("LOGGED_IN_USER", "LOGGED_IN_NAME", "CURRENT_DATE"):
                    env_lines.append(f"- {key}: {value}")
            if env_lines:
                env_context = "\n## Environment Context\n" + "\n".join(env_lines) + "\n"

        # Add environment-specific hints
        env_key = self.task_config.get("env_key") or self.task_config.get("env_id")
        env_hints = ""
        if env_key == "fostgres":
            env_hints = """
## Database Exploration
Before writing SQL queries, first explore the database schema:
- List tables: SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'
- List columns: SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'your_table'
"""

        # Add computer_use-specific guidance
        modality = self.task_config.get("task_modality", "tool_use")
        computer_use_hints = ""
        if modality == "computer_use":
            computer_use_hints = """
## Browser Interaction Strategy
You are controlling a web browser via screenshots. Follow this loop:

1. **Act**: Perform ONE action (click, type, scroll, etc.)
2. **Observe**: Take a screenshot to see the result
3. **Adapt**: If the screen hasn't changed, try a DIFFERENT action

Key rules:
- After clicking or typing, ALWAYS take a screenshot next to see what happened
- NEVER repeat the same action more than twice. If it didn't work, try something different:
  - Can't find an element by scrolling? Use the search bar or navigation menu instead
  - Page not loading after a click? Try refreshing with key("F5") or clicking a different element
  - Form not submitting? Check if required fields are missing
- Use wait() only ONCE after a page navigation, then screenshot to check. Do not wait repeatedly
- When the task is fully complete, say <done>. Do not keep clicking after finishing
"""
        tool_names = [t["function"]["name"] for t in self.tools if "function" in t]
        tool_names_str = ", ".join(tool_names)

        system_content = f"""You are a helpful agent. Complete the task by calling tools.

## Current Date
Today's date is {current_date}. When dates are mentioned without a year, assume the current year ({datetime.now().year}) or a future date.
{env_context}{env_hints}{computer_use_hints}
## Available Tools
{tools_json}

## Tool Call Format
Use the tools listed above by name ({tool_names_str}). Format each call as:
<tool_call>{{"name": "<tool_name_from_above>", "arguments": {{...}}}}</tool_call>

## Error Handling
If a tool call returns an error:
- Read the error message carefully
- Do NOT repeat the same call with identical arguments
- Change your approach: use different parameters, try a different tool, or break the task into smaller steps

## Response Format
EVERY response MUST end with exactly ONE of:
1. A tool call: <tool_call>...</tool_call> - to perform an action
2. Done signal: <done> - ONLY when the task is fully complete

IMPORTANT: When the task is complete, first output your final answer with the requested information, THEN say <done>. Do not just say <done> without providing the answer.

NEVER respond with just a message. NEVER say "feel free to ask" or offer further help.
If the task is complete, provide your answer then say <done>. Otherwise, make a tool call."""

        # Build conversation with system prompt
        system_message = {"role": "system", "content": system_content}

        # For computer_use with initial screenshot, create multimodal user message
        initial_screenshot = obs.get("initial_screenshot")
        if initial_screenshot and isinstance(initial_screenshot, list):
            # Build multimodal content: task prompt + screenshot
            user_content = [{"type": "text", "text": task_prompt}]
            # Add images from screenshot result
            for item in initial_screenshot:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    user_content.append(item)
            user_message = {"role": "user", "content": user_content}
            logger.info(f"Task {self.task_key}: included initial screenshot in user message")
        else:
            user_message = {"role": "user", "content": task_prompt}

        self.chat_history = [system_message, user_message]

        metadata = {
            "task_key": self.task_key,
            "env_key": self.task_config.get("env_key") or self.task_config.get("env_id"),
            "tools": self.tools,
            "modality": self.task_config.get("task_modality", "tool_use"),
        }

        return self.chat_history.copy(), metadata

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        """
        Initialize the Fleet environment and return initial observation (sync wrapper).

        For async contexts, use init_async() instead.
        """
        return asyncio.run(self.init_async(prompt))

    async def step_async(self, action: str) -> BaseTextEnvStepOutput:
        """
        Execute one step in the Fleet environment (async version).

        Parses the action for tool calls, executes via OpenEnv's FleetTaskEnv,
        and returns observation. Reward is computed by the verifier on completion.
        """
        step_start = time.time()
        self.turns += 1
        assistant_msg = {"role": "assistant", "content": action}
        self.chat_history.append(assistant_msg)
        if self.context_manager:
            self.context_manager.track_message(assistant_msg)

        max_turns_reached = self.turns >= self.max_turns

        # Parse tool call from LLM response
        tool_call = parse_tool_call(action)

        # Convert Qwen's [0, 1000] normalized coordinates to pixel coordinates
        if tool_call:
            self._convert_qwen_coordinates(tool_call)

        # Check if agent signals completion
        has_done_signal = "<done>" in action.lower() or "[done]" in action.lower()

        # Log the model output for debugging early termination
        if self.turns == 1:
            # Log full output for turn 1 to diagnose early termination
            logger.info(
                f"Task {self.task_key} turn 1 FULL OUTPUT:\n"
                f"has_tool_call={tool_call is not None}\n"
                f"has_done_signal={has_done_signal}\n"
                f"agent_done={has_done_signal and not tool_call}\n"
                f"action_length={len(action)}\n"
                f"action='''{action[:500]}'''"
            )

        # Only consider <done> if there's NO tool call - if there's a tool call,
        # we need to execute it and see the result before ending the episode
        agent_done = has_done_signal and not tool_call

        tool_result = None
        error = None
        reward = 0.0
        mcp_time = 0.0

        # Handle context management tools locally (no MCP call)
        if tool_call and self.context_manager and self.context_manager.is_context_tool(tool_call["name"]):
            tool_result, self.chat_history = self.context_manager.execute_tool(
                tool_call["name"], tool_call.get("arguments", {}), self.chat_history
            )
        # Execute tool call if present via OpenEnv
        elif tool_call and self.openenv_task_env:
            self.tool_calls += 1
            # Build action dict for OpenEnv
            openenv_action = {
                "tool": tool_call["name"],
                "params": tool_call.get("arguments", {}),
                "done": agent_done,
            }

            try:
                # Use async step method
                mcp_start = time.time()
                obs, reward, done, info = await self.openenv_task_env.step_async(openenv_action)
                mcp_time = time.time() - mcp_start
                tool_result = obs.get("observation")
                if "tool_error" in info:
                    error = info["tool_error"]

                # Truncate long tool outputs and store full version for retrieval if context management is enabled
                if tool_result and isinstance(tool_result, str) and self.context_manager:
                    tool_result = self.context_manager.truncate_output(tool_result)
            except Exception as e:
                mcp_time = time.time() - mcp_start
                error = str(e)
        elif agent_done and self.openenv_task_env:
            # Agent signaled done without tool call
            openenv_action = {"done": True}
            try:
                mcp_start = time.time()
                obs, reward, done, info = await self.openenv_task_env.step_async(openenv_action)
                mcp_time = time.time() - mcp_start
            except Exception as e:
                mcp_time = time.time() - mcp_start
                error = str(e)

        # Detect error patterns in tool_result (tool returned error as successful response)
        if not error and tool_result:
            result_str = str(tool_result) if not isinstance(tool_result, str) else tool_result
            # Check for common error patterns in tool response
            if result_str.strip().startswith("Error:") or result_str.strip().startswith("error:"):
                error = result_str
                tool_result = None
            elif isinstance(tool_result, dict) and tool_result.get("error"):
                error = tool_result["error"]
                tool_result = None

        # Check if episode is done
        episode_done = agent_done or max_turns_reached

        # Upload trace at episode end if trace config is set
        if episode_done and FleetTaskEnv._trace_config:
            try:
                from envs.fleet_env.trace import upload_trace

                inst_id = None
                orch = getattr(self.openenv_task_env, "_orch", None)
                if orch:
                    fleet_env = getattr(orch, "_fleet_env", None)
                    if fleet_env:
                        inst_id = getattr(fleet_env, "instance_id", None)
                await upload_trace(
                    api_key=self.api_key,
                    job_id=FleetTaskEnv._trace_config["job_id"],
                    task_key=self.task_key,
                    model=FleetTaskEnv._trace_config["model"],
                    chat_history=self.chat_history,
                    reward=reward,
                    instance_id=inst_id,
                    metadata={"env_key": self.task_config.get("env_key"), "turns": self.turns},
                )
            except Exception as e:
                logger.warning(f"Failed to upload trace for {self.task_key}: {e}")

        # Build observation message
        if max_turns_reached:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=True,
                metadata={"done_reason": "max_turns", "task_key": self.task_key},
            )

        # Build response observation
        # For VL models, tool_result may contain images in OpenAI-compatible format:
        # [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "data:..."}}]
        if error:
            self.tool_errors += 1
            obs_content = f"Error: {error}"
            new_obs = {"role": "user", "content": obs_content}
        elif tool_result:
            # Check if tool_result contains images (list with image_url items)
            if isinstance(tool_result, list) and any(
                isinstance(item, dict) and item.get("type") == "image_url" for item in tool_result
            ):
                # Multimodal result - create OpenAI-compatible content array
                content_parts = []
                for item in tool_result:
                    if isinstance(item, dict):
                        if item.get("type") == "image_url":
                            content_parts.append(item)
                        elif item.get("type") == "text":
                            content_parts.append({"type": "text", "text": f"Tool result:\n{item.get('text', '')}"})
                        else:
                            # Unknown type - convert to text
                            content_parts.append(
                                {"type": "text", "text": f"Tool result:\n{json.dumps(item, indent=2)}"}
                            )
                    else:
                        content_parts.append({"type": "text", "text": f"Tool result:\n{str(item)}"})
                new_obs = {"role": "user", "content": content_parts}
            elif isinstance(tool_result, dict):
                obs_content = f"Tool result:\n{json.dumps(tool_result, indent=2)}"
                new_obs = {"role": "user", "content": obs_content}
            else:
                obs_content = f"Tool result:\n{tool_result}"
                new_obs = {"role": "user", "content": obs_content}
        elif agent_done:
            obs_content = "Task marked as complete."
            new_obs = {"role": "user", "content": obs_content}
        elif not tool_call:
            obs_content = 'No tool call found. Use <tool_call>{"name": "...", "arguments": {...}}</tool_call> format.'
            new_obs = {"role": "user", "content": obs_content}
        else:
            obs_content = "Action executed."
            new_obs = {"role": "user", "content": obs_content}
        self.chat_history.append(new_obs)
        if self.context_manager:
            self.context_manager.track_message(new_obs)

        step_time = time.time() - step_start
        metadata = {
            "task_key": self.task_key,
            "turn": self.turns,
            "tool_call": tool_call,
            "tool_result": tool_result,
            "error": error,
            "done_reason": "agent_done" if agent_done else None,
            "step_time": step_time,
            "mcp_time": mcp_time,
        }

        # If context was modified (e.g., manage_context dropped turns), return full chat_history
        # so the generator can replace its copy. This is required for stepwise training to work.
        if tool_call and self.context_manager and self.context_manager.is_context_tool(tool_call["name"]):
            if tool_call["name"] == "manage_context":
                metadata["modified_chat_history"] = self.chat_history.copy()

        return BaseTextEnvStepOutput(
            observations=[new_obs],
            reward=reward,
            done=episode_done,
            metadata=metadata,
        )

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Execute one step in the Fleet environment (sync wrapper).

        For async contexts, use step_async() instead.
        """
        return asyncio.run(self.step_async(action))

    def close(self):
        """Close the Fleet environment and cleanup resources."""
        if self.openenv_task_env:
            try:
                self.openenv_task_env.close()
            except Exception as e:
                print(f"Warning: Failed to close Fleet environment: {e}")
            self.openenv_task_env = None

    def get_metrics(self) -> Dict[str, Any]:
        """Return environment metrics for this episode."""
        return {
            "task_key": self.task_key,
            "env_key": self.task_config.get("env_key") or self.task_config.get("env_id"),
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "tool_errors": self.tool_errors,
        }

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across episodes with per-env breakdown.

        Handles two types of metrics:
        1. Normal episode metrics: {env_key, turns, tool_calls, tool_errors}
        2. Init failure metrics: {env_init_failed, env_init_failed/{env_key}}
        """
        if not metrics:
            return {}

        # Track init failures per env (from env_init_failed/{env_key} pattern)
        env_init_failures: Dict[str, int] = {}
        total_init_failures = 0

        # Group normal metrics by env_key
        env_data: Dict[str, Dict[str, List[int]]] = {}
        for m in metrics:
            # Check for init failure metrics (env_init_failed/{env_key} pattern)
            for key, value in m.items():
                if key.startswith("env_init_failed/"):
                    env_key = key.split("/", 1)[1]
                    env_init_failures[env_key] = env_init_failures.get(env_key, 0) + int(value)
                    total_init_failures += int(value)
                elif key == "env_init_failed":
                    # Already counted via per-env key
                    pass

            # Normal episode metrics (has env_key field)
            env_key = m.get("env_key")
            if env_key:
                if env_key not in env_data:
                    env_data[env_key] = {"turns": [], "tool_calls": [], "tool_errors": [], "timing": []}
                env_data[env_key]["turns"].append(m.get("turns", 0))
                env_data[env_key]["tool_calls"].append(m.get("tool_calls", 0))
                env_data[env_key]["tool_errors"].append(m.get("tool_errors", 0))
                timing = {k: v for k, v in m.items() if k.startswith("timing/")}
                if timing:
                    env_data[env_key]["timing"].append(timing)

        result: Dict[str, Any] = {}
        total_turns = 0
        total_tool_calls = 0
        total_tool_errors = 0
        total_episodes = 0

        # Per-env_key metrics
        for env_key, data in env_data.items():
            turns_list = data["turns"]
            tool_calls_list = data["tool_calls"]
            tool_errors_list = data["tool_errors"]

            avg_turns = sum(turns_list) / len(turns_list)
            avg_tool_calls = sum(tool_calls_list) / len(tool_calls_list)
            avg_tool_errors = sum(tool_errors_list) / len(tool_errors_list)
            # Tool calls per turn (avoid div by zero)
            total_env_turns = sum(turns_list)
            total_env_tool_calls = sum(tool_calls_list)
            total_env_tool_errors = sum(tool_errors_list)
            tool_calls_per_turn = total_env_tool_calls / total_env_turns if total_env_turns > 0 else 0
            # Tool error rate (errors per tool call)
            tool_error_rate = total_env_tool_errors / total_env_tool_calls if total_env_tool_calls > 0 else 0

            result[f"{env_key}/avg_turns"] = avg_turns
            result[f"{env_key}/min_turns"] = min(turns_list)
            result[f"{env_key}/max_turns"] = max(turns_list)
            result[f"{env_key}/avg_tool_calls"] = avg_tool_calls
            result[f"{env_key}/tool_calls_per_turn"] = tool_calls_per_turn
            result[f"{env_key}/avg_tool_errors"] = avg_tool_errors
            result[f"{env_key}/total_tool_errors"] = total_env_tool_errors
            result[f"{env_key}/tool_error_rate"] = tool_error_rate
            result[f"{env_key}/num_episodes"] = len(turns_list)

            total_turns += total_env_turns
            total_tool_calls += total_env_tool_calls
            total_tool_errors += total_env_tool_errors
            total_episodes += len(turns_list)

        # Aggregate timing metrics across all envs
        all_timing = []
        for env_key, data in env_data.items():
            timing_list = data.get("timing", [])
            if timing_list:
                all_timing.extend(timing_list)
                for tkey in timing_list[0].keys():
                    vals = [t[tkey] for t in timing_list if tkey in t]
                    if vals:
                        result[f"{env_key}/{tkey}"] = sum(vals) / len(vals)
        if all_timing:
            for tkey in all_timing[0].keys():
                vals = [t[tkey] for t in all_timing if tkey in t]
                if vals:
                    result[tkey] = sum(vals) / len(vals)

        # Overall metrics
        result["avg_turns"] = total_turns / total_episodes if total_episodes > 0 else 0
        result["avg_tool_calls"] = total_tool_calls / total_episodes if total_episodes > 0 else 0
        result["tool_calls_per_turn"] = total_tool_calls / total_turns if total_turns > 0 else 0
        result["avg_tool_errors"] = total_tool_errors / total_episodes if total_episodes > 0 else 0
        result["total_tool_errors"] = total_tool_errors
        result["tool_error_rate"] = total_tool_errors / total_tool_calls if total_tool_calls > 0 else 0
        result["total_episodes"] = total_episodes

        # Add init failure metrics (per-env and total)
        for env_key, failures in env_init_failures.items():
            result[f"{env_key}/env_init_failed"] = failures
        if total_init_failures > 0:
            result["total_env_init_failed"] = total_init_failures

        return result


def clear_caches():
    """Clear global caches. Useful for testing."""
    global _TASK_CACHE
    _TASK_CACHE = {}
