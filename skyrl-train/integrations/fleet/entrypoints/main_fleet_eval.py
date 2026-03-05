"""
Fleet Task Eval-Only Entrypoint for SkyRL.

Registers the FleetTaskEnv and runs evaluation only (no training).
Uses the same eval logic as main_generate but with Fleet env registration.
"""

import asyncio

import hydra
import ray
from loguru import logger
from omegaconf import DictConfig
from typing import Any

from skyrl_gym.envs import register
from skyrl_train.entrypoints.main_generate import EvalOnlyEntrypoint
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.utils.utils import validate_generator_cfg, initialize_ray


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg: DictConfig) -> dict:
    register(
        id="fleet_task",
        entry_point="integrations.fleet.env:FleetTaskEnv",
    )

    exp = EvalOnlyEntrypoint(cfg)
    return asyncio.run(exp.run())


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    metrics = ray.get(eval_entrypoint.remote(cfg))
    logger.info(f"Metrics from Fleet eval-only run: {metrics}")


if __name__ == "__main__":
    main()
