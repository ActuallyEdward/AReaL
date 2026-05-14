from __future__ import annotations

import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from agent.mini_swe_bench_agent import MiniSweBenchAgent
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.openai import ArealOpenAI
from areal.utils import stats_tracker
from areal.utils.perf_tracer import atrace_scope

from .pre_build_tasks_utils import build_docker_image, get_instance_id


DEFAULT_SWEBENCH_ROOT = Path("/Users/edwardwang/Desktop/projects/datasets/swe-bench")


class MiniSweBenchWorkflow(RolloutWorkflow):
    """AReaL workflow for SWE-bench using mini-swe-agent and Docker Compose tasks."""

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        dump_dir: str | None = None,
        rollout_stat_scope: str = "rollout",
        max_tokens: int = 32768,
        max_iteration: int = 250,
        max_workers: int = 25,
        swebench_root: str | Path = DEFAULT_SWEBENCH_ROOT,
        reset_env_timeout: float = 1200.0,
        eval_timeout: int = 1200,
        command_timeout: int = 60,
        cleanup: bool = True,
        prebuild_images: bool = True,
    ):
        # AReaL wraps this workflow in GroupedRolloutWorkflow when
        # gconfig.n_samples > 1. Each workflow execution should produce exactly
        # one trajectory.
        self.gconfig = gconfig.new(n_samples=1)
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.rollout_stat_scope = rollout_stat_scope
        self.max_tokens = max(max_tokens, self.gconfig.max_tokens)
        self.gconfig.max_tokens = self.max_tokens
        self.max_iteration = max_iteration
        self.swebench_root = Path(swebench_root)
        self.reset_env_timeout = reset_env_timeout
        self.eval_timeout = eval_timeout
        self.command_timeout = command_timeout
        self.cleanup = cleanup
        self.prebuild_images = prebuild_images
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._prebuild_locks: dict[str, asyncio.Lock] = {}
        self._prebuilt_images: set[str] = set()
        if self.dump_dir is not None:
            os.makedirs(self.dump_dir, exist_ok=True)

    async def arun_episode(self, engine, data):
        instance_id = get_instance_id(data)
        client = ArealOpenAI(
            engine=engine,
            tokenizer=self.tokenizer,
            tool_call_parser="qwen25",
            engine_max_tokens=self.max_tokens,
        )
        uid = uuid.uuid4().hex[:8]

        if self.prebuild_images and not await self._ensure_image_built(instance_id, data):
            return None

        print(f"\n{'=' * 70}")
        print(f"[EPISODE START] SWE-bench task {instance_id}")
        print(f"{'=' * 70}\n")

        output_path = (
            f"{self.dump_dir}/MiniSweBenchAgent_Output"
            if self.dump_dir is not None
            else "MiniSweBenchAgent_Output"
        )
        reward = await MiniSweBenchAgent(
            output_path=output_path,
            max_iteration=self.max_iteration,
            executor=self.executor,
            swebench_root=self.swebench_root,
            eval_timeout=self.eval_timeout,
            command_timeout=self.command_timeout,
            cleanup=self.cleanup,
            max_completion_tokens=self.gconfig.max_new_tokens,
            max_total_tokens=self.max_tokens,
        ).run_agent(
            data=data,
            client=client,
            uid=uid,
            traj_i=0,
        )

        print(f"\n{'=' * 70}")
        print(f"[EPISODE END] SWE-bench task {instance_id}")
        print(f"{'=' * 70}\n")

        if reward is None:
            print(f"Rank {os.getenv('RANK', '0')} - Task {instance_id}, trajectory failed.")
            if self.dump_dir is not None:
                failed_dir = Path(self.dump_dir) / "failed_tasks"
                failed_dir.mkdir(parents=True, exist_ok=True)
                (failed_dir / f"{instance_id}_{uid}.txt").write_text(
                    f"Task {instance_id} trajectory {uid} failed.\n"
                )
            stats_tracker.get(self.rollout_stat_scope).scalar(num_trajectories_failed=1)
            return None

        print(f"Rank {os.getenv('RANK', '0')} - Task {instance_id}, reward: {reward}")
        stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)
        client.apply_reward_discount(turn_discount=0.9)
        completions_with_reward = client.export_interactions(style="individual")

        if len(completions_with_reward) == 0:
            print(f"Task {instance_id} produced no exportable interactions.")
            return None

        stats_tracker.get(self.rollout_stat_scope).scalar(
            num_full_passes=1 if reward == 1.0 else 0,
            num_trajectories_failed=0,
        )

        print(f"Rank {os.getenv('RANK', '0')} - Task {instance_id} completed.")
        return completions_with_reward

    async def _ensure_image_built(self, instance_id: str, data: dict) -> bool:
        if instance_id in self._prebuilt_images:
            return True

        lock = self._prebuild_locks.setdefault(instance_id, asyncio.Lock())
        async with lock:
            if instance_id in self._prebuilt_images:
                return True

            loop = asyncio.get_running_loop()
            try:
                async with atrace_scope(
                    f"build_docker_image:{instance_id}",
                    args={"timeout": self.reset_env_timeout},
                ):
                    await asyncio.wait_for(
                        loop.run_in_executor(
                            self.executor,
                            partial(
                                build_docker_image,
                                task=data,
                                swebench_root=self.swebench_root,
                                timeout=self.reset_env_timeout,
                            ),
                        ),
                        timeout=self.reset_env_timeout + 60.0,
                    )
            except TimeoutError:
                print(f"Timeout while building docker image for task {instance_id}")
                return False

            self._prebuilt_images.add(instance_id)
            return True
