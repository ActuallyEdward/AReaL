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
        n_trajs: int = 1,
        max_tokens: int = 32768,
        max_iteration: int = 250,
        max_workers: int = 25,
        swebench_root: str | Path = DEFAULT_SWEBENCH_ROOT,
        reset_env_timeout: float = 1200.0,
        eval_timeout: int = 1200,
        command_timeout: int = 60,
        filter_uniform_reward: bool = False,
        cleanup: bool = True,
        prebuild_images: bool = True,
    ):
        self.gconfig = gconfig
        self.gconfig.n_samples = 1
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.rollout_stat_scope = rollout_stat_scope
        self.n_trajs = n_trajs
        self.max_tokens = max_tokens
        self.max_iteration = max_iteration
        self.swebench_root = Path(swebench_root)
        self.reset_env_timeout = reset_env_timeout
        self.eval_timeout = eval_timeout
        self.command_timeout = command_timeout
        self.filter_uniform_reward = filter_uniform_reward
        self.cleanup = cleanup
        self.prebuild_images = prebuild_images
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        if self.dump_dir is not None:
            os.makedirs(self.dump_dir, exist_ok=True)

    async def arun_episode(self, engine, data):
        instance_id = get_instance_id(data)
        clients = [
            ArealOpenAI(
                engine=engine,
                tokenizer=self.tokenizer,
                tool_call_parser="qwen25",
            )
            for _ in range(self.n_trajs)
        ]
        uids = [uuid.uuid4().hex[:8] for _ in range(self.n_trajs)]

        if self.prebuild_images:
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
                return None

        print(f"\n{'=' * 70}")
        print(f"[EPISODE START] SWE-bench task {instance_id}")
        print(f"{'=' * 70}\n")

        output_path = (
            f"{self.dump_dir}/MiniSweBenchAgent_Output"
            if self.dump_dir is not None
            else "MiniSweBenchAgent_Output"
        )
        rewards = await asyncio.gather(
            *[
                MiniSweBenchAgent(
                    output_path=output_path,
                    max_iteration=self.max_iteration,
                    executor=self.executor,
                    swebench_root=self.swebench_root,
                    eval_timeout=self.eval_timeout,
                    command_timeout=self.command_timeout,
                    cleanup=self.cleanup,
                ).run_agent(
                    data=data,
                    client=clients[i],
                    uid=uids[i],
                    traj_i=i,
                )
                for i in range(self.n_trajs)
            ]
        )

        print(f"\n{'=' * 70}")
        print(f"[EPISODE END] SWE-bench task {instance_id}")
        print(f"{'=' * 70}\n")

        completions_with_reward = {}
        if self.filter_uniform_reward:
            valid_rewards = [reward for reward in rewards if reward is not None]
            if valid_rewards and all(reward == valid_rewards[0] for reward in valid_rewards):
                print(
                    f"Rank {os.getenv('RANK', '0')} - Task {instance_id} "
                    "has uniform reward across trajectories. Discarding all."
                )
                return completions_with_reward
            if not valid_rewards:
                print(f"Rank {os.getenv('RANK', '0')} - Task {instance_id} all trajectories failed.")
                return completions_with_reward

        for i, (reward, client) in enumerate(zip(rewards, clients)):
            if reward is None:
                print(f"Rank {os.getenv('RANK', '0')} - Task {instance_id}, Trajectory {i} failed.")
                if self.dump_dir is not None:
                    failed_dir = Path(self.dump_dir) / "failed_tasks"
                    failed_dir.mkdir(parents=True, exist_ok=True)
                    (failed_dir / f"{instance_id}_traj_{i}.txt").write_text(
                        f"Task {instance_id} trajectory {i} failed.\n"
                    )
                continue

            print(f"Rank {os.getenv('RANK', '0')} - Task {instance_id}, Trajectory {i} reward: {reward}")
            stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)
            client.apply_reward_discount(turn_discount=0.9)
            completions_with_reward.update(client.export_interactions(style="individual"))

        if len(completions_with_reward) == 0:
            print(f"All trajectories failed for task {instance_id}.")
            completions_with_reward = None

        stats_tracker.get(self.rollout_stat_scope).scalar(
            num_full_passes=sum(1 for reward in rewards if reward == 1.0)
        )
        stats_tracker.get(self.rollout_stat_scope).scalar(
            num_trajectories_failed=sum(1 for reward in rewards if reward is None)
        )

        print(f"Rank {os.getenv('RANK', '0')} - Task {instance_id} completed.")
        return completions_with_reward
