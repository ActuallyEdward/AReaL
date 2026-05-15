from __future__ import annotations

import asyncio
import json
import os
import subprocess
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from areal.utils.perf_tracer import Category, atrace_scope, session_context, trace_perf

from ._mini_path import ensure_mini_swe_agent_on_path
from .areal_mini_model import ArealMiniModel
from .compose_environment import SweBenchComposeEnvironment
from .prompts import (
    FORMAT_ERROR_TEMPLATE,
    INSTANCE_TEMPLATE,
    OBSERVATION_TEMPLATE,
    SYSTEM_TEMPLATE,
)

ensure_mini_swe_agent_on_path()

from minisweagent.agents.default import DefaultAgent  # noqa: E402


DEFAULT_SWEBENCH_ROOT = Path("/Users/edwardwang/Desktop/projects/datasets/swe-bench")


class MiniSweBenchAgent:
    """AReaL rollout agent using mini-swe-agent on generated SWE-bench Compose tasks."""

    def __init__(
        self,
        output_path: str = "MiniSweBenchAgent_Output",
        max_iteration: int = 250,
        executor: ThreadPoolExecutor | None = None,
        swebench_root: str | Path = DEFAULT_SWEBENCH_ROOT,
        eval_timeout: int = 1200,
        command_timeout: int = 60,
        cleanup: bool = True,
        max_completion_tokens: int = 1024,
        max_total_tokens: int | None = None,
    ):
        self.output_path = Path(output_path)
        self.max_iteration = max_iteration
        self.executor = executor
        self.swebench_root = Path(swebench_root)
        self.eval_timeout = eval_timeout
        self.command_timeout = command_timeout
        self.cleanup = cleanup
        self.max_completion_tokens = max_completion_tokens
        self.max_total_tokens = max_total_tokens
        assert self.executor is not None, "Executor must be provided to MiniSweBenchAgent"

    @session_context()
    @trace_perf("MiniSweBenchAgent.run_agent", category=Category.COMPUTE)
    async def run_agent(self, data: dict, client, uid: str | None = None, traj_i: int = 0) -> float | None:
        uid = uid or uuid.uuid4().hex[:8]
        instance_id = data.get("instance_id") or data.get("task_name")
        if not instance_id:
            raise ValueError("data must include instance_id or task_name")
        traj_id = f"{uid}.traj{traj_i}"
        self.output_path.mkdir(parents=True, exist_ok=True)

        try:
            async with atrace_scope(f"mini_swebench:{instance_id}:run_agent"):
                event_loop = asyncio.get_running_loop()
                reward = await self._run_in_executor(
                    self._run_sync,
                    data,
                    client,
                    event_loop,
                    instance_id,
                    traj_id,
                    timeout=self.eval_timeout + self.command_timeout * self.max_iteration,
                )
            try:
                client.set_last_reward(reward)
            except RuntimeError:
                if reward != 0.0:
                    raise
                print(
                    f"MiniSweBenchAgent zero-reward trajectory for {instance_id} "
                    "has no cached model interaction to export."
                )
            return reward
        except Exception as exc:
            if self._is_context_limit_error(exc):
                reward = 0.0
                print(
                    f"MiniSweBenchAgent context limit for {instance_id} "
                    f"trajectory {traj_id}; assigning reward 0.0."
                )
                try:
                    client.set_last_reward(reward)
                except RuntimeError:
                    print(
                        f"MiniSweBenchAgent zero-reward trajectory for {instance_id} "
                        "has no cached model interaction to export."
                    )
                return reward
            print(f"MiniSweBenchAgent error for {instance_id}: {exc}")
            traceback.print_exc()
            return None

    async def _run_in_executor(self, fn, *args, timeout: float | None = None, **kwargs):
        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(self.executor, partial(fn, *args, **kwargs))
        if timeout is not None:
            return await asyncio.wait_for(task, timeout=timeout)
        return await task

    def _run_sync(
        self,
        data: dict,
        client,
        event_loop: asyncio.AbstractEventLoop,
        instance_id: str,
        traj_id: str,
    ) -> float:
        task_dir = self.swebench_root / "swe-bench-team-tasks" / instance_id
        env = SweBenchComposeEnvironment(
            task_dir=task_dir,
            traj_id=traj_id,
            timeout=self.command_timeout,
            eval_timeout=self.eval_timeout,
            cleanup=self.cleanup,
        )
        traj_path = self.output_path / f"{instance_id}.{traj_id}.traj.json"
        model = ArealMiniModel(
            client,
            event_loop=event_loop,
            observation_template=OBSERVATION_TEMPLATE,
            format_error_template=FORMAT_ERROR_TEMPLATE,
            model_kwargs={
                "temperature": 0.0,
                "parallel_tool_calls": True,
                "max_completion_tokens": self.max_completion_tokens,
                "max_total_tokens": self.max_total_tokens,
            },
        )
        agent = DefaultAgent(
            model,
            env,
            system_template=SYSTEM_TEMPLATE,
            instance_template=INSTANCE_TEMPLATE,
            step_limit=self.max_iteration,
            cost_limit=0.0,
            output_path=traj_path,
        )
        problem_statement = data.get("problem_statement") or data.get("instruction") or ""
        info = {"exit_status": "", "submission": ""}
        try:
            try:
                info = agent.run(problem_statement)
            except Exception as exc:
                if not self._is_context_limit_error(exc):
                    raise
                reward = 0.0
                info = {
                    "exit_status": "ContextLimitExceeded",
                    "submission": "",
                    "error": str(exc),
                }
                eval_output = {
                    "returncode": None,
                    "output": f"Trajectory exceeded context limit before evaluation: {exc}",
                }
                metrics = {
                    "swebench": {
                        "reward": reward,
                        "binary_reward": reward,
                        "partial_reward": reward,
                        "training_reward": reward,
                    },
                    "context_limit_exceeded": True,
                    "error": str(exc),
                }
                print(
                    f"MiniSweBenchAgent context limit for {instance_id} "
                    f"trajectory {traj_id}; assigning reward 0.0."
                )
            else:
                eval_output = env.run_eval()
                reward, metrics = self._score_eval(instance_id, traj_id, eval_output)
            self._write_result(instance_id, traj_id, reward, info, eval_output, metrics)
            return reward
        finally:
            env.cleanup()

    def _is_context_limit_error(self, exc: BaseException) -> bool:
        messages = []
        current: BaseException | None = exc
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            messages.append(str(current).lower())
            current = current.__cause__ or current.__context__
        text = "\n".join(messages)
        return any(
            marker in text
            for marker in (
                "exceeds max_total_tokens",
                "exceeds engine_max_tokens",
                "max_new_tokens",
                "non-positive",
                "context length",
                "sequence length",
            )
        )

    def _score_eval(self, instance_id: str, traj_id: str, eval_output: dict) -> tuple[float, dict]:
        result_dir = self.output_path / "eval_results" / instance_id / traj_id
        result_dir.mkdir(parents=True, exist_ok=True)
        log_path = result_dir / "eval.log"
        metrics_path = result_dir / "test_results.json"
        log_path.write_text(eval_output.get("output", ""))

        parser_script = self.swebench_root / "run_eval_with_metrics.py"
        if not parser_script.exists():
            return 0.0, {
                "fallback": True,
                "error": f"missing parser script: {parser_script}",
                "returncode": eval_output.get("returncode"),
            }

        result = subprocess.run(
            [
                "python3",
                str(parser_script),
                "--instance-id",
                instance_id,
                "--log",
                str(log_path),
                "--output",
                str(metrics_path),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            return self._reward_from_metrics(metrics), metrics
        return 0.0, {
            "fallback": True,
            "error": "parser did not produce metrics output",
            "parser_output": result.stdout,
            "returncode": eval_output.get("returncode"),
            "swebench": {
                "reward": 0.0,
                "binary_reward": 0.0,
                "partial_reward": 0.0,
                "training_reward": 0.0,
            },
        }

    def _reward_from_metrics(self, metrics: dict) -> float:
        swebench_metrics = metrics.setdefault("swebench", {})
        binary_reward = float(swebench_metrics.get("reward", 0.0))
        partial_reward = float(swebench_metrics.get("partial_reward", binary_reward))
        swebench_metrics.setdefault("binary_reward", binary_reward)
        swebench_metrics["training_reward"] = partial_reward
        return partial_reward

    def _write_result(
        self,
        instance_id: str,
        traj_id: str,
        reward: float,
        info: dict,
        eval_output: dict,
        metrics: dict,
    ) -> None:
        result_path = self.output_path / f"{instance_id}.{traj_id}.result.json"
        swebench_metrics = metrics.get("swebench", {})
        binary_reward = float(swebench_metrics.get("binary_reward", reward))
        partial_reward = float(swebench_metrics.get("partial_reward", reward))
        result = {
            "instance_id": instance_id,
            "traj_id": traj_id,
            "reward": reward,
            "reward_type": "partial_reward",
            "binary_reward": binary_reward,
            "partial_reward": partial_reward,
            "agent_info": info,
            "eval": {
                "returncode": eval_output.get("returncode"),
                "output": eval_output.get("output", ""),
            },
            "metrics": metrics,
        }
        result_path.write_text(json.dumps(result, indent=2))
