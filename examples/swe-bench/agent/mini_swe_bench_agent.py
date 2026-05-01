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
                reward = await self._run_in_executor(
                    self._run_sync,
                    data,
                    client,
                    instance_id,
                    traj_id,
                    timeout=self.eval_timeout + self.command_timeout * self.max_iteration,
                )
            client.set_last_reward(reward)
            return reward
        except Exception as exc:
            print(f"MiniSweBenchAgent error for {instance_id}: {exc}")
            traceback.print_exc()
            return None

    async def _run_in_executor(self, fn, *args, timeout: float | None = None, **kwargs):
        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(self.executor, partial(fn, *args, **kwargs))
        if timeout is not None:
            return await asyncio.wait_for(task, timeout=timeout)
        return await task

    def _run_sync(self, data: dict, client, instance_id: str, traj_id: str) -> float:
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
            info = agent.run(problem_statement)
            eval_output = env.run_eval()
            reward, metrics = self._score_eval(instance_id, traj_id, eval_output)
            self._write_result(instance_id, traj_id, reward, info, eval_output, metrics)
            return reward
        finally:
            env.cleanup()

    def _score_eval(self, instance_id: str, traj_id: str, eval_output: dict) -> tuple[float, dict]:
        result_dir = self.output_path / "eval_results" / instance_id / traj_id
        result_dir.mkdir(parents=True, exist_ok=True)
        log_path = result_dir / "eval.log"
        metrics_path = result_dir / "test_results.json"
        log_path.write_text(eval_output.get("output", ""))

        parser_script = self.swebench_root / "run_eval_with_metrics.py"
        if not parser_script.exists():
            fallback = 1.0 if eval_output.get("returncode") == 0 else 0.0
            return fallback, {"fallback": True, "returncode": eval_output.get("returncode")}

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
            return float(metrics.get("swebench", {}).get("reward", 0.0)), metrics
        fallback = 1.0 if eval_output.get("returncode") == 0 else 0.0
        return fallback, {"fallback": True, "parser_output": result.stdout}

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
        result = {
            "instance_id": instance_id,
            "traj_id": traj_id,
            "reward": reward,
            "agent_info": info,
            "eval": {
                "returncode": eval_output.get("returncode"),
                "output": eval_output.get("output", ""),
            },
            "metrics": metrics,
        }
        result_path.write_text(json.dumps(result, indent=2))
