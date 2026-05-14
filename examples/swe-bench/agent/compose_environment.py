from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ._mini_path import ensure_mini_swe_agent_on_path

ensure_mini_swe_agent_on_path()

from minisweagent.exceptions import Submitted  # noqa: E402
from minisweagent.utils.serialize import recursive_merge  # noqa: E402


class SweBenchComposeEnvironmentConfig(BaseModel):
    task_dir: Path
    traj_id: str
    project_root: Path | None = None
    run_root: Path | None = None
    source_repo_root: Path | None = None
    compose_project_name: str | None = None
    reset_repo: bool = True
    timeout: int = 60
    eval_timeout: int = 1200
    interpreter: list[str] = ["bash", "-lc"]
    env: dict[str, str] = {}
    cleanup: bool = True


class SweBenchComposeEnvironment:
    """mini-swe-agent Environment for the generated SWE-bench Compose tasks."""

    def __init__(self, *, config_class: type = SweBenchComposeEnvironmentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.task_dir = self.config.task_dir.resolve()
        self.metadata = json.loads((self.task_dir / "task_metadata.json").read_text())
        self.instance_id = self.metadata["instance_id"]
        self.repo_dir = self.metadata["repo_dir"]
        self.cwd = f"{self.metadata['container_repo_root']}/{self.repo_dir}"
        self.compose_file = self.task_dir / "docker-compose.yaml"
        self.compose_project_name = self.config.compose_project_name or self._compose_project_name()
        self.project_root = self.config.project_root or self.task_dir.parents[1]
        self.run_root = self.config.run_root or (
            self.project_root / "runs" / self.instance_id / self.config.traj_id
        )
        self.task_repo_root = self.run_root / "task_repo"
        self.container_id = ""
        self._start()

    def _compose_project_name(self) -> str:
        raw = f"swebench-{self.instance_id}-{self.config.traj_id}".lower()
        return "".join(ch if ch.isalnum() or ch in "_-" else "-" for ch in raw)

    def _base_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.config.env)
        env.update(
            {
                "TRAJ_ID": self.config.traj_id,
                "COMPOSE_PROJECT_NAME": self.compose_project_name,
                "PROJECT_ROOT": str(self.project_root),
                "RUN_ROOT": str(self.run_root),
                "TASK_REPO_ROOT": str(self.task_repo_root),
                "RESET_REPO": "1" if self.config.reset_repo else "0",
            }
        )
        if self.config.source_repo_root:
            env["SOURCE_REPO_ROOT"] = str(self.config.source_repo_root)
        return env

    def _run(self, cmd: list[str], *, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            cwd=self.task_dir,
            env=self._base_env(),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )

    def _start(self) -> None:
        result = self._run([str(self.task_dir / "docker_compose_start.sh")], timeout=self.config.eval_timeout)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start SWE-bench compose environment:\n{result.stdout}")
        self.container_id = result.stdout.strip().splitlines()[-1]

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "")
        exec_cmd = [
            "docker",
            "compose",
            "-p",
            self.compose_project_name,
            "-f",
            str(self.compose_file),
            "exec",
            "-T",
            "-w",
            cwd or self.cwd,
            "client",
            *self.config.interpreter,
            command,
        ]
        start = time.time()
        try:
            result = self._run(exec_cmd, timeout=timeout or self.config.timeout)
            output = {
                "output": result.stdout,
                "returncode": result.returncode,
                "exception_info": "",
                "extra": {"runtime": time.time() - start, "command": shlex.join(exec_cmd)},
            }
        except Exception as exc:
            raw_output = getattr(exc, "output", "") or ""
            if isinstance(raw_output, bytes):
                raw_output = raw_output.decode("utf-8", errors="replace")
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {exc}",
                "extra": {"exception_type": type(exc).__name__, "exception": str(exc)},
            }
        self._check_finished(output)
        return output

    def run_eval(self) -> dict[str, Any]:
        result = self._run([str(self.task_dir / "docker_compose_eval.sh")], timeout=self.config.eval_timeout)
        return {
            "output": result.stdout,
            "returncode": result.returncode,
            "exception_info": "",
        }

    def cleanup(self) -> None:
        if self.config.cleanup:
            self._run([str(self.task_dir / "docker_compose_stop.sh")], timeout=120)
            self._cleanup_run_root()

    def _cleanup_run_root(self) -> None:
        run_root = self.run_root.resolve()
        runs_root = (self.project_root / "runs").resolve()
        instance_runs_root = (runs_root / self.instance_id).resolve()

        try:
            run_root.relative_to(instance_runs_root)
        except ValueError:
            print(f"Skipping SWE-bench run cleanup outside instance runs root: {run_root}")
            return

        if run_root in (runs_root, instance_runs_root):
            print(f"Skipping unsafe SWE-bench run cleanup path: {run_root}")
            return

        shutil.rmtree(run_root, ignore_errors=True)

    def _check_finished(self, output: dict) -> None:
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(
            self.config.model_dump(mode="json"),
            self.metadata,
            platform.uname()._asdict(),
            {
                "cwd": self.cwd,
                "compose_project_name": self.compose_project_name,
                "container_id": self.container_id,
            },
            kwargs,
        )

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "compose_project_name": self.compose_project_name,
                "container_id": self.container_id,
            }
        }

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
