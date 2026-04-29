from __future__ import annotations

import json
import subprocess
from pathlib import Path


DEFAULT_SWEBENCH_ROOT = Path("/Users/edwardwang/Desktop/projects/datasets/swe-bench")


def get_instance_id(task: dict) -> str:
    instance_id = task.get("instance_id") or task.get("task_name")
    if not instance_id:
        raise ValueError("task must include instance_id or task_name")
    return instance_id


def get_task_dir(task: dict, swebench_root: str | Path = DEFAULT_SWEBENCH_ROOT) -> Path:
    return Path(swebench_root) / "swe-bench-team-tasks" / get_instance_id(task)


def build_docker_image(
    task: dict,
    swebench_root: str | Path = DEFAULT_SWEBENCH_ROOT,
    timeout: float = 1200.0,
) -> None:
    task_dir = get_task_dir(task, swebench_root)
    build_script = task_dir / "docker_build.sh"
    if not build_script.exists():
        raise FileNotFoundError(build_script)
    result = subprocess.run(
        [str(build_script)],
        cwd=task_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to build image for {get_instance_id(task)}:\n{result.stdout}")


def load_task_index(swebench_root: str | Path = DEFAULT_SWEBENCH_ROOT) -> dict[str, dict]:
    index_path = Path(swebench_root) / "swe-bench-team-tasks" / "docker_task_index.json"
    rows = json.loads(index_path.read_text())
    return {row["instance_id"]: row for row in rows}
