import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from agent_rl_config import AgentRLConfig
from datasets import load_dataset

from areal import PPOTrainer
from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import load_expr_config
from areal.utils import seeding
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger


WORKFLOW_PATH = "workflow.mini_swe_bench_workflow.MiniSweBenchWorkflow"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _sanitize_row_for_scheduler(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_safe(value) for key, value in row.items()}


class _JsonSafeDataset:
    """Dataset wrapper that sanitizes examples at dataloader read time."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return _sanitize_row_for_scheduler(self.dataset[index])


def main(args):
    config, _ = load_expr_config(args, AgentRLConfig)

    rank = int(os.getenv("RANK", "0"))
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    assert allocation_mode.train is not None

    dataset = load_dataset(
        path="parquet",
        split="train",
        data_files=[
            str(
                Path(__file__).parent.parent.parent
                / "dataset"
                / config.train_dataset.path
            )
        ],
    )
    dataset = _JsonSafeDataset(dataset)

    workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        max_tokens=config.max_tokens_per_trajectory,
        dump_dir=os.path.join(
            StatsLogger.get_log_path(config.stats_logger), "generated"
        ),
        max_iteration=config.max_iteration,
        max_workers=config.max_workers,
        swebench_root=config.swebench_root,
        reset_env_timeout=config.task_timeouts._reset_env,
        eval_timeout=int(config.task_timeouts._evaluate_completion_sync),
        command_timeout=config.command_timeout,
        cleanup=config.cleanup,
        prebuild_images=config.prebuild_images,
    )

    eval_workflow_kwargs = workflow_kwargs.copy()

    with PPOTrainer(
        config,
        train_dataset=dataset,
        valid_dataset=dataset,
    ) as trainer:
        trainer.train(
            workflow=WORKFLOW_PATH,
            workflow_kwargs=workflow_kwargs,
            eval_workflow=WORKFLOW_PATH,
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
