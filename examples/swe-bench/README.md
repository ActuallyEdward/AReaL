# SWE-bench Agent Training

This example mirrors `examples/terminal_bench`, but uses mini-swe-agent as the
agent loop and the generated SWE-bench Docker Compose task folders as the runtime.

## Required Layout

The training command is run from the AReaL repo root:

```bash
cd /Users/edwardwang/Desktop/projects/datasets/AReaL
```

The SWE-bench runtime assets are expected here:

```text
/Users/edwardwang/Desktop/projects/datasets/swe-bench/
  swe-bench-team-tasks/<instance_id>/
    Dockerfile
    docker-compose.yaml
    docker_build.sh
    docker_compose_start.sh
    docker_compose_eval.sh
    docker_compose_stop.sh
    eval.sh
    task_metadata.json
  run_eval_with_metrics.py
  task_repo/
```

The mini-swe-agent source is expected here:

```text
/Users/edwardwang/Desktop/projects/datasets/mini-swe-agent/src
```

## Dataset Location

`train.py` loads parquet files from AReaL's dataset directory:

```text
/Users/edwardwang/Desktop/projects/datasets/AReaL/dataset/
```

Put your train and validation parquet files there, for example:

```text
/Users/edwardwang/Desktop/projects/datasets/AReaL/dataset/swe-bench/swe-bench-team-tasks.parquet
/Users/edwardwang/Desktop/projects/datasets/AReaL/dataset/swe-bench/swe-bench-val.parquet
```

Then update these fields in the config:

```yaml
train_dataset:
  path: swe-bench/swe-bench-team-tasks.parquet

valid_dataset:
  path: swe-bench/swe-bench-val.parquet
```

Each dataset row should include:

```text
instance_id
problem_statement
```

`task_name` can be used instead of `instance_id`, and `instruction` can be used instead
of `problem_statement`, but the recommended schema is:

```json
{
  "instance_id": "django__django-13212",
  "problem_statement": "..."
}
```

The `instance_id` must match a folder under:

```text
/Users/edwardwang/Desktop/projects/datasets/swe-bench/swe-bench-team-tasks/
```

## Docker Images

The repo-level arm64 images were exported as tar files under:

```text
/Users/edwardwang/Desktop/projects/datasets/swe-bench/docker-arm64-venv/image-tars/
```

On the Linux/arm64 runtime machine, load them before training:

```bash
for tar in /path/to/docker-arm64-venv/image-tars/*.tar; do
  docker load -i "$tar"
done
```

The workflow can build the thin per-task images automatically:

```yaml
prebuild_images: true
```

If you prebuild them yourself, set:

```yaml
prebuild_images: false
```

## Docker Runtime Requirement

This workflow starts task environments using Docker Compose from inside the AReaL
runtime. Like the terminal-bench example, the AReaL runtime container needs access to
host Docker:

```bash
-v /var/run/docker.sock:/var/run/docker.sock
-v /usr/bin/docker:/usr/bin/docker:ro
-v /usr/libexec/docker/cli-plugins:/usr/libexec/docker/cli-plugins:ro
```

## Run Training

SGLang config:

```bash
python3 examples/swe-bench/train.py \
  --config examples/swe-bench/config_swebench_sglang.yaml
```

vLLM/NPU config:

```bash
python3 examples/swe-bench/train.py \
  --config examples/swe-bench/config_swebench_vllm_npu.yaml
```

Before running, edit the config values for:

```yaml
swebench_root: /Users/edwardwang/Desktop/projects/datasets/swe-bench
train_dataset.path: swe-bench/swe-bench-team-tasks.parquet
valid_dataset.path: swe-bench/swe-bench-val.parquet
actor.path: Qwen/Qwen3-8B
```

## Runtime Flow

For each task and trajectory, the workflow:

1. Builds the thin task image with `docker_build.sh` if `prebuild_images` is true.
2. Starts one Docker Compose project for that trajectory.
3. Runs mini-swe-agent inside the task container.
4. Evaluates with `/eval.sh`.
5. Parses the eval log with `run_eval_with_metrics.py`.
6. Uses `metrics["swebench"]["reward"]` as the rollout reward.
7. Stops the Compose project.

Parallel rollouts are isolated by unique trajectory ids and Compose project names.
