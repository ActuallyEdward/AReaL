# SWE-bench Workflow

This mirrors `examples/terminal_bench/workflow`, but launches the generated
SWE-bench Docker Compose tasks and runs the mini-swe-agent based agent.

Main entry point:

```text
workflow.mini_swe_bench_workflow.MiniSweBenchWorkflow
```

Expected task data:

- `instance_id` or `task_name`
- `problem_statement` or `instruction`

Expected runtime assets:

```text
/Users/edwardwang/Desktop/projects/datasets/swe-bench/swe-bench-team-tasks/<instance_id>/
```

The workflow optionally prebuilds the task image, runs `n_trajs` parallel rollouts, parses
reward through the agent/eval layer, records rollout stats, and exports AReaL interactions.
