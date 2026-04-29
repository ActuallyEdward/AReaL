# SWE-bench Agent

This package mirrors the role of `examples/terminal_bench/agent`, but uses the
mini-swe-agent control loop instead of CAMEL's terminal agent.

Main pieces:

- `MiniSweBenchAgent`: high-level rollout agent for AReaL workflows.
- `ArealMiniModel`: mini-swe-agent `Model` adapter backed by AReaL's OpenAI-compatible client.
- `SweBenchComposeEnvironment`: mini-swe-agent `Environment` adapter backed by the generated
  SWE-bench Docker Compose task folders.

The expected task runtime lives under:

```text
/Users/edwardwang/Desktop/projects/datasets/swe-bench/swe-bench-team-tasks/<instance_id>/
```

Each rollout uses a unique `traj_id`, starts that task's Compose project, lets the mini agent
edit the mounted repo, runs `/eval.sh`, parses metrics through `run_eval_with_metrics.py`, and
then stops the Compose project.

