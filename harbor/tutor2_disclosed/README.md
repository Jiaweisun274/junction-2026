# Harbor task: AfterQuery tutor2

This packages `probe/tutor2.py` as a Harbor task. OpenCode sees the rules and calls the tutor through MCP; the simulator implementation lives only in the `tutor` sidecar image. The sidecar writes the final result to a named volume mounted read-only in the agent container. Harbor collects that artifact and passes it to a separate verifier container, which writes the numeric reward.

## Local setup

Install Harbor and make sure Docker is running. Supply the model provider key in the shell that launches Harbor; do not put it in this task package.

```sh
# Put your key in the ignored .env file using a local editor, then:
AFTERQUERY_SEED=2001 .venv/bin/harbor run --env-file .env -p harbor/tutor2 -a opencode -m openai/<model-id> -k 1
```

Set `AFTERQUERY_SEED` to another integer to play a different deterministic instance. The default is `2001`. Repeat with fresh task trials per seed and model. The task defaults to OpenCode; the model identifier and provider must match the credentials configured in your Harbor environment.

The task image installs the MCP Python SDK for both the sidecar server and the reference client. The agent container receives no simulator source. It can read the final score artifact after the game, but cannot modify it through its read-only volume mount.

## Anti-exploit settings

- **Network allowlist** (`task.toml`): the agent container can reach only the `tutor` sidecar and model API hosts (Harbor adds the run's inference host automatically). It cannot download this public repository and replay the simulator.
- **Secret seed salt**: set `AFTERQUERY_SEED_SALT` (any string kept out of the repo) in the shell that launches Harbor. The sidecar mixes it with `AFTERQUERY_SEED` (SHA-256), so the hidden child differs from the public seed even if the agent knew the simulator. The salt lives only in the sidecar's environment. Without a salt the public seed is used directly (fine for local debugging, not for evaluation).
- The agent container never contains simulator source; the result file is written by the sidecar and mounted read-only; the verifier runs in a separate container.

```sh
AFTERQUERY_SEED=2001 AFTERQUERY_SEED_SALT=<secret> .venv/bin/harbor run --env-file .env -p harbor/tutor2 -a opencode -m openai/<model-id> -k 1
```

## Scoring

The reward is `(agent exam - no-teaching exam) / (full-information rollout - no-teaching exam)`. It may exceed 1 because the rollout is a strong reference, not a proven optimum. `reward.json` also records the raw exam score, literature-reference score, oracle comparison score, and turns used.

The reference solver in `solution/` uses the observation-only literature policy from `probe/lit_ref.py` and plays through the same MCP tools. The task instance seed is set through `AFTERQUERY_SEED` for both agent and reference runs.

## Current scope

Harbor 0.25.0 is installed in `.venv`; the ignored `.env` supplies provider keys.
Verified on Sat 14 Nov with the allowlist and a salted seed (`AFTERQUERY_SEED=2001`, salt `smoketest`):
- OpenCode + `openai/gpt-5.4-mini`: completed 30 turns through MCP, reward 0.304.
- Reference (`-a oracle`, runs `solution/solve.sh`): reward 0.607, identical to the verifier's own reference computation.
Harbor job folders (`jobs/`) are not committed; re-run to reproduce.
