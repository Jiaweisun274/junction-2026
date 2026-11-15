# Example Harbor runs

Complete Harbor output of `./eval.sh gemini/gemini-3.5-flash 3001`, run from a fresh clone of this repository on a
MacBook (Apple Silicon, Docker Desktop). Six children, salt `eval1`.

| folder | what played | reward |
|---|---|---|
| `reference/` | PPO reference (`-a oracle`) | 0.675 |
| `agent/` | OpenCode with `gemini/gemini-3.5-flash` | 0.358 |
| `nop/` | do-nothing baseline (`-a nop`) | 0.000 |
| `stop/` | stop-early baseline (`StopEarlyAgent`) | 0.000 |

In each run, `result.json` holds the verifier's reward. In `agent/`, `agent/opencode.txt` is the full playthrough:
every tool call the model made and every answer the child gave. OpenCode's internal state folder was left out.
These runs predate `reward.txt`; the verifier now writes it as well.
