# The Hidden-State Classroom: an RL environment where frontier agents teach worse than a trained policy

AfterQuery "Break the Agent", Junction 2026 (team Sauna Renaissance).

An agent tutors six simulated 10-year-olds in fractions, one after another. Each child has a hidden state the agent
never sees: which of six prerequisite-linked concepts they have mastered, how fast each memory fades, and whether
they hold one of three documented misconceptions. The agent has 30 turns per child (three sessions, Monday /
Wednesday / Friday) to `explain`, `quiz`, `correct` a named misconception, or `ask` (free, unreliable self-report).
The reward is the children's exam gain next Monday, pooled over the six children and normalised so that no teaching
scores 0 and a teacher who could see each child's hidden state scores about 1.

**Demo page:** `viz/simple.html` (teach a child yourself, x-ray the hidden state, watch an AI agent and the reference
teach the same child, and the evidence). It loads a data file, so serve the folder:
`cd viz && python3 -m http.server 8000`, then open http://localhost:8000/simple.html. A fuller version is `viz/game.html`.

## Result

Through the standard harness (Harbor + OpenCode over MCP), on salted evaluation children (salt `eval1`):

| policy | mean reward, batches 3001-3005 (6 children each) |
|---|---|
| **trained PPO reference** (observation-only) | **0.70** |
| gpt-6.1-sol (3 runs) | 0.61 |
| claude-opus-5-5 | 0.59 |
| gemini-3.1-pro-preview | 0.55 |
| gemini-pro-latest | 0.52 |
| gemini-3.5-flash | 0.52 |
| do-nothing / stop-early / Monday-only | 0.00 / 0.00 / 0.07 |

On ten more batches (3006-3015) the reference wins 19 of 20 Gemini runs. **Overall the reference wins 39 of 43
frontier-agent runs.**

### Where the agents go wrong
One decision explains the gap. An unlearned child still answers a quiz right 1 time in 4 (the agent is told this).
After one right answer the chance that the idea was learned is only about 71%; after two in a row about 90%. Agents
move on after a single right answer, and so build on ideas the child never learned: 1.4-2.0 times per child, against
0.5 for the reference. Switching this one habit off in our rule-based teacher costs about 8 points on a child's exam,
about the size of the whole AI-reference gap (7 points). In one logged run, Gemini 3.1 Pro (given every simulator
parameter) wrote that it would "use multiple quizzes to overcome the 25% guess chance", then moved on after four lucky
guesses with one child, who learned nothing all week. A high quiz share is *not* the cause: our rule-based teacher
quizzes as much as the agents (67% vs 68% of turns) and still wins clearly.

### Two references
- **PPO (trained)**: a small MLP (2 x 128) trained with PPO on about half a million simulated children with random
  seeds, never the evaluation children. It sees only the agent's observations. It is the reference the harness reruns
  (`-a oracle`). 0.73 on 1,200 fresh children.
- **Five rules (hand-written)**: written by us around the principle the trained policy also follows, never build on
  an unconfirmed idea; six numbers tuned on training seeds (`probe/planned_ref.py`). 0.67 on the same 1,200 children.
  It beats Gemini in 18 of 20 paired batches. PPO reaches the same goal differently: it often teaches ahead and checks
  earlier ideas later (`probe/checkback_ref.py`).

### Is the gap our fault? Four checks

| objection | what we did | result |
|---|---|---|
| the agent lacks information | gave it every simulator parameter (`harbor/tutor2_disclosed`) | Gemini +0.01 and +0.03 over 15 paired batches, Opus +0.07 over 3; the reference wins 29 of 33 |
| it doesn't know good tutoring | added four lines of tutoring advice (`harbor/tutor2_hint`) | no consistent gain (Gemini +0.02 / -0.05, Opus +0.02); same way of playing |
| the simulator is unrealistic and real-world knowledge misleads it | same simulator and children told as abstract "nodes" (`harbor/tutor2_abstract`) | no model does better without the story (Gemini -0.07 / -0.06 over 15 batches, Opus +0.03 over 3); the reference wins all Gemini runs |
| our numbers make it lose | 27 variations of the testing effect, forgetting speed and learning speed (`probe/sensitivity.py`) | the agents' habits lose in all 27 |

### Tiebreaker: a small model learning from the environment
Gemma 4 E4B trained by expert iteration on its own reward-filtered episodes (`gemma/`): 0.37 -> 0.39 -> 0.41 over two
rounds (paired +0.04, se 0.04: a trend, not yet significant). The PPO reference is itself a small network trained only
on the environment's experience.

Full lab notebook, including the experiments that failed and every correction: `probe/RESULTS.md`.

## For judges: how to rerun

Requirements: Docker, Python 3.12+, and a model provider key. Tested from a clean copy of the repository.

```sh
cd afterquery
python3 -m venv .venv && .venv/bin/pip install "harbor==0.25.0" numpy
cp .env.example .env            # put your provider key in .env (never commit it)
./eval.sh gemini/gemini-pro-latest 3001 3002 3003
```

`eval.sh` runs, through Harbor, on each seed: the reference (`-a oracle`, which executes
`harbor/tutor2/solution/solve.sh`: the PPO policy playing through the same MCP tools), the OpenCode agent with the given
model, the do-nothing baseline (`-a nop`) and the stop-early baseline (`harbor/baseline_agents.py:StopEarlyAgent`), then
prints a table. Model names use Harbor's provider prefixes (`gemini/...`, `openai/...`, `anthropic/...`).

Our numbers to compare with (salt `eval1`, the default in `harbor/run_seeds.sh`):

| seed | reference | do-nothing | stop-early |
|---|---|---|---|
| 3001 | 0.675 | 0.00 | 0.00 |
| 3002 | 0.776 | 0.00 | 0.00 |
| 3003 | 0.848 | 0.00 | 0.00 |

The reference is deterministic, so these reproduce exactly. Agent scores vary run to run (on 3001-3003 the frontier
models scored 0.37-0.75; the reference wins on average, not on every single run). For children nobody has seen, set your own secret salt: `AFTERQUERY_SEED_SALT=<anything>`.
Other variants: `TASK=tutor2_disclosed`, `TASK=tutor2_hint`, `TASK=tutor2_abstract`. Single runs:
`harbor/run_seeds.sh <agent> <model|-> <tag> <seed...>`.

## Layout

| path | what |
|---|---|
| `harbor/tutor2/` | the Harbor task: `instruction.md` (what the agent reads), `task.toml`, `environment/` (agent container + MCP server sidecar holding the simulator), `solution/` (PPO reference agent), `tests/` (separate verifier) |
| `harbor/tutor2_disclosed/`, `tutor2_hint/`, `tutor2_abstract/` | the three check variants |
| `harbor/baseline_agents.py` | stop-early and Monday-only baselines |
| `harbor/results/` | every run's result (one TSV per experiment) |
| `probe/tutor2.py` | the simulator (single child); `probe/rules_tutor2.txt` the rules |
| `probe/rl_ref.py`, `probe/ppo_policy.py`, `probe/ppo_ref.npz` | PPO training and the numpy policy used at run time |
| `probe/planned_ref.py`, `probe/checkback_ref.py` | the five-rule teacher and the check-back variant |
| `probe/failure_analysis.py`, `probe/find_quotes.py`, `probe/sensitivity.py` | failure analysis, quote finder, sensitivity analysis |
| `probe/RESULTS.md` | every experiment, including the ones that failed |
| `viz/` | demo page, full game, JavaScript port of the simulator (checked against Python on 228 replays) |
| `gemma/` | Gemma tiebreaker: text harness, training, results |
| `REFERENCES.md` | the sources for every mechanism, and which numbers are our own settings |

## Integrity

- The simulator lives only in the `tutor` sidecar; the agent container has no simulator code and, under the network
  allowlist, cannot reach anything but the MCP server and model APIs.
- `AFTERQUERY_SEED_SALT` mixes a secret into every child's seed, so a public copy of the simulator cannot replay them.
- The verifier runs in a separate container and reads a result file the agent can only read, not write.
- `ask` uses independent random draws per concept (an earlier shared draw leaked the true state; fixed).
- A child's exam runs automatically after the 30th turn; a run that does not complete all six children scores 0;
  finishing early loses turns.
- Normalised rewards can exceed 1 on rare children: the full-information teacher is strong but not a proven optimum.

## Why these mechanics (model card)

| mechanism | source | simplification |
|---|---|---|
| mastery as a hidden binary state, quiz slip 0.10 / guess 0.25 | Bayesian Knowledge Tracing (Corbett & Anderson 1995) | binary mastery; one exercise per concept |
| forgetting as a half-life, new 2.6-7.8 days | FSRS-5 default first-review stabilities (Anki), converted to half-lives | exponential forgetting |
| correct recall x2.5 the half-life, larger when recall had faded | testing effect (Roediger & Karpicke 2006); SM-2 ease factor 2.5; spacing | fixed multipliers |
| repeated explanation in one session helps less (x0.6 each time) | massed vs spaced practice (Cepeda et al. 2006) | fixed factor |
| prerequisite chain | knowledge space theory; fraction curriculum order | a single chain |
| three misconceptions with signature answers | documented fraction errors (Ni & Zhou 2005; Brown & Burton 1978) | three only; targeted correction succeeds 70% |
| over-confident self-report | illusion of explanatory depth (Rozenblit & Keil 2002) | fixed probabilities |

The sources back each effect; the specific numbers are our settings, except the forgetting speeds (FSRS defaults).
Full list: `REFERENCES.md`. This is a stylised model of tutoring, not a model of real children; its job is to pose a
fair, hard sequential decision problem under hidden state. Calibrating it on real student logs (ASSISTments, Eedi) is
the natural next step.

MIT licensed. All code was written during the event.
