# Fractions Tutor — an RL environment where frontier agents teach worse than a trained policy

AfterQuery "Break the Agent", Junction 2026 (team Sauna Renaissance).

An agent tutors six simulated 10-year-olds in fractions, one after another. Each child has a hidden state the
agent never sees: which of six prerequisite-linked concepts they have mastered, how fast each memory fades, and
whether they hold one of three documented misconceptions. The agent has 30 turns per child (three sessions, Monday /
Wednesday / Friday) to `explain`, `quiz`, `correct` a named misconception, or `ask` (free, unreliable self-report).
The reward is the children's exam gain next Monday, pooled over the six children and normalised so that no teaching
scores 0 and a teacher who could see each child's hidden state scores about 1.

**Result.** Through the standard harness (Harbor + OpenCode over MCP), on fresh salted children:

| policy | mean reward (5 runs x 6 children) |
|---|---|
| trained PPO reference (observation-only) | **0.70** |
| claude-opus-5-5 | 0.59 |
| gpt-6.1-sol (3 runs) | 0.61 |
| gemini-3.1-pro-preview | 0.55 |
| gemini-pro-latest | 0.52 |
| gemini-3.5-flash | 0.52 |
| learning-science rulebook (BKT mastery + spaced review), no training | 0.56 |
| do-nothing / stop-early / Monday-only | 0.00 / 0.00 / 0.07 |

The reference wins 20 of 23 frontier-agent runs. Telling the agent **every simulator parameter** does not close the gap
(Gemini: 0.52 -> 0.56 and 0.55 -> 0.52; the reference still wins 9 of 10), so the gap comes from decisions, not from
withheld information. Frontier agents reach the level of a textbook rulebook; the decisions that beat it (when to stop
verifying, checking the prerequisite after a failed concept, spending the last turns teaching instead of re-quizzing)
are learned from experience in the environment. Full lab notebook: `probe/RESULTS.md`.

## Run it

Requirements: Docker, Python 3.12+, and a model provider key.

```sh
cd afterquery
python3 -m venv .venv && .venv/bin/pip install "harbor==0.25.0" numpy
cp .env.example .env            # then put your provider key in .env (never commit it)
./eval.sh gemini/gemini-pro-latest 3001 3002 3003
```

`eval.sh` runs, through Harbor: the reference run on each seed (`-a oracle`, which executes `harbor/tutor2/solution/solve.sh`,
the trained PPO policy playing through the same MCP tools), the OpenCode agent with the given model on each seed, the
do-nothing baseline (`-a nop`) and the stop-early baseline, then prints a table. Model names use Harbor's provider
prefixes (`openai/...`, `anthropic/...`, `gemini/...`). Use `TASK=tutor2_disclosed` to give the agent every simulator
parameter. Single runs: `harbor/run_seeds.sh <agent> <model|-> <tag> <seed...>`.

## Layout

| path | what |
|---|---|
| `harbor/tutor2/` | the Harbor task: `instruction.md` (what the agent reads), `task.toml`, `environment/` (agent container + MCP server sidecar holding the simulator), `solution/` (PPO reference agent), `tests/` (separate verifier) |
| `harbor/tutor2_disclosed/` | same task, instruction lists every simulator parameter |
| `harbor/baseline_agents.py` | stop-early and Monday-only baselines |
| `probe/tutor2.py` | the simulator (single child); `probe/rules_tutor2.txt` the rules |
| `probe/rl_ref.py`, `probe/ppo_policy.py`, `probe/ppo_ref.npz` | PPO training (observation-only features) and the numpy policy used at run time |
| `probe/lit_ref.py`, `probe/rule_ref.py` | training-free references (learning-science rulebook; simple mastery rule) |
| `probe/RESULTS.md` | every experiment, including the ones that failed |
| `fable_review/` | an independent design review and critique |

## Integrity

- The simulator lives only in the `tutor` sidecar; the agent container has no simulator code and, under the network
  allowlist, cannot reach anything but the MCP server and model APIs.
- `AFTERQUERY_SEED_SALT` mixes a secret into every child's seed, so a public copy of the simulator cannot replay them.
- The verifier runs in a separate container and reads a result file the agent can only read, not write.
- `ask` uses independent random draws per concept (an earlier shared draw leaked the true state; fixed).
- A child's exam runs automatically after the 30th turn; forgetting `finish` costs nothing; finishing early loses turns.
- Normalised rewards can exceed 1 on rare children: the full-information teacher is strong but not a proven optimum.

## Why these mechanics (model card)

| mechanism | source | simplification |
|---|---|---|
| mastery as a hidden binary state, quiz slip 0.10 / guess 0.25 | Bayesian Knowledge Tracing (Corbett & Anderson 1995) | binary mastery; one exercise per concept |
| forgetting as a half-life, new 2.6-7.8 days | FSRS-5 default first-review stabilities (Anki), converted to half-lives | exponential forgetting |
| correct recall x2.5 the half-life, larger when recall had faded | testing effect (Roediger & Karpicke 2006); SM-2 ease factor 2.5; spacing | fixed multipliers |
| repeated explanation in one session helps less (x0.6 each time) | massed vs spaced practice | fixed factor |
| prerequisite chain | knowledge space theory; fraction curriculum order | a single chain |
| three misconceptions with signature answers | documented fraction errors (e.g. whole-number bias, adding across) | three only; targeted correction succeeds 70% |
| over-confident self-report | metacognition / illusion of explanatory depth | fixed probabilities |

This is a stylised model of tutoring, not a model of real children; its job is to pose a fair, hard sequential
decision problem under hidden state. Calibrating it on real student logs (ASSISTments, Eedi) is the natural next step.

MIT licensed. All code was written during the event.
