# Phase 2 — critique of the tutor2 environment and an 18-hour plan

Read: `HANDOFF.md`, `afterquery/probe/RESULTS.md`, `afterquery/harbor/tutor2/` (instruction, task.toml,
compose, MCP server, tests, solution), `probe/tutor2.py`, `rl_ref.py`, `ppo_policy.py`.
New evidence in this file comes from `fable/analysis/check_tutor2.py`, which runs scripted policies on the
existing simulator (100 held-out seeds 2041–2140, no model APIs). It reproduces the notebook's PPO numbers
exactly (0.699 held-out; 0.96 / 0.56 / 0.78 / 0.90 / 0.40 on the five salted instances).

**Verdict: keep the tutoring environment. Do not switch and do not re-skin.** The gap is not yet
established, the reference is thinner than it looks, and there is one real information leak. All three are
fixable tonight. The order matters: fix and freeze, then measure on fresh seeds, then write.

## 1. What is strong

- **The lab notebook.** Five probes, failures recorded, a designer-induced flaw found and fixed (4a), a
  calibration check that hurt the team's own result. Few teams will have this. It is the best asset for
  the 10-minute demo and for a PhD application.
- **The harness work is done.** Harbor + OpenCode + MCP sidecar, separate verifier, read-only result
  volume, salted seed, network allowlist, reference replayed through MCP with an identical score.
- **Server-side scoring.** The agent reports nothing; the exam is a deterministic expectation (no exam
  sampling noise); do-nothing scores exactly 0; stop-early is roughly proportional (PPO stopped after
  5 / 10 / 20 turns: 0.15 / 0.23 / 0.47).
- **The failure story is decision-level.** Moving on after one lucky quiz, then re-explaining a concept
  whose prerequisite is missing. It is concrete and visible in trajectories.
- **The reference acts on observations only** and its training seeds are disjoint from test seeds.

## 2. What is weak or risky

### 2.1 The gap is not statistically established (largest risk to the 30-point item)
Paired differences on the five Harbor seeds, PPO minus agent:

| Model | Per-seed differences | Mean | SD | t (n=5) |
|---|---|---|---|---|
| gpt-6.1-sol | 0.03, 0.21, 0.18, 0.46, −0.27 | 0.12 | 0.27 | 1.0 |
| Opus 5.5 | 0.40, 0.32, 0.23, 0.46, −0.31 | 0.22 | 0.31 | 1.6 |

Neither is significant. One child's score is dominated by luck: PPO's per-child SD is 0.31, and the
paired SD between PPO and a simple heuristic is 0.42. With a true gap of 0.15 and paired SD 0.30, three
single-child judge runs put the reference ahead with probability 0.81. That is one failure in five on the
rubric's biggest item. On the default seed 2001 the scores are 0.96 against 0.93, a tie.

### 2.2 A 30-line rule nearly matches PPO, and it beats both frontier agents
Scripted policy: explain the frontier concept, quiz it, move on after two correct answers in a row, use
`correct` when a misconception signature shows. Nothing trained, nothing tuned.

| Policy (100 held-out seeds) | Mean | SD | Beats PPO on a seed |
|---|---|---|---|
| PPO reference | 0.699 | 0.309 | — |
| two-in-a-row + misconception correction | 0.671 | 0.353 | 40% |
| three-in-a-row + correction | 0.641 | 0.313 | 46% |
| one correct + correction | 0.460 | 0.315 | 22% |

On the five salted instances the two-in-a-row rule scores 0.94, 0.79, 0.89, 0.56, 0.13, mean 0.66. That
is above gpt-6.1-sol (0.60) and Opus (0.50) and below PPO (0.72).

Two consequences. First, PPO's lead over a trivial rule is 0.03, which is inside the noise (standard
error 0.04). Calling PPO a "stronger reference" overstates it. Second, this is good news if used
properly: "frontier agents lose to a 30-line mastery rule they could have written" is a better
difficulty story than "they lose to a neural network", and it removes the black-box objection. The
notebook reports naive at 0.54 because that version lacks the `correct` step.

Note also that the confirmation rule that wins has flipped between simulator versions. In Probe 3 the
agents lost by quizzing twice; now quizzing twice is the near-best rule. A judge reading the notebook
will see that.

### 2.3 Fairness: the reference had the simulator, the agent has prose
PPO trained on 15 million simulator steps. The agent gets qualitative rules ("some chance", "varies by
child", "helps less each time") with no numbers for learn rates (0.3–0.6), the prerequisite leak (0.05),
massed damping (0.6), half-life ranges, misconception rates (0.4 × 0.6) or fix rates (0.7 / 0.1). The
network allowlist is there to stop the agent getting the simulator. The rubric gives zero for a gap that
comes from withheld information. "The reference only sees observations at run time" does not answer
this, because the knowledge sits in its weights. This is the attack I would make as a judge.

### 2.4 Reward integrity: one real leak, several small holes
- **`ask` is a free side channel.** `Sim.ask` uses the same random numbers `u[turn][3:6]` for every
  concept within a turn. On 20% of turns the six answers are mixed, and then "no" means not known and
  "yes" means known. Asking all six every turn costs nothing. It lifts the one-correct heuristic from
  0.46 to 0.63. My quick attempt did not beat PPO with it, but the instructions describe `ask` as
  unreliable while the implementation makes it a periodic state read-out. A careful agent or judge can
  find this.
- **No `finish`, no score.** `tests/test.sh` gives 0 if the result is not marked completed. An agent
  that plays all 30 turns and hits the 1800 s timeout, or forgets `finish`, scores 0. A judge can call
  that a trick, and it adds variance to agent runs.
- **The normaliser is not an upper bound.** PPO scores above 1 on 9% of held-out seeds (maximum 1.68);
  the denominator ranges from 0.19 to 0.63 across children, so some children count three times as much
  as others.
- **Massed quizzing has no diminishing returns.** Each correct quiz multiplies the half-life by at
  least 2.5, even seconds after the previous one, while explanations are damped at 0.6. In my test,
  lock-in quizzes add only 0.02–0.04, so it is a realism inconsistency more than an exploit.
- **The salt is public and optional.** `eval1` is the default in `run_seeds.sh`; with no salt the
  public seed is used. The README command must set one.
- `npm i -g opencode-ai@latest` is unpinned, so a rerun next week may behave differently.

### 2.5 Realism
- The honest description is "a stylised POMDP with literature-shaped dynamics", not "a simulated child".
  Parameters are chosen, not fitted, except the half-life range. The team's own Duolingo fit disagreed
  with the first version by a factor of 90.
- Mechanisms were added after seeing agent results (forgetting, then calibration, then damping), and the
  reference changed three times (tuned heuristic, literature, PPO). The notebook is honest about it, but
  the final table must come from a frozen version on seeds never used before.
- One fixed question per concept, a strict chain, and one misconception per concept are visibly toy.
  Say so first, before a judge does.
- Citations were written from memory (handoff issue 6). Check each before submission. Prior work to
  check and cite if correct: Rafferty et al. 2016, "Faster teaching via POMDP planning"; Corbett and
  Anderson 1995 (BKT); Settles and Meeder 2016 (half-life regression).

### 2.6 Craft
- `harbor/tutor2/README.md` says the reference solver is the literature policy; `solve.sh` runs PPO.
- `HANDOFF.md` section 2–3 describes the 1-day half-life version and the old results.
- No single command yet that runs reference, do-nothing, stop-early and prints a table.
- PPO retraining needs torch and is not wired into a documented command.

## 3. What was missed

1. **Batching children per run.** The cheapest variance fix. Six children per Harbor run takes the
   3-run win probability from 0.81 to 0.98 under the assumptions in 2.1. A 30-turn game took about two
   minutes in OpenCode, so six take about twelve, inside the timeout.
2. **Giving the agent the simulator.** The salt already makes the hidden child unreproducible, so the
   source can be placed in the agent container. This turns the fairness attack into a strength.
3. **A transparent reference.** Section 2.2.
4. **Counterfactual hand-over.** The simulator is deterministic per turn index. Replay an agent's
   trajectory up to turn t, let the reference finish, and plot final score against t. The turn where the
   curve drops is the costly decision. This ties mistakes to decisions with numbers, which is what the
   second 30-point item asks for, and it is a research-style result.
5. **Pooled normalisation.** Score a batch as sum of gains over sum of full-information gains. This
   removes the scores above 1 and the unequal child weights.
6. **Telling the agent it may script the tools.** The MCP endpoint is on localhost and the `mcp` client
   is installed in the agent image. One sentence in the instructions makes "tools were available"
   undeniable.

## 4. Keep, re-skin or switch

| Option | For | Against | Decision |
|---|---|---|---|
| Keep tutor2, fix and freeze | Harness done, agent evidence exists, notebook is an asset | Needs the fixes below | **Yes** |
| Re-skin as an abstract game | Ends the realism debate | Loses the literature anchor, changes agent priors, needs every run repeated, adds nothing to the gap | No |
| Switch to the Phase 1 wind-farm design | Cleaner reward (money), reference needs no training, scripted gap 0.57 to 0.83 | Zero agent runs; Phase 1 found that a design that looks hard on paper can fail; no Harbor packaging; 18 hours alone | No |

The Phase 1 prototypes are useful in the write-up as two more negative or untested probes, nothing more.

## 5. Plan for the remaining hours (one person; clock times assume it is about 16:00)

Rule for the night: no new mechanisms. Bug fixes, measurement and writing only.

**Block 1, 16:00–17:30 — fix and freeze (v4)**
1. `ask`: draw from an independent stream per (turn, concept), for example
   `np.random.default_rng([int(seed) + 99, turn, k]).random(3)`. PPO does not use `ask`, so no retraining.
2. Unfinished games: the sidecar writes a provisional result after every turn, scored on the current
   state. Reaching turn 30 finalises automatically.
3. Batch mode: `AFTERQUERY_CHILDREN=6`; `finish` moves to the next child; reward =
   sum(exam − no-teaching) / sum(full-info − no-teaching). Update `ppo_agent.py` to loop.
4. Instructions: add a parameter table with every prior and rate; copy `tutor2.py` into the agent image
   and say so; say the tools may be scripted at `$AFTERQUERY_MCP_URL`.
5. Add the two-in-a-row rule as `solution/rule_agent.py`; add do-nothing and stop-early (10 turns)
   scripts.
6. Pin the OpenCode version. Sync the three copies of `tutor2.py`. Commit and tag `v4-frozen`.
7. In the background: retrain PPO for 8,000 iterations instead of 2,000 (about 40 minutes of CPU). Use
   it only if it beats the current weights on seeds 2041–2140; state which seeds chose it.

Ask the mentors now, in parallel: is a trained policy acceptable as the reference, and do judges rerun
one seed three times or three seeds.

**Block 2, 17:30–19:30 — the measurement that decides everything**
Fresh seeds 3001–3030, never used so far, salt set in the shell. Per model, 5 runs of 6 children:
- Condition A: rules with numbers and simulator source.
- Condition B (if credits allow, 2–3 runs): current prose-only rules.
Also reference (PPO), two-in-a-row rule, do-nothing, stop-early on the same batches.

Decision at 19:30:
- Both agents below the rule and PPO in at least 4 of 5 batches under A: done. Headline is "agents with
  the simulator in hand lose to a 30-line rule and to a small trained policy".
- Gap holds under B only: report both honestly as an ablation. The claim becomes "agents do not use the
  model even when they have it" if A is close, or the gap is conceded as information-driven. Do not
  hide A.
- No gap in either: the submission is the notebook plus an honest negative result with integrity and
  craft points intact. Do not start a new environment at 19:30.

**Block 3, 19:30–22:00 — why the agents lose**
From the trajectories, per run and for PPO and the rule on the same children: share of concepts left
after a single correct quiz; longest run of explanations of one concept in a session; prerequisite
re-checks after a failed explanation; highest concept reached; review share in the Friday session.
Then the counterfactual hand-over curve for three or four losing runs. One table and one figure.

**Block 4, 22:00–23:30 — craft**
`./eval.sh` (or `make eval`) builds the images and runs reference, rule, do-nothing and stop-early
through Harbor, then prints the table; a flag adds three agent runs when a key is present.
`make train` reproduces the PPO weights. Rewrite the harbor README, `HANDOFF.md` and root README to the
frozen version. Add a reward-integrity section listing each attack tried and its result, including the
`ask` leak that was found and fixed. Add a limitations section (stylised, not validated on children,
normaliser is a benchmark). Make the repo public.

**Block 5, 23:30–00:30 — video draft, then sleep.** Two minutes: setting, table, one failure trajectory,
one command.

**Sunday 07:00–10:00.** Rerun `./eval.sh` from a clean clone. Final video. Submit by 09:30.

**Cut list.** Gemma fine-tune: only if Block 4 ends before 22:00, which it will not. The PPO policy is
itself a 26k-parameter model trained on environment experience; say that for the tiebreaker. Spread:
no v0 exists and the kill rule passed at noon. The final occupies 07:30–08:35 on Sunday, directly
before the deadline. Skip it, or start the unmodified starter and do not watch it.

## 6. Top five, in order
1. Batch six children per run and pool the normalisation; rerun on fresh seeds 3001–3030.
2. Publish all parameters and the simulator source to the agent, then measure whether the gap survives.
3. Add the 30-line two-in-a-row rule as a second, transparent reference next to PPO.
4. Fix the `ask` leak and the unfinished-game zero, then freeze the simulator.
5. Counterfactual hand-over analysis and a one-command `eval.sh`; cut Gemma and Spread.
