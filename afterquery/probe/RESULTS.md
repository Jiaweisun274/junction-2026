# Probe 1 (Fri 13 Nov evening): two small daily-life games, blind Claude subagent

Setup: subagent (Claude Opus-class, Python allowed, could not read the source) played 3 seeds per game from a rules text.

## A. Sell a used bike (optimal stopping, unknown value, Normal prior given)
| seed | agent | reference (Bayesian DP) | oracle (known value) | 37% rule |
|---|---|---|---|---|
| 1 | 351 | 351 | 351 | 50 |
| 7 | 265* | 337 | 337 | 50 |
| 10 | 165 | 165 | 247 | 50 |
*execution slip: batched several `reject` calls and skipped an offer its own model said to accept.
Agent recognised the problem and wrote the same Bayesian DP. No decision gap.

## B. Shared apartment sauna (opponent modelling, 4 routine types listed in rules)
| seed | agent | reference (fit routines, myopic) | oracle (true routines) | always 19 |
|---|---|---|---|---|
| 6 | 46.5 | 51.0 | 55.0 | 19.5 |
| 9 | 54.0 | 54.0 | 55.0 | 19.5 |
| 10 | 52.5 | 52.0 | 55.0 | 24.0 |
Agent fitted every neighbour's routine within ~4 days and planned 7 evenings ahead. No gap.

## Lesson
When the rules hand over the full model class (prior, noise, list of routine types), a frontier agent with Python
does "fit the model, then optimise" as well as our reference. Gap must come from something that is not a
one-shot fit-then-solve: model class not enumerated, long horizon with many interacting parts,
costly exploration in a large space, or the agent's own actions confounding its data.

# Probe 2 (Fri 13 Nov late): ferry boarding gate, faster-is-slower, shape not given, unannounced degradation
Throughput f(q) = c*q/(1+(q/qs)^2), unknown qs/peak per seed; at a random minute 20-35 qs halves, peak -20%. 50 steps.
Rules only say throughput depends on gate crowding "in a way you have to find out" and "conditions may change".
| seed | agent run 1 | agent run 2 | reference (sweep, hold best, re-sweep on drop) | oracle | call 10/min | flood 40/min |
|---|---|---|---|---|---|---|
| 1 | 402 | 389 | 402 | 444 | 283 | 55 |
| 3 | 382 | 384 | 341 | 400 | 246 | 31 |
| 4 | 394 | 396 | 328 | 373 | 199 | 26 |
Both agents discovered the non-monotone curve (best crowd ~12-16), noticed the mid-game drop, and wrote adaptive
scripts. They beat our quickly-written reference on seeds 3-4. Caveat: each agent played 3 seeds in a row and
carried learning across games, which a fresh Harbor episode would not allow; on the first game (seed 1) they scored
402 and 389 vs reference 402. Still no gap.

## Lesson after 3 probes
Small, low-dimensional problems with clear per-step feedback are solved: the agent explores, fits, adapts,
even without being told the model. A gap probably needs what Vending-Bench has: many concurrent decisions competing
for attention, long horizons (hundreds of steps, growing context), delayed or confounded consequences,
or truly adaptive (learning) opponents. Trade-off: this conflicts with "keep it small".

# Probe 3 (Fri 13 Nov, ~midnight): tutoring a simulated child (hidden mastery, prerequisite chain, over-confident self-reports)
6 concepts in a prerequisite chain, 30 turns. explain (chance to learn, ~0 if prerequisite missing), quiz (slip 0.1 / guess 0.25, stated),
ask (free, "yes" 95% if mastered / 75-90% if not, stated qualitatively). Reward = (exam - exam with no teaching) / (oracle exam - same).
One fresh blind agent per seed (no carry-over between games).
| seed | agent | first reference (hand-written) | tuned reference | oracle |
|---|---|---|---|---|
| 1 | 1.00 | 0.80 | 1.00 | 1 |
| 4 | 0.33 | 0.50 | 0.83 | 1 |
| 9 | 0.50 | 0.50 | 1.00 | 1 |
| mean | 0.61 | 0.60 | 0.94 | 1 |
Tuned reference = same Bayesian quiz/explain rule with thresholds picked on seeds 1-40 (start belief 0.3, explain while belief < 0.55,
move on at > 0.9, assumed learn rate 0.3). Held-out seeds 41-140: tuned 0.83, first reference 0.55, naive "explain, ask, move on" 0.25.

Agents were not fooled by "I get it": all three ignored `ask` and relied on quizzes. Their loss came from over-verification:
too many confirmation quizzes (they wanted two correct in a row), which teach nothing, so they ran out of turns for later concepts.
Two of three said so themselves afterwards (a knowing-doing gap). One also believed concept 3 was mastered when it was not.
First sign of a real gap, but only 3 runs, and the reference was tuned on seeds that include the agent seeds
(held-out result 0.83 says it is not just overfitting). Next: more agent runs on held-out seeds, other models.

## Probe 3b: held-out reference, more runs, warning ablation (Sat 14 Nov, ~01:00)
Reference re-tuned only on training seeds 1001-1200 (start 0.3, explain while belief < 0.7, move on at > 0.9, learn rate 0.3).
Test seeds 2001-2100: reference mean 0.86, naive 0.24. Reference hits the ceiling (1.0) on 65% of test seeds; on the other 35% it averages 0.60.
One fresh blind agent per game. "warn" = rules say children often claim to understand when they don't; "no-warn" = that sentence removed.
| seed | agent warn | agent no-warn | reference (held-out tuned) |
|---|---|---|---|
| 1 | 1.00 | | 1.00 |
| 4 | 0.33 | | 1.00 |
| 9 | 0.50 | | 1.00 |
| 2001 | 1.00 | 1.00 | 1.00 |
| 2002 | 1.00 | 1.00 | 1.00 |
| 2003 | 1.00 | 0.60 | 1.00 |
| 2004 | 1.00 | | 1.00 |
| 2005 | 0.80 | | 1.00 |
| mean | 0.83 (8 runs) | 0.87 (3 runs) | 1.00 |
Findings:
- The gap is real but only shows on hard seeds (slow learners, tight budget): 0.33 / 0.50 / 0.80 vs 1.00. On easy seeds both hit the ceiling.
- Failure mode is the same in every losing run: a fixed "two correct in a row" confirmation rule and an up-front diagnosis phase that spend turns on quizzes, which teach nothing.
- Warning ablation: even without the warning, agents tried `ask`, saw "yes" to everything and ignored it. They were not fooled by self-reports in this tool-based setting.
- Design lever: remove the ceiling (lower learn rates, or a longer chain / fewer turns) so most seeds are budget-tight; that is where decisions matter.

# Probe 4: tutor v2 (sessions Mon/Wed/Fri, exam next Mon; forgetting, testing effect, spacing, misconceptions)
Model and parameters: see tutor2.py docstring and rules_tutor2.txt. Oracle = best of a full-information greedy rule and a
full-information rollout planner (not guaranteed optimal). Reference tuned on seeds 1001-1060; test seeds 2001-2060: reference 0.71, naive 0.53.

## 4a: first agent runs (design flaw, results contaminated)
| seed | agent | reference | naive |
|---|---|---|---|
| 2001 | 0.63 | 0.67 | 0.12 |
| 2002 | 0.25 | 0.85 | 0.70 |
| 2003 | 0.73 | 0.35 | 0.09 |
| 2004 | 0.28 | 0.75 | 0.37 |
| 2005 | 0.72 | 0.76 | 0.96 |
| mean | 0.52 | 0.68 | 0.45 |
Flaw: two "generic" wrong answers looked like misconception signatures ("both" for 2/3 = 3/4 or 4/6 implies 2/3 = 3/4;
"2/6" for 1/4 + 1/2 is adding across). In 3 of 5 runs the agent reasonably read them as misconceptions and wasted turns.
That is designer-induced misleading, not a fair difficulty. Fixed: generic wrong answers are now neutral, and add_across on
concept 5 shows in the word problem (6) the add-across way; the rules say so.
Genuine failure seen anyway (seed 2004): raced ahead on Monday on single correct answers, so later concepts had no base.
Agents consistently did spaced review at the start of each session; on seed 2003 the agent beat the reference by a lot.

## 4b: fixed version, 8 fresh agents (one game each)
| seed | agent exam | agent (norm.) | reference | naive | agent's main loss |
|---|---|---|---|---|---|
| 2001 | 65% | 0.86 | 0.67 | 0.12 | never reached 5-6 |
| 2002 | 41% | 0.23 | 0.85 | 0.70 | took a lucky-guess correct on prerequisite 2 as mastery, then explained 3 nine times (it could not click) |
| 2003 | 57% | 0.56 | 0.35 | 0.09 | 3 faded, re-taught on the last turn |
| 2004 | 40% | 0.30 | 0.75 | 0.37 | hidden misconception on 2 surfaced only on turn 18; then 4 explanations of 3 that never clicked |
| 2005 | 58% | 0.82 | 0.76 | 0.96 | 7 turns on concept 4 that never clicked |
| 2006 | 78% | 0.85 | 0.74 | 0.83 | 3 faded at the end |
| 2007 | 67% | 0.61 | 0.80 | 0.87 | 13 turns on concept 5 with no return |
| 2008 | 83% | 0.95 | 0.74 | 0.35 | - |
| mean | | 0.65 | 0.71 | 0.54 | |
Agent wins 5 of 8 seeds; the mean gap (0.06) is not consistent. But both are far from the full-information oracle (1.0).
Recurring, decision-level failure: perseveration. When a concept does not click, the agent keeps re-explaining it
(7-13 turns) instead of re-checking the prerequisite (which may be forgotten or was only guessed right) or moving
the turns to spaced review. Agents did spaced review well and diagnosed the misconception whenever its signature appeared.
Next: a stronger, principled reference (belief-state POMDP planner: particle filter over the hidden state + rollouts),
which should beat both and make the gap consistent.

# Probe 5: class of five (tutor3), 60 turns, lecture / classquiz added (Sat 14 Nov, ~00:30)
References (no tuning): RTI tiered instruction (rti_class.py) 0.54 mean on seeds 2001-2010; per-child literature policy
(lit_class.py) 0.45; naive "lecture twice + class quiz" 0.41. Oracle = best of full-info greedy and a rollout planner
with a small candidate set — NOT a reliable upper bound here (one agent scored above it).
| seed | agent | RTI | naive |
|---|---|---|---|
| 2001 | 0.54 | 0.42 | 0.34 |
| 2002 | 0.69 | 0.48 | 0.37 |
| 2003 | 1.02 | 0.62 | 0.33 |
| 2004 | 0.73 | 0.48 | 0.56 |
| 2005 | 0.91 | 0.50 | 0.26 |
| mean | 0.78 | 0.50 | 0.37 |
The agent beats every reference on every seed. It independently ran RTI-like teaching (lecture, class quiz, one-to-one
remediation for those who failed, spaced class quizzes at session start, targeted misconception correction, and it
inferred an add_across bug on concept 5 from a word-problem answer). Its own stated mistakes: pushing new material on
Wednesday instead of spaced review; long one-to-one effort on one stubborn child.

# Calibration check: Duolingo half-life regression (Settles & Meeder 2016 data, first 2M rows)
fit_hlr.py: log2 h = 7.13 + 0.184 sqrt(1+correct) - 0.795 sqrt(1+wrong); MAE on recall 0.15.
Implied half-life ~90 days, +3-5% per correct recall. Our simulator uses 1 day, x2.5 per correct recall.
Caveats: vocabulary, adult self-selected learners, items mostly already practised (mean recall 0.90), our quick
least-squares fit without the paper's regularisation. Still, it says our forgetting is far too fast and our
testing-effect multiplier far too strong compared with this real data.

# First API runs with run_agent.py (tutor2 fixed, one child, fresh game each)
Reference = untuned literature policy (lit_ref.py), on the same seeds.
| seed | gpt-6.1-sol reasoning medium | gpt-6.1-sol reasoning high | literature reference |
|---|---|---|---|
| 2001 | 0.97 | 0.95 | 0.89 |
| 2002 | 0.23 | 0.51 | 0.85 |
| 2003 | 0.00 | 0.08 | 0.74 |
| 2004 | 0.21 | 0.21 | 0.73 |
| 2005 | 0.49 | 0.35 | 0.71 |
| mean | 0.38 | 0.42 | 0.78 |
Cost per game ~70k input tokens (~95% cached), ~600-1100 output tokens. Raising reasoning effort did not close the gap.
Failure pattern: moves on after one quiz (often a lucky guess on a prerequisite), then re-explains a concept that cannot
click; never backs off to the prerequisite. Earlier Claude subagent runs (heavy reasoning) averaged 0.65 on 2001-2008.
| seed | claude-opus-5-5 effort medium (API, tools only) |
|---|---|
| 2001 | 0.95 |
| 2002 | 0.18 |
| 2003 | 0.30 |
| 2004 | 0.13 |
| 2005 | 0.44 |
| mean | 0.40 (reference 0.78) |
Note the harness effect: Claude as a Claude Code subagent (shell, own scripts, long thinking) averaged 0.65 on 2001-2008;
the same family through plain tool calls at medium effort averages 0.40. OpenCode is a coding agent with a shell, so the
judges' runs may look more like the subagent runs. Must test inside Harbor/OpenCode before claiming the gap.

# Harbor + OpenCode runs (Sat 14 Nov, salt "eval1", seeds 2001-2005; reference = Harbor `-a oracle` running solve.sh)
| seed | reference | OpenCode + gpt-6.1-sol | OpenCode + claude-opus-5-5 |
|---|---|---|---|
| 2001 | 0.52 | 0.89 | 0.53 |
| 2002 | 0.59 | 0.20 | 0.24 |
| 2003 | 0.54 | 0.63 | 0.69 |
| 2004 | 0.65 | 0.47 | 0.51 |
| 2005 | 0.68 | 0.38 | 0.66 |
| mean | 0.60 | 0.52 | 0.52 |
Harness effect confirmed: inside OpenCode both frontier models rise from ~0.40 (plain tool calls) to ~0.52.
The reference still leads on average but wins only 3 of 5 seeds per model; the gap is not yet "consistent".
Results TSVs: afterquery/harbor/results/. Batch script: afterquery/harbor/run_seeds.sh.

# Calibration of forgetting + POMDP planner attempt (Sat 14 Nov)
tutor2.py now uses FSRS-5-derived half-lives (new concept U(2.6, 7.8) d, pre-known U(8, 20) d; x2.5 on correct recall
= SM-2 ease). Harbor copies NOT yet synced (still old 1-day model).
40 seeds (2001-2040), calibrated: literature 0.62, naive (2-in-a-row, no review) 0.58 (was 0.69 vs 0.53 with 1-day
half-life). Most of the old reference advantage came from the unrealistically fast forgetting.
pomdp_ref.py: particle filter (wide priors) + Monte Carlo one-step improvement over the literature policy, common
random numbers, 48 rollouts/candidate: 0.62 (= literature). Diagnostic with the simulator's TRUE prior: 0.59.
=> one-step policy improvement on this base adds nothing; simple observation-based policies plateau near 0.6 while
the full-information policy defines 1.0. A stronger reference needs real search (e.g. POMCP) or the env needs a
different source of difficulty.

# RL reference (PPO, observation-only features) on calibrated tutor2 (Sat 14 Nov)
rl_ref.py: 63 observation features (counts, last result, signature seen, days since contact, BKT belief, time),
15 actions, reward = exam gain at the end, 2-layer MLP. Trained on random seeds in [1e6, 1e9), 2000 iters x 256
children (~15M steps), 9 min on a laptop CPU.
Held-out seeds 2041-2140 (never used for training or checkpoint choice):
| policy | normalised score |
|---|---|
| PPO, last checkpoint (rl_ref_last.pt) | **0.78** |
| PPO, checkpoint picked on 2001-2040 | 0.78 |
| literature policy | 0.58 |
| naive (2-in-a-row, no review) | 0.60 |
Matches the estimated observation-only ceiling (~0.7-0.8). Use rl_ref_last.pt (no selection bias).

# Realism check: diminishing returns of massed re-explanation (Sat 14 Nov)
tutor2.py: each earlier explanation of the same concept in the same session multiplies the chance of it clicking
by 0.6; a new session starts fresh (massed vs spaced practice). Rules text and Harbor instruction updated; RL
observation gains "explanations of this concept this session". Harbor simulator copies synced to this version.
Held-out seeds 2041-2140:
| policy | before (no damping) | with massed damping |
|---|---|---|
| PPO (rl_ref_massed.pt) | 0.78 | **0.70** |
| literature | 0.58 | 0.59 |
| naive | 0.60 | 0.54 |
PPO still leads the literature policy by ~0.11. Its behaviour changed: explain 58% / quiz 39% (was 73/25), runs of
1.3 explanations (was 2.1), and it now often quizzes the PREREQUISITE right after teaching the next concept
("E3 Q2-" -> back to 2). It still keeps teaching new material to the end (20% quizzes in the last 5 turns vs 68%
for the literature policy) and reaches concept 5.7 on average (literature 4.0).
