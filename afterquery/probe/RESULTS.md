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
