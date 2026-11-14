# Phase 1 — independent design (written before reading `afterquery/` or `HANDOFF.md`)

Pick: **Wind-farm O&M desk** — a maintenance-scheduling POMDP with biased sensors and one crew.
Rejected after prototyping: single-product dynamic pricing with reference-price effects (too forgiving).
All numbers below come from the two prototypes in `fable/prototype/`. No LLM was run; the "agent-like"
rows are scripted heuristics standing in for what an agent tends to do. The real gap is unverified.

## 1. What makes a frontier coding agent lose fairly

A top model in OpenCode with a shell will recognise a textbook problem, write the DP or the Bayesian
update, and fit a model to immediate feedback. So the gap cannot come from "the maths is hard". It has
to come from properties that stay hard even with code:

| Property | Why it hurts an agent | Why it is still fair |
|---|---|---|
| Errors are first-order, not second-order | Revenue curves are flat near the optimum, so a rough answer scores ~0.9. Thresholds, failures and irreversible events punish a rough answer fully | The costs are stated in the rules |
| Hidden state plus a misleading cheap signal | The agent's strength, fitting to immediate feedback, becomes the trap when the feedback is biased or confounded | The bias mechanism and its prior are disclosed; the history contains enough to infer it |
| Scarce action slot per step | Forces prioritisation among many components; "do the obviously good thing everywhere" is not available | Same constraint for the reference |
| One life, no reset, per-step new observations | Brute-force experimentation is impossible; a one-off script is not enough, the agent must keep a state pipeline running for 60 steps | The agent may write that pipeline; nothing blocks it |
| Per-instance hidden parameters | The answer cannot be memorised or computed once | Priors are public |

Two properties I would not rely on: long tedious horizons alone (the gap then measures patience, and a
judge will say so), and anything the instructions do not state (zero points under the rubric).

A consequence worth stating plainly: **no design guarantees the gap on paper.** A disciplined agent can
always rebuild the reference from the disclosed rules. What a design can do is make the lazy path score
clearly lower and make the disciplined path long enough that agents rarely finish it. This has to be
measured with real agent runs in the first hour, before investing further.

## 2. Three candidates

### A. Discount Trap — pricing with reference-price memory (prototyped, rejected)
One seasonal product, 50 days, price ladder, limited stock, hidden demand parameters. Customers remember
past prices (Popescu and Wu 2007 style): discounts lift demand today and lower the reference price, so
later demand at normal prices drops. Reference: particle belief over 5 parameters plus a fluid DP on
(day, stock, reference price), re-solved daily. Files: `env.py`, `policies.py`, `run_eval.py`.

Result (30 seeds, score = revenue / full-information revenue):

| Policy | Mean | SD |
|---|---|---|
| do-nothing | 0.000 | 0.000 |
| stop-early (10 of 50 days) | 0.172 | 0.024 |
| fixed list price | 0.658 | 0.143 |
| analyst: regress log sales on price, myopic, paced | 0.914 | 0.074 |
| Bayesian belief, myopic | 0.948 | 0.060 |
| reference: belief + DP lookahead | 0.992 | 0.008 |

The naive analyst reaches 0.91. With my first parameter set a fixed list price scored 0.89. Pricing
revenue is flat near the optimum, so even a wrong model loses only a few points. A frontier agent
would land at 0.90–0.97 and three judge runs could not separate it from the reference. This matches the
risk already noted in `CLAUDE.md`. It is the best fit for a pricing PhD story and the worst fit for the
rubric. Rejected.

### B. Wind-farm O&M desk (prototyped, picked) — section 3

### C. Night-train network revenue management
Six itineraries share three legs, hidden demand curves, daily price per itinerary. Reference:
certainty-equivalent LP re-solve with posterior sampling. Realistic and on theme, but the reference is
exactly what a coding agent writes first (Poisson fit + LP), and it shares A's flat-revenue problem.
Not prototyped. Rejected.

## 3. The pick: Wind-farm O&M desk

### Setting
An operator runs 8 offshore turbines for 60 days with **one crew vessel: one job per workable day**.

- Each turbine has hidden wear 0–4. Wear 4 means failed, which is visible. Wear advances by one level
  with daily probability `q_i * (1 + 0.5 * wear)`. The rate `q_i` is hidden, `Uniform(0.03, 0.15)`.
- Every morning each running turbine reports a vibration reading `wear + b_i + noise`, noise SD 1. The
  sensor offset `b_i` is hidden and constant, `Normal(0, 1)`. The dial is informative and miscalibrated.
- Weather is a two-state Markov chain; about two days in three are workable.
- Jobs: `service(i)` resets wear to 0, costs 2, turbine idle that day. `repair(i)` on a failed turbine
  costs 10 and takes 5 days. `inspect(i)` reveals true wear, costs 0.3, uses the crew day.
- Profit J = running turbine-days minus job costs.
- Score = (J − J_nothing) / (J_full-info − J_nothing), same seed, same random numbers.

Everything above, including priors and costs, goes in the instructions. The agent's workspace also gets
the simulator source without the seed secret, so it holds the same generative model the reference uses.

### Prototype result (`python3 fable/prototype/windfarm.py 200`, 6 seconds)

| Policy | Mean | SD | Share of seeds below reference |
|---|---|---|---|
| do-nothing | 0.000 | 0.000 | 1.00 |
| stop-early (reference for 12 of 60 days) | 0.171 | 0.142 | 1.00 |
| reactive: repair failures only | 0.340 | 0.131 | 1.00 |
| periodic service every 2nd day | 0.383 | 0.184 | 0.98 |
| trust the dial: service highest 3-day mean reading above 2.5 | 0.573 | 0.191 | 0.94 |
| ablation: Bayes filter that ignores sensor bias | 0.629 | 0.218 | 0.86 |
| variant: reference that also inspects | 0.789 | 0.169 | 0.65 |
| **reference: Bayes filter over (wear, bias, rate) + cost index** | **0.830** | 0.167 | — |

Paired gap, reference minus trust-the-dial: mean 0.257, SD 0.196. Probability that the mean paired gap
over 3 seeds is positive: 0.994; above 0.05: 0.978.

Two honest findings from the prototype:
- Inspection is a dominated action with these costs (0.789 with, 0.830 without). Either make inspection
  not consume the crew day, or drop it. As built, it is a decoy, and a judge may call a decoy a trick.
- The reference reaches 0.83 of full information, so there is headroom above it. Good for credibility:
  the reference is a strong demonstrated run, not a ceiling.

### Against the rubric

**The gap (30).** The lazy paths an LLM takes when hand-playing (repair what broke, service what reads
high) sit at 0.34–0.57. The reference sits at 0.83. The gap is 4–6 times the standard error of a
3-run mean (0.11). To tighten further, one judge run should average 3 farms; that cuts the SD by 1.7
at the cost of 180 tool calls. Both sides are deterministic given the seed: demand luck is pre-drawn
per seed, so the same actions give the same profit.

**Meaningful difficulty (30).** Each expected mistake maps to one decision and one ablation row:
- Trusting the dial. A turbine with a +1.5 offset gets serviced again and again while a −1.5 turbine
  fails unnoticed. Evidence: bias-blind filter 0.63 against 0.83.
- No filtering over time. One noisy reading decides the job. Evidence: dial rule 0.57.
- Repair-first myopia. A failed turbine loses 1 per day; a wear-3 turbine left alone loses about 4 per
  day in expectation. The reference compares the two; reactive play does not. Evidence: reactive 0.34.
- Calendar thinking. Periodic service wastes the scarce crew day on healthy turbines. Evidence: 0.38.

The information is available: the reading history identifies each sensor's offset, because a turbine
just serviced is known to be at wear 0, so its next readings reveal `b_i`. No tool is broken, no rule is
hidden, and the reference uses only observations.

**Reward integrity (25).**
- Do-nothing scores exactly 0 by construction. Stop-early scores 0.17 because turbines keep failing.
- Score is computed by the server from its own log. The agent reports nothing.
- Hidden wear, `q_i`, `b_i` and the random stream never leave the server. The seed is salted with a
  secret that is not in the agent's container, so having the simulator source does not reveal the draws.
- One episode per run id; no reset tool; steps are serialised; a finished episode rejects further calls.
- Invalid jobs (repairing a running turbine, working in bad weather) return an error and still consume
  the day, so probing for information through error messages costs a day and returns nothing hidden.
- Spamming service is bounded by one job per day and costs 2 each; periodic service scores 0.38.
- Unfinished episodes are scored as if the remaining days had no jobs.

**Craft (15).** Rules read like a real O&M contract: money in, money out. One command runs reference,
baselines and the table. The reference is 60 lines of numpy with no trained weights, so "reproduce the
reference" takes seconds on a judge's laptop.

**Realism.** Condition-based maintenance under partial observability is a standard POMDP application
and sensor miscalibration is a real nuisance in condition monitoring. The numbers (rates, costs) are
stylised and should be labelled as such.

**Tiebreaker.** Log (history → job) pairs from the reference on 2,000 seeds and fine-tune a small model
on them; evaluate before and after on held-out seeds.

### Risks
1. A disciplined agent writes the same filter and closes the gap. This is the main risk and it cannot be
   ruled out without runs. Test: three agent runs on three seeds. If two reach 0.75, harden with 16
   turbines, spare-part lead times and a weather forecast (crew planning becomes a restless bandit), or
   abandon.
2. The reference is a hand-built index rule. A judge may ask whether it is tuned on the test seeds. Keep
   tuning seeds and evaluation seeds disjoint and say so.
3. SD per farm is large (0.17–0.20). Use the 3-farm average per run.

## 4. What this phase says about any environment, including an existing one
- Check the dumb baselines first. Candidate A looked hard on paper and a fixed price scored 0.89.
- Give the agent the simulator source. It removes the "withheld information" attack in one move.
- Prefer a reference that needs no trained weights, or show the trained one next to a transparent one.
- Average several instances per run. Three single-episode runs are too noisy for gaps under 0.15.
