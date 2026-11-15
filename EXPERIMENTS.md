# Experiments

Every number in the README, where it comes from, and how to reproduce it. Agent runs: Harbor + OpenCode, six children
per run, salt `eval1`. Raw results: one TSV per experiment in `harbor/results/` (`reward` is the agent's score,
`reference_reward` the PPO reference on the same children).

| # | experiment | how | raw results | result |
|---|---|---|---|---|
| 1 | Main gap, batches 3001-3005 | `harbor/tutor2`, five models | `v4_*.tsv` | reference 0.70; GPT 0.61, Opus 0.59, Gemini 0.52-0.55 |
| 2 | Main gap, batches 3006-3015 | same, two Gemini models | `v6_*.tsv` | reference wins 19/20 |
| 3 | Baselines | `-a nop`, `StopEarlyAgent`, `StopAfterMondayAgent` | `base_*.tsv` | 0.00, 0.00, 0.07 |
| 4 | Every parameter disclosed | `harbor/tutor2_disclosed` | `v5d_*`, `v11d_*`, `v10d_opus55.tsv` | Gemini +0.01 / +0.03 (15 batches), Opus +0.07 (3); reference wins 29/33 |
| 5 | Tutoring advice in the prompt | `harbor/tutor2_hint` | `v7h_*`, `v9h_*`, `v10h_opus55.tsv` | Gemini +0.02 / -0.05, Opus +0.02; no consistent gain |
| 6 | No teaching story | `harbor/tutor2_abstract` | `v8a_*`, `v9a_*`, `v10a_opus55.tsv` | Gemini -0.07 / -0.06, Opus +0.03; no model does better |
| 7 | 27 parameter settings | `probe/sensitivity.py` | `probe/sensitivity_results.txt` | the agents' habits lose in all 27 |
| 8 | PPO vs five rules, 1,200 fresh children | `probe/compare_refs.py fresh` | printed | PPO 0.725, five rules 0.671 |
| 9 | Five-rule ablation | `probe/compare_refs.py ablation` | printed | rule 2 off: -0.078 per child; others -0.005 to -0.016 |
| 10 | Lucky-guess move-ons, quotes | `probe/replay_traces.py`, `failure_analysis.py`, `find_quotes.py` | needs the run logs | AI 1.4-2.0 per child, reference 0.5 |
| 11 | "Check back later" variant | `probe/checkback_ref.py` | printed | as good as rule 2 with fewer quizzes |
| 12 | README command from a fresh clone | `./eval.sh gemini/gemini-3.5-flash 3001` | `fresh_clone_*.tsv`, `example_runs/` | Linux x86: reference 0.675, agent 0.388; MacBook (Apple Silicon): reference 0.675, agent 0.358; baselines 0 on both |
| 13 | Gemini 4 Argon, one run | `harbor/tutor2` | `v12_argon_smoke.tsv` | 0.57 vs reference 0.67 |

Notes
- Runs that crashed (GPT on 3002 and 3003, rate limits) are excluded. A later retry of GPT on 3002 finished
  (`v4_gpt61sol_3002.tsv`, 0.72 vs reference 0.78); it is not in the 39/43 count.
- Changes of mind along the way: the first simulator forgot far too fast and was recalibrated to FSRS half-lives;
  "testing instead of teaching" was our first explanation and is wrong (the five rules quiz as much and win);
  the five-rule teacher is hand-written, not extracted from PPO. The full working notes are in the git history.
