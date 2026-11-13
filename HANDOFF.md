# Handoff — Junction 2026, team Sauna Renaissance

Last updated: Sat 14 Nov 2026, early morning. Repo: https://github.com/Jiaweisun274/junction-2026 (private now; make it **public before Sun 10:00**, Junction requires it).

## 1. What we are doing

| Challenge | Status | Folder |
|---|---|---|
| **AfterQuery — Break the Agent** (main) | Environment chosen and working; first evidence of a gap; not yet in Harbor | `afterquery/` |
| Bahn Express — Spread (side) | Registered, starter runs; no v0 rules yet | `spread/` |
| Verda / Gemma tech challenges (stretch) | Not started | — |

Preliminary submissions are saved as drafts on the Junction platform (one project per challenge). Final submissions open Sat 12:00, close **Sun 10:00**. Each needs a ≤2 min video.

### AfterQuery rubric (what judges score)
- **Gap (30):** frontier agent consistently below our reference run, reproducible.
- **Meaningful difficulty (30):** agent's mistakes tie to decisions; it had the info and tools it needed.
- **Reward integrity (25):** judges hunt exploits for 15 min; do-nothing and stop-early must score low.
- **Craft (15):** rules make sense; one command from the README reproduces results.
- Judges rerun with the standard harness (**Harbor + OpenCode over MCP**): our reference, 3 agent runs, do-nothing, stop-early. Tiebreaker: a small model trained on the env improves.

## 2. The environment: one-to-one fractions tutor (`afterquery/probe/tutor2.py`)

A tutor teaches one 10-year-old six fraction concepts in a prerequisite chain (equal parts → comparing → equivalent → common denominators → adding → word problems). Three sessions (Mon/Wed/Fri), 10 turns each; exam next Monday. Score = exam score, normalised: (agent − no teaching) / (full-information policy − no teaching).

Hidden per-concept state: unlearned / mastered (decays) / misconception. Mechanisms, all stated in the rules (`afterquery/probe/rules_tutor2.txt`):
- **Forgetting:** recall = 2^(−days/half-life), half-life 1 day after learning.
- **Testing effect + spacing:** a correct quiz multiplies half-life by 2.5 (re-explaining by 1.5), more when recall had dropped.
- **Prerequisites:** a concept almost never clicks while the previous one isn't mastered.
- **Misconceptions:** three documented fraction bugs (bigger_denominator, add_to_top_and_bottom, add_across); they show as a typical wrong answer, also downstream (add-across shows in word problems); targeted `correct` fixes them, re-explaining rarely does.
- **Self-report:** `ask` is free but children over-claim understanding.
- Actions: `explain K`, `quiz K` (shows the child's answer), `correct K BUG`, `ask K`, `finish`.

**Reference policy** (`afterquery/probe/lit_ref.py`): standard components from education research, parameters fixed a priori, never tuned on our seeds — BKT mastery learning (Corbett & Anderson 1995, threshold 0.95), Leitner/half-life spaced review at 0.9 target retention, prerequisite back-off after 3 failed explanations (knowledge space theory), targeted misconception correction. Robustness check: its score does not change for assumed half-lives 0.5–10 days; assumed learn rate 0.15–0.5 gives 0.58–0.69 on 40 test seeds.

Other files in `afterquery/probe/`: earlier probes (`sell.py`, `sauna.py`, `gate.py`, `tutor.py`), the class-of-five version (`tutor3.py`, `rti_class.py`, `lit_class.py`), tuning script `sweep_tutor2.py`, and **`RESULTS.md` — the full lab notebook of every probe, including failed ones and a design flaw we found and fixed.**

## 3. Results so far (tutor2, seeds 2001–2005, one fresh game per seed)

| seed | gpt-6.1-sol (reasoning high) | claude-opus-5-5 (effort medium) | literature reference |
|---|---|---|---|
| 2001 | 0.95 | 0.95 | 0.89 |
| 2002 | 0.51 | 0.18 | 0.85 |
| 2003 | 0.08 | 0.30 | 0.74 |
| 2004 | 0.21 | 0.13 | 0.73 |
| 2005 | 0.35 | 0.44 | 0.71 |
| **mean** | **0.42** | **0.40** | **0.78** |

Reference mean over 40 test seeds is 0.69 (these 5 seeds favour it a bit). Naive "teach in order, two correct in a row, no review" ≈ 0.53.

**Recurring agent failure (decision-level):** moves on after a single quiz answer (often a lucky guess on a prerequisite), then re-explains a concept that cannot click because its prerequisite is missing; never backs off; little spaced review before the exam.

**Big caveat — harness effect:** Claude playing as a Claude Code subagent (shell, own scripts, long thinking) averaged **0.65** on seeds 2001–2008. OpenCode is also a shell-capable coding agent, so judges' runs may look like 0.65, not 0.40. **The gap must be confirmed inside Harbor + OpenCode before we claim it.**

What did NOT work (see RESULTS.md): bike selling (optimal stopping), sauna opponent modelling, ferry-gate faster-is-slower, class-of-five tutoring — frontier agents matched or beat our references whenever feedback was immediate and the problem small.

## 4. How to run things

```bash
cd afterquery
.venv/bin/python run_agent.py --backend responses --model gpt-6.1-sol --reasoning high --seeds 2001-2005 --parallel 5 --tag my_run
.venv/bin/python run_agent.py --backend anthropic --model claude-opus-5-5 --effort medium --seeds 2001-2005 --parallel 5 --tag my_run
.venv/bin/python run_agent.py --backend openai --base-url http://<host>:8000/v1 --model <vllm-model> --seeds 2001-2010   # self-hosted (Verda)
```
- `.venv` is Python 3.13 with `anthropic`, `openai`, `numpy` (system Python 3.9 is too old for the Anthropic SDK). Recreate: `python3.13 -m venv .venv && .venv/bin/pip install anthropic openai numpy`.
- API keys: put them in `afterquery/.env` (`OPENAI_API_KEY=...`, `ANTHROPIC_API_KEY=...`). It is gitignored and chmod 600; `run_agent.py` loads it. **Never commit keys or paste them in chat.**
- Output: `afterquery/runs/<tag>/seed_<n>.jsonl` (full trajectory) and `summary.tsv`. Cost ≈ 70k input tokens per game, mostly cached.
- Play by hand: `GAME_DIR=/tmp/g python3 afterquery/probe/tutor2.py new 2001`, then `quiz 1`, `explain 2`, `correct 3 add_to_top_and_bottom`, `ask 1`, `finish`.
- Reference/benchmarks: `cd afterquery/probe && ../.venv/bin/python lit_ref.py`.

Data: Duolingo half-life regression traces (Settles & Meeder 2016, CC BY-NC) in `data/duolingo/` (gitignored, 379 MB). `afterquery/calibration/fit_hlr.py` gives half-lives of ~90 days for adult vocabulary — our 1-day half-life is deliberately much faster (see open issue 2).

## 5. Next steps (in priority order)

1. **Ask the AfterQuery mentors (Daksh Raghuwanshi, Ved Vedere)** first thing: Harbor task format, how AfterQuery model credits are used, whether a stylised simulator is acceptable, and ask them to try to exploit the reward.
2. **Package as a Harbor task** (`instruction.md`, `task.toml`, `environment/docker-compose.yaml` with the simulator in a separate MCP-server container the agent cannot read, `solution/solve.sh` = reference policy, `tests/test.sh` writing `/logs/verifier/reward.json`). Docs: https://docs.harborframework.com/tasks/overview
3. **Run OpenCode inside Harbor** on seeds 2001–2005. If the gap survives, scale to 20+ seeds; if not, see open issue 1.
4. **Baselines for judges:** do-nothing (finish at once → 0) and stop-early (finish after a few turns) — check both score low; write the reward-integrity notes (no source access, invalid inputs rejected, repeated `ask` gives the same answer, score cannot be raised without teaching).
5. **README with one-command eval** that reproduces the results table.
6. Stretch: Verda GPU (€200 credit, document the process) → vLLM-host Gemma, collect trajectories, fine-tune, show improvement (AfterQuery tiebreaker + Gemma + Verda challenges).
7. Spread (only if time): v0 rules in `CLAUDE.md`-style plan — driver-density filter, quote list × 1.3, rising fee schedule, backlog cap; kill if not ≥90% on time by Sat noon. Key is in `spread/.env` (gitignored). Test finals Sat 14:00 and 20:00; final Sun 08:00, register before 07:30.

## 6. Open issues

1. **Harness effect** (above): the biggest risk to the gap claim.
2. **Realism of forgetting:** 1-day half-life is far faster than the Duolingo fit. Either slow it down and re-check the gap, or state clearly that it is a deliberately compressed week.
3. **Few seeds:** 5 per model so far; need 20+.
4. **"Oracle" is not a true upper bound** (full-information heuristic + rollout); scores can in principle exceed 1. Say so in the write-up.
5. Same quiz question each time (an agent noted possible memorisation); minor.
6. Citations in RESULTS.md / this file were written from memory in places; verify before putting them in the submission.

## 7. Credits and accounts
- OpenAI: $100 Junction credit (Codex) — works with the API key in `afterquery/.env`.
- Anthropic: API key in `afterquery/.env`; Junction gives $20/person.
- AfterQuery: frontier model credits via their harness (ask mentors).
- Verda: €200 team credit (console.verda.com; create the project before redeeming the coupon).
- Modal $100/person, Featherless $50/person (Gemma hosting) if needed.
