# The Hidden-State Classroom

Team Sauna Renaissance · Junction 2026 (Helsinki, 13–15 Nov) · AfterQuery "Break the Agent"

An RL environment where frontier AI agents tutor simulated children in fractions and lose to a small trained policy.
Each child's knowledge, memory and misconceptions are hidden; the agent only sees the answers. Across five frontier
models (Claude Opus, GPT, three Gemini models) the reference wins 39 of 43 runs. The agents know the rules, and are
even told that an unlearned child guesses right 1 time in 4, yet they move on after a single right answer, which
alone costs about as much as the whole gap.

- **Everything is in [`afterquery/`](afterquery/)**: the Harbor task, the simulator, the references, every result and
  the one-command evaluation. Start with [`afterquery/README.md`](afterquery/README.md).
- **Demo page**: `afterquery/viz/simple.html` (serve the folder with `python3 -m http.server`).
- **Lab notebook**: [`afterquery/probe/RESULTS.md`](afterquery/probe/RESULTS.md), every experiment including the
  ones that failed and every correction.
- **Sources**: [`afterquery/REFERENCES.md`](afterquery/REFERENCES.md).

Rerun it:

```sh
cd afterquery
python3 -m venv .venv && .venv/bin/pip install "harbor==0.25.0" numpy
cp .env.example .env            # add your model provider key
./eval.sh gemini/gemini-pro-latest 3001 3002 3003
```

All code here was written during the event. MIT licensed.
