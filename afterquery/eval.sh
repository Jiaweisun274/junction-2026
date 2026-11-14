#!/bin/bash
# One-command evaluation for the AfterQuery "Break the Agent" task (fractions tutor, six children per run).
#
#   ./eval.sh <model> [seed ...]            e.g. ./eval.sh gemini/gemini-pro-latest 3001 3002 3003
#
# Runs, through Harbor with the standard OpenCode harness:
#   1. the reference run (trained PPO policy, solution/solve.sh) on every seed
#   2. the OpenCode agent with <model> on every seed (one run per seed = three agent runs for three seeds)
#   3. the do-nothing baseline (Harbor's nop agent) and the stop-early baseline on the first seed
# and prints a summary table. Needs Docker, the venv (.venv with harbor==0.25.0) and the provider key in .env.
# Seeds default to 3001 3002 3003. AFTERQUERY_SEED_SALT (default "eval1") picks the hidden children.
set -u
cd "$(dirname "$0")"
model=${1:?usage: ./eval.sh <model> [seed ...]}; shift
seeds=("$@"); [ ${#seeds[@]} -eq 0 ] && seeds=(3001 3002 3003)
tag=eval_$(date +%m%d_%H%M)
export PYTHONPATH=harbor
echo "Reference (PPO) ..."
harbor/run_parallel.sh oracle - ${tag}_reference "${seeds[@]}" > /dev/null
echo "Agent: $model ..."
harbor/run_parallel.sh opencode "$model" ${tag}_agent "${seeds[@]}" > /dev/null
echo "Baselines ..."
harbor/run_seeds.sh nop - ${tag}_nop "${seeds[0]}" > /dev/null 2>&1
harbor/run_seeds.sh baseline_agents:StopEarlyAgent - ${tag}_stop "${seeds[0]}" > /dev/null 2>&1
python3 - "$tag" "$model" <<'PY'
import csv, sys
tag, model = sys.argv[1], sys.argv[2]
def rows(name):
    try:
        return list(csv.DictReader(open(f"harbor/results/{tag}_{name}.tsv"), delimiter="\t"))
    except FileNotFoundError:
        return []
def num(x):
    try: return float(x)
    except (TypeError, ValueError): return float("nan")
ref, agent = rows("reference"), rows("agent")
print(f"\nResults ({tag}); reward = pooled normalised exam gain over six children (1.0 = full-information teacher)")
print(f"{'seed':>6} {'reference':>10} {'agent':>8} {'literature':>11}")
a = {r["seed"]: r for r in agent}
for r in ref:
    print(f"{r['seed']:>6} {num(r['reward']):10.3f} {num(a.get(r['seed'], {}).get('reward')):8.3f} {num(r['literature_reward']):11.3f}")
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
print(f"{'mean':>6} {mean([num(r['reward']) for r in ref]):10.3f} {mean([num(r['reward']) for r in agent]):8.3f} {mean([num(r['literature_reward']) for r in ref]):11.3f}")
for name, label in (("nop", "do-nothing"), ("stop", "stop-early")):
    rr = rows(name)
    print(f"{label:>11}: {num(rr[0]['reward']) if rr else float('nan'):.3f}")
print(f"agent model: {model}")
PY
