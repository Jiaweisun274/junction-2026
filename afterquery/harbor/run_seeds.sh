#!/bin/bash
# Run one Harbor trial per seed for one agent/model and collect rewards into a TSV.
# Usage: harbor/run_seeds.sh <agent> <model-or-dash> <tag> <seed> [<seed> ...]
#   harbor/run_seeds.sh opencode openai/gpt-6.1-sol gpt61sol 2001 2002 2003
#   harbor/run_seeds.sh oracle - reference 2001 2002 2003
# The seed salt comes from AFTERQUERY_SEED_SALT in the calling shell (default "eval1").
set -u
cd "$(dirname "$0")/.."
agent=$1; model=$2; tag=$3; shift 3
salt=${AFTERQUERY_SEED_SALT:-eval1}
out=harbor/results/$tag.tsv
mkdir -p harbor/results jobs/$tag
echo -e "seed\treward\texam\treference_reward\tliterature_reward\tturns\tjob" > "$out"
for seed in "$@"; do
  args=(run --env-file .env -p harbor/tutor2 -a "$agent" -k 1 -o "jobs/$tag/seed_$seed")
  [ "$model" != "-" ] && args+=(-m "$model")
  AFTERQUERY_SEED=$seed AFTERQUERY_SEED_SALT=$salt .venv/bin/harbor "${args[@]}" > "jobs/$tag/seed_$seed.log" 2>&1
  res=$(ls jobs/$tag/seed_$seed/*/*/result.json 2>/dev/null | head -1)
  python3 - "$res" "$seed" >> "$out" <<'PY'
import json, sys
path, seed = sys.argv[1], sys.argv[2]
try:
    r = json.load(open(path)); v = (r.get("verifier_result") or {}).get("rewards", {})
    print(f"{seed}\t{v.get('reward', 'NA')}\t{v.get('exam_score', 'NA')}\t{v.get('reference_reward', 'NA')}\t{v.get('literature_reward', 'NA')}\t{v.get('turns_used', 'NA')}\t{path}")
except Exception as e:
    print(f"{seed}\tERROR\t\t\t\t{path} {e}")
PY
done
cat "$out"
