#!/bin/bash
# Launch one Harbor trial per (agent, seed) in parallel and merge the results.
# Usage: harbor/run_parallel.sh <agent> <model-or-dash> <tag> <seed> [<seed> ...]
cd "$(dirname "$0")/.."
agent=$1; model=$2; tag=$3; shift 3
for seed in "$@"; do harbor/run_seeds.sh "$agent" "$model" "${tag}_$seed" "$seed" > /dev/null 2>&1 & done
wait
head -1 harbor/results/${tag}_$1.tsv > harbor/results/$tag.tsv
for seed in "$@"; do tail -n +2 harbor/results/${tag}_$seed.tsv; done >> harbor/results/$tag.tsv
rm -f harbor/results/${tag}_*.tsv
cat harbor/results/$tag.tsv
