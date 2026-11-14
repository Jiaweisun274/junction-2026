#!/bin/sh
set -eu

mkdir -p /logs/verifier
python - <<'PY'
import json
from pathlib import Path

try:
    result = json.loads(Path("/grader/result.json").read_text(encoding="utf-8"))
except FileNotFoundError:
    result = {"completed": False, "reward": 0.0, "turns_used": 0}

if not result.get("completed"):
    output = {"reward": 0.0, "completed": 0.0, "turns_used": float(result.get("turns_used", 0))}
else:
    output = {
        "reward": float(result["reward"]),
        "completed": 1.0,
        "exam_score": float(result["exam_score"]),
        "no_teaching_score": float(result["no_teaching_score"]),
        "reference_exam_score": float(result["reference_exam_score"]),
        "reference_reward": float(result["reference_reward"]),
        "oracle_score": float(result["oracle_score"]),
        "turns_used": float(result["turns_used"]),
    }

with open("/logs/verifier/reward.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
PY
