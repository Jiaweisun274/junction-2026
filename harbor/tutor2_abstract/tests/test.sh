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
        "literature_reward": float(result["literature_reward"]),
        "full_info_score": float(result["full_info_score"]),
        "turns_used": float(result["turns_used"]),
    }

# One score in [0, 1] (the AfterQuery convention). The pooled reward can exceed 1 on rare batches, because the
# full-information teacher used for normalisation is strong but not a proven optimum; it is clipped here and the
# raw value is kept as reward_unclipped.
output["reward_unclipped"] = output["reward"]
output["reward"] = min(1.0, max(0.0, output["reward"]))
with open("/logs/verifier/reward.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
with open("/logs/verifier/reward.txt", "w", encoding="utf-8") as f:
    f.write(f"{output['reward']}\n")
PY
