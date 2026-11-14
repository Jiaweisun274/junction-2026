"""Expert iteration step: keep each child's best sampled episodes (by exam gain) and fine-tune a LoRA on Gemma's own
turns from them. The reward is the only supervision: no human or reference-policy labels.

  python train.py --samples runs/r1_sample.json [runs/r2_sample.json ...] --out lora_r1 --top 2
"""
import argparse, collections, json
import numpy as np


def select(paths, top):
    rows, kept, all_g = [], 0, []
    for p in paths:
        eps = json.load(open(p))["episodes"]
        by = collections.defaultdict(list)
        for e in eps:
            by[e["seed"]].append(e); all_g.append(e["gain"])
        for seed, group in by.items():
            mean = np.mean([e["gain"] for e in group])
            best = sorted(group, key=lambda e: -e["gain"])[:top]
            for e in best:
                if e["gain"] > mean and e["gain"] > 0 and e["invalid"] == 0:
                    kept += 1
                    rows += [{"prompt": [{"role": "user", "content": x["prompt"]}],
                              "completion": [{"role": "assistant", "content": x["reply"]}]} for x in e["turns"]]
    print(f"kept {kept} episodes, {len(rows)} turns; mean gain of all sampled {np.mean(all_g):.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=1.0)
    a = ap.parse_args()
    import os
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    rows = select(a.samples, a.top)
    ds = Dataset.from_list(rows).shuffle(seed=0)
    cfg = SFTConfig(output_dir=a.out, num_train_epochs=a.epochs, per_device_train_batch_size=8,
                    gradient_accumulation_steps=2, learning_rate=1e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
                    bf16=True, logging_steps=20, save_strategy="no", report_to=[], max_length=6144,
                    completion_only_loss=True, gradient_checkpointing=True)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    tr = SFTTrainer(model=os.environ.get("GEMMA_MODEL", "google/gemma-4-E4B-it"), args=cfg, train_dataset=ds,
                    peft_config=lora)
    tr.train()
    tr.model.save_pretrained(a.out)
    tr.processing_class.save_pretrained(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
