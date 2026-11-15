"""Reference teachers without any LLM: PPO vs the five-rule teacher on fresh children, and the five-rule ablation.

  python compare_refs.py fresh      # pooled reward on batches 3016-3215 (salt eval1, 1,200 children)
  python compare_refs.py ablation   # raw exam gain per child on seeds 5001-5400, one rule changed at a time
"""
import sys
import numpy as np
import tutor2 as t, sweep_tutor2 as w, ppo_policy as P, planned_ref as Q, replay_traces as RT

FIVE_RULES = dict(confirm=2, final=0, review=0, maxexpl=2, diag=True, prereq=True)


def fresh(first=3016, last=3215):
    out = {"PPO": [], "five rules": []}
    for b in range(first, last + 1):
        cs = RT.child_seeds(b)
        base = sum(t.Sim(c).exam() for c in cs); full = sum(w.oracle_exam(c) for c in cs)
        out["PPO"].append((sum(P.run(c).exam() for c in cs) - base) / (full - base))
        out["five rules"].append((sum(Q.run(c, **FIVE_RULES).exam() for c in cs) - base) / (full - base))
    for k, v in out.items():
        print(f"{k:10s} mean pooled reward {np.mean(v):.3f} over {len(v)} batches")
    d = np.array(out["five rules"]) - np.array(out["PPO"])
    print(f"five rules minus PPO {d.mean():+.3f} (se {d.std(ddof=1) / np.sqrt(len(d)):.3f}), five rules win {int((d > 0).sum())}/{len(d)}")


def ablation(seeds=range(5001, 5401)):
    base = {sd: t.Sim(str(sd)).exam() for sd in seeds}
    g = lambda **kn: float(np.mean([Q.run(sd, **kn).exam() - base[sd] for sd in seeds]))
    ref = g(**FIVE_RULES)
    print(f"five rules: exam gain per child {ref:.4f}")
    for label, change in [("rule 2 off: move on after one right answer", dict(confirm=1)),
                          ("rule 3 off: no prerequisite check", dict(prereq=False)),
                          ("rule 4 off: no Monday diagnosis", dict(diag=False)),
                          ("rule 5 off: one review quiz at session start", dict(review=1))]:
        print(f"{label:46s} {g(**dict(FIVE_RULES, **change)) - ref:+.4f}")


if __name__ == "__main__":
    {"fresh": fresh, "ablation": ablation}[sys.argv[1] if len(sys.argv) > 1 else "ablation"]()
