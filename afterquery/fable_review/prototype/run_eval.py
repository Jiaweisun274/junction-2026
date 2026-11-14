"""python3 run_eval.py [n_seeds]  -- reference vs baselines, score = revenue / full-information revenue."""
import sys
import numpy as np
from env import Market, T, PRICES
from policies import fixed, Analyst, BeliefMyopic, BeliefDP, fluid_dp_q


def play(seed, make_policy, stop_after=T):
    m, pol = Market(seed), make_policy()
    while not m.done and m.t < stop_after:
        m.step(pol(m.observe(), m.stock0))
    return m.revenue


def oracle(seed):
    """Full-information benchmark: knows theta and r, re-solves the fluid DP every day, same demand luck."""
    m = Market(seed)
    while not m.done:
        m.step(float(PRICES[np.argmax(fluid_dp_q(m.theta, m.t, m.stock, m.r, m.stock0))]))
    return m.revenue


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    seeds = range(1000, 1000 + n)
    orc = np.array([oracle(s) for s in seeds])
    rows = {
        "do-nothing": lambda s: 0.0,
        "stop-early (reference, 10 of 50 days)": lambda s: play(s, BeliefDP, stop_after=10),
        "fixed list price 100": lambda s: play(s, lambda: fixed(100.0)),
        "fixed price 50": lambda s: play(s, lambda: fixed(50.0)),
        "analyst (regress on price, myopic, paced)": lambda s: play(s, Analyst),
        "belief + myopic (ablation: no lookahead)": lambda s: play(s, BeliefMyopic),
        "REFERENCE belief + DP lookahead": lambda s: play(s, BeliefDP),
    }
    print(f"{n} seeds, T={T}. score = revenue / full-information revenue on the same seed")
    print(f"{'policy':44s} mean   sd    min   P(score<ref)")
    res = {k: np.array([f(s) for s in seeds]) / orc for k, f in rows.items()}
    ref = res["REFERENCE belief + DP lookahead"]
    for k, v in res.items():
        print(f"{k:44s} {v.mean():.3f} {v.std(ddof=1):.3f} {v.min():.3f}  {np.mean(v < ref):.2f}")
