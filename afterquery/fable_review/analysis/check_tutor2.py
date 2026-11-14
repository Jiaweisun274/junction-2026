"""Free checks on the existing tutor2 simulator (no model APIs). Run: python3 check_tutor2.py [n_seeds]
1. Is the free `ask` a side channel?  2. Does massed quizzing lock memory in?  3. How noisy is a 3-run comparison?
"""
import os, sys
from math import erf, sqrt
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "afterquery", "probe"))
import tutor2 as t
import sweep_tutor2 as w
import ppo_policy as P

K = t.K


def is_yes(ans):
    return ans in t.YES_T


def play(seed, use_ask=False, lock_in=0, confirm=1):
    """Simple mastery-learning tutor: explain the frontier, quiz it, move on after `confirm` correct answers,
    fix a misconception when its signature shows. Options:
      use_ask  - before each turn ask all six concepts (free). If the answers are mixed, 'no' means not known and
                 'yes' means known, so the frontier is read off directly.
      lock_in  - after a concept is confirmed, quiz it this many extra times straight away."""
    s = t.Sim(str(seed))
    k, streak, need_explain, pending_lock, bug = 0, 0, True, [], {}
    while s.turn < t.T:
        if use_ask:
            ans = [is_yes(s.ask(j)) for j in range(K)]
            if any(ans) and not all(ans):                      # informative turn
                known = [a for a in ans]
                if s.last_explained is not None and ans[s.last_explained]:
                    known[s.last_explained] = None             # ambiguous: just explained
                front = next((j for j in range(K) if known[j] is False), None)
                if front is not None and front != k and all(known[j] for j in range(front)):
                    k, streak, need_explain = front, 0, True
                elif known[min(k, K - 1)] is False:
                    streak, need_explain = 0, True
            elif not any(ans) and k != 0:                      # everyone says no: nothing is known
                k, streak, need_explain = 0, 0, True
        if pending_lock:
            s.quiz(pending_lock.pop()); continue
        kk = min(k, K - 1)
        if kk in bug:
            s.correct(kk, bug.pop(kk)); need_explain = False; continue
        if need_explain:
            s.explain(kk); need_explain = False; continue
        ok, q, a, right = s.quiz(kk)
        if ok:
            streak += 1
            if streak >= confirm:
                pending_lock = [kk] * lock_in
                if k < K - 1:
                    k, streak, need_explain = k + 1, 0, True
        else:
            streak, need_explain = 0, True
            if a == t.QUIZ[kk][2]:
                owner = t.CARRY.get(kk, kk)
                bug[owner] = t.BUGS[owner]; need_explain = False
                if owner != kk:
                    k = owner
    return w.norm(str(seed), s.exam())


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seeds = list(range(2041, 2041 + n))
    pol = P.Policy()
    res = {
        "PPO reference": [w.norm(str(sd), P.run(sd, pol).exam()) for sd in seeds],
        "naive (2 correct in a row)": [play(sd, confirm=2) for sd in seeds],
        "simple (1 correct)": [play(sd) for sd in seeds],
        "simple + 1 lock-in quiz": [play(sd, lock_in=1) for sd in seeds],
        "simple + 2 lock-in quizzes": [play(sd, lock_in=2) for sd in seeds],
        "simple + free ask side channel": [play(sd, use_ask=True) for sd in seeds],
        "simple + ask + 1 lock-in": [play(sd, use_ask=True, lock_in=1) for sd in seeds],
    }
    ref = np.array(res["PPO reference"])
    g = np.random.default_rng(0)
    print(f"{n} held-out seeds {seeds[0]}-{seeds[-1]}, normalised score")
    print(f"{'policy':34s} mean   sd    P(beats PPO on a seed)  P(3-seed mean >= PPO 3-seed mean)")
    for name, v in res.items():
        v = np.array(v); d = v - ref
        idx = g.integers(0, n, (20000, 3))
        print(f"{name:34s} {v.mean():.3f} {v.std(ddof=1):.3f}   {np.mean(d > 0):.2f}                  {np.mean(d[idx].mean(1) >= 0):.2f}")
    inf = []
    for sd in seeds:
        s = t.Sim(str(sd))
        for turn in range(t.T):
            a = [is_yes(s.ask(j)) for j in range(K)]
            inf.append(any(a) and not all(a)); s.explain(min(turn // 5, K - 1))
    print(f"share of turns where the six free `ask` answers are mixed: {np.mean(inf):.2f}")
    print(f"PPO per-child sd {ref.std(ddof=1):.3f}. If agent-minus-PPO has mean -0.15 and paired sd 0.30 per child:")
    for kids in (1, 3, 6, 10):
        z = 0.15 / (0.30 / sqrt(3 * kids))
        print(f"  {kids:2d} children per run, 3 runs: P(reference mean > agent mean) = {0.5 * (1 + erf(z / sqrt(2))):.3f}")
