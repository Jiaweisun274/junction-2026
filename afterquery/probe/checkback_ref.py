"""Test of an explanation for why PPO quizzes less than the five-rule teacher: "teach ahead, check back later".

Variant of planned_ref.run: after ONE right answer the teacher moves on, but the idea it left is marked unverified;
after the first explanation of the next idea it quizzes the unverified one (a delayed check, which also benefits more
from the testing effect). Right: verified. Wrong: go back to it. Everything else as in the five rules.
"""
import itertools, sys
import numpy as np
import tutor2 as t, planned_ref as Q


def run(seed, maxexpl=2, diag=True, prereq=True, checkback=True, confirm=1):
    s = t.Sim(str(seed))
    k, streak, bug, need_explain = 0, 0, {}, not diag
    sess_expl, unverified = {}, None
    while s.turn < t.T:
        sess = s.turn // t.TURNS_PER
        kk = min(k, t.K - 1)
        if bug:
            b_k = next(iter(bug)); s.correct(b_k, bug.pop(b_k)); need_explain = False; continue
        # delayed check of the idea we left after a single right answer
        if checkback and unverified is not None and sess_expl.get((sess, kk), 0) + sum(v for (ss, kx), v in sess_expl.items() if kx == kk and ss < sess) >= 1:
            u = unverified; unverified = None
            ok, q, a, right = s.quiz(u)
            if not ok:
                k, streak, need_explain = u, 0, True
                if a == t.QUIZ[u][2]:
                    owner = t.CARRY.get(u, u); bug[owner] = t.BUGS[owner]; need_explain = False
            continue
        n_ex = sess_expl.get((sess, kk), 0)
        if prereq and kk > 0 and n_ex >= maxexpl and need_explain:
            p = kk - 1; ok, q, a, right = s.quiz(p)
            if ok: sess_expl[(sess, kk)] = 0
            else:
                k, streak = p, 0
                if a == t.QUIZ[p][2]:
                    owner = t.CARRY.get(p, p); bug[owner] = t.BUGS[owner]; need_explain = False
            continue
        if need_explain:
            s.explain(kk); sess_expl[(sess, kk)] = n_ex + 1; need_explain = False; continue
        ok, q, a, right = s.quiz(kk)
        if ok:
            streak += 1
            if streak >= confirm and k < t.K - 1:
                if checkback and confirm == 1: unverified = kk
                k, streak = k + 1, 0
                need_explain = not (diag and sess == 0 and s.turn < 4)
        else:
            streak, need_explain = 0, True
            if a == t.QUIZ[kk][2]:
                owner = t.CARRY.get(kk, kk); bug[owner] = t.BUGS[owner]; need_explain = False
                if owner != kk: k = owner
    return s


def quiz_share(fn, seeds, **kn):
    calls = {"q": 0, "n": 0}
    oq, oe, oc = t.Sim.quiz, t.Sim.explain, t.Sim.correct
    def q(self, k): calls["q"] += 1; calls["n"] += 1; return oq(self, k)
    def e(self, k): calls["n"] += 1; return oe(self, k)
    def c(self, k, b): calls["n"] += 1; return oc(self, k, b)
    t.Sim.quiz, t.Sim.explain, t.Sim.correct = q, e, c
    try:
        for sd in seeds: fn(sd, **kn)
    finally:
        t.Sim.quiz, t.Sim.explain, t.Sim.correct = oq, oe, oc
    return calls["q"] / calls["n"]


if __name__ == "__main__":
    train, test = range(1001, 1401), range(5001, 5401)
    gain = lambda fn, seeds, **kn: float(np.mean([fn(sd, **kn).exam() - t.Sim(str(sd)).exam() for sd in seeds]))
    best = max(((gain(run, train, maxexpl=m, diag=d, prereq=p), dict(maxexpl=m, diag=d, prereq=p))
                for m, d, p in itertools.product([1, 2, 3], [False, True], [False, True])), key=lambda x: x[0])
    print("check-back variant, tuned on 1001-1400:", best)
    P = dict(confirm=2, final=0, review=0, maxexpl=2, diag=True, prereq=True)
    for name, fn, kn in [("five rules (two in a row)", Q.run, P), ("check back later", run, best[1]),
                         ("one right answer, no check back", run, dict(best[1], checkback=False))]:
        print(f"{name:34s} held-out gain {gain(fn, test, **kn):.4f}  quiz share {quiz_share(fn, list(test)[:100], **kn):.0%}")
