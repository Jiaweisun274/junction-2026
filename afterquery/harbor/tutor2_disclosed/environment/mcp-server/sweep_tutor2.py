"""Tune the observation-only reference of tutor2 on training seeds; report on held-out test seeds."""
import itertools, numpy as np, tutor2 as t

def oracle_exam(sd, cache={}):
    if sd not in cache:
        cache[sd] = max(t.run_policy(sd, t.oracle_step).exam(), t.run_oracle2(sd).exam())
    return cache[sd]

def norm(sd, e):
    base, orc = t.Sim(sd).exam(), oracle_exam(sd)
    return (e - base) / (orc - base)

def evaluate(seeds):
    return float(np.mean([norm(sd, t.run_policy(sd, t.reference_step, obs=True).exam()) for sd in seeds]))

if __name__ == "__main__":
    train = [str(s) for s in range(1001, 1061)]
    test = [str(s) for s in range(2001, 2061)]
    grid = dict(start=[0.2, 0.3, 0.5], lo=[0.5, 0.7], hi=[0.8, 0.9], review=[0.4, 0.6, 0.8], endrev=[2, 4, 6], mult=[2.0, 3.0])
    best = (-1, None)
    for vals in itertools.product(*grid.values()):
        t.REF.update(dict(zip(grid, vals)))
        v = evaluate(train)
        if v > best[0]:
            best = (v, dict(zip(grid, vals)))
    print("best on train", round(best[0], 3), best[1])
    t.REF.update(best[1])
    print("reference on test", round(evaluate(test), 3))
    nv = np.mean([norm(sd, t.run_policy(sd, t.naive_step, obs=True).exam()) for sd in test])
    print("naive on test", round(float(nv), 3))
