"""POMDP planner reference for tutor2 (observation-only).

Belief: a particle filter over the child's hidden state AND the unknown per-child parameters. Each particle is a
complete hypothetical child (mastery / misconception per concept, memory half-lives, learn rates, susceptibilities).
The prior uses only what the rules state plus wide priors on what they leave open:
  - stated in the rules: quiz reliability (0.9 / 0.25), prerequisite chain, misconception signatures and that they
    carry into the next concept, exam scoring, that the child may know some early concepts;
  - wide priors (NOT the simulator's distributions): learn chance U(0.15, 0.75), new-concept half-life U(1, 10) days,
    pre-known half-life U(5, 25), each misconception possible with prob 0.35, starting knowledge 0-3 concepts uniform.
  - qualitative words mapped to fixed numbers: "rarely" = 0.1, "usually" = 0.7, "very hard" = 0.05; quiz recall
    strengthens memory 2.5x (SM-2 default ease factor), re-explaining 1.5x.
Filtering is likelihood-free: each particle is stepped forward with the action, and after a quiz it is kept only if
it would have produced the same kind of answer (correct / misconception signature / other wrong). Survivors are
resampled with fresh random futures.

Planning: Monte Carlo policy improvement over the literature policy (lit_ref). For each candidate action, sample
particles, take the action, then let the literature policy continue (seeing only simulated observations) to the
exam; pick the action with the best average exam score. The literature policy's own choice is always a candidate,
so the planner can only improve on it (up to Monte Carlo noise).
"""
import copy
import numpy as np
import tutor2 as t
import lit_ref as L

N_PART, M_ROLL = 400, 48


def _new_particle(rng, real):
    s = object.__new__(t.Sim)
    s.p = dict(plearn=rng.uniform(0.15, 0.75, t.K),
               suscept={b: bool(rng.random() < 0.35) for b in t.BUGS.values()},
               h0=rng.uniform(1.0, 10.0, t.K),
               u=rng.random((t.T + 2, 8)))
    s.state, s.h, s.last = ["unlearned"] * t.K, [0.0] * t.K, [0.0] * t.K
    s.turn, s.last_explained = real.turn, None
    m0 = int(rng.integers(0, 4))
    for k in range(m0):
        b = t.BUGS.get(k)
        if b and s.p["suscept"][b] and rng.random() < 0.5:
            s.state[k] = "buggy"
        else:
            s.state[k], s.h[k], s.last[k] = "mastered", rng.uniform(5.0, 25.0), -3.0
    return s


def _clone(s, rng):
    c = object.__new__(t.Sim)
    c.p = dict(s.p)
    u = s.p["u"].copy(); u[s.turn:] = rng.random(u[s.turn:].shape); c.p["u"] = u
    c.state, c.h, c.last = list(s.state), list(s.h), list(s.last)
    c.turn, c.last_explained = s.turn, s.last_explained
    c._sess_expl = dict(getattr(s, "_sess_expl", {}))
    return c


def _category(k, ok, ans):
    if ok:
        return "correct"
    return "sig" if ans == t.QUIZ[k][2] else "wrong"


def _apply(s, a):
    kind, k, bug = a
    if kind == "explain":
        s.explain(k); return None
    if kind == "correct":
        s.correct(k, bug); return None
    ok, q, ans, right = s.quiz(k)
    return (ok, ans)


class Planner:
    def __init__(self, seed):
        self.rng = np.random.default_rng(int(seed) * 7 + 11)
        self.parts = None
        self.lit_mem = {}
        self.last_obs = None

    def _init(self, real):
        self.parts = [_new_particle(self.rng, real) for _ in range(N_PART)]

    def update(self, real, action, obs):
        """Filter the particles with the real action and its observation."""
        kept = []
        for s in self.parts:
            o = _apply(s, action)
            if action[0] != "quiz" or _category(action[1], *o) == _category(action[1], *obs):
                kept.append(s)
        if len(kept) < 20:   # belief collapse: restart from the prior, replay is not possible, so keep what matched
            fresh = [_new_particle(self.rng, real) for _ in range(N_PART)]
            kept = kept + [p for p in fresh if action[0] != "quiz" or _category(action[1], *_apply(_clone(p, self.rng), action)) == _category(action[1], *obs)]
            if not kept:
                kept = fresh
        idx = self.rng.integers(0, len(kept), N_PART)
        self.parts = [_clone(kept[i], self.rng) for i in idx]
        for p in self.parts:
            p.turn = real.turn

    def _marginal(self, k, state):
        return np.mean([p.state[k] == state for p in self.parts])

    def _candidates(self, real, lit_choice):
        cands = {lit_choice}
        m = [self._marginal(k, "mastered") for k in range(t.K)]
        first_gap = next((k for k in range(t.K) if m[k] < 0.9), t.K - 1)
        for k in range(min(first_gap + 2, t.K)):
            cands.add(("explain", k, None))
            cands.add(("quiz", k, None))
        for k in t.BUGS:
            if self._marginal(k, "buggy") > 0.1:
                cands.add(("correct", k, t.BUGS[k]))
        return list(cands)

    def _rollout_value(self, a, mem, starts):
        """Value of action a on a fixed set of sampled futures (common random numbers across candidates)."""
        total = 0.0
        for st in starts:
            s = object.__new__(t.Sim)
            s.p = st.p   # read-only during the rollout
            s.state, s.h, s.last = list(st.state), list(st.h), list(st.last)
            s.turn, s.last_explained = st.turn, st.last_explained
            m = copy.deepcopy(mem)
            o = _apply(s, a)
            lo = (a[0], a[1], o)
            while s.turn < t.T:
                act = L.lit_step(s, m, lo)
                o = _apply(s, act)
                lo = (act[0], act[1], o)
            total += s.exam()
        return total / len(starts)

    def act(self, real, last_obs):
        if self.parts is None:
            self._init(real)
        # keep the literature policy's own memory in step with what really happened
        lit_choice = L.lit_step(real, self.lit_mem, last_obs)
        mem = copy.deepcopy(self.lit_mem)
        idx = self.rng.integers(0, len(self.parts), M_ROLL)
        starts = [_clone(self.parts[i], self.rng) for i in idx]
        best, best_v = lit_choice, self._rollout_value(lit_choice, mem, starts)
        for a in self._candidates(real, lit_choice):
            if a == lit_choice:
                continue
            v = self._rollout_value(a, mem, starts)
            if v > best_v + 1e-9:
                best, best_v = a, v
        return best


def run(seed):
    real = t.Sim(str(seed))
    pl = Planner(seed)
    last_obs = None
    while real.turn < t.T:
        a = pl.act(real, last_obs)
        obs = _apply(real, a)
        pl.update(real, a, obs)
        last_obs = (a[0], a[1], obs)
    return real


if __name__ == "__main__":
    import sys, time
    import sweep_tutor2 as w
    seeds = [str(s) for s in (range(int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else range(2001, 2004))]
    P, R, N = [], [], []
    for sd in seeds:
        t0 = time.time()
        p = w.norm(sd, run(sd).exam())
        r = w.norm(sd, t.run_policy(sd, L.lit_step, obs=True).exam())
        n = w.norm(sd, t.run_policy(sd, t.naive_step, obs=True).exam())
        P.append(p); R.append(r); N.append(n)
        print(f"{sd} pomdp {p:.2f} literature {r:.2f} naive {n:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    P, R, N = map(np.array, (P, R, N))
    print(f"mean: pomdp {P.mean():.2f} literature {R.mean():.2f} naive {N.mean():.2f} | pomdp beats literature on {np.mean(P > R):.0%}")
