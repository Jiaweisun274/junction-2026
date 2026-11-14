"""Wind-farm O&M desk (prototype of the Phase 1 pick). python3 windfarm.py [n_seeds]

N turbines, T days, ONE crew job per workable day. Each turbine has hidden wear 0..4
(4 = failed, visible). Hidden per-turbine wear rate q_i and hidden per-turbine sensor
bias b_i. Daily vibration reading y = wear + b_i + noise. Jobs: inspect (reveals wear),
service (wear -> 0), repair (failed -> running after a lead time).
Profit = running turbine-days - job costs. Score = (J - J_nothing) / (J_fullinfo - J_nothing).
Standard library + numpy only.
"""
import sys
import numpy as np

N, T = 8, 60
Q_RANGE, BIAS_SD, NOISE_SD = (0.03, 0.15), 1.0, 1.0
C_INSPECT, C_SERVICE, C_REPAIR, REPAIR_DAYS = 0.3, 2.0, 10.0, 5
P_STAY_GOOD, P_STAY_BAD = 0.8, 0.6


def hazard(q, w):                       # P(wear advances today | wear w)
    return q * (1 + 0.5 * w)


class Farm:
    def __init__(self, seed):
        g = np.random.default_rng(seed)
        self.q = g.uniform(*Q_RANGE, N)                 # hidden
        self.b = g.normal(0, BIAS_SD, N)                # hidden
        self._uw, self._ey, self._uwx = g.uniform(size=(T, N)), g.normal(0, NOISE_SD, (T, N)), g.uniform(size=T)
        self.w = np.zeros(N, int)                       # hidden
        self.down = np.zeros(N, int)                    # days until a repaired turbine is back
        self.t, self.good, self.profit = 0, True, 0.0

    def observe(self):
        run = (self.w < 4)
        y = np.where(run, self.w + self.b + self._ey[self.t], np.nan)
        return {"day": self.t, "workable": self.good, "failed": ~run & (self.down == 0),
                "in_repair": self.down > 0, "reading": y}

    def step(self, job=None):
        """job = None | ("inspect"|"service"|"repair", i). Returns inspected wear or None."""
        out, off = None, None
        if job and self.good:
            kind, i = job
            if kind == "inspect" and self.w[i] < 4:
                self.profit -= C_INSPECT; out = int(self.w[i])
            elif kind == "service" and self.w[i] < 4:
                self.profit -= C_SERVICE; self.w[i] = 0; off = i
            elif kind == "repair" and self.w[i] == 4 and self.down[i] == 0:
                self.profit -= C_REPAIR; self.down[i] = REPAIR_DAYS
        run = self.w < 4
        self.profit += run.sum() - (1 if off is not None else 0)
        adv = run & (self._uw[self.t] < hazard(self.q, self.w))
        self.w[adv] += 1
        back = self.down == 1
        self.down[self.down > 0] -= 1
        self.w[back] = 0
        self.good = self._uwx[self.t] < (P_STAY_GOOD if self.good else 1 - P_STAY_BAD)
        self.t += 1
        return out


# ---------------- policies: act(obs, last_inspection) -> job ----------------

def nothing(obs, _):
    return None


def reactive(obs, _):
    f = np.flatnonzero(obs["failed"])
    return ("repair", f[0]) if len(f) else None


class RawThreshold:
    """Hand-play heuristic: trust the dial. Service the turbine whose 3-day mean reading is highest and > tau."""
    def __init__(self, tau=2.5):
        self.tau, self.hist = tau, []

    def __call__(self, obs, _):
        self.hist.append(obs["reading"])
        f = np.flatnonzero(obs["failed"])
        if len(f):
            return ("repair", f[0])
        m = np.nan_to_num(np.array(self.hist[-3:]), nan=-9.0).mean(axis=0)
        m = np.where(np.isnan(obs["reading"]), -9, m)
        i = int(np.argmax(m))
        return ("service", i) if m[i] > self.tau else None


class Periodic:
    """Calendar maintenance: repair failures, otherwise service turbines round-robin every `gap` days."""
    def __init__(self, gap=2):
        self.gap, self.k = gap, 0

    def __call__(self, obs, _):
        f = np.flatnonzero(obs["failed"])
        if len(f):
            return ("repair", f[0])
        if obs["day"] % self.gap == 0:
            for _ in range(N):
                self.k = (self.k + 1) % N
                if not np.isnan(obs["reading"][self.k]):
                    return ("service", self.k)
        return None


BG = np.linspace(-3, 3, 25)                       # bias grid
QG = np.linspace(Q_RANGE[0], Q_RANGE[1], 7)       # rate grid


class BayesIndex:
    """Reference: exact grid Bayes filter per turbine over (wear, bias, rate) + a cost-based index rule.
    Uses only public rules and observations."""
    def __init__(self, k_inspect=0.0, s_min=1.0, bias_sd=BIAS_SD):
        prior = np.exp(-0.5 * (BG / bias_sd) ** 2)[None, :, None] * np.ones((1, len(BG), len(QG)))
        self.B = np.zeros((N, 5, len(BG), len(QG)))
        self.B[:, 0] = prior / prior.sum()
        self.k_inspect, self.s_min, self.job = k_inspect, s_min, None

    def _advance(self, i):                         # one day of wear progression
        B, new = self.B[i], np.zeros_like(self.B[i])
        for w in range(4):
            h = np.clip(hazard(QG, w), 0, 1)[None, :]
            new[w] += B[w] * (1 - h); new[w + 1] += B[w] * h
        self.B[i] = new

    def __call__(self, obs, inspected):
        # 1. fold in yesterday's job and overnight progression
        if self.job:
            kind, i = self.job
            if kind == "inspect" and inspected is not None:
                keep = self.B[i, inspected].copy(); self.B[i] = 0; self.B[i, inspected] = keep
            elif kind == "service":
                tot = self.B[i].sum(axis=0); self.B[i] = 0; self.B[i, 0] = tot
        if obs["day"] > 0:
            for i in range(N):
                self._advance(i)
        # 2. condition on what is visible this morning
        for i in range(N):
            y = obs["reading"][i]
            if np.isnan(y):                        # failed or in repair
                if obs["in_repair"][i]:
                    tot = self.B[i].sum(axis=0); self.B[i] = 0; self.B[i, 0] = tot  # comes back as new
                else:
                    self.B[i, :4] = 0
            else:
                self.B[i, 4] = 0
                lik = np.exp(-0.5 * ((y - np.arange(5)[:, None] - BG[None, :]) / NOISE_SD) ** 2)
                self.B[i] *= lik[:, :, None]
            s = self.B[i].sum()
            self.B[i] = self.B[i] / s if s > 0 else self.B[i]
        # 3. index rule
        self.job = None
        if not obs["workable"]:
            return None
        best, val = None, 0.0
        F = C_REPAIR + REPAIR_DAYS + 1.0           # cost of one failure
        for i in range(N):
            if obs["failed"][i]:
                v, job = 1.0 + 0.01 * i, ("repair", i)          # each idle day loses 1
            elif np.isnan(obs["reading"][i]):
                continue
            else:
                pw = self.B[i].sum(axis=(1, 2))
                pfail = (self.B[i, 3].sum(axis=0) * np.clip(hazard(QG, 3), 0, 1)).sum()
                p3next = pw[3] + (self.B[i, 2].sum(axis=0) * hazard(QG, 2)).sum()
                serve = (pfail + 0.35 * p3next) * F - (C_SERVICE + 1) * (pw[0] + 0.5 * pw[1]) - self.s_min
                unsure = min(pw[2] + pw[3], 1 - pw[2] - pw[3])
                insp = self.k_inspect * unsure * F * 0.25 - C_INSPECT
                v, job = (serve, ("service", i)) if serve >= insp else (insp, ("inspect", i))
            if v > val:
                best, val = job, v
        self.job = best
        return best


def full_info(farm):
    """Benchmark that sees true wear and rates (not available to any agent)."""
    def act(obs, _):
        run3 = np.flatnonzero(farm.w == 3)
        if len(run3):
            return ("service", run3[np.argmax(farm.q[run3])])
        f = np.flatnonzero(obs["failed"])
        if len(f):
            return ("repair", f[0])
        run2 = np.flatnonzero((farm.w == 2) & (farm.q > 0.1))
        return ("service", run2[np.argmax(farm.q[run2])]) if len(run2) else None
    return act


def play(seed, make, stop_after=T, oracle=False):
    farm = Farm(seed)
    pol = full_info(farm) if oracle else make()
    last = None
    while farm.t < T:
        job = pol(farm.observe(), last) if farm.t < stop_after else None
        last = farm.step(job)
    return farm.profit


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seeds = range(5000, 5000 + n)
    j0 = np.array([play(s, lambda: nothing) for s in seeds])
    j1 = np.array([play(s, None, oracle=True) for s in seeds])
    rows = {
        "do-nothing": lambda s: play(s, lambda: nothing),
        "stop-early (reference, 12 of 60 days)": lambda s: play(s, BayesIndex, stop_after=12),
        "reactive (repair failures only)": lambda s: play(s, lambda: reactive),
        "periodic service (every 2nd day)": lambda s: play(s, Periodic),
        "raw 3-day reading > 2.5 (trust the dial)": lambda s: play(s, RawThreshold),
        "raw 3-day reading > 2.0": lambda s: play(s, lambda: RawThreshold(2.0)),
        "ablation: filter ignores sensor bias": lambda s: play(s, lambda: BayesIndex(bias_sd=0.05)),
        "variant: filter + inspections": lambda s: play(s, lambda: BayesIndex(k_inspect=1.0)),
        "REFERENCE Bayes filter + index rule": lambda s: play(s, BayesIndex),
    }
    print(f"{n} seeds. score = (J - J_nothing)/(J_fullinfo - J_nothing); mean J_nothing={j0.mean():.0f}, J_fullinfo={j1.mean():.0f}")
    print(f"{'policy':44s} mean   sd    P(score<ref)  3-seed-mean sd")
    res = {k: (np.array([f(s) for s in seeds]) - j0) / (j1 - j0) for k, f in rows.items()}
    ref = res["REFERENCE Bayes filter + index rule"]
    for k, v in res.items():
        print(f"{k:44s} {v.mean():.3f} {v.std(ddof=1):.3f}   {np.mean(v < ref):.2f}        {v.std(ddof=1)/np.sqrt(3):.3f}")
    d = ref - res["raw 3-day reading > 2.5 (trust the dial)"]
    g = np.random.default_rng(0)
    m3 = d[g.integers(0, n, (20000, 3))].mean(axis=1)
    print(f"paired gap reference - dial: mean {d.mean():.3f}, sd {d.std(ddof=1):.3f}; "
          f"P(3-seed paired mean gap > 0) = {np.mean(m3 > 0):.3f}, > 0.05: {np.mean(m3 > 0.05):.3f}")
