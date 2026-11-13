"""RTI reference for the class: Response to Intervention / differentiated instruction (tiered instruction).
Tier 1: whole-class lecture of the class's next concept, then a class quiz to see who got it.
Tier 2/3: one-to-one help (explain, or targeted misconception correction) for children who failed the class quiz,
          with prerequisite back-off after 3 failed explanations.
Spaced review: class quiz at the start of each later session on concepts most children have learned
(Leitner/HLR-style half-life estimate, review below 0.9 predicted recall), and exam-day review at the end.
Beliefs and parameters are the same a-priori values as lit_class.py (no tuning on our seeds).
"""
import numpy as np
import tutor2 as t2, tutor3 as t3, lit_class as L


class RTI(L.LitClass):
    def __init__(self):
        super().__init__()
        self.stage = "lecture"     # lecture -> check -> remediate
        self.k = 0

    def class_next(self):
        fr = [self.frontier(i) for i in range(t3.N)]
        fr = [k for k in fr if k is not None]
        return min(fr) if fr else None

    def act(self, c):
        N, K, now = t3.N, t2.K, c.clock.now()
        turns_left = t3.T - c.ct
        session = c.ct // t3.TURNS_PER
        in_sess = c.ct % t3.TURNS_PER
        for i in range(N):
            for k in range(K):
                if self.bug[i][k]:
                    return ("correct", k, i, self.bug[i][k])
        # exam-day review at the very end
        if session == 2 and turns_left <= 3:
            k = min(range(K), key=lambda j: sum(self.recall(i, j, t3.EXAM_DAY) for i in range(N) if self.learned[i][j]) - 10 * (sum(self.learned[i][j] for i in range(N)) == 0))
            return ("classquiz", k, None, None)
        # spaced review at session start
        due = {}
        for i in range(N):
            for k in range(K):
                if self.learned[i][k] and self.recall(i, k, now) < L.TARGET:
                    due.setdefault(k, []).append(i)
        if due and in_sess < 4:
            k, kids = max(due.items(), key=lambda x: len(x[1]))
            if len(kids) >= 2:
                return ("classquiz", k, None, None)
        k = self.class_next()
        if k is None:
            k = min(range(K), key=lambda j: np.mean([self.recall(i, j, t3.EXAM_DAY) for i in range(N)]))
            return ("classquiz", k, None, None)
        behind = [i for i in range(N) if self.frontier(i) is not None and self.frontier(i) <= k]
        # Tier 1: lecture when most children still need this concept and it has not been lectured yet
        if len(behind) >= 3 and all(self.expl[i][k] == 0 for i in behind):
            return ("lecture", k, None, None)
        if len(behind) >= 3 and any(self.b[i][k] < 0.7 for i in behind) and all(self.expl[i][k] <= 1 for i in behind) and max(self.b[i][k] for i in behind) < 0.9:
            # after one lecture: check the class
            if all(self.b[i][k] != L.P_L0 for i in behind) and sum(1 for i in behind if self.b[i][k] > 0.6) == 0:
                return ("lecture", k, None, None)
            return ("classquiz", k, None, None)
        # Tier 2/3: individual help for the child furthest behind on this concept
        i = min(behind, key=lambda j: (self.b[j][k], j)) if behind else None
        if i is None:
            return ("classquiz", k, None, None)
        kk = self.frontier(i)
        if self.backoff[i] is not None:
            return ("quiz", self.backoff[i], i, None)
        if kk > 0 and self.expl[i][kk] >= L.BACKOFF + 1:
            self.backoff[i] = kk - 1; self.learned[i][kk - 1] = False
            self.b[i][kk - 1] = min(self.b[i][kk - 1], 0.6)
            return ("quiz", kk - 1, i, None)
        return ("explain", kk, i, None) if self.b[i][kk] < 0.7 else ("quiz", kk, i, None)


def run(seed):
    c = t3.Class(seed)
    pol = RTI()
    while c.ct < t3.T:
        a = pol.act(c)
        now = c.clock.now()
        res = c.apply(a)
        kind, k, s, bug = a
        if kind in ("quiz", "classquiz"):
            for i, (ok, q, ans, right) in res:
                pol.observe(i, "quiz", k, now, (ok, ans))
        elif kind == "explain":
            pol.observe(s, "explain", k, now)
        elif kind == "lecture":
            for i in range(t3.N):
                pol.observe(i, "explain", k, now, lecture=True)
        elif kind == "correct":
            pol.observe(s, "correct", k, now)
    return c


if __name__ == "__main__":
    import sys
    seeds = [int(x) for x in sys.argv[1:]] or list(range(2001, 2011))
    R, Lt, Nv = [], [], []
    for sd in seeds:
        base = t3.Class(sd).exam()
        orc = max(t3.greedy_oracle(sd).exam(), t3.rollout_oracle(sd).exam())
        nz = lambda e: (e - base) / (orc - base)
        r, l, n = nz(run(sd).exam()), nz(L.run(sd).exam()), nz(t3.naive_class(sd).exam())
        R.append(r); Lt.append(l); Nv.append(n)
        print(sd, "rti %.2f lit %.2f naive %.2f" % (r, l, n))
    print("mean: rti %.2f lit %.2f naive %.2f" % (np.mean(R), np.mean(Lt), np.mean(Nv)))
