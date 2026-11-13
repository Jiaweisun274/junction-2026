"""Literature reference for the class (tutor3). Same components as lit_ref.py, written for one teacher and five
children; every parameter fixed a priori (no tuning on our seeds).
Per child and concept: BKT belief (P(L0)=0.5, P(T)=0.3, slip/guess from the rules, mastery at 0.95; Corbett &
Anderson 1995), half-life estimate (1 day, doubled on each correct recall; Leitner/HLR), misconception flag
(signature answer; Brown & Burton 1978), explanation count for prerequisite back-off after 3 (knowledge space theory).
Teacher rule each turn, in order:
  1. a known misconception -> targeted correction for that child
  2. a concept due for review (predicted recall < 0.9) for 2+ children -> class quiz; for one child -> quiz that child
  3. the last session's final turns -> class quiz on the concept with the lowest predicted exam-day recall
  4. 3+ children whose next concept is the same and not yet explained to them -> lecture it
  5. otherwise teach the child who is furthest behind: back off to the prerequisite after 3 failed explanations,
     explain while belief < 0.7, quiz to confirm.
"""
import numpy as np
import tutor2 as t2, tutor3 as t3

P_L0, P_T, MASTERY, H0, H_MULT, TARGET, BACKOFF = 0.5, 0.3, 0.95, 1.0, 2.0, 0.9, 3


class LitClass:
    def __init__(self):
        N, K = t3.N, t2.K
        self.b = [[P_L0] * K for _ in range(N)]
        self.learned = [[False] * K for _ in range(N)]
        self.h = [[H0] * K for _ in range(N)]
        self.last = [[0.0] * K for _ in range(N)]
        self.bug = [[None] * K for _ in range(N)]
        self.expl = [[0] * K for _ in range(N)]
        self.backoff = [None] * N

    def recall(self, i, k, at):
        return 2 ** (-(at - self.last[i][k]) / self.h[i][k]) if self.learned[i][k] else 0.0

    def observe(self, i, kind, k, now, obs=None, lecture=False):
        b = self.b[i]
        if kind == "quiz":
            ok, ans = obs
            if not ok and k in t2.BUGS and ans == t2.QUIZ[k][2]:
                self.bug[i][k] = t2.BUGS[k]; b[k] = 0.0; self.learned[i][k] = False; return
            if not ok and k in t2.CARRY and ans == t2.QUIZ[k][2]:
                j = t2.CARRY[k]; self.bug[i][j] = t2.BUGS[j]; b[j] = 0.0; self.learned[i][j] = False; return
            lm, ln = (1 - t2.SLIP, t2.GUESS) if ok else (t2.SLIP, 1 - t2.GUESS)
            b[k] = b[k] * lm / (b[k] * lm + (1 - b[k]) * ln)
            if self.learned[i][k]:
                if ok:
                    self.h[i][k] *= H_MULT
                else:
                    self.learned[i][k] = False; b[k] = min(b[k], 0.5)
                self.last[i][k] = now
            if self.backoff[i] == k and b[k] > 0.8:
                self.backoff[i] = None
                if k + 1 < t2.K:
                    self.expl[i][k + 1] = 0
        elif kind == "explain":
            if self.bug[i][k] is None:
                b[k] += (1 - b[k]) * P_T * (0.5 if lecture else 1.0)
            self.expl[i][k] += 1
            if self.learned[i][k]:
                self.last[i][k] = now
        elif kind == "correct":
            self.bug[i][k] = None; b[k] = 0.5
        if b[k] >= MASTERY and not self.learned[i][k] and self.bug[i][k] is None:
            self.learned[i][k] = True; self.last[i][k] = now; self.h[i][k] = H0

    def frontier(self, i):
        return next((k for k in range(t2.K) if not self.learned[i][k]), None)

    def act(self, c):
        N, K, now = t3.N, t2.K, c.clock.now()
        turns_left = t3.T - c.ct
        session = c.ct // t3.TURNS_PER
        # 1. misconceptions
        for i in range(N):
            for k in range(K):
                if self.bug[i][k]:
                    return ("correct", k, i, self.bug[i][k])
        # 3. end of course: exam-day review
        if session == 2:
            weak = [(k, sum(1 for i in range(N) if self.learned[i][k] and self.recall(i, k, t3.EXAM_DAY) < TARGET)) for k in range(K)]
            need = sum(1 for k, n in weak if n >= 2)
            if need and turns_left <= need:
                k = max(weak, key=lambda x: x[1])[0]
                return ("classquiz", k, None, None)
        # 2. due reviews
        due = {}
        for i in range(N):
            for k in range(K):
                if self.learned[i][k] and self.recall(i, k, now) < TARGET:
                    due.setdefault(k, []).append(i)
        if due:
            k, kids = max(due.items(), key=lambda x: len(x[1]))
            if len(kids) >= 2:
                return ("classquiz", k, None, None)
            i = kids[0]
            return ("quiz", k, i, None)
        # 4. lecture
        fr = [(i, self.frontier(i)) for i in range(N)]
        fresh = [(i, k) for i, k in fr if k is not None and self.expl[i][k] == 0 and self.backoff[i] is None]
        if fresh:
            ks = [k for _, k in fresh]
            k = max(set(ks), key=ks.count)
            if ks.count(k) >= 3:
                return ("lecture", k, None, None)
        # 5. furthest-behind child
        open_kids = [(i, k) for i, k in fr if k is not None]
        if not open_kids:
            k = min(range(K), key=lambda j: np.mean([self.recall(i, j, t3.EXAM_DAY) for i in range(N)]))
            return ("classquiz", k, None, None)
        i, k = min(open_kids, key=lambda x: (sum(self.learned[x[0]]), x[0]))
        if self.backoff[i] is not None:
            return ("quiz", self.backoff[i], i, None)
        if k > 0 and self.expl[i][k] >= BACKOFF:
            self.backoff[i] = k - 1; self.learned[i][k - 1] = False
            self.b[i][k - 1] = min(self.b[i][k - 1], 0.6)
            return ("quiz", k - 1, i, None)
        return ("explain", k, i, None) if self.b[i][k] < 0.7 else ("quiz", k, i, None)


def run(seed):
    c = t3.Class(seed)
    pol = LitClass()
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
    L, Nv, G = [], [], []
    for sd in seeds:
        base = t3.Class(sd).exam()
        g = t3.greedy_oracle(sd).exam()
        orc = max(g, t3.rollout_oracle(sd).exam())
        lit = run(sd).exam(); nv = t3.naive_class(sd).exam()
        nz = lambda e: (e - base) / (orc - base)
        L.append(nz(lit)); Nv.append(nz(nv)); G.append(nz(g))
        print(sd, "base %.3f oracle %.3f | lit %.2f naive %.2f greedy-full-info %.2f" % (base, orc, nz(lit), nz(nv), nz(g)))
    print("mean: lit %.2f naive %.2f greedy-full-info %.2f" % (np.mean(L), np.mean(Nv), np.mean(G)))
