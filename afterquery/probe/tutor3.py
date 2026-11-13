"""Probe 5: a small class. Five children, three sessions (Mon/Wed/Fri, 20 turns each), exam next Monday.
Each child follows the tutor2 student model (forgetting, testing effect, spacing, misconceptions).
Each turn the teacher does ONE thing:
  explain K S / quiz K S / correct K BUG S   -> one-to-one with child S (1..5)
  lecture K                                   -> explain K to the whole class; each child learns with half the
                                                 one-to-one chance (Bloom 1984: one-to-one tutoring is far more effective)
  classquiz K                                 -> every child answers the same exercise; you see every answer
  ask K S                                     -> free self-report of child S
Score = class average exam score.

CLI ($GAME_DIR/tutor3_state.json): new SEED | state | <action...> | finish
"""
import copy, json, os, sys
import numpy as np
import tutor2 as t2

N = 5
SESSIONS = [0.0, 2.0, 4.0]
TURNS_PER = 20
T = TURNS_PER * len(SESSIONS)
EXAM_DAY = 7.0
LECTURE_FACTOR = 0.5
NAMES = ["Aino", "Eetu", "Helmi", "Leo", "Venla"]
DIR = os.environ.get("GAME_DIR", ".")
PATH = os.path.join(DIR, "tutor3_state.json")


class Clock:
    def __init__(self):
        self.ct = 0

    def now(self):
        s = min(self.ct // TURNS_PER, len(SESSIONS) - 1)
        return SESSIONS[s] + 0.005 * (self.ct % TURNS_PER)


class Student(t2.Sim):
    def __init__(self, seed, clock):
        super().__init__(seed)
        r = np.random.default_rng(int(seed) + 999)
        self.p = dict(self.p)
        self.p["u"] = r.random((400, 8))        # draws indexed by this child's own action count
        self.clock = clock

    def now(self, turn=None):
        return self.clock.now()

    def explain_scaled(self, k, factor):
        old = self.p["plearn"].copy()
        self.p["plearn"] = old * factor
        old_leak = t2.LEAK
        t2.LEAK = old_leak * factor
        try:
            self.explain(k)
        finally:
            self.p["plearn"] = old
            t2.LEAK = old_leak


class Class:
    def __init__(self, seed):
        self.seed = int(seed)
        self.clock = Clock()
        self.kids = [Student(self.seed * 10 + i, self.clock) for i in range(N)]

    @property
    def ct(self):
        return self.clock.ct

    def apply(self, a):
        """a = (kind, k, student or None, bug or None). Returns list of (student, quiz result) for quizzes."""
        kind, k, s, bug = a
        out = []
        if kind == "explain":
            self.kids[s].explain(k)
        elif kind == "quiz":
            out.append((s, self.kids[s].quiz(k)))
        elif kind == "correct":
            self.kids[s].correct(k, bug)
        elif kind == "lecture":
            for kid in self.kids:
                kid.explain_scaled(k, LECTURE_FACTOR)
        elif kind == "classquiz":
            for i, kid in enumerate(self.kids):
                out.append((i, kid.quiz(k)))
        self.clock.ct += 1
        return out

    def exam(self):
        return float(np.mean([kid.exam() for kid in self.kids]))

    def clone(self, rng=None):
        c = copy.copy(self)
        c.clock = Clock(); c.clock.ct = self.clock.ct
        c.kids = []
        for kid in self.kids:
            n = copy.copy(kid)
            n.state, n.h, n.last = list(kid.state), list(kid.h), list(kid.last)
            n.p = dict(kid.p)
            if rng is not None:
                u = kid.p["u"].copy(); u[kid.turn:] = rng.random(u[kid.turn:].shape); n.p["u"] = u
            n.clock = c.clock
            c.kids.append(n)
        return c


# ---------------- full-information policies (for the oracle) ----------------
def kid_gain(kid):
    """How much this child's exam could still rise: unmastered concepts plus fading ones."""
    g = 0.0
    for k in range(t2.K):
        if kid.state[k] == "mastered":
            g += 0.65 * (1 - kid.recall(k, EXAM_DAY))
        else:
            g += 0.65
    return g


def greedy_full(c):
    """Base policy: help the child with most to gain, using the single-child full-information rule."""
    i = max(range(N), key=lambda j: kid_gain(c.kids[j]))
    kid = c.kids[i]
    kid.turn_saved = kid.turn
    a, k, b = t2.oracle_step(kid)
    return (a, k, i, b)


def candidates(c):
    cands = []
    for i, kid in enumerate(c.kids):
        a, k, b = t2.oracle_step(kid)
        cands.append((a, k, i, b))
    fr = [next((k for k in range(t2.K) if kid.state[k] != "mastered"), None) for kid in c.kids]
    fr = [k for k in fr if k is not None]
    if fr:
        k = max(set(fr), key=fr.count)
        cands.append(("lecture", k, None, None))
    fading = [(k, sum(1 for kid in c.kids if kid.state[k] == "mastered" and kid.recall(k) < 0.7)) for k in range(t2.K)]
    k, cnt = max(fading, key=lambda x: x[1])
    if cnt >= 2:
        cands.append(("classquiz", k, None, None))
    return cands


def rollout_oracle(seed, n=6):
    c = Class(seed)
    rng = np.random.default_rng(seed + 5)
    while c.ct < T:
        cands = candidates(c)
        futures = [np.random.default_rng(rng.integers(1 << 30)) for _ in range(n)]
        seeds = [f.integers(1 << 30) for f in futures]
        best, bv = None, -1
        for a in cands:
            v = 0.0
            for sd in seeds:
                cc = c.clone(np.random.default_rng(sd))
                cc.apply(a)
                while cc.ct < T:
                    cc.apply(greedy_full(cc))
                v += cc.exam()
            if v > bv + 1e-12:
                best, bv = a, v
        c.apply(best)
    return c


def greedy_oracle(seed):
    c = Class(seed)
    while c.ct < T:
        c.apply(greedy_full(c))
    return c


# ---------------- observation-only policies ----------------
def make_lit_class():
    """Literature reference: the tutor2 literature policy per child (BKT mastery + spaced review + prerequisite
    back-off + misconception remediation), plus a teacher rule: serve the most urgent need in this order —
    misconception, due review, teaching the child who is furthest behind. Lecture when 3+ children share the
    same next concept and none of them has started it."""
    import lit_ref
    mems = [dict() for _ in range(N)]
    lasts = [None] * N

    class Proxy:
        """Lets the single-child policy see class time and the child's turn index."""
        def __init__(self, c, i):
            self.c, self.i = c, i
            self.turn = min(c.ct * t2.T // T, t2.T - 1)    # map class time onto the 30-turn single-child schedule
        def now(self):
            return self.c.clock.now()

    def step(c, last_by_kid):
        props = []
        for i in range(N):
            lo = last_by_kid.get(i)
            if lo is not None:
                lasts[i] = lo
            px = Proxy(c, i)
            a = lit_ref.lit_step(px, mems[i], lasts[i])
            lasts[i] = None
            props.append((i, a))
        def urgency(ia):
            i, (a, k, b) = ia
            m = mems[i]
            if a == "correct":
                return (0, 0)
            if a == "quiz" and m["learned"][k]:
                return (1, 0)
            return (2, -sum(m["learned"]))          # furthest behind first
        # lecture if 3+ children want to explain the same not-yet-started concept
        wants = [(i, k) for i, (a, k, b) in props if a == "explain" and mems[i]["explains"][k] == 0]
        if wants:
            ks = [k for _, k in wants]
            k = max(set(ks), key=ks.count)
            if ks.count(k) >= 3:
                for i, kk in wants:
                    if kk == k:
                        mems[i]["explains"][k] += 0      # counted on observation
                return ("lecture", k, None, None), [i for i, kk in wants if kk == k]
        i, (a, k, b) = min(props, key=urgency)
        return (a, k, i, b), [i]

    return step, mems


def run_obs_policy(seed, maker):
    c = Class(seed)
    step, mems = maker()
    pending = {}
    while c.ct < T:
        act, involved = step(c, pending)
        res = c.apply(act)
        pending = {}
        kind, k, s, b = act
        if kind in ("quiz", "classquiz"):
            for i, (ok, q, ans, right) in res:
                pending[i] = ("quiz", k, (ok, ans))
        elif kind == "explain":
            pending[s] = ("explain", k, None)
        elif kind == "correct":
            pending[s] = ("correct", k, None)
        elif kind == "lecture":
            for i in range(N):
                pending[i] = ("explain", k, None)
        # children not involved keep their previous proposal state (no new observation)
    return c


def naive_class(seed):
    """Lecture each concept in order twice, then a class quiz; no review, no individual help."""
    c = Class(seed)
    k = 0
    while c.ct < T:
        for a in [("lecture", k, None, None), ("lecture", k, None, None), ("classquiz", k, None, None)]:
            if c.ct < T:
                c.apply(a)
        k = min(k + 1, t2.K - 1)
    return c


def score_all(seed, with_rollout=True):
    base = Class(seed).exam()
    orc = greedy_oracle(seed).exam()
    if with_rollout:
        orc = max(orc, rollout_oracle(seed).exam())
    lit = run_obs_policy(seed, make_lit_class).exam()
    nv = naive_class(seed).exam()
    nz = lambda e: (e - base) / (orc - base)
    return dict(seed=seed, base=round(base, 3), oracle=round(orc, 3), lit=round(nz(lit), 2), naive=round(nz(nv), 2))


# ---------------- CLI ----------------
def load():
    with open(PATH) as fh:
        return json.load(fh)


def save(st):
    with open(PATH, "w") as fh:
        json.dump(st, fh)


def replay(st):
    c = Class(st["seed"])
    for a in st["actions"]:
        c.apply(tuple(a))
    return c


DAYS = ["Monday", "Wednesday", "Friday"]


def show(st, c, msg=None):
    for line in st["log"][-60:]:
        print(line)
    if len(st["log"]) > 60:
        print(f"(showing the last 60 of {len(st['log'])} log lines; `state full` shows all)")
    if msg:
        print(">> " + msg)
    if st["done"]:
        print(st["result"]); return
    if c.ct >= T:
        print("All turns used. Use `finish` to run the exam."); return
    s = c.ct // TURNS_PER
    print(f"Now: session {s+1} of 3 ({DAYS[s]}), turn {c.ct % TURNS_PER + 1} of {TURNS_PER} "
          f"(turn {c.ct+1} of {T} overall). The exam is next Monday.")


def main():
    cmd = sys.argv[1]
    if cmd == "bench":
        seeds = [int(x) for x in sys.argv[2:]]
        for sd in seeds:
            print(score_all(sd))
        return
    if cmd == "new":
        st = dict(seed=int(sys.argv[2]), actions=[], log=[], done=False, result=None)
        save(st)
        print("Children: " + ", ".join(f"{i+1}={n}" for i, n in enumerate(NAMES)))
        print("Concepts (each needs the one before it): " + "; ".join(f"{i+1}={x}" for i, x in enumerate(t2.CONCEPTS)))
        print("Known misconceptions: " + "; ".join(f"{b}: {d}" for b, d in t2.BUG_DESC.items()))
        show(st, Class(st["seed"])); return
    st = load()
    c = replay(st)
    if cmd == "state":
        if len(sys.argv) > 2 and sys.argv[2] == "full":
            for line in st["log"]:
                print(line)
            st2 = dict(st); st2["log"] = []
            show(st2, c); return
        show(st, c); return
    if st["done"]:
        show(st, c); return
    if cmd == "finish":
        e = c.exam()
        st.update(done=True, exam=e, result=f"FINISHED after {c.ct} turns. Class average exam score next Monday: {round(e*100)}%.")
        save(st); show(st, c); return
    if c.ct >= T and cmd != "ask":
        show(st, c, "No turns left. Use finish."); return
    day = DAYS[min(c.ct // TURNS_PER, 2)]
    try:
        if cmd in ("explain", "quiz", "ask"):
            k, s = int(sys.argv[2]) - 1, int(sys.argv[3]) - 1
        elif cmd == "correct":
            k, bug, s = int(sys.argv[2]) - 1, sys.argv[3], int(sys.argv[4]) - 1
            if bug not in t2.BUG_DESC:
                raise ValueError
        elif cmd in ("lecture", "classquiz"):
            k, s = int(sys.argv[2]) - 1, None
        else:
            raise ValueError
        assert 0 <= k < t2.K and (s is None or 0 <= s < N)
    except (ValueError, IndexError, AssertionError):
        print("usage: explain K S | quiz K S | correct K BUG S | ask K S | lecture K | classquiz K | state | finish  (K 1..6, S 1..5)")
        return
    if cmd == "ask":
        ans = c.kids[s].ask(k)
        st["log"].append(f"   (free) ask {NAMES[s]} about {k+1} -> \"{ans}\"")
        save(st); show(st, c, f"{NAMES[s]}: \"{ans}\""); return
    if cmd == "explain":
        a = ("explain", k, s, None); c.apply(a)
        st["log"].append(f"turn {c.ct:2} ({day}): explain {k+1} to {NAMES[s]}"); msg = "Done."
    elif cmd == "correct":
        a = ("correct", k, s, bug); c.apply(a)
        st["log"].append(f"turn {c.ct:2} ({day}): correct {bug} on {k+1} with {NAMES[s]}"); msg = "Done."
    elif cmd == "lecture":
        a = ("lecture", k, None, None); c.apply(a)
        st["log"].append(f"turn {c.ct:2} ({day}): lecture {k+1} to the class"); msg = "Done."
    elif cmd == "quiz":
        a = ("quiz", k, s, None); res = c.apply(a)
        ok, q, ans, right = res[0][1]
        r = "correct" if ok else f"wrong; right answer {right}"
        st["log"].append(f"turn {c.ct:2} ({day}): quiz {k+1} {NAMES[s]}: \"{ans}\" ({r})"); msg = f"{NAMES[s]}: \"{ans}\" ({r})"
    else:
        a = ("classquiz", k, None, None); res = c.apply(a)
        parts = [f"{NAMES[i]} \"{ans}\" ({'ok' if ok else 'wrong'})" for i, (ok, q, ans, right) in res]
        q, right = t2.QUIZ[k][0], t2.QUIZ[k][1]
        st["log"].append(f"turn {c.ct:2} ({day}): classquiz {k+1} [right: {right}]: " + "; ".join(parts)); msg = "Quiz: " + q
    st["actions"].append(list(a))
    save(st); show(st, c, msg)


if __name__ == "__main__":
    main()
