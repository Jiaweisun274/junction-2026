"""Probe 4: tutoring v2 — three sessions over a week, forgetting, testing effect, spacing, misconceptions.

Time: sessions on day 0, 2, 4 (10 turns each); exam on day 7.
Per concept k the child is in one of: unlearned | mastered (half-life h, last exposure) | buggy (a misconception).
- Recall probability of a mastered concept: p = 2 ** (-(now - last) / h)   (half-life regression, Settles & Meeder 2016)
- explain k: unlearned -> learns w.p. plearn[k] if prerequisite mastered (else LEAK); a susceptible child may learn it wrongly (bug).
             mastered -> restudy: h *= RESTUDY * spacing; buggy -> fixed only rarely.
- quiz k: mastered & recalled -> correct w.p. 0.9, retrieval practice: h *= RETRIEVE * spacing (testing effect).
          mastered & forgotten -> guess; feedback re-exposes (last = now).
          buggy -> the misconception's typical wrong answer w.p. 0.9.  unlearned -> guess.
- correct k BUG: targeted fix of a named misconception (w.p. 0.7 if that is the bug).
- ask k: free self-report, over-confident; buggy children are almost always sure.
Exam per concept: mastered 0.9 p + 0.25 (1 - p); buggy 0.10; unlearned 0.25.

CLI ($GAME_DIR/tutor2_state.json): new SEED | state | explain K | quiz K | ask K | correct K BUG | finish
Internal: bench [train|test]
"""
import json, os, sys
import numpy as np

CONCEPTS = ["equal parts", "comparing fractions (numerator and denominator)", "equivalent fractions",
            "common denominators", "adding fractions", "word problems with fractions"]
K = len(CONCEPTS)
SESSIONS = [0.0, 2.0, 4.0]
TURNS_PER = 10
T = TURNS_PER * len(SESSIONS)
EXAM_DAY = 7.0
SLIP, GUESS = 0.10, 0.25
LEAK = 0.05
H0 = 1.0  # legacy; per-child half-lives now come from params(): see CALIBRATION below
# CALIBRATION (Sat 14 Nov): FSRS-5 default first-review stabilities (days to 90% recall) are 0.40 (Again),
# 1.18 (Hard), 3.17 (Good). As exponential half-lives (x ln2/ln(1/0.9) = x6.58): ~2.6, ~7.8, ~21 days.
# Newly learned concept: half-life U(2.6, 7.8) days (Again..Hard). Known before the course: U(8, 20) (Hard..Good).
# Correct recall multiplies the half-life by 2.5 = the SM-2 default ease factor.
RETRIEVE, RESTUDY = 2.5, 1.5
MASSED = 0.6  # diminishing returns of re-explaining within a session (massed practice)
FIX_EXPLAIN, FIX_TARGETED = 0.10, 0.70
BUG_EXAM = 0.10
BUGS = {1: "bigger_denominator", 2: "add_to_top_and_bottom", 4: "add_across"}
BUG_DESC = {
    "bigger_denominator": "thinks a bigger denominator means a bigger fraction (says 1/8 > 1/4)",
    "add_to_top_and_bottom": "thinks adding the same number to top and bottom keeps a fraction equal (says 2/3 = 3/4)",
    "add_across": "adds numerators and denominators separately (says 1/2 + 1/3 = 2/5)",
}
QUIZ = [
    ("A cake is cut into 4 pieces and you take one. Is that 1/4 if the pieces are different sizes?", "No", None, ["Yes", "I don't know"]),
    ("Which is bigger, 1/4 or 1/8?", "1/4", "1/8", ["they are the same", "I don't know"]),
    ("Which fraction equals 2/3: 3/4 or 4/6?", "4/6", "3/4", ["neither", "I'm not sure"]),
    ("Write 1/2 and 1/3 with the same denominator.", "3/6 and 2/6", None, ["1/6 and 1/6", "I don't know"]),
    ("What is 1/2 + 1/3?", "5/6", "2/5", ["1/6", "I don't know"]),
    ("Mia eats 1/4 of a pizza and Leo eats 1/2. How much of the pizza is eaten?", "3/4", "2/6", ["1/8", "I don't know"]),
]
# generic wrong answers never imply a listed misconception. A misconception also shows in the next concept that uses it:
# add_across on concept 5 makes the child answer the word problem (6) the add-across way.
CARRY = {5: 4}   # concept 6 shows the misconception of concept 5


YES_T = ["Yeah, that makes sense!", "Oh okay, I get it now.", "Mm-hm, yes, I understand.", "Yep, got it.", "I'm sure, it's easy!"]
NO_T = ["Hmm... not really.", "I'm a bit confused, sorry.", "Can you say it again?", "I don't get it."]
DIR = os.environ.get("GAME_DIR", ".")
PATH = os.path.join(DIR, "tutor2_state.json")


def params(seed):
    r = np.random.default_rng(int(seed) + 77)
    m0 = int(r.choice([0, 1, 2, 3], p=[0.2, 0.35, 0.3, 0.15]))
    plearn = r.uniform(0.3, 0.6, K)
    suscept = {b: bool(r.random() < 0.4) for b in BUGS.values()}
    h_init = r.uniform(8.0, 20.0, K)
    u = r.random((T + 2, 8))
    u_init = r.random(K)
    h0 = r.uniform(2.6, 7.8, K)
    return dict(m0=m0, plearn=plearn, suscept=suscept, h_init=h_init, u=u, u_init=u_init, h0=h0)


class Sim:
    def __init__(self, seed):
        self.p = P = params(seed)
        self.state = ["unlearned"] * K
        self.h = [0.0] * K
        self.last = [0.0] * K
        for k in range(P["m0"]):
            b = BUGS.get(k)
            if b and P["suscept"][b] and P["u_init"][k] < 0.6:
                self.state[k] = "buggy"
            else:
                self.state[k] = "mastered"
                self.h[k] = P["h_init"][k]
                self.last[k] = -3.0
        self.turn = 0
        self.last_explained = None

    # --- time ---
    def now(self, turn=None):
        t = self.turn if turn is None else turn
        s = min(t // TURNS_PER, len(SESSIONS) - 1)
        return SESSIONS[s] + 0.01 * (t % TURNS_PER)

    def recall(self, k, at=None):
        if self.state[k] != "mastered":
            return 0.0
        at = self.now() if at is None else at
        return 2 ** (-(at - self.last[k]) / self.h[k])

    def mastered_ok(self, k):
        return k == 0 or self.state[k - 1] == "mastered"

    def _learn(self, k, u_bug):
        b = BUGS.get(k)
        if b and self.p["suscept"][b] and u_bug < 0.6:
            self.state[k] = "buggy"
        else:
            self.state[k] = "mastered"
            self.h[k], self.last[k] = self.p["h0"][k], self.now()

    # --- actions ---
    def explain(self, k):
        u = self.p["u"][self.turn]
        st = self.state[k]
        # Massed repetition helps less: each earlier explanation of the same concept in the same session
        # multiplies the chance of it clicking by MASSED (a new session starts fresh: spacing).
        if not hasattr(self, "_sess_expl"):
            self._sess_expl = {}
        key = (self.turn // TURNS_PER, k)
        damp = MASSED ** self._sess_expl.get(key, 0)
        self._sess_expl[key] = self._sess_expl.get(key, 0) + 1
        if st == "unlearned":
            if u[0] < (self.p["plearn"][k] if self.mastered_ok(k) else LEAK) * damp:
                self._learn(k, u[1])
        elif st == "mastered":
            sp = 1 + (1 - self.recall(k))
            self.h[k] *= RESTUDY * sp
            self.last[k] = self.now()
        elif st == "buggy" and u[0] < FIX_EXPLAIN:
            self.state[k] = "mastered"; self.h[k], self.last[k] = self.p["h0"][k], self.now()
        self.turn += 1
        self.last_explained = k

    def quiz(self, k):
        u = self.p["u"][self.turn]
        q, right, bug_ans, wrong = QUIZ[k]
        st = self.state[k]
        if st == "mastered":
            rp = self.recall(k)
            if u[0] < rp:
                ok = u[1] < 1 - SLIP
                if ok:
                    self.h[k] *= RETRIEVE * (1 + (1 - rp))
                self.last[k] = self.now()
            else:
                ok = u[1] < GUESS
                self.last[k] = self.now()     # feedback re-exposes
        elif st == "buggy":
            ok = u[1] < 0.10
        else:
            ok = u[1] < GUESS
        carried = k in CARRY and self.state[CARRY[k]] == "buggy" and st != "mastered"
        if carried:
            ok = u[1] < 0.10
        if ok:
            ans = right
        elif st == "buggy" or carried:
            ans = bug_ans
        else:
            ans = wrong[int(u[2] * len(wrong))]
        self.turn += 1
        self.last_explained = None
        return ok, q, ans, right

    def correct(self, k, bug):
        u = self.p["u"][self.turn]
        if self.state[k] == "buggy" and BUGS.get(k) == bug and u[0] < FIX_TARGETED:
            self.state[k] = "mastered"; self.h[k], self.last[k] = self.p["h0"][k], self.now()
        self.turn += 1
        self.last_explained = None

    def ask(self, k):
        u = self.p["u"][self.turn]
        st = self.state[k]
        if st == "buggy":
            py = 0.97
        elif st == "mastered" and u[3] < self.recall(k):
            py = 0.95
        else:
            py = 0.90 if self.last_explained == k else 0.75
        tmpl = YES_T if u[4] < py else NO_T
        return tmpl[int(u[5] * len(tmpl))]

    def exam_k(self, k):
        st = self.state[k]
        if st == "mastered":
            p = self.recall(k, EXAM_DAY)
            return 0.9 * p + 0.25 * (1 - p)
        return BUG_EXAM if st == "buggy" else 0.25

    def exam(self):
        return float(np.mean([self.exam_k(k) for k in range(K)]))


# ---------------- oracle: full-information greedy rollout ----------------
def exam_if_idle(s):
    return s.exam()


def oracle_step(s):
    """Knows the true state. Fix bugs on the path, teach the frontier, else review the weakest memory;
    in the last session, prefer reviews that raise expected exam score most."""
    for k in range(K):
        if s.state[k] == "buggy" and (k == 0 or s.state[k - 1] == "mastered"):
            return ("correct", k, BUGS[k])
    frontier = next((k for k in range(K) if s.state[k] != "mastered"), None)
    session = s.turn // TURNS_PER
    turns_left = T - s.turn
    mastered = [k for k in range(K) if s.state[k] == "mastered"]
    def gain(k):
        p = s.recall(k, EXAM_DAY)
        return 0.65 * (1 - p)
    if frontier is not None and s.mastered_ok(frontier) and not (session == 2 and turns_left <= 4 and mastered and max(gain(k) for k in mastered) > 0.15):
        return ("explain", frontier, None)
    if mastered:
        k = max(mastered, key=gain)
        return ("quiz" if s.recall(k) > 0.3 else "explain", k, None)
    return ("explain", frontier if frontier is not None else 0, None)


def run_policy(seed, step_fn, obs=False):
    s = Sim(seed)
    mem = {}
    last_obs = None
    while s.turn < T:
        a, k, b = step_fn(s, mem, last_obs) if obs else step_fn(s)
        if a == "explain":
            s.explain(k); last_obs = ("explain", k, None)
        elif a == "quiz":
            ok, q, ans, right = s.quiz(k); last_obs = ("quiz", k, (ok, ans))
        elif a == "correct":
            s.correct(k, b); last_obs = ("correct", k, None)
    return s


# ---------------- reference: observation-only ----------------
BUG_ANS = {k: QUIZ[k][2] for k in BUGS}


REF = dict(start=0.2, lo=0.5, hi=0.9, review=0.4, endrev=2, h0=1.0, mult=3.0, lr=0.3)  # tuned on seeds 1001-1060


def reference_step(s, mem, last_obs):
    """Belief per concept: p = P(mastered and remembered). Bug detected from the typical wrong answer.
    Assumed memory model: half-life 1 day after learning, x2 per correct recall (not the true values)."""
    if not mem:
        mem.update(b=[REF["start"]] * K, bug=[None] * K, h=[REF["h0"]] * K, last=[None] * K, k=0, phase="learn")
    b = mem["b"]
    now = s.now()
    if last_obs:
        a, k, o = last_obs
        if a == "quiz":
            ok, ans = o
            if not ok and k in BUG_ANS and ans == BUG_ANS[k]:
                mem["bug"][k] = BUGS[k]; b[k] = 0.0
            elif not ok and k in CARRY and ans == QUIZ[k][2]:
                j = CARRY[k]                      # misconception lives in the prerequisite
                mem["bug"][j] = BUGS[j]; b[j] = 0.0; mem["k"] = min(mem["k"], j)
            else:
                lm, ln = (1 - SLIP, GUESS) if ok else (SLIP, 1 - GUESS)
                b[k] = b[k] * lm / (b[k] * lm + (1 - b[k]) * ln)
                if ok and b[k] > 0.8:
                    mem["h"][k] = mem["h"][k] * REF["mult"] if mem["last"][k] is not None else REF["h0"]
                    mem["last"][k] = now
        elif a == "explain" and mem["bug"][k] is None:
            b[k] = b[k] + (1 - b[k]) * REF["lr"]
            if mem["last"][k] is None:
                mem["last"][k] = now
        elif a == "correct":
            mem["bug"][k] = None; b[k] = 0.5

    def est_recall(k, at):
        if mem["last"][k] is None:
            return 1.0
        return 2 ** (-(at - mem["last"][k]) / mem["h"][k])

    session = s.turn // TURNS_PER
    in_sess = s.turn % TURNS_PER
    turns_left = T - s.turn
    # session start (2 and 3): one review quiz on known concepts that are likely fading
    if in_sess == 0:
        mem["review"] = [k for k in range(mem["k"]) if est_recall(k, now) < REF["review"]]
    if mem.get("review"):
        k = mem["review"].pop(0)
        b[k] = min(b[k], 0.6)          # unsure again after days away
        return ("quiz", k, None)
    # end of course: spend the last 3 turns reviewing the weakest-by-exam concepts
    if session == 2 and turns_left <= REF["endrev"] and mem["k"] > 0:
        k = min(range(mem["k"]), key=lambda j: est_recall(j, EXAM_DAY))
        mem["last"][k] = now
        return ("quiz", k, None)
    k = min(mem["k"], K - 1)
    if mem["bug"][k]:
        return ("correct", k, mem["bug"][k])
    if b[k] > REF["hi"] and mem["k"] < K:
        mem["k"] += 1
        k = min(mem["k"], K - 1)
        if mem["k"] >= K:
            k = min(range(K), key=lambda j: est_recall(j, EXAM_DAY))
            return ("quiz", k, None)
    if b[k] < REF["lo"]:
        return ("explain", k, None)
    return ("quiz", k, None)


# ---------------- naive baseline: linear mastery learning, no review ----------------
def naive_step(s, mem, last_obs):
    if not mem:
        mem.update(k=0, streak=0, need_explain=True)
    if last_obs and last_obs[0] == "quiz":
        mem["streak"] = mem["streak"] + 1 if last_obs[2][0] else 0
        mem["need_explain"] = not last_obs[2][0]
    if mem["streak"] >= 2 and mem["k"] < K - 1:
        mem.update(k=mem["k"] + 1, streak=0, need_explain=True)
    k = mem["k"]
    if mem["need_explain"]:
        mem["need_explain"] = False
        return ("explain", k, None)
    return ("quiz", k, None)


def score(seed, kind):
    base = Sim(seed).exam()
    orc = run_policy(seed, oracle_step).exam()
    if kind == "oracle":
        e = orc
    elif kind == "reference":
        e = run_policy(seed, reference_step, obs=True).exam()
    elif kind == "naive":
        e = run_policy(seed, naive_step, obs=True).exam()
    return (e - base) / (orc - base) if orc > base else 1.0


def bench(which):
    seeds = range(1001, 1101) if which == "train" else range(2001, 2101)
    for kind in ["reference", "naive"]:
        v = np.array([score(str(s), kind) for s in seeds])
        print(kind, "mean", round(v.mean(), 3), "share>=1", round((v >= 0.999).mean(), 2), "min", round(v.min(), 2))
    for s in list(seeds)[:6]:
        sm = Sim(str(s))
        print(s, "m0", sm.p["m0"], "suscept", [b for b, v in sm.p["suscept"].items() if v],
              "base", round(sm.exam(), 3), "oracle", round(run_policy(str(s), oracle_step).exam(), 3),
              "ref", round(score(str(s), "reference"), 2), "naive", round(score(str(s), "naive"), 2))


# ---------------- CLI ----------------
def load():
    with open(PATH) as fh:
        return json.load(fh)


def save(st):
    with open(PATH, "w") as fh:
        json.dump(st, fh)


def replay(st):
    s = Sim(st["seed"])
    for a in st["actions"]:
        if a[0] == "explain":
            s.explain(a[1])
        elif a[0] == "quiz":
            s.quiz(a[1])
        elif a[0] == "correct":
            s.correct(a[1], a[2])
    return s


DAYS = ["Monday", "Wednesday", "Friday"]


def show(st, s, msg=None):
    for line in st["log"]:
        print(line)
    if msg:
        print(">> " + msg)
    if st["done"]:
        print(st["result"])
    elif s.turn >= T:
        print("All 30 turns used. Use `finish` to run the exam.")
    else:
        sess = s.turn // TURNS_PER
        print(f"Now: session {sess+1} of 3 ({DAYS[sess]}), turn {s.turn % TURNS_PER + 1} of {TURNS_PER} "
              f"(turn {s.turn+1} of {T} overall). The exam is next Monday.")


def main():
    cmd = sys.argv[1]
    if cmd == "bench":
        bench(sys.argv[2] if len(sys.argv) > 2 else "train"); return
    if cmd == "new":
        st = dict(seed=sys.argv[2], actions=[], log=[], done=False, result=None)
        save(st)
        print("Concepts (each needs the one before it): " + "; ".join(f"{i+1}={c}" for i, c in enumerate(CONCEPTS)))
        print("Known misconceptions: " + "; ".join(f"{b}: {d}" for b, d in BUG_DESC.items()))
        show(st, Sim(st["seed"])); return
    st = load()
    s = replay(st)
    if st["done"]:
        show(st, s); return
    msg = None
    day = lambda: DAYS[min(s.turn // TURNS_PER, 2)]
    if cmd in ("explain", "quiz", "ask", "correct"):
        k = int(sys.argv[2]) - 1
        if not 0 <= k < K:
            print("K must be 1..6"); return
        if cmd != "ask" and s.turn >= T:
            msg = "No turns left. Use finish."
        elif cmd == "explain":
            d = day(); s.explain(k); st["actions"].append(["explain", k])
            st["log"].append(f"turn {s.turn:2} ({d}): explain {k+1} ({CONCEPTS[k]})")
            msg = "You explain it. The child listens."
        elif cmd == "quiz":
            d = day(); ok, q, ans, right = s.quiz(k); st["actions"].append(["quiz", k])
            res = "correct" if ok else f"wrong, the right answer is {right}"
            st["log"].append(f"turn {s.turn:2} ({d}): quiz {k+1}: \"{q}\" child: \"{ans}\" ({res})")
            msg = f"Child answers \"{ans}\" ({res}). You show the right answer."
        elif cmd == "correct":
            if len(sys.argv) < 4 or sys.argv[3] not in BUG_DESC:
                print("usage: correct K BUG, BUG one of " + ", ".join(BUG_DESC)); return
            d = day(); s.correct(k, sys.argv[3]); st["actions"].append(["correct", k, sys.argv[3]])
            st["log"].append(f"turn {s.turn:2} ({d}): correct {k+1} {sys.argv[3]}")
            msg = "You work through that misconception with the child."
        else:
            ans = s.ask(k)
            st["log"].append(f"   (free) ask {k+1} -> \"{ans}\"")
            msg = f"\"{ans}\""
    elif cmd == "finish":
        e = s.exam()
        st["done"] = True
        st["exam"] = e
        st["result"] = f"FINISHED after {s.turn} turns. Exam next Monday: score {round(e*100)}%."
    save(st)
    show(st, s, msg)



# ---------------- stronger oracle: full-information rollout planner ----------------
import copy


def _clone_fresh(s, rng):
    c = copy.copy(s)
    c.state, c.h, c.last = list(s.state), list(s.h), list(s.last); c._sess_expl = dict(getattr(s, "_sess_expl", {}))
    c.p = dict(s.p)
    u = s.p["u"].copy()
    u[s.turn:] = rng.random(u[s.turn:].shape)      # future draws unknown to the planner
    c.p["u"] = u
    return c


def _apply(s, a):
    if a[0] == "explain":
        s.explain(a[1])
    elif a[0] == "quiz":
        s.quiz(a[1])
    else:
        s.correct(a[1], a[2])


def _candidates(s):
    acts = [("explain", k, None) for k in range(K)] + [("quiz", k, None) for k in range(K) if s.state[k] == "mastered"]
    acts += [("correct", k, BUGS[k]) for k in range(K) if s.state[k] == "buggy"]
    return acts


def rollout_oracle_step(s, n=16, rng=np.random.default_rng(0)):
    """Policy-improvement step over the greedy oracle, with common random numbers across candidates."""
    futures = [rng.random(s.p["u"][s.turn:].shape) for _ in range(n)]
    best, best_v = None, -1
    for a in _candidates(s):
        v = 0.0
        for fu in futures:
            c = copy.copy(s)
            c.state, c.h, c.last = list(s.state), list(s.h), list(s.last); c._sess_expl = dict(getattr(s, "_sess_expl", {}))
            c.p = dict(s.p); u = s.p["u"].copy(); u[s.turn:] = fu; c.p["u"] = u
            _apply(c, a)
            while c.turn < T:
                _apply(c, oracle_step(c))
            v += c.exam()
        if v > best_v + 1e-12:
            best, best_v = a, v
    return best


def run_oracle2(seed):
    s = Sim(seed)
    rng = np.random.default_rng(int(seed))
    while s.turn < T:
        _apply(s, rollout_oracle_step(s, rng=rng))
    return s


def bench2():
    import time
    t0 = time.time()
    rows = []
    for sd in [str(x) for x in range(1001, 1021)]:
        base = Sim(sd).exam()
        o1 = run_policy(sd, oracle_step).exam()
        o2 = run_oracle2(sd).exam()
        rf = run_policy(sd, reference_step, obs=True).exam()
        nv = run_policy(sd, naive_step, obs=True).exam()
        rows.append((base, o1, o2, rf, nv))
    a = np.array(rows)
    print("mean exam: base %.3f greedy-oracle %.3f rollout-oracle %.3f reference %.3f naive %.3f" % tuple(a.mean(0)))
    print("rollout beats greedy on", (a[:, 2] > a[:, 1] + 1e-9).mean(), "of seeds; naive beats rollout on", (a[:, 4] > a[:, 2] + 1e-9).mean())
    print("secs", round(time.time() - t0, 1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench2":
        bench2()
    else:
        main()
