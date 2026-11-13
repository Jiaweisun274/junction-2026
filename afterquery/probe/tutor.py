"""Probe 3: tutoring a simulated child in fractions. Hidden mastery (BKT-style), prerequisite chain,
over-confident self-reports. CLI ($GAME_DIR/tutor_state.json):
  new SEED | state | explain K | quiz K | ask K | finish        (K = 1..6)
Internal: bench SEED...
"""
import json, os, sys
import numpy as np

CONCEPTS = ["equal parts", "numerator and denominator", "equivalent fractions",
            "common denominators", "adding fractions", "word problems with fractions"]
K = len(CONCEPTS)
T = 30
SLIP, GUESS = 0.10, 0.25
LEAK = 0.05                    # chance to learn a concept whose prerequisite is not mastered
YES_MASTERED = 0.95
YES_NOT, YES_NOT_FRESH = 0.75, 0.90   # 'fresh' = asked right after explaining that concept
DIR = os.environ.get("GAME_DIR", ".")
PATH = os.path.join(DIR, "tutor_state.json")


def params(seed):
    r = np.random.default_rng(int(seed) + 7)
    m0 = int(r.choice([0, 1, 2, 3], p=[0.2, 0.35, 0.3, 0.15]))   # student already knows first m0 concepts
    plearn = r.uniform(0.3, 0.6, K)
    u = r.random((T + 1, 4))                                       # per-turn uniforms (common random numbers)
    return dict(m0=m0, plearn=plearn, u=u)


def exam(mastered):
    return float(np.mean([0.9 if x else 0.25 for x in mastered]))


YES_T = ["Yeah, that makes sense!", "Oh okay, I get it now.", "Mm-hm, yes, I understand.",
         "I think so, yeah!", "Yep, got it."]
NO_T = ["Hmm... not really.", "I'm a bit confused, sorry.", "Can you say it again?", "I don't get it."]
RIGHT_T = ["(answers correctly)", "(gets it right)"]
WRONG_T = ["(answers wrong)", "(gives a wrong answer)"]


class Sim:
    def __init__(self, seed):
        self.p = params(seed)
        self.mastered = [i < self.p["m0"] for i in range(K)]
        self.turn = 0                       # turns used
        self.last_explained = None
        self.base = exam(self.mastered)

    def prereq_ok(self, k):
        return k == 0 or self.mastered[k - 1]

    def explain(self, k):
        u = self.p["u"][self.turn][0]
        p = self.p["plearn"][k] if self.prereq_ok(k) else LEAK
        if not self.mastered[k] and u < p:
            self.mastered[k] = True
        self.turn += 1
        self.last_explained = k
        return "You explain " + CONCEPTS[k] + ". The child listens."

    def quiz(self, k):
        u = self.p["u"][self.turn][1]
        ok = u < (1 - SLIP if self.mastered[k] else GUESS)
        self.turn += 1
        self.last_explained = None
        tmpl = RIGHT_T if ok else WRONG_T
        return ok, tmpl[int(self.p["u"][self.turn - 1][3] * len(tmpl))]

    def ask(self, k):
        # free; same turn -> same draw, so repeated asking gives the same answer
        u = self.p["u"][self.turn][2]
        if self.mastered[k]:
            py = YES_MASTERED
        else:
            py = YES_NOT_FRESH if self.last_explained == k else YES_NOT
        yes = u < py
        tmpl = YES_T if yes else NO_T
        return tmpl[int(self.p["u"][self.turn][3] * len(tmpl))]

    def score(self):
        e, base = exam(self.mastered), self.base
        return e


def norm(score, base, orc):
    return (score - base) / (orc - base) if orc > base else 1.0


# ---------- policies for bench ----------
def run(seed, policy):
    s = Sim(seed)
    policy(s)
    return s


def oracle(s):
    while s.turn < T:
        todo = [k for k in range(K) if not s.mastered[k]]
        if not todo:
            break
        k = next((k for k in todo if s.prereq_ok(k)), todo[0])
        s.explain(k)


def reference(s):
    """Uses only quizzes (slip/guess are stated in the rules) and an assumed learn rate of 0.3."""
    k, b = 0, 0.5
    fails = 0
    while s.turn < T and k < K:
        if b > 0.9:
            k, b, fails = k + 1, 0.5, 0
            continue
        if b < 0.25:
            s.explain(k)
            b = b + (1 - b) * 0.3
            fails += 1
            if fails >= 4 and k > 0:          # suspect a prerequisite gap: re-check the previous concept
                k, b, fails = k - 1, 0.5, 0
            continue
        ok, _ = s.quiz(k)
        lm, ln = (1 - SLIP, GUESS) if ok else (SLIP, 1 - GUESS)
        b = b * lm / (b * lm + (1 - b) * ln)


def naive(s):
    """Explain each concept once, ask, move on if 'yes'; else explain again."""
    for k in range(K):
        if s.turn >= T:
            break
        s.explain(k)
        if s.ask(k) not in YES_T and s.turn < T:
            s.explain(k)


def explain_twice(s):
    for k in range(K):
        for _ in range(2):
            if s.turn < T:
                s.explain(k)


def bench(seeds):
    rows = []
    for sd in seeds:
        o = run(sd, oracle)
        base, orc = o.base, exam(o.mastered)
        r = {"seed": sd, "m0": params(sd)["m0"], "base_exam": round(base, 3), "oracle_exam": round(orc, 3)}
        for name, pol in [("reference", reference), ("naive_ask", naive), ("explain_twice", explain_twice)]:
            s = run(sd, pol)
            r[name] = round(norm(exam(s.mastered), base, orc), 2)
        rows.append(r)
        print(r)
    print("mean", {k: round(float(np.mean([r[k] for r in rows])), 2) for k in ["reference", "naive_ask", "explain_twice"]})


# ---------- CLI ----------
def load():
    with open(PATH) as fh:
        return json.load(fh)


def save(st):
    with open(PATH, "w") as fh:
        json.dump(st, fh)


def replay(st):
    s = Sim(st["seed"])
    for a, k in st["actions"]:
        if a == "explain":
            s.explain(k)
        elif a == "quiz":
            s.quiz(k)
    return s


def show(st, msg=None):
    for line in st["log"]:
        print(line)
    if msg:
        print(">> " + msg)
    if st["done"]:
        print(st["result"])
    else:
        print(f"Turns used: {st['turns']} of {T}.")


def main():
    cmd = sys.argv[1]
    if cmd == "bench":
        bench(sys.argv[2:]); return
    if cmd == "new":
        st = dict(seed=sys.argv[2], actions=[], log=[], turns=0, done=False, result=None)
        save(st)
        print("Concepts (each needs the one before it): " +
              "; ".join(f"{i+1}={c}" for i, c in enumerate(CONCEPTS)))
        show(st); return
    st = load()
    if st["done"]:
        show(st); return
    s = replay(st)
    msg = None
    if cmd in ("explain", "quiz", "ask"):
        k = int(sys.argv[2]) - 1
        if not 0 <= k < K:
            print("K must be 1..6"); return
        if cmd != "ask" and s.turn >= T:
            msg = "No turns left. Use finish."
        elif cmd == "explain":
            msg = s.explain(k)
            st["actions"].append(["explain", k]); st["turns"] = s.turn
            st["log"].append(f"turn {s.turn:2}: explain {k+1} ({CONCEPTS[k]})")
        elif cmd == "quiz":
            ok, txt = s.quiz(k)
            st["actions"].append(["quiz", k]); st["turns"] = s.turn
            st["log"].append(f"turn {s.turn:2}: quiz {k+1} ({CONCEPTS[k]}) -> child {txt}")
            msg = "Child " + txt
        else:
            ans = s.ask(k)
            st["log"].append(f"   (free) ask {k+1} ({CONCEPTS[k]}) -> \"{ans}\"")
            msg = f'"{ans}"'
    elif cmd == "finish":
        e = exam(s.mastered)
        st["done"] = True
        st["result"] = f"FINISHED after {s.turn} turns. Exam score: {round(e*100)}%."
        st["exam"] = e
    save(st)
    show(st, msg)


if __name__ == "__main__":
    main()
