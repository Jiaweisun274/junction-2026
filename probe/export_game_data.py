"""Export the 30 evaluation children (seeds 3001-3005, salt eval1) for the browser teaching game (viz/simple.html).

For each child: the simulator's random draws and parameters (so the JavaScript port reproduces the Python simulator
exactly), the untaught and full-information exam scores, and the action sequence of every teacher that taught this
child (PPO, the planned 5-rule teacher, the literature rulebook, and the frontier agents from the replayed Harbor runs).
Output: viz/game_data.json
"""
import json, os
import numpy as np
import tutor2 as t, sweep_tutor2 as w, planned_ref as Q, replay_traces as RT

HERE = os.path.dirname(os.path.abspath(__file__))
TR = json.load(open(os.path.join(HERE, "../viz/traces.json")))
PLANNED = dict(confirm=2, final=0, review=0, maxexpl=2, diag=True, prereq=True)
TEACHERS = {"PPO": "PPO (trained)", "literature rulebook": "Textbook rulebook", "claude-opus-5-5": "Claude Opus 5.5",
            "gpt-6.1-sol": "GPT 6.1 Sol", "gemini-3.1-pro-preview": "Gemini 3.1 Pro", "gemini-pro-latest": "Gemini Pro",
            "gemini-3.5-flash": "Gemini 3.5 Flash"}


def planned_actions(cs):
    acts = []
    oq, oe, oc = t.Sim.quiz, t.Sim.explain, t.Sim.correct
    t.Sim.quiz = lambda self, k: (acts.append(["quiz", k]), oq(self, k))[1]
    t.Sim.explain = lambda self, k: (acts.append(["explain", k]), oe(self, k))[1]
    t.Sim.correct = lambda self, k, b: (acts.append(["correct", k, b]), oc(self, k, b))[1]
    try:
        s = Q.run(cs, **PLANNED)
    finally:
        t.Sim.quiz, t.Sim.explain, t.Sim.correct = oq, oe, oc
    return acts, s.exam()


def r8(a):
    return np.round(np.asarray(a, dtype=float), 8).tolist()


def main():
    out = {"concepts": t.CONCEPTS, "quiz": [list(q) for q in t.QUIZ], "bugs": {str(k): b for k, b in t.BUGS.items()},
           "bug_desc": t.BUG_DESC, "carry": {str(k): v for k, v in t.CARRY.items()},
           "const": dict(SESSIONS=t.SESSIONS, TURNS_PER=t.TURNS_PER, EXAM_DAY=t.EXAM_DAY, SLIP=t.SLIP, GUESS=t.GUESS,
                         LEAK=t.LEAK, RETRIEVE=t.RETRIEVE, RESTUDY=t.RESTUDY, MASSED=t.MASSED,
                         FIX_EXPLAIN=t.FIX_EXPLAIN, FIX_TARGETED=t.FIX_TARGETED, BUG_EXAM=t.BUG_EXAM),
           "yes": t.YES_T, "no": t.NO_T, "children": []}
    for seed in range(3001, 3006):
        run = TR["runs"][str(seed)]
        for i, cs in enumerate(RT.child_seeds(seed)):
            p = t.params(cs)
            c = {"id": f"{seed}-{i + 1}", "batch": seed, "idx": i, "seed": cs, "m0": p["m0"], "plearn": r8(p["plearn"]),
                 "suscept": p["suscept"], "h_init": r8(p["h_init"]), "h0": r8(p["h0"]), "u": r8(p["u"]),
                 "u_init": r8(p["u_init"]), "u_ask": r8(p["u_ask"]),
                 "base": round(t.Sim(cs).exam(), 6), "full": round(w.oracle_exam(cs), 6), "teachers": {}}
            acts, ex = planned_actions(cs)
            c["teachers"]["Planned (5 rules)"] = {"actions": acts, "exam": round(ex, 6)}
            for key, label in TEACHERS.items():
                pol = run["policies"].get(key)
                if not pol:
                    continue
                kid = pol["kids"][i]
                a = [[s["act"], s["k"] - 1] + ([s["bug"]] if s["act"] == "correct" else []) for s in kid["steps"]]
                c["teachers"][label] = {"actions": a, "exam": kid["exam"]}
            out["children"].append(c)
    path = os.path.join(HERE, "../viz/game_data.json")
    json.dump(out, open(path, "w"), separators=(",", ":"))
    print("wrote", path, os.path.getsize(path) // 1024, "KB")


if __name__ == "__main__":
    main()
