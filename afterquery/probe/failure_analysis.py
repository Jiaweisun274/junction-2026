"""Failure analysis from the replayed traces (viz/traces.json).

1. Behaviour profile per teacher: action mix, quizzes in the last 5 turns, concepts reached, final state.
2. Decision-level mistakes per child (same definitions as the visualization).
3. Counterfactual hand-over: replay the agent's own first t turns for a child, then let the PPO reference finish the
   remaining turns (it rebuilds its observation record from the agent's history). The exam gain over the agent's own
   result shows how much of the loss is still recoverable at Monday's end (t=10) and Wednesday's end (t=20).
Writes FAILURE_ANALYSIS_DATA.md next to this file's parent folder.
"""
import json, os, statistics as st
import tutor2 as t, ppo_policy as P

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "../viz/traces.json")))
ORDER = ["PPO", "literature rulebook", "claude-opus-5-5", "gpt-6.1-sol", "gemini-pro-latest", "gemini-3.1-pro-preview",
         "gemini-3.5-flash", "gemini-pro-latest (all parameters disclosed)", "gemini-3.1-pro-preview (all parameters disclosed)"]
KEYS = ["guess", "prereq", "massed", "overquiz", "missedbug"]


def analyse(kid, start):
    marks, sess, saw, corr = [None] * len(kid["steps"]), {}, set(), set()
    for i, s in enumerate(kid["steps"]):
        k = s["k"] - 1; before = kid["steps"][i - 1] if i else start
        if s["act"] == "explain":
            key = (s["day"], k); sess[key] = sess.get(key, 0) + 1
            if k > 0 and before["state"][k - 1] != "M": marks[i] = "prereq"
            elif sess[key] >= 3: marks[i] = "massed"
        if s["act"] == "quiz":
            if s.get("sig"): saw.add(5 if s["k"] == 6 else s["k"])
            if s["ok"] and before["state"][k] != "M":
                nxt = kid["steps"][i + 1] if i + 1 < len(kid["steps"]) else None
                if nxt and nxt["k"] == s["k"] + 1: marks[i] = "guess"
            if s["ok"] and before["state"][k] == "M" and before["recall"][k] > 0.9:
                if any(p["act"] == "quiz" and p["k"] == s["k"] and p["ok"] and p["day"] == s["day"] for p in kid["steps"][:i]):
                    marks[i] = "overquiz"
        if s["act"] == "correct": corr.add(s["k"])
    return marks, len(saw - corr)


def handover(child_seed, steps, t_cut):
    """Agent plays its first t_cut steps; PPO finishes. Returns the exam score."""
    s, tr, pol = t.Sim(child_seed), P.Tracker(), P.Policy()
    for st_ in steps[:t_cut]:
        k, turn = st_["k"] - 1, s.turn
        if st_["act"] == "explain": s.explain(k); tr.update(turn, ("explain", k, None))
        elif st_["act"] == "correct": s.correct(k, st_["bug"]); tr.update(turn, ("correct", k, st_["bug"]))
        else:
            ok, q, ans, right = s.quiz(k); tr.update(turn, ("quiz", k, None), (ok, ans))
    while s.turn < t.T:
        turn = s.turn; a = P.decode(pol.act(tr.obs(turn)))
        if a[0] == "quiz":
            ok, q, ans, right = s.quiz(a[1]); tr.update(turn, a, (ok, ans))
        elif a[0] == "explain": s.explain(a[1]); tr.update(turn, a)
        else: s.correct(a[1], a[2]); tr.update(turn, a)
    return s.exam()


def main():
    rows, hand = [], []
    for n in ORDER:
        kids = []; rew = []
        for seed, run in D["runs"].items():
            pol = run["policies"].get(n)
            if not pol: continue
            rew.append(pol["reward"])
            for ci, kid in enumerate(pol["kids"]):
                kids.append((seed, ci, kid, run["start"][ci], run["children"][ci]))
        if not kids: continue
        tot = {k: 0 for k in KEYS}; acts = {"explain": 0, "quiz": 0, "correct": 0}; steps = 0; late_q = []; reach = []; final_m = []; bugs_left = []
        for seed, ci, kid, start, cs in kids:
            marks, missed = analyse(kid, start)
            for m in marks:
                if m: tot[m] += 1
            tot["missedbug"] += missed
            for s in kid["steps"]: acts[s["act"]] += 1; steps += 1
            late_q.append(sum(1 for s in kid["steps"][-5:] if s["act"] == "quiz") / 5)
            reach.append(max([s["k"] for s in kid["steps"]] or [0]))
            last = kid["steps"][-1] if kid["steps"] else start
            final_m.append(sum(1 for x in last["state"] if x == "M")); bugs_left.append(sum(1 for x in last["state"] if x == "B"))
        nk = len(kids)
        rows.append(dict(n=n, runs=len(rew), reward=st.mean(rew), kids=nk, per={k: tot[k] / nk for k in KEYS},
                         explain=acts["explain"] / steps, quiz=acts["quiz"] / steps, correct=acts["correct"] / steps,
                         late_q=st.mean(late_q), reach=st.mean(reach), final_m=st.mean(final_m), bugs_left=st.mean(bugs_left)))
        if n not in ("PPO", "literature rulebook") and "disclosed" not in n:
            g0, g10, g20 = [], [], []
            for seed, ci, kid, start, cs in kids:
                if len(kid["steps"]) < 30: continue
                g0.append(handover(cs, kid["steps"], 0) - kid["exam"]); g10.append(handover(cs, kid["steps"], 10) - kid["exam"]); g20.append(handover(cs, kid["steps"], 20) - kid["exam"])
            hand.append((n, st.mean(g0), st.mean(g10), st.mean(g20), len(g10)))
    L = ["# Failure analysis data (generated by probe/failure_analysis.py)", "",
         "## Behaviour profile (per child, 30 turns)", "",
         "| teacher | runs | reward | explain | quiz | correct | quiz share, last 5 turns | highest concept reached | concepts mastered at end | misconceptions left |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['n']} | {r['runs']} | {r['reward']:.2f} | {r['explain']:.0%} | {r['quiz']:.0%} | {r['correct']:.1%} | {r['late_q']:.0%} | {r['reach']:.1f} | {r['final_m']:.1f} | {r['bugs_left']:.2f} |")
    L += ["", "## Decision-level mistakes (count per child)", "",
          "guess = moved on after a correct quiz on a concept the child had not learned; prereq = explained a concept whose prerequisite was not mastered; massed = third or later explanation of a concept in one session; overquiz = re-quizzed a concept that was mastered, well remembered and already answered correctly that session; missedbug = a misconception answer was seen and never corrected.", "",
          "| teacher | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
    for r in rows:
        L.append(f"| {r['n']} | " + " | ".join(f"{r['per'][k]:.2f}" for k in KEYS) + " |")
    L += ["", "## Counterfactual hand-over to the PPO reference (mean exam gain per child)", "",
          "| agent | children | PPO teaches all week (t=0) | PPO takes over after Monday (t=10) | PPO takes over after Wednesday (t=20) |", "|---|---|---|---|---|"]
    for n, z, a, b, c in hand:
        L.append(f"| {n} | {c} | {z:+.3f} | {a:+.3f} | {b:+.3f} |")
    out = os.path.join(HERE, "../FAILURE_ANALYSIS_DATA.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
