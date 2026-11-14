"""Replay Harbor/OpenCode trajectories through the simulator to recover the hidden state at every step.

For each run (model, seed) we read the agent's MCP tool calls from opencode.txt, rebuild the six salted children,
apply the same actions in order, and record after every turn-taking action: the child's true per-concept state
(unlearned / mastered / misconception), current recall of mastered concepts, and the observation the agent got.
The replay is checked against the log (quiz answers must match) and against the Harbor reward.
PPO and the literature policy are played on the same children for side-by-side comparison.
Output: viz/traces.json
"""
import glob, hashlib, json, os, sys, csv
import tutor2 as t, sweep_tutor2 as w, ppo_policy as P, lit_ref as L

N_CHILD, SALT = 6, "eval1"
MODELS = {"claude-opus-5-5": "v4_opus55", "gpt-6.1-sol": "v4_gpt61sol", "gemini-pro-latest": "v4_gemini",
          "gemini-3.1-pro-preview": "v4_gemini31pro", "gemini-3.5-flash": "v4_gemini35flash",
          "gemini-pro-latest (all parameters disclosed)": "v5d_gemini",
          "gemini-3.1-pro-preview (all parameters disclosed)": "v5d_gemini31pro"}
DAYS = ["Mon", "Wed", "Fri"]


def child_seeds(seed):
    return [str(int(hashlib.sha256(f"{SALT}:{seed}:{i}".encode()).hexdigest(), 16) % 1_000_000_000) for i in range(N_CHILD)]


def snapshot(s):
    return {"state": ["MUB"[["mastered", "unlearned", "buggy"].index(x)] for x in s.state],
            "recall": [round(s.recall(k), 3) for k in range(t.K)],
            "exam_recall": [round(s.recall(k, t.EXAM_DAY), 3) for k in range(t.K)]}


def step_record(s, turn, kind, k, bug=None, ok=None, ans=None):
    sess = min(turn // t.TURNS_PER, 2)
    d = {"turn": turn, "day": DAYS[sess], "act": kind, "k": k + 1}
    if bug: d["bug"] = bug
    if kind == "quiz":
        d["ok"] = bool(ok); d["ans"] = ans
        d["sig"] = (not ok) and ans == t.QUIZ[k][2]
    d.update(snapshot(s))
    return d


def tool_calls(path):
    calls = []
    for line in open(path):
        try:
            p = json.loads(line).get("part", {})
        except json.JSONDecodeError:
            continue
        if p.get("type") != "tool" or p.get("state", {}).get("status") != "completed":
            continue
        name = p.get("tool", "").split("_")[-1]
        try:
            out = json.loads(p["state"].get("output") or "{}")
        except json.JSONDecodeError:
            out = {}
        calls.append((name, p["state"].get("input") or {}, out))
    return calls


def replay_agent(seed, path):
    seeds = child_seeds(seed)
    kids = [{"steps": [], "asks": 0} for _ in range(N_CHILD)]
    ci, s, mism = 0, t.Sim(seeds[0]), 0
    exams = []

    def end_child():
        nonlocal ci, s
        kids[ci]["exam"] = round(s.exam(), 4); exams.append(s.exam())
        ci += 1
        if ci < N_CHILD:
            s = t.Sim(seeds[ci])

    for name, inp, out in tool_calls(path):
        if ci >= N_CHILD:
            break
        if name == "ask":
            kids[ci]["asks"] += 1; continue
        if name == "finish":
            end_child(); continue
        if name not in ("explain", "quiz", "correct"):
            continue
        if out.get("message", "").startswith("The run is over"):
            break
        k = int(inp["K"]) - 1
        turn = s.turn
        if name == "explain":
            s.explain(k); rec = step_record(s, turn, "explain", k)
        elif name == "correct":
            s.correct(k, inp["BUG"]); rec = step_record(s, turn, "correct", k, inp["BUG"])
        else:
            ok, q, ans, right = s.quiz(k)
            if "answer" in out and out["answer"] != ans:
                mism += 1
            rec = step_record(s, turn, "quiz", k, ok=ok, ans=ans)
        kids[ci]["steps"].append(rec)
        if s.turn >= t.T:
            end_child()
    while ci < N_CHILD:
        end_child()
    return kids, exams, mism


def play_policy(seed, which):
    kids, exams = [], []
    for sd in child_seeds(seed):
        s = t.Sim(sd); steps = []
        if which == "PPO":
            tr, pol = P.Tracker(), P.Policy()
            while s.turn < t.T:
                turn = s.turn; a = P.decode(pol.act(tr.obs(turn)))
                if a[0] == "quiz":
                    ok, q, ans, right = s.quiz(a[1]); tr.update(turn, a, (ok, ans)); steps.append(step_record(s, turn, "quiz", a[1], ok=ok, ans=ans))
                elif a[0] == "explain":
                    s.explain(a[1]); tr.update(turn, a); steps.append(step_record(s, turn, "explain", a[1]))
                else:
                    s.correct(a[1], a[2]); tr.update(turn, a); steps.append(step_record(s, turn, "correct", a[1], a[2]))
        else:
            mem, lo = {}, None
            while s.turn < t.T:
                turn = s.turn; a, k, b = L.lit_step(s, mem, lo)
                if a == "quiz":
                    ok, q, ans, right = s.quiz(k); lo = (a, k, (ok, ans)); steps.append(step_record(s, turn, "quiz", k, ok=ok, ans=ans))
                elif a == "explain":
                    s.explain(k); lo = (a, k, None); steps.append(step_record(s, turn, "explain", k))
                else:
                    s.correct(k, b); lo = (a, k, None); steps.append(step_record(s, turn, "correct", k, b))
        kids.append({"steps": steps, "exam": round(s.exam(), 4), "asks": 0}); exams.append(s.exam())
    return kids, exams


def pooled(seed, exams):
    seeds = child_seeds(seed)
    base = [t.Sim(s).exam() for s in seeds]; full = [w.oracle_exam(s) for s in seeds]
    return (sum(exams) - sum(base)) / (sum(full) - sum(base)), base, full


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = {"concepts": t.CONCEPTS, "bugs": t.BUG_DESC, "bug_concept": {b: k + 1 for k, b in t.BUGS.items()}, "runs": {}}
    for seed in (3001, 3002, 3003, 3004, 3005):
        run = {"children": child_seeds(seed), "policies": {}, "start": [snapshot(t.Sim(c)) for c in child_seeds(seed)]}
        for name in ("PPO", "literature rulebook"):
            kids, ex = play_policy(seed, "PPO" if name == "PPO" else "lit")
            r, base, full = pooled(seed, ex)
            run["policies"][name] = {"reward": round(r, 4), "kids": kids}
        run["base"] = [round(x, 4) for x in base]; run["full"] = [round(x, 4) for x in full]
        for name, tag in MODELS.items():
            paths = glob.glob(os.path.join(here, f"../jobs/{tag}_{seed}/seed_{seed}/*/*/agent/opencode.txt"))
            if not paths:
                continue
            kids, ex, mism = replay_agent(seed, paths[0])
            r, _, _ = pooled(seed, ex)
            logged = None
            tsv = os.path.join(here, f"../harbor/results/{tag}.tsv")
            if os.path.exists(tsv):
                for row in csv.DictReader(open(tsv), delimiter="\t"):
                    if row["seed"] == str(seed) and row["reward"] not in ("ERROR", "NA"):
                        logged = float(row["reward"])
            ok = logged is not None and abs(logged - r) < 1e-6
            print(f"{seed} {name:48s} replay {r:.4f} logged {logged} {'OK' if ok else 'MISMATCH'} quiz-answer mismatches {mism}")
            if ok:
                run["policies"][name] = {"reward": round(r, 4), "kids": kids}
        out["runs"][str(seed)] = run
    os.makedirs(os.path.join(here, "../viz"), exist_ok=True)
    json.dump(out, open(os.path.join(here, "../viz/traces.json"), "w"), separators=(",", ":"))
    print("wrote viz/traces.json", os.path.getsize(os.path.join(here, "../viz/traces.json")) // 1024, "KB")


if __name__ == "__main__":
    main()
