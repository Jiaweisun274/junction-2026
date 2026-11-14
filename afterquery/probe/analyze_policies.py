"""Behaviour statistics for tutor2 policies: PPO, literature, and frontier agents' Harbor traces."""
import json, glob, numpy as np, torch
import tutor2 as t, lit_ref as L, rl_ref as R

def ppo_trace(sd, net):
    e = R.Env(sd); tr = []
    while e.s.turn < t.T:
        with torch.no_grad():
            a = int(torch.argmax(net(torch.from_numpy(e.obs()))[0]))
        kind, k, bug = R.decode(a)
        before = e.s.turn
        if kind == "quiz":
            ok_before = e.qr[k]
        e.step(a)
        res = None
        if kind == "quiz":
            res = "ok" if e.qr[k] > ok_before else ("sig" if e.sig[t.CARRY.get(k, k)] and e.last[k] == -1 else "wrong")
        tr.append((kind, k, res))
    return tr

def lit_trace(sd):
    s = t.Sim(str(sd)); mem = {}; lo = None; tr = []
    while s.turn < t.T:
        a, k, b = L.lit_step(s, mem, lo)
        if a == "quiz":
            ok, q, ans, right = s.quiz(k); lo = (a, k, (ok, ans))
            tr.append((a, k, "ok" if ok else ("sig" if ans == t.QUIZ[k][2] else "wrong")))
        else:
            (s.explain(k) if a == "explain" else s.correct(k, b)); lo = (a, k, None); tr.append((a, k, None))
    return tr

def harbor_traces(tag):
    out = []
    for f in glob.glob(f"../jobs/{tag}/seed_*/*/*/agent/opencode.txt"):
        tr = []
        for line in open(f):
            try: p = json.loads(line).get("part", {})
            except Exception: continue
            if p.get("type") == "tool" and p.get("state", {}).get("status") == "completed":
                name = p.get("tool", "").split("_")[-1]
                if name in ("explain", "quiz", "correct"):
                    k = int(p["state"]["input"].get("K", 1)) - 1
                    res = None
                    if name == "quiz":
                        o = p["state"].get("output", "")
                        res = "ok" if '"correct": true' in o or '"correct":true' in o else "wrong"
                    tr.append((name, k, res))
        if tr: out.append(tr)
    return out

def stats(traces):
    n = sum(len(x) for x in traces)
    c = lambda kind: sum(1 for x in traces for a in x if a[0] == kind) / n
    # explanations in a row before the first quiz on that concept
    runs, moves_after, backoffs = [], [], 0
    for tr in traces:
        for i, (a, k, r) in enumerate(tr):
            if a == "explain":
                j = i; m = 0
                while j < len(tr) and tr[j][0] == "explain" and tr[j][1] == k: m += 1; j += 1
                if i == 0 or not (tr[i-1][0] == "explain" and tr[i-1][1] == k): runs.append(m)
            # after a wrong quiz on k, does the next turn-taking action go to k-1 (back-off)?
            if a == "quiz" and r != "ok" and k > 0 and i + 1 < len(tr) and tr[i+1][1] == k - 1: backoffs += 1
        # consecutive correct answers on k right before first action on a higher concept
        best_k = -1
        for i, (a, k, r) in enumerate(tr):
            if k > best_k and best_k >= 0:
                streak = 0; j = i - 1
                while j >= 0 and tr[j][0] == "quiz" and tr[j][1] == best_k and tr[j][2] == "ok": streak += 1; j -= 1
                moves_after.append(streak)
            best_k = max(best_k, k)
    reach = np.mean([max(a[1] for a in tr) + 1 for tr in traces])
    last5_quiz = np.mean([sum(1 for a in tr[-5:] if a[0] == "quiz") / 5 for tr in traces])
    return dict(explain=round(c("explain"), 2), quiz=round(c("quiz"), 2), correct=round(c("correct"), 3),
                expl_run=round(float(np.mean(runs)), 2), moves_after_streak=round(float(np.mean(moves_after)), 2),
                backoffs_per_game=round(backoffs / len(traces), 2), highest_concept=round(float(reach), 2),
                last5_quiz_share=round(float(last5_quiz), 2))

if __name__ == "__main__":
    net = R.Net(); net.load_state_dict(torch.load("rl_ref_last.pt")); net.eval()
    seeds = range(2041, 2141)
    rows = {"PPO": [ppo_trace(s, net) for s in seeds], "literature": [lit_trace(s) for s in seeds],
            "gpt-6.1-sol (Harbor, old sim)": harbor_traces("h_gpt61sol"), "opus-5.5 (Harbor, old sim)": harbor_traces("h_opus55")}
    for name, tr in rows.items():
        print(f"{name:30s} n={len(tr):3d}", stats(tr))
    print("\nPPO example traces (E=explain Q=quiz C=correct; quiz result +/-/!):")
    for tr in rows["PPO"][:4]:
        print(" ".join(f"{a[0][0].upper()}{a[1]+1}{ {'ok':'+','wrong':'-','sig':'!',None:''}[a[2]] }" for a in tr))
