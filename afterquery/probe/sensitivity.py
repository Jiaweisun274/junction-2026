"""Sensitivity analysis: does "quiz less, teach more" depend on the simulator's parameters?

We vary three parameters that set the value of a quiz versus an explanation:
  retrieve  - half-life multiplier after a correct quiz (testing effect), base 2.5
  forget    - scale on all memory half-lives (smaller = faster forgetting), base 1.0
  learn     - scale on the per-concept learn chance, base 1.0
For each setting we score a family of scripted teachers whose knobs move the quiz share
(how many correct answers before moving on, explanations before the first quiz, review at session start,
review turns before the exam), and an "agent-like" teacher that copies the frontier agents' habits
(about 65% quizzes: explain once, quiz, move on after one correct, review at each session start, review
the last five turns). Output: best quiz share per setting, the agent-like teacher's regret, and the best
teacher's knobs.
"""
import itertools, sys
import numpy as np
import tutor2 as t

BASE_PARAMS = t.params


def make_params(retrieve, forget, learn):
    def p(seed):
        d = BASE_PARAMS(seed)
        d = dict(d)
        d["plearn"] = np.clip(d["plearn"] * learn, 0.02, 0.98)
        d["h0"] = d["h0"] * forget
        d["h_init"] = d["h_init"] * forget
        return d
    return p


def teacher(seed, confirm, expl_first, start_review, final_review):
    s = t.Sim(str(seed))
    k, streak, need, bug, lastsess = 0, 0, expl_first, {}, 0
    known = set()
    while s.turn < t.T:
        sess = s.turn // t.TURNS_PER
        left = t.T - s.turn
        # session-start review: one quiz on each concept already confirmed
        if start_review and sess != lastsess:
            lastsess = sess
            queue = sorted(known)
        else:
            queue = getattr(teacher, "_q", [])
        if sess != lastsess:
            lastsess = sess
        if start_review and queue:
            kk = queue.pop(0); teacher._q = queue
            ok, q, a, r = s.quiz(kk)
            if not ok:
                known.discard(kk); k = min(k, kk); streak = 0; need = expl_first
            continue
        teacher._q = []
        if left <= final_review and known:
            kk = sorted(known)[(t.T - left) % len(known)]
            s.quiz(kk); continue
        kk = min(k, t.K - 1)
        if kk in bug:
            s.correct(kk, bug.pop(kk)); need = 0; continue
        if need > 0:
            s.explain(kk); need -= 1; continue
        ok, q, a, right = s.quiz(kk)
        if ok:
            streak += 1
            if streak >= confirm:
                known.add(kk)
                if k < t.K - 1:
                    k, streak, need = k + 1, 0, expl_first
        else:
            streak, need = 0, 1
            if a == t.QUIZ[kk][2]:
                owner = t.CARRY.get(kk, kk); bug[owner] = t.BUGS[owner]; need = 0
                if owner != kk: k = owner
    return s


def quiz_share_and_gain(seeds, **knobs):
    gains, qs = [], []
    for sd in seeds:
        teacher._q = []
        base = t.Sim(str(sd)).exam()
        s = teacher(sd, **knobs)
        gains.append(s.exam() - base)
    return float(np.mean(gains))


def count_quiz_share(seed, **knobs):
    # replay the teacher while counting actions
    calls = {"q": 0, "n": 0}
    orig_q, orig_e, orig_c = t.Sim.quiz, t.Sim.explain, t.Sim.correct
    def q(self, k): calls["q"] += 1; calls["n"] += 1; return orig_q(self, k)
    def e(self, k): calls["n"] += 1; return orig_e(self, k)
    def c(self, k, b): calls["n"] += 1; return orig_c(self, k, b)
    t.Sim.quiz, t.Sim.explain, t.Sim.correct = q, e, c
    try:
        teacher._q = []; teacher(seed, **knobs)
    finally:
        t.Sim.quiz, t.Sim.explain, t.Sim.correct = orig_q, orig_e, orig_c
    return calls["q"] / max(calls["n"], 1)


AGENT_LIKE = dict(confirm=1, expl_first=1, start_review=True, final_review=5)
GRID = [dict(confirm=c, expl_first=e, start_review=sr, final_review=fr)
        for c, e, sr, fr in itertools.product([1, 2, 3], [1, 2, 3], [False, True], [0, 2, 5])]


def main(n_seeds=120):
    seeds = list(range(5001, 5001 + n_seeds))
    settings = [(r, f, l) for r in (1.5, 2.5, 4.0) for f in (0.3, 1.0, 3.0) for l in (0.7, 1.0, 1.3)]
    print("retrieve forget learn | best gain  best quiz-share  best knobs                         | agent-like gain  quiz-share  regret")
    rows = []
    for r, f, l in settings:
        t.RETRIEVE = r
        t.params = make_params(r, f, l)
        scored = []
        for kn in GRID:
            g = quiz_share_and_gain(seeds, **kn)
            scored.append((g, kn))
        g_best, kn_best = max(scored, key=lambda x: x[0])
        qs_best = np.mean([count_quiz_share(sd, **kn_best) for sd in seeds[:30]])
        g_agent = quiz_share_and_gain(seeds, **AGENT_LIKE)
        qs_agent = np.mean([count_quiz_share(sd, **AGENT_LIKE) for sd in seeds[:30]])
        rows.append((r, f, l, g_best, qs_best, kn_best, g_agent, qs_agent))
        print(f"{r:8.1f} {f:6.1f} {l:5.1f} | {g_best:9.3f}  {qs_best:15.0%}  {str(kn_best):34s} | {g_agent:15.3f}  {qs_agent:10.0%}  {g_best - g_agent:6.3f}", flush=True)
    t.params = BASE_PARAMS; t.RETRIEVE = 2.5
    return rows


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
