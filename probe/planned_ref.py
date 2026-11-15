"""A readable, hand-planned teacher for tutor2: the PPO reference's habits written as a short rule list.

Uses only what the teacher can observe (its own actions and the child's answers). Rules, in priority order:
  1. A misconception's typical wrong answer was seen  -> run the targeted correction.
  2. Last FINAL turns of the week                      -> keep teaching: explain the frontier concept
                                                          (no re-testing right before the exam).
  3. First REVIEW turns of Wednesday and Friday        -> quiz the confirmed concept seen longest ago;
                                                          a wrong answer sends the frontier back to it.
  4. The frontier concept failed after MAXEXPL explanations this session
                                                       -> quiz its prerequisite (it may have been a lucky guess);
                                                          a wrong answer moves the frontier back.
  5. Otherwise explain the frontier concept once, then quiz it; move on after CONFIRM correct answers in a row.
     On Monday the first concepts are quizzed before explaining (the child may already know them).
The knobs are few and tuned on training seeds only (1001-1400), never on the evaluation children.
"""
import itertools, sys
import numpy as np
import tutor2 as t


def run(seed, confirm=2, final=4, review=1, maxexpl=2, diag=True, prereq=True):
    s = t.Sim(str(seed))
    k, streak, bug = 0, 0, {}
    need_explain = not diag
    known, last_ok = set(), {}
    sess_expl = {}
    review_left, cur_sess = 0, 0
    while s.turn < t.T:
        sess = s.turn // t.TURNS_PER
        if sess != cur_sess:
            cur_sess, review_left = sess, review
        left = t.T - s.turn
        kk = min(k, t.K - 1)
        # 1. misconception
        if bug:
            b_k = next(iter(bug)); s.correct(b_k, bug.pop(b_k)); need_explain = False; continue
        # 2. last turns: teach
        if left <= final:
            s.explain(kk); continue
        # 3. session-start review
        if review_left > 0 and known:
            review_left -= 1
            r = min(known, key=lambda c: last_ok.get(c, -1))
            ok, q, a, right = s.quiz(r)
            if ok:
                last_ok[r] = s.turn
            else:
                known.discard(r)
                if a == t.QUIZ[r][2]:
                    owner = t.CARRY.get(r, r); bug[owner] = t.BUGS[owner]
                if r < k:
                    k, streak, need_explain = r, 0, True
            continue
        review_left = 0
        # 4. stuck on the frontier: check the prerequisite
        n_ex = sess_expl.get((sess, kk), 0)
        if prereq and kk > 0 and n_ex >= maxexpl and need_explain:
            p = kk - 1
            ok, q, a, right = s.quiz(p)
            if ok:
                last_ok[p] = s.turn
                sess_expl[(sess, kk)] = 0  # prerequisite holds: try explaining again
            else:
                known.discard(p); k, streak = p, 0
                if a == t.QUIZ[p][2]:
                    owner = t.CARRY.get(p, p); bug[owner] = t.BUGS[owner]; need_explain = False
            continue
        # 5. explain once, then quiz
        if need_explain:
            s.explain(kk); sess_expl[(sess, kk)] = n_ex + 1; need_explain = False; continue
        ok, q, a, right = s.quiz(kk)
        if ok:
            streak += 1; last_ok[kk] = s.turn
            if streak >= confirm:
                known.add(kk)
                if k < t.K - 1:
                    k, streak = k + 1, 0
                    need_explain = not (diag and sess == 0 and s.turn < 4)
        else:
            streak, need_explain = 0, True
            if a == t.QUIZ[kk][2]:
                owner = t.CARRY.get(kk, kk); bug[owner] = t.BUGS[owner]; need_explain = False
                if owner != kk:
                    k = owner
    return s


GRID = [dict(confirm=c, final=f, review=r, maxexpl=m, diag=d, prereq=p)
        for c, f, r, m, d, p in itertools.product([1, 2], [0, 2, 4, 6], [0, 1, 2], [1, 2, 3], [False, True], [False, True])]


def gain(seeds, **kn):
    return float(np.mean([run(sd, **kn).exam() - t.Sim(str(sd)).exam() for sd in seeds]))


def tune(train=range(1001, 1401)):
    scored = sorted(((gain(train, **kn), kn) for kn in GRID), key=lambda x: -x[0])
    for g, kn in scored[:5]:
        print(f"{g:.4f} {kn}")
    return scored[0][1]


if __name__ == "__main__":
    best = tune()
    print("BEST", best)
