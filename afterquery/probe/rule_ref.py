"""Training-free rule reference for tutor2 (from the Fable review, afterquery/fable_review/analysis/check_tutor2.py).

Uses only what the rules state and no experience with the simulator:
  explain the next concept, quiz it, move on after two correct answers in a row, re-explain after a wrong answer,
  and when a misconception's typical wrong answer appears, correct it on the concept where it lives.
No reviews, no use of `ask`, no tuned numbers.
"""
import tutor2 as t


def run(seed, confirm=2):
    s = t.Sim(str(seed))
    k, streak, need_explain, bug = 0, 0, True, {}
    while s.turn < t.T:
        kk = min(k, t.K - 1)
        if kk in bug:
            s.correct(kk, bug.pop(kk)); need_explain = False; continue
        if need_explain:
            s.explain(kk); need_explain = False; continue
        ok, q, a, right = s.quiz(kk)
        if ok:
            streak += 1
            if streak >= confirm and k < t.K - 1:
                k, streak, need_explain = k + 1, 0, True
        else:
            streak, need_explain = 0, True
            if a == t.QUIZ[kk][2]:
                owner = t.CARRY.get(kk, kk)
                bug[owner] = t.BUGS[owner]; need_explain = False
                if owner != kk:
                    k = owner
    return s
