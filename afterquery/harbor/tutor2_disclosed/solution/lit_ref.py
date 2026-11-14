"""Literature reference policy for tutor2: standard components from education research, parameters fixed a priori
(never tuned on our seeds).

1. Mastery learning with Bayesian Knowledge Tracing (Bloom 1968; Corbett & Anderson 1995):
   practise a concept until P(mastered) >= 0.95. Slip/guess are the values stated in the rules;
   P(L0) = 0.5 (uninformative), P(T) = 0.3 (fixed guess at a typical learn rate, not fitted).
2. Spaced review (Leitner 1972; Settles & Meeder 2016; Tabibian et al. 2019): keep a half-life estimate per learned
   concept, start at 1 day, double on each correct recall (Leitner-style doubling); review a concept when its predicted
   recall drops below the target retention 0.9 (the FSRS/Anki default).
3. Prerequisite back-off (knowledge space theory, ALEKS; Doignon & Falmagne 1999): after 3 explanations without
   reaching mastery, re-test the prerequisite and fall back to it if it fails.
4. Misconception remediation (Brown & Burton 1978; conceptual change): on the misconception's signature answer,
   use the targeted correction.
5. Exam preparation: in the last session, keep enough turns to review every concept whose predicted exam-day recall
   is below the target.
"""
import numpy as np
import tutor2 as t

P_L0, P_T, MASTERY = 0.5, 0.3, 0.95
H0, H_MULT, TARGET = 1.0, 2.0, 0.9
BACKOFF_AFTER = 3


def lit_step(s, mem, last_obs):
    if not mem:
        mem.update(b=[P_L0] * t.K, bug=[None] * t.K, h=[H0] * t.K, last=[None] * t.K,
                   learned=[False] * t.K, explains=[0] * t.K, k=0, backoff=None)
    b, now = mem["b"], s.now()

    # ---- update beliefs from the last observation ----
    if last_obs:
        a, k, o = last_obs
        if a == "quiz":
            ok, ans = o
            if not ok and k in t.BUGS and ans == t.QUIZ[k][2]:
                mem["bug"][k] = t.BUGS[k]; b[k] = 0.0; mem["learned"][k] = False
            elif not ok and k in t.CARRY and ans == t.QUIZ[k][2]:
                j = t.CARRY[k]; mem["bug"][j] = t.BUGS[j]; b[j] = 0.0; mem["learned"][j] = False
                mem["k"] = min(mem["k"], j)
            else:
                lm, ln = (1 - t.SLIP, t.GUESS) if ok else (t.SLIP, 1 - t.GUESS)
                b[k] = b[k] * lm / (b[k] * lm + (1 - b[k]) * ln)
                if mem["learned"][k]:
                    if ok:
                        mem["h"][k] *= H_MULT
                    mem["last"][k] = now
                    if not ok:                 # forgotten: needs re-learning
                        b[k] = min(b[k], 0.5); mem["learned"][k] = False
            if mem["backoff"] is not None and k == mem["backoff"]:
                if b[k] >= MASTERY or (ok and b[k] > 0.8):
                    mem["backoff"] = None; mem["explains"][k + 1] = 0
        elif a == "explain":
            if mem["bug"][k] is None:
                b[k] = b[k] + (1 - b[k]) * P_T
            mem["explains"][k] += 1
            if mem["learned"][k]:
                mem["last"][k] = now
        elif a == "correct":
            mem["bug"][k] = None; b[k] = 0.5
    for k in range(t.K):
        if b[k] >= MASTERY and not mem["learned"][k] and mem["bug"][k] is None:
            mem["learned"][k] = True; mem["last"][k] = now; mem["h"][k] = H0

    def recall(k, at):
        return 2 ** (-(at - mem["last"][k]) / mem["h"][k]) if mem["learned"][k] else 0.0

    turns_left = t.T - s.turn
    session = s.turn // t.TURNS_PER
    learned = [k for k in range(t.K) if mem["learned"][k]]

    # ---- 5. exam preparation in the last session ----
    if session == len(t.SESSIONS) - 1:
        weak = [k for k in learned if recall(k, t.EXAM_DAY) < TARGET]
        if weak and turns_left <= len(weak):
            return ("quiz", min(weak, key=lambda k: recall(k, t.EXAM_DAY)), None)

    # ---- 2. spaced review of concepts below target retention now ----
    due = [k for k in learned if recall(k, now) < TARGET]
    if due:
        return ("quiz", min(due, key=lambda k: recall(k, now)), None)

    # ---- frontier concept ----
    k = mem["k"]
    while k < t.K and mem["learned"][k]:
        k += 1
    mem["k"] = k
    if k >= t.K:
        # all learned: review the weakest for the exam
        return ("quiz", min(learned, key=lambda j: recall(j, t.EXAM_DAY)), None)

    # ---- 4. misconception ----
    if mem["bug"][k]:
        return ("correct", k, mem["bug"][k])

    # ---- 3. prerequisite back-off ----
    if mem["backoff"] is not None:
        return ("quiz", mem["backoff"], None)
    if k > 0 and mem["explains"][k] >= BACKOFF_AFTER and b[k] < MASTERY:
        mem["backoff"] = k - 1
        mem["learned"][k - 1] = False; b[k - 1] = min(b[k - 1], 0.6)
        return ("quiz", k - 1, None)

    # ---- 1. mastery learning with BKT ----
    return ("explain", k, None) if b[k] < 0.7 else ("quiz", k, None)


if __name__ == "__main__":
    import sweep_tutor2 as w
    rows = []
    for sd in [str(x) for x in range(2001, 2009)]:
        lit = w.norm(sd, t.run_policy(sd, lit_step, obs=True).exam())
        ref = w.norm(sd, t.run_policy(sd, t.reference_step, obs=True).exam())
        rows.append((lit, ref)); print(sd, "literature", round(lit, 2), "tuned-ref", round(ref, 2))
    print("mean 2001-2008: literature %.2f tuned-ref %.2f" % tuple(np.mean(rows, 0)))
    test = [str(x) for x in range(2001, 2061)]
    v = [w.norm(sd, t.run_policy(sd, lit_step, obs=True).exam()) for sd in test]
    print("60 test seeds: literature %.3f" % np.mean(v))
