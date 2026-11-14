"""Numpy-only PPO reference policy for tutor2 (no torch needed at run time).

Tracker rebuilds the same observation features as rl_ref.Env from what a player sees (its own actions, quiz answers,
the turn counter). act() is the greedy action of the trained network (weights in ppo_ref.npz, exported from
rl_ref_massed.pt). Used by the Harbor reference agent (over MCP) and by the sidecar's reference score.
"""
import os
import numpy as np
import tutor2 as t

BUG_K = sorted(t.BUGS)
F_PER = 10


def now_of(turn):
    s = min(turn // t.TURNS_PER, len(t.SESSIONS) - 1)
    return t.SESSIONS[s] + 0.01 * (turn % t.TURNS_PER)


def decode(a):
    if a < t.K:
        return ("explain", a, None)
    if a < 2 * t.K:
        return ("quiz", a - t.K, None)
    k = BUG_K[a - 2 * t.K]
    return ("correct", k, t.BUGS[k])


class Tracker:
    def __init__(self):
        K = t.K
        self.ne = np.zeros(K); self.qr = np.zeros(K); self.qw = np.zeros(K); self.last = np.zeros(K)
        self.sig = np.zeros(K); self.nc = np.zeros(K); self.touch = np.full(K, -1.0); self.b = np.full(K, 0.5)
        self.se = np.zeros(K); self.sess = 0

    def obs(self, turn):
        now = now_of(turn)
        since = np.where(self.touch < 0, 0.0, (now - self.touch) / 7.0)
        per = np.stack([self.ne / 5, self.qr / 5, self.qw / 5, self.last, self.sig, self.nc / 3,
                        since, self.b, (self.touch >= 0).astype(float), self.se / 3], 1).ravel()
        sess = turn // t.TURNS_PER
        g = [turn / t.T, (t.T - turn) / t.T, float(sess == 0), float(sess == 1), float(sess == 2),
             (t.TURNS_PER - turn % t.TURNS_PER) / t.TURNS_PER]
        return np.concatenate([per, g]).astype(np.float32)

    def update(self, turn, action, quiz=None):
        """turn = turn index when the action was taken; quiz = (correct, answer) for quizzes."""
        kind, k, bug = action
        now = now_of(turn)
        if turn // t.TURNS_PER != self.sess:
            self.sess = turn // t.TURNS_PER; self.se[:] = 0
        if kind == "explain":
            self.ne[k] += 1; self.se[k] += 1
            self.b[k] = self.b[k] + (1 - self.b[k]) * 0.3
        elif kind == "correct":
            self.nc[k] += 1; self.sig[k] = 0; self.b[k] = 0.5
        else:
            ok, ans = quiz
            if ok:
                self.qr[k] += 1; self.last[k] = 1; lm, ln = 1 - t.SLIP, t.GUESS
            else:
                self.qw[k] += 1; self.last[k] = -1; lm, ln = t.SLIP, 1 - t.GUESS
                if ans == t.QUIZ[k][2]:
                    self.sig[t.CARRY.get(k, k)] = 1
            self.b[k] = self.b[k] * lm / (self.b[k] * lm + (1 - self.b[k]) * ln)
        self.touch[k] = now


class Policy:
    def __init__(self, path=None):
        path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_ref.npz")
        w = np.load(path)
        self.W = [(w["w0"], w["b0"]), (w["w1"], w["b1"])]
        self.Wpi, self.bpi = w["wpi"], w["bpi"]

    def act(self, obs):
        h = obs
        for W, b in self.W:
            h = np.tanh(h @ W.T + b)
        return int(np.argmax(h @ self.Wpi.T + self.bpi))


def run(seed, policy=None):
    """Play one game on the simulator directly (used by the sidecar for the reference score)."""
    policy = policy or Policy()
    s = t.Sim(str(seed)); tr = Tracker()
    while s.turn < t.T:
        a = decode(policy.act(tr.obs(s.turn)))
        turn = s.turn
        if a[0] == "explain":
            s.explain(a[1]); tr.update(turn, a)
        elif a[0] == "correct":
            s.correct(a[1], a[2]); tr.update(turn, a)
        else:
            ok, q, ans, right = s.quiz(a[1]); tr.update(turn, a, (ok, ans))
    return s
