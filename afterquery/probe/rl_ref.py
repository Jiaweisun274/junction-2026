"""RL reference for tutor2: PPO on observation-only features (minimal local version).

Observation (only what an agent sees): per concept - explanations given, quizzes right / wrong, last quiz result,
misconception signature seen (and not yet corrected), corrections done, days since last contact, and a BKT-style
mastery belief using the rule-stated slip/guess; plus turn, session and turns left.
Actions (15): explain 1-6, quiz 1-6, correct the misconception that belongs to concept 2 / 3 / 5.
Reward: exam score minus the no-teaching exam score, given at the end.
Training seeds are drawn from [10^6, 10^9); evaluation uses test seeds 2001-2040 (disjoint).
"""
import sys, time
import numpy as np
import torch
import torch.nn as nn
import tutor2 as t

BUG_K = sorted(t.BUGS)                     # concepts (0-based) that can hold a misconception: [1, 2, 4]
N_ACT = 2 * t.K + len(BUG_K)
F_PER = 9
OBS_DIM = t.K * F_PER + 6


def decode(a):
    if a < t.K:
        return ("explain", a, None)
    if a < 2 * t.K:
        return ("quiz", a - t.K, None)
    k = BUG_K[a - 2 * t.K]
    return ("correct", k, t.BUGS[k])


class Env:
    def __init__(self, seed):
        self.s = t.Sim(str(seed))
        self.base = self.s.exam()
        K = t.K
        self.ne = np.zeros(K); self.qr = np.zeros(K); self.qw = np.zeros(K); self.last = np.zeros(K)
        self.sig = np.zeros(K); self.nc = np.zeros(K); self.touch = np.full(K, -1.0); self.b = np.full(K, 0.5)

    def obs(self):
        now = self.s.now()
        since = np.where(self.touch < 0, 0.0, (now - self.touch) / 7.0)
        per = np.stack([self.ne / 5, self.qr / 5, self.qw / 5, self.last, self.sig, self.nc / 3,
                        since, self.b, (self.touch >= 0).astype(float)], 1).ravel()
        sess = self.s.turn // t.TURNS_PER
        g = [self.s.turn / t.T, (t.T - self.s.turn) / t.T, float(sess == 0), float(sess == 1), float(sess == 2),
             (t.TURNS_PER - self.s.turn % t.TURNS_PER) / t.TURNS_PER]
        return np.concatenate([per, g]).astype(np.float32)

    def step(self, a):
        kind, k, bug = decode(a)
        now = self.s.now()
        if kind == "explain":
            self.s.explain(k); self.ne[k] += 1
            self.b[k] = self.b[k] + (1 - self.b[k]) * 0.3
        elif kind == "correct":
            self.s.correct(k, bug); self.nc[k] += 1; self.sig[k] = 0; self.b[k] = 0.5
        else:
            ok, q, ans, right = self.s.quiz(k)
            if ok:
                self.qr[k] += 1; self.last[k] = 1
                lm, ln = 1 - t.SLIP, t.GUESS
            else:
                self.qw[k] += 1; self.last[k] = -1
                lm, ln = t.SLIP, 1 - t.GUESS
                if ans == t.QUIZ[k][2]:
                    owner = t.CARRY.get(k, k)
                    self.sig[owner] = 1
            self.b[k] = self.b[k] * lm / (self.b[k] * lm + (1 - self.b[k]) * ln)
        self.touch[k] = now
        done = self.s.turn >= t.T
        r = (self.s.exam() - self.base) if done else 0.0
        return r, done


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(OBS_DIM, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.pi = nn.Linear(128, N_ACT)
        self.v = nn.Linear(128, 1)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def evaluate(net, seeds):
    import sweep_tutor2 as w
    out = []
    for sd in seeds:
        e = Env(sd)
        while e.s.turn < t.T:
            with torch.no_grad():
                logits, _ = net(torch.from_numpy(e.obs()))
            e.step(int(torch.argmax(logits)))
        out.append(w.norm(str(sd), e.s.exam()))
    return float(np.mean(out))


def train(iters=300, n_env=128, seed=0, eval_every=50, out="rl_ref.pt"):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    test = list(range(2001, 2041))
    t0 = time.time()
    for it in range(1, iters + 1):
        envs = [Env(int(rng.integers(10**6, 10**9))) for _ in range(n_env)]
        O, A, LP, V, R = [], [], [], [], []
        for step in range(t.T):
            obs = torch.from_numpy(np.stack([e.obs() for e in envs]))
            with torch.no_grad():
                logits, v = net(obs)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            rew = np.zeros(n_env, dtype=np.float32)
            for i, e in enumerate(envs):
                r, _ = e.step(int(a[i])); rew[i] = r
            O.append(obs); A.append(a); LP.append(dist.log_prob(a)); V.append(v); R.append(torch.from_numpy(rew))
        O, A, LP, V, R = map(torch.stack, (O, A, LP, V, R))      # [T, n_env, ...]
        ret = torch.flip(torch.cumsum(torch.flip(R, [0]), 0), [0])   # undiscounted return-to-go (reward only at end)
        adv = ret - V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        O, A, LP, ret, adv = O.reshape(-1, OBS_DIM), A.reshape(-1), LP.reshape(-1), ret.reshape(-1), adv.reshape(-1)
        for _ in range(4):
            perm = torch.randperm(len(A))
            for mb in perm.split(1024):
                logits, v = net(O[mb])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = torch.exp(dist.log_prob(A[mb]) - LP[mb])
                pl = -torch.min(ratio * adv[mb], torch.clamp(ratio, 0.8, 1.2) * adv[mb]).mean()
                vl = ((v - ret[mb]) ** 2).mean()
                loss = pl + 0.5 * vl - 0.01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        if it % eval_every == 0 or it == 1:
            mean_r = float(R.sum(0).mean())
            print(f"iter {it:4d}  train exam gain {mean_r:.3f}  test norm {evaluate(net, test):.3f}  ({time.time()-t0:.0f}s)", flush=True)
            torch.save(net.state_dict(), out)
    return net


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    train(iters=iters)
