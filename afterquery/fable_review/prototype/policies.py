"""Policies for the Discount Trap prototype. All use only Market.observe() + public rules."""
import numpy as np
from env import T, PRICES, R0, rate, draw_theta

XG, RG = 41, 19  # DP grid sizes (stock, reference price)


def fluid_dp_q(theta, t, stock, r, stock0):
    """Q(price) at (t, stock, r) from a deterministic (fluid) DP under parameters theta."""
    xs = np.linspace(0, stock0, XG)
    rs = np.linspace(PRICES[0], PRICES[-1], RG)
    alpha = theta[2]

    def q_at(x, rr, V):  # x, rr broadcastable arrays -> Q[..., price]
        x, rr = x[..., None], rr[..., None]
        s = np.minimum(rate(theta, PRICES, rr), x)
        x2, r2 = x - s, alpha * rr + (1 - alpha) * PRICES
        fx = np.clip(x2 / xs[-1] * (XG - 1), 0, XG - 1 - 1e-9)
        fr = np.clip((r2 - rs[0]) / (rs[-1] - rs[0]) * (RG - 1), 0, RG - 1 - 1e-9)
        i, j = fx.astype(int), fr.astype(int)
        a, b = fx - i, fr - j
        v = ((1 - a) * (1 - b) * V[i, j] + a * (1 - b) * V[i + 1, j]
             + (1 - a) * b * V[i, j + 1] + a * b * V[i + 1, j + 1])
        return PRICES * s + v

    V = np.zeros((XG, RG))
    X, R = np.meshgrid(xs, rs, indexing="ij")
    for _ in range(T - 1, t, -1):
        V = q_at(X, R, V).max(axis=-1)
    return q_at(np.array(float(stock)), np.array(float(r)), V)


class Belief:
    """Importance-weighted prior particles over (M, beta, alpha, gg, gl); exact Bayes up to MC error."""

    def __init__(self, n=40000, seed=0):
        rng = np.random.default_rng(seed)
        self.th = draw_theta(rng, n)
        self.r = np.full(n, R0)
        self.logw = np.zeros(n)
        self.rng = rng
        self.seen = 0

    def update(self, history):
        for price, sales, stock_after in history[self.seen:]:
            lam = rate(self.th, price, self.r)
            if stock_after > 0:  # uncensored Poisson log-lik (dropping constants)
                self.logw += sales * np.log(lam) - lam
            else:                # sold out: demand >= sales; normal approx to the tail
                z = (sales - 0.5 - lam) / np.sqrt(lam)
                self.logw += np.log(np.clip(0.5 * (1 - np.tanh(0.8 * z)), 1e-12, 1))
            self.r = self.th[2] * self.r + (1 - self.th[2]) * price
        self.seen = len(history)

    def weights(self):
        w = np.exp(self.logw - self.logw.max())
        return w / w.sum()

    def sample(self, k):
        idx = self.rng.choice(len(self.logw), size=k, p=self.weights())
        return [(tuple(a[i] for a in self.th), self.r[i]) for i in idx]

    def mean(self):
        w = self.weights()
        return tuple(float(w @ a) for a in self.th), float(w @ self.r)


# ---------- policies: f(obs, stock0) -> price ----------

def fixed(price):
    return lambda obs, stock0: price


class Analyst:
    """What a competent-but-myopic analyst does: regress log sales on price, pick the
    price that maximises today's revenue, paced so stock lasts the season."""
    probe = [100.0, 80.0, 60.0, 100.0, 80.0, 60.0]

    def __call__(self, obs, stock0):
        h, t = obs["history"], obs["day"]
        if t < len(self.probe):
            return self.probe[t]
        p = np.array([x[0] for x in h])
        y = np.log(np.array([x[1] for x in h]) + 0.5)
        b, a = np.polyfit(p, y, 1)
        b = min(b, -0.002)
        lam = np.exp(a + b * PRICES)
        pace = obs["stock"] / max(1, obs["days_left"])
        return float(PRICES[np.argmax(PRICES * np.minimum(lam, pace))])


class BeliefMyopic:
    """Correct Bayesian belief over the true model, but still myopic + paced (no lookahead)."""

    def __init__(self):
        self.b = Belief()

    def __call__(self, obs, stock0):
        self.b.update(obs["history"])
        th, r = self.b.mean()
        pace = obs["stock"] / max(1, obs["days_left"])
        return float(PRICES[np.argmax(PRICES * np.minimum(rate(th, PRICES, r), pace))])


class BeliefDP:
    """Reference: Bayesian belief + lookahead. Average fluid-DP Q-values over K posterior samples."""

    def __init__(self, k=6):
        self.b, self.k = Belief(), k

    def __call__(self, obs, stock0):
        self.b.update(obs["history"])
        q = sum(fluid_dp_q(th, obs["day"], obs["stock"], r, stock0) for th, r in self.b.sample(self.k))
        return float(PRICES[np.argmax(q)])
