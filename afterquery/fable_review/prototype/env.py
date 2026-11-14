"""Discount Trap: one-season pricing with reference-price memory (prototype).

A shop sells C units over T days. Each day the agent picks a price from a ladder.
Customers remember past prices (reference price r). Paying less than r feels like
a gain, paying more feels like a loss. Demand parameters are hidden; their prior
is public. Only sales (censored by stock) are observed.

    lam_t  = M * exp(-beta*(p_t/100 - 1)) * exp(g(r_t - p_t)/100)
    g(x)   = gamma_gain * x  if x >= 0  else  gamma_loss * x
    D_t    ~ Poisson(lam_t),  sales_t = min(D_t, stock_t)
    r_{t+1}= alpha * r_t + (1 - alpha) * p_t,     r_0 = 100 (public)

Standard library + numpy only.
"""
import math
import numpy as np

T = 50
PRICES = np.arange(20, 121, 10).astype(float)  # 20..120
R0 = 100.0
PRIOR = {  # independent uniforms, all public
    "M": (20.0, 60.0),
    "beta": (1.5, 4.0),
    "alpha": (0.70, 0.97),
    "gg": (0.0, 5.0),
    "gl": (0.0, 8.0),
}
STOCK_RANGE = (1200, 6000)  # drawn per seed, public


def rate(theta, p, r):
    """Mean demand. theta = dict or tuple of arrays (M, beta, alpha, gg, gl)."""
    M, beta, _, gg, gl = theta
    d = (r - p) / 100.0
    return M * np.exp(-beta * (p / 100.0 - 1.0)) * np.exp(np.where(d >= 0, gg * d, gl * d))


def draw_theta(rng, n=None):
    return tuple(rng.uniform(lo, hi, size=n) for lo, hi in PRIOR.values())


def _poisson_quantile(u, lam):
    """Inverse CDF so one uniform per day gives common random numbers across policies."""
    k, pmf = 0, math.exp(-lam)
    cdf = pmf
    while cdf < u and k < 10000:
        k += 1
        pmf *= lam / k
        cdf += pmf
    return k


class Market:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.theta = draw_theta(rng)            # hidden
        self.stock0 = int(rng.integers(*STOCK_RANGE))
        self._u = rng.uniform(size=T)           # hidden demand luck, fixed per seed
        self.t, self.stock, self.r, self.revenue = 0, self.stock0, R0, 0.0
        self.history = []                        # (price, sales, stock_after)

    @property
    def done(self):
        return self.t >= T

    def observe(self):
        """Everything the agent may see."""
        return {"day": self.t, "days_left": T - self.t, "stock": self.stock,
                "history": list(self.history)}

    def step(self, price):
        if self.done:
            raise ValueError("season over")
        if price not in PRICES:
            raise ValueError("price must be on the ladder")  # no state change
        lam = float(rate(self.theta, price, self.r))
        sales = min(_poisson_quantile(self._u[self.t], lam), self.stock)
        self.stock -= sales
        self.revenue += price * sales
        self.r = self.theta[2] * self.r + (1 - self.theta[2]) * price
        self.t += 1
        self.history.append((float(price), int(sales), int(self.stock)))
        return self.observe()
