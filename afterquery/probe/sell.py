"""Game A: sell a used bike. Optimal stopping with an unknown offer distribution.

CLI (state lives in $GAME_DIR/sell_state.json):
  python sell.py new SEED   start a game
  python sell.py state      show the current day, offer and history
  python sell.py accept     sell to today's buyer
  python sell.py reject     turn today's buyer down and wait for tomorrow
Internal (not for players):
  python sell.py bench SEED...   run reference and oracle policies
"""
import json, os, sys
import numpy as np

T = 20          # days
C = 5.0         # cost of keeping the ad up one more day (paid per rejected offer)
SIG = 50.0      # sd of a single offer around the bike's market value
M0, S0 = 300.0, 60.0   # prior on the market value
SALV = 150.0    # bike shop buys it at the end of day 20 if unsold

DIR = os.environ.get("GAME_DIR", ".")
PATH = os.path.join(DIR, "sell_state.json")


def gen(seed):
    r = np.random.default_rng(int(seed))
    mu = r.normal(M0, S0)
    offers = np.maximum(0, r.normal(mu, SIG, T)).round().tolist()
    return float(mu), offers


# ---------- reference: Bayesian DP over the posterior mean ----------
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)
GH_W = GH_W / GH_W.sum()
GRID = np.arange(-200, 900, 2.0)


def post_var(n):
    return 1.0 / (1.0 / S0**2 + n / SIG**2)


def build_bayes():
    # G[t](m): value at start of day t (1..T+1) before seeing day t's offer, posterior mean m after t-1 offers
    G = {T + 1: np.full_like(GRID, SALV)}
    for t in range(T, 0, -1):
        v = post_var(t - 1)
        sd = np.sqrt(v + SIG**2)
        vals = np.zeros_like(GRID)
        for x_std, w in zip(GH_X, GH_W):
            x = GRID + sd * x_std
            m_new = (GRID / v + x / SIG**2) / (1 / v + 1 / SIG**2)
            cont = -C + np.interp(m_new, GRID, G[t + 1])
            vals += w * np.maximum(x, cont)
        G[t] = vals
    return G


def build_oracle(mu):
    g = {T + 1: SALV}
    for t in range(T, 0, -1):
        x = mu + SIG * GH_X
        g[t] = float(np.sum(GH_W * np.maximum(x, -C + g[t + 1])))
    return g


def play(offers, accept_fn):
    for t in range(1, T + 1):
        if accept_fn(t, offers[:t]):
            return offers[t - 1] - C * (t - 1), t
    return SALV - C * T, None


def bench(seeds):
    G = build_bayes()

    def bayes(t, hist):
        n = len(hist)
        v = post_var(n)
        m = (M0 / S0**2 + sum(hist) / SIG**2) * v
        return hist[-1] >= -C + float(np.interp(m, GRID, G[t + 1]))

    out = []
    for s in seeds:
        mu, offers = gen(s)
        g = build_oracle(mu)
        ref, rday = play(offers, bayes)
        orc, oday = play(offers, lambda t, h: h[-1] >= -C + g[t + 1])
        # naive 37% rule: reject first 7, then take first offer above the best seen
        def rule37(t, h):
            return t > 7 and h[-1] > max(h[:7])
        r37, d37 = play(offers, rule37)
        out.append(dict(seed=s, mu=round(mu), ref=ref, ref_day=rday, oracle=orc, oracle_day=oday,
                        rule37=r37, best_offer=max(offers)))
    return out


# ---------- CLI ----------
def load():
    with open(PATH) as f:
        return json.load(f)


def save(s):
    with open(PATH, "w") as f:
        json.dump(s, f)


def show(s):
    if s["done"]:
        print(f"GAME OVER. {s['result']}")
        return
    t = s["day"]
    print(f"Day {t} of {T}. Today's offer: EUR {s['offers'][t-1]:.0f}. "
          f"Ad costs paid so far: EUR {C*(t-1):.0f}.")
    print("Offers so far (day: amount): " + ", ".join(f"{i+1}: {o:.0f}" for i, o in enumerate(s["offers"][:t])))


def main():
    cmd = sys.argv[1]
    if cmd == "bench":
        for row in bench(sys.argv[2:]):
            print(row)
        return
    if cmd == "new":
        mu, offers = gen(sys.argv[2])
        s = dict(seed=sys.argv[2], day=1, offers=offers, done=False, result=None)
        save(s); show(s); return
    s = load()
    if s["done"]:
        show(s); return
    t = s["day"]
    if cmd == "accept":
        net = s["offers"][t - 1] - C * (t - 1)
        s["done"] = True
        s["result"] = f"Sold on day {t} for EUR {s['offers'][t-1]:.0f}. Net = EUR {net:.0f}."
    elif cmd == "reject":
        if t == T:
            net = SALV - C * T
            s["done"] = True
            s["result"] = f"No sale by day {T}. Bike shop pays EUR {SALV:.0f}. Net = EUR {net:.0f}."
        else:
            s["day"] = t + 1
    elif cmd != "state":
        print("unknown command"); return
    save(s); show(s)


if __name__ == "__main__":
    main()
