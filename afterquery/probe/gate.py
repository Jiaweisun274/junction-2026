"""Probe 2: ferry boarding gate. Faster-is-slower: throughput is non-monotone in gate crowding,
the shape is unknown, and the gate degrades at an unannounced minute.

Each minute: you call A passengers from the lounge to the gate area -> they join the gate crowd q
-> boarded ~ min(q, Poisson(f(q))) with f(q) = c*q / (1 + (q/qs)^2)  (peak c*qs/2 at q = qs)
-> q decreases by boarded. Departure after T minutes. Score = passengers boarded.

CLI ($GAME_DIR/gate_state.json):  new SEED | state | call N
Internal: bench SEED...
"""
import json, os, sys
import numpy as np

T = 50
P = 800
AMAX = 40
DIR = os.environ.get("GAME_DIR", ".")
PATH = os.path.join(DIR, "gate_state.json")


def params(seed):
    r = np.random.default_rng(int(seed) + 1000)
    qs = r.uniform(12, 24)
    peak = r.uniform(10, 13)
    tc = int(r.integers(20, 36))           # minute the gate degrades (1-based, applies from tc on)
    noise = r.random((T, 64))               # common random numbers for Poisson draws
    return dict(qs=qs, peak=peak, tc=tc, noise=noise)


def phase(p, t):
    if t >= p["tc"]:
        return p["qs"] * 0.5, p["peak"] * 0.8
    return p["qs"], p["peak"]


def f(q, qs, peak):
    c = 2 * peak / qs
    return c * q / (1 + (q / qs) ** 2)


def poisson_from_u(lam, u):
    # inverse-CDF Poisson using a fixed uniform -> deterministic per seed/minute
    k, pk = 0, np.exp(-lam)
    cdf = pk
    while u > cdf and k < 200:
        k += 1
        pk *= lam / k
        cdf += pk
    return k


def step(p, t, q, lounge, a):
    a = max(0, min(int(a), AMAX, lounge))
    lounge -= a
    q += a
    qs, peak = phase(p, t)
    b = min(q, poisson_from_u(f(q, qs, peak), p["noise"][t - 1][0]))
    q -= b
    return q, lounge, a, b


def run(seed, policy):
    p = params(seed)
    q, lounge, total, hist = 0, P, 0, []
    for t in range(1, T + 1):
        a = policy(t, q, lounge, hist, p)
        q0 = q
        q, lounge, a, b = step(p, t, q, lounge, a)
        hist.append((t, q0 + a, b))   # (minute, crowd at gate after calling, boarded)
        total += b
    return total


def oracle(t, q, lounge, hist, p):
    qs, _ = phase(p, t)
    return max(0, round(qs) - q)


def flood(t, q, lounge, hist, p):
    return AMAX


def make_steady(k):
    return lambda t, q, lounge, hist, p: k


def reference(t, q, lounge, hist, p):
    """Uses only observations. Sweep target crowd 4..40 for 12 minutes, then hold the target with
    the best recent throughput (window 10 min); a drop in throughput at the held target triggers a re-sweep."""
    if not hist:
        reference.state = dict(mode="sweep", start=1, targets=list(range(4, 41, 3)))
    st = reference.state
    if st["mode"] == "sweep":
        i = t - st["start"]
        if i < len(st["targets"]):
            return max(0, st["targets"][i] - q)
        st["mode"] = "hold"
    window = [h for h in hist if h[0] >= max(st["start"], t - 12)]
    bins = {}
    for _, crowd, b in window:
        k = int(round(crowd / 3.0)) * 3
        bins.setdefault(k, []).append(b)
    best = max(bins, key=lambda k: np.mean(bins[k]) - 0.3 * abs(k - 12) / 12) if bins else 12
    # change detection: last 4 minutes well below what this target delivered before
    recent = [b for _, crowd, b in hist[-4:]]
    if st["mode"] == "hold" and len(hist) >= st["start"] + 16:
        ref_rate = np.mean([b for _, crowd, b in hist[st["start"] + 11:-4]] or [0])
        if np.mean(recent) < 0.7 * ref_rate:
            reference.state = dict(mode="sweep", start=t, targets=list(range(4, 25, 3)))
            return max(0, 4 - q)
    return max(0, best - q)


def bench(seeds):
    for s in seeds:
        p = params(s)
        print(dict(seed=s, qs=round(p["qs"], 1), peak=round(p["peak"], 1), tc=p["tc"],
                   oracle=run(s, oracle), reference=run(s, reference), flood=run(s, flood),
                   steady10=run(s, make_steady(10)), steady15=run(s, make_steady(15)), nothing=0))


# ---------------- CLI ----------------
def load():
    with open(PATH) as fh:
        return json.load(fh)


def save(s):
    with open(PATH, "w") as fh:
        json.dump(s, fh)


def show(s):
    if s["log"]:
        print("min | called | gate crowd after calling | boarded")
        for t, a, crowd, b in s["log"]:
            print(f"{t:3} | {a:6} | {crowd:24} | {b}")
    print(f"Boarded so far: {s['total']}. Waiting at gate now: {s['q']}. Still in lounge: {s['lounge']}.")
    if s["t"] > T:
        print(f"SHIP DEPARTED. Final score: {s['total']} passengers boarded.")
    else:
        print(f"Next: minute {s['t']} of {T}. Use `call N` (0..{AMAX}).")


def main():
    cmd = sys.argv[1]
    if cmd == "bench":
        bench(sys.argv[2:]); return
    if cmd == "new":
        s = dict(seed=sys.argv[2], t=1, q=0, lounge=P, total=0, log=[])
        save(s); show(s); return
    s = load()
    if cmd == "call" and s["t"] <= T:
        p = params(s["seed"])
        q0 = s["q"]
        q, lounge, a, b = step(p, s["t"], s["q"], s["lounge"], int(sys.argv[2]))
        s["log"].append([s["t"], a, q0 + a, b])
        s.update(q=q, lounge=lounge, total=s["total"] + b, t=s["t"] + 1)
    save(s); show(s)


if __name__ == "__main__":
    main()
