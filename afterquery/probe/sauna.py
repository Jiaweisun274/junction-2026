"""Game B: the shared sauna of a Finnish apartment building. Opponent modelling.

CLI (state lives in $GAME_DIR/sauna_state.json):
  python sauna.py new SEED    start a game
  python sauna.py state       show the booking log so far
  python sauna.py go SLOT     tonight, go at SLOT (17..21) or 'home'
Internal:
  python sauna.py bench SEED...
"""
import json, os, random, sys
from itertools import permutations

SLOTS = [17, 18, 19, 20, 21]
PREF = {17: 1.0, 18: 2.0, 19: 3.0, 20: 2.0, 21: 1.0}
PEN = 1.5
T = 30
N = 6
SKIP = 0.1
DIR = os.environ.get("GAME_DIR", ".")
PATH = os.path.join(DIR, "sauna_state.json")


def gen(seed):
    r = random.Random(int(seed))
    types = ["fixed", "fixed", "least", "least", "most", "alt"]
    r.shuffle(types)
    nb = []
    for ty in types:
        d = {"type": ty}
        if ty == "fixed":
            d["slot"] = r.choice(SLOTS)
        elif ty == "alt":
            d["a"], d["b"] = r.sample(SLOTS, 2)
        else:
            d["fav"] = r.choice(SLOTS)
        nb.append(d)
    skips = [[r.random() < SKIP for _ in range(N)] for _ in range(T)]
    return nb, skips


def tie(cands, fav):
    return min(cands, key=lambda s: (abs(s - fav), s))


def rule(d, day, prev_counts):
    """Slot a neighbour with routine d picks on day (1-based), given yesterday's counts incl. you."""
    ty = d["type"]
    if ty == "fixed":
        return d["slot"]
    if ty == "alt":
        return d["a"] if day % 2 == 1 else d["b"]
    if prev_counts is None:
        return d["fav"]
    if ty == "least":
        m = min(prev_counts[s] for s in SLOTS)
    else:
        m = max(prev_counts[s] for s in SLOTS)
    return tie([s for s in SLOTS if prev_counts[s] == m], d["fav"])


def counts_of(choices, mine):
    c = {s: 0 for s in SLOTS}
    for x in choices + [mine]:
        if x in c:
            c[x] += 1
    return c


def simulate_day(nb, skips, day, prev_counts):
    return [None if skips[day - 1][i] else rule(d, day, prev_counts) for i, d in enumerate(nb)]


def payoff(slot, nbr_choices):
    if slot == "home":
        return 0.0
    return PREF[slot] - PEN * sum(1 for x in nbr_choices if x == slot)


# ---------- reference: fit each neighbour's routine from the log ----------
HYPS = ([{"type": "fixed", "slot": s} for s in SLOTS]
        + [{"type": "alt", "a": a, "b": b} for a, b in permutations(SLOTS, 2)]
        + [{"type": t, "fav": f} for t in ("least", "most") for f in SLOTS])


def predict_probs(log, day):
    """log: list of (nbr_choices, my_choice). Returns per-neighbour dict slot->prob for `day`."""
    preds = []
    for i in range(N):
        best, best_err = [], None
        for h in HYPS:
            err = 0
            prev = None
            for k, (ch, mine) in enumerate(log):
                if ch[i] is not None and rule(h, k + 1, prev) != ch[i]:
                    err += 1
                prev = counts_of(ch, mine)
            if best_err is None or err < best_err:
                best, best_err = [h], err
            elif err == best_err:
                best.append(h)
        prev = counts_of(*log[-1]) if log else None
        p = {}
        for h in best:
            s = rule(h, day, prev)
            p[s] = p.get(s, 0) + 1.0 / len(best)
        preds.append(p)
    return preds


def best_slot(preds):
    vals = {"home": 0.0}
    for s in SLOTS:
        vals[s] = PREF[s] - PEN * sum((1 - SKIP) * p.get(s, 0) for p in preds)
    return max(vals, key=vals.get)


def run_policy(seed, policy):
    nb, skips = gen(seed)
    log, total, prev = [], 0.0, None
    for day in range(1, T + 1):
        mine = policy(log, day, nb)
        ch = simulate_day(nb, skips, day, prev)
        total += payoff(mine, ch)
        log.append((ch, mine))
        prev = counts_of(ch, mine)
    return total


def ref_policy(log, day, nb):
    return best_slot(predict_probs(log, day))


def oracle_policy(log, day, nb):
    prev = counts_of(*log[-1]) if log else None
    preds = [{rule(d, day, prev): 1.0} for d in nb]
    return best_slot(preds)


def bench(seeds):
    for s in seeds:
        print(dict(seed=s, ref=run_policy(s, ref_policy), oracle=run_policy(s, oracle_policy),
                   always19=run_policy(s, lambda l, d, n: 19),
                   home=0.0))


# ---------- CLI ----------
def load():
    with open(PATH) as f:
        return json.load(f)


def save(s):
    with open(PATH, "w") as f:
        json.dump(s, f)


def show(s):
    names = ["Aino", "Eero", "Kaisa", "Mikko", "Liisa", "Juha"]
    if s["log"]:
        print("Booking log (who sat in which slot; '-' = did not go):")
        print("day | you  | " + " | ".join(f"{n:5}" for n in names) + " | your score")
        for k, e in enumerate(s["log"]):
            row = " | ".join(f"{str(x) if x else '-':5}" for x in e["nbrs"])
            print(f"{k+1:3} | {str(e['mine']):4} | {row} | {e['pay']:+.1f}")
    print(f"Total score so far: {s['total']:+.1f}")
    if s["day"] > T:
        print("GAME OVER.")
    else:
        print(f"Next: evening of day {s['day']} of {T}. Choose a slot 17-21 or 'home'.")


def main():
    cmd = sys.argv[1]
    if cmd == "bench":
        bench(sys.argv[2:]); return
    if cmd == "new":
        s = dict(seed=sys.argv[2], day=1, log=[], total=0.0)
        save(s); show(s); return
    s = load()
    if cmd == "go":
        if s["day"] > T:
            show(s); return
        arg = sys.argv[2]
        mine = "home" if arg == "home" else int(arg)
        if mine != "home" and mine not in SLOTS:
            print("invalid slot"); return
        nb, skips = gen(s["seed"])
        prev = None
        if s["log"]:
            e = s["log"][-1]
            prev = counts_of(e["nbrs"], e["mine"])
        ch = simulate_day(nb, skips, s["day"], prev)
        pay = payoff(mine, ch)
        s["log"].append(dict(nbrs=ch, mine=mine, pay=pay))
        s["total"] += pay
        s["day"] += 1
    save(s); show(s)


if __name__ == "__main__":
    main()
