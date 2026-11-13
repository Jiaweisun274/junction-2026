"""Fit a half-life regression (Settles & Meeder 2016) to the Duolingo learning traces, to calibrate tutor2's
forgetting model: recall p = 2^(-delta/h), log2 h = theta0 + theta1*sqrt(1+n_correct) + theta2*sqrt(1+n_wrong).
Uses the first N rows (the file is in time order across users)."""
import gzip, sys, csv
import numpy as np

path = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
rows = []
with gzip.open(path, "rt") as fh:
    r = csv.reader(fh); next(r)
    for i, row in enumerate(r):
        if i >= N:
            break
        p, delta = float(row[0]), float(row[2]) / 86400.0
        seen, correct = int(row[8]), int(row[9])
        rows.append((p, delta, correct, seen - correct))
a = np.array(rows)
p, d, c, w = a.T
p = np.clip(p, 0.0001, 0.9999)
mask = d > 0.01
h_obs = np.clip(-d / np.log2(p), 1.0 / 96, 274.0)        # same clipping as the paper (15 min .. 9 months)
X = np.column_stack([np.ones(mask.sum()), np.sqrt(1 + c[mask]), np.sqrt(1 + w[mask])])
y = np.log2(h_obs[mask])
theta, *_ = np.linalg.lstsq(X, y, rcond=None)
pred_h = 2 ** (X @ theta)
pred_p = 2 ** (-d[mask] / pred_h)
mae = np.mean(np.abs(pred_p - p[mask]))
print("rows used", int(mask.sum()), "theta", np.round(theta, 3), "MAE on p", round(float(mae), 3))
for nc in [0, 1, 2, 3, 5]:
    h = 2 ** (theta[0] + theta[1] * np.sqrt(1 + nc) + theta[2] * 1.0)
    print(f"correct recalls {nc}: half-life {h:.2f} days")
h = [2 ** (theta[0] + theta[1] * np.sqrt(1 + nc) + theta[2]) for nc in range(6)]
print("multiplier per extra correct recall:", [round(h[i + 1] / h[i], 2) for i in range(5)])
print("median delta (days)", round(float(np.median(d)), 2), " mean p_recall", round(float(p.mean()), 3))
