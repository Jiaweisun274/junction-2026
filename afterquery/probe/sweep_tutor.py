import sys, numpy as np
sys.path.insert(0, '.')
import tutor as t
seeds = [str(s) for s in range(1, 41)]
def make(hi, lo, lr, start):
    def pol(s):
        k, b = 0, start
        while s.turn < t.T and k < t.K:
            if b > hi: k, b = k+1, start; continue
            if b < lo:
                s.explain(k); b = b + (1-b)*lr; continue
            ok,_ = s.quiz(k)
            lm, ln = (1-t.SLIP, t.GUESS) if ok else (t.SLIP, 1-t.GUESS)
            b = b*lm/(b*lm+(1-b)*ln)
    return pol
def score(pol):
    out=[]
    for sd in seeds:
        o=t.run(sd,t.oracle); base,orc=o.base,t.exam(o.mastered)
        s=t.run(sd,pol); out.append(t.norm(t.exam(s.mastered),base,orc))
    return np.mean(out)
print('reference', round(score(t.reference),3), 'naive', round(score(t.naive),3))
best=[]
for hi in [0.7,0.8,0.9]:
  for lo in [0.25,0.4,0.55]:
    for lr in [0.3,0.45]:
      for start in [0.3,0.5]:
        best.append((round(score(make(hi,lo,lr,start)),3),hi,lo,lr,start))
best.sort(reverse=True); print(best[:5])
