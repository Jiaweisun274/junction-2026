"""Replay agent logs, find each move-on after a lucky guess, and print the reasoning the agent wrote just before it."""
import sys, json, glob
sys.path.insert(0, ".")
import tutor2 as t, replay_traces as RT
out=[]
for p in sorted(glob.glob("../jobs/v4_*/seed_*/*/*/agent/opencode.txt")+glob.glob("../jobs/v5d_*/seed_*/*/*/agent/opencode.txt")):
    tag=p.split("/")[2]; seed=int(p.split("/")[3].split("_")[1])
    seeds=RT.child_seeds(seed); ci=0; s=t.Sim(seeds[0]); buf=[]; prev=None
    for line in open(p):
        try: part=json.loads(line).get("part",{})
        except: continue
        if part.get("type") in ("reasoning","text") and part.get("text"): buf.append(part["text"])
        if part.get("type")!="tool" or part.get("state",{}).get("status")!="completed": continue
        name=part.get("tool","").split("_")[-1]; inp=part["state"].get("input") or {}
        if name=="finish":
            ci+=1; prev=None
            if ci>=6: break
            s=t.Sim(seeds[ci]); buf=[]; continue
        if name not in ("explain","quiz","correct"): continue
        k=int(inp["K"])-1; reason=" ".join(buf)[-900:]; buf=[]
        st_before=s.state[k]
        if prev and prev["lucky"] and k==prev["k"]+1:
            out.append(dict(tag=tag,seed=seed,child=ci+1,turn=s.turn+1,prevk=prev["k"]+1,act=name,k=k+1,reason=reason,prevreason=prev["reason"]))
        if name=="explain": s.explain(k); prev=None
        elif name=="correct": s.correct(k,inp["BUG"]); prev=None
        else:
            ok,q,a,r=s.quiz(k); prev=dict(k=k,lucky=ok and st_before!="mastered",reason=reason)
        if s.turn>=t.T:
            ci+=1; prev=None
            if ci>=6: break
            s=t.Sim(seeds[ci])
print(len(out), "lucky move-ons with reasoning")
good=[o for o in out if len(o["reason"])>80]
print(len(good),"with reasoning text")
for o in good[:14]:
    print("\n==",o["tag"],o["seed"],"child",o["child"],"turn",o["turn"],f"quiz {o['prevk']} right (lucky) -> {o['act']} {o['k']}")
    print("REASONING BEFORE MOVE:", o["reason"].replace("\n"," ")[-600:])
