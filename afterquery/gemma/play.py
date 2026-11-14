"""Gemma plays the tutoring environment as text (same simulator, same rules, same scoring as the Harbor task).

Each turn the model sees the rules plus this child's full history (its own actions and the child's answers) and
answers with one line of reasoning and one ACTION line. Rollouts are batched across many children with vLLM.

  python play.py eval   --out runs/base_eval.json [--lora DIR]          # batches 3001-3015, salt eval1, greedy
  python play.py sample --out runs/r1_sample.json --n-children 128 --k 8 [--lora DIR] [--start 0]

Differences from the Harbor task (disclosed): no `ask` (free self-report; frontier agents rarely used it) and no
early `finish` (every child gets all 30 turns). A reply without a valid ACTION line wastes the turn.
"""
import argparse, hashlib, json, os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim"))
import tutor2 as t            # noqa: E402
import sweep_tutor2 as w      # noqa: E402

MODEL = os.environ.get("GEMMA_MODEL", "google/gemma-4-E4B-it")
DAYS = ["Monday", "Wednesday", "Friday"]
RULES = """You are a private tutor for a 10-year-old child learning fractions. The child gets one week: three sessions (Monday, Wednesday, Friday) of 10 turns, 30 turns in total, followed by an exam next Monday (three days after Friday).

The exam covers six concepts in prerequisite order:
1. Equal parts
2. Comparing fractions (numerator and denominator)
3. Equivalent fractions
4. Common denominators
5. Adding fractions
6. Word problems with fractions

The child's mastery, memory, and misconceptions are hidden. You only see the child's answers.

The exam: all six concepts count equally. On a concept the child has mastered and still remembers on exam day, they score about 90%. On a concept they do not know (never learned, or forgotten), about 25% (guessing). A child who holds a misconception about a concept scores about 10% on it.

Quizzes: a child who knows and remembers concept K answers a quiz on it correctly about 9 times in 10. A child who does not know it still guesses right about 1 time in 4.

How learning works:
- A concept learned in one session can fade within days unless it is reviewed. Forgetting speed varies by child and concept.
- Quizzing strengthens memory more than another explanation. Reviewing after time has passed is more effective than reviewing immediately.
- A concept is very hard to learn until its prerequisite is mastered.
- Re-explaining the same concept again in the same session helps less each time; explaining it again in a later session works as well as the first time.
- Some children develop one of these misconceptions:
  - bigger_denominator: thinks a bigger denominator means a bigger fraction; says 1/8 > 1/4.
  - add_to_top_and_bottom: thinks adding the same number to numerator and denominator keeps a fraction equivalent; says 2/3 = 3/4.
  - add_across: adds numerators and denominators separately; says 1/2 + 1/3 = 2/5.
- A misconception shows up as its typical wrong answer (the examples above), also in the next concept that uses it: a child who adds across will also add across in word problems. Other wrong answers just mean the child does not know or has forgotten.
- Re-explaining rarely fixes a misconception. The targeted correct action is designed for that.
- The child may already know some early concepts, possibly with a misconception.

Actions (each uses one turn):
- explain K: explain concept K. Each explanation has some chance of making it click.
- quiz K: quiz concept K and see the child's answer.
- correct K BUG: work through the named misconception BUG on concept K.

Reply with one short line of reasoning, then a final line of the form
ACTION: explain 3
or ACTION: quiz 2
or ACTION: correct 2 bigger_denominator"""

ACT_RE = re.compile(r"ACTION:\s*(explain|quiz|correct)\s+([1-6])(?:\s+(bigger_denominator|add_to_top_and_bottom|add_across))?", re.I)


def child_seeds(seed, salt="eval1", n=6):
    return [str(int(hashlib.sha256(f"{salt}:{seed}:{i}".encode()).hexdigest(), 16) % 1_000_000_000) for i in range(n)]


class Episode:
    def __init__(self, seed):
        self.seed, self.sim, self.hist, self.turns, self.invalid = seed, t.Sim(seed), [], [], 0

    def prompt(self):
        tn = self.sim.turn
        h = "\n".join(self.hist) if self.hist else "(nothing yet)"
        return (f"{RULES}\n\nHistory with this child:\n{h}\n\nNow: {DAYS[tn // 10]}, turn {tn + 1} of 30 "
                f"({30 - tn} turns left). Choose the next action.")

    def step(self, reply):
        tn, day = self.sim.turn, DAYS[self.sim.turn // 10][:3]
        m = ACT_RE.findall(reply or "")
        if not m:
            self.invalid += 1; self.sim.turn += 1
            self.hist.append(f"{day} turn {tn + 1}: (no valid action, turn lost)"); return None
        a, k, bug = m[-1][0].lower(), int(m[-1][1]) - 1, m[-1][2] or None
        if a == "explain":
            self.sim.explain(k); line = f"explain {k + 1}"
        elif a == "quiz":
            ok, q, ans, right = self.sim.quiz(k)
            line = f'quiz {k + 1} -> child answered "{ans}" ({"correct" if ok else "wrong; right answer " + right})'
        else:
            if bug is None:
                self.invalid += 1; self.sim.turn += 1
                self.hist.append(f"{day} turn {tn + 1}: (correct without a misconception name, turn lost)"); return None
            self.sim.correct(k, bug); line = f"correct {k + 1} {bug}"
        self.hist.append(f"{day} turn {tn + 1}: {line}")
        return (a, k, bug)


def rollout(llm, seeds, temperature, lora=None, max_tokens=160):
    from vllm import SamplingParams
    eps = [Episode(s) for s in seeds]
    sp = SamplingParams(temperature=temperature, top_p=0.95 if temperature > 0 else 1.0, max_tokens=max_tokens)
    for _ in range(t.T):
        live = [e for e in eps if e.sim.turn < t.T]
        if not live:
            break
        msgs = [[{"role": "user", "content": e.prompt()}] for e in live]
        outs = llm.chat(msgs, sp, lora_request=lora, use_tqdm=False)
        for e, m, o in zip(live, msgs, outs):
            reply = o.outputs[0].text
            e.turns.append({"prompt": m[0]["content"], "reply": reply})
            e.step(reply)
    return eps


def make_llm(lora_dir):
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    llm = LLM(MODEL, max_model_len=6144, gpu_memory_utilization=0.85, enable_prefix_caching=True,
              enable_lora=lora_dir is not None, max_lora_rank=32, max_num_seqs=512)
    return llm, (LoRARequest("a", 1, lora_dir) if lora_dir else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "sample"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora")
    ap.add_argument("--n-children", type=int, default=128)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--first", type=int, default=3001)
    ap.add_argument("--last", type=int, default=3015)
    a = ap.parse_args()
    llm, lora = make_llm(a.lora)
    t0 = time.time()
    if a.mode == "eval":
        batches = list(range(a.first, a.last + 1))
        seeds = [s for b in batches for s in child_seeds(b)]
        eps = rollout(llm, seeds, 0.0, lora)
        res, per = [], {}
        for bi, b in enumerate(batches):
            ch = eps[bi * 6:(bi + 1) * 6]
            ex = [e.sim.exam() for e in ch]; base = [t.Sim(e.seed).exam() for e in ch]; full = [w.oracle_exam(e.seed) for e in ch]
            r = (sum(ex) - sum(base)) / (sum(full) - sum(base)); res.append(r); per[b] = r
        acts = [h.split(": ")[1].split()[0] for e in eps for h in e.hist]
        mix = {x: round(acts.count(x) / len(acts), 3) for x in ("explain", "quiz", "correct", "(no")}
        out = {"model": MODEL, "lora": a.lora, "mean_reward": float(np.mean(res)), "per_batch": per, "action_mix": mix,
               "invalid_rate": sum(e.invalid for e in eps) / (len(eps) * t.T), "seconds": time.time() - t0,
               "example": eps[0].hist}
        print(json.dumps({k: v for k, v in out.items() if k != "example"}, indent=1))
    else:
        seeds = [str(20_000_000 + a.start + i) for i in range(a.n_children)]
        eps = rollout(llm, [s for s in seeds for _ in range(a.k)], 1.0, lora)
        out = {"model": MODEL, "lora": a.lora, "episodes": [
            {"seed": e.seed, "gain": e.sim.exam() - t.Sim(e.seed).exam(), "invalid": e.invalid, "turns": e.turns}
            for e in eps], "seconds": time.time() - t0}
        g = [e["gain"] for e in out["episodes"]]
        print(f"sampled {len(g)} episodes, mean gain {np.mean(g):.4f}, invalid {np.mean([e['invalid'] for e in out['episodes']]):.2f}/episode, {out['seconds']:.0f}s")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
