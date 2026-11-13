"""Run an LLM agent on the tutoring environment (tutor2: one child, three sessions, exam next Monday).

The game runs in-process; the model only sees the rules and the tools' text results.
Backends:
  --backend anthropic  official Anthropic SDK (reads ANTHROPIC_API_KEY or an `ant auth login` profile)
  --backend openai     any OpenAI-compatible server (vLLM on Verda, other providers): --base-url, OPENAI_API_KEY
Each game writes a JSONL trajectory to runs/<tag>/seed_<seed>.jsonl; a summary table goes to runs/<tag>/summary.tsv.

Example:
  .venv/bin/python run_agent.py --seeds 2001-2003 --model claude-opus-5-5 --tag opus_try
  .venv/bin/python run_agent.py --backend openai --base-url http://<verda-ip>:8000/v1 --model google/gemma-3-27b-it --seeds 2001-2010
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe"))
import tutor2 as t2  # noqa: E402

def load_env_file():
    """Read KEY=VALUE lines from afterquery/.env into the environment (values never printed)."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if v and not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v


load_env_file()
RULES = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe", "rules_tutor2.txt")).read()
DAYS = ["Monday", "Wednesday", "Friday"]
SYSTEM = ("You are playing a decision game through tools. Maximise your score. Use exactly one tool per turn and "
          "read its result before deciding the next action. Call `finish` when the 30 turns are used.\n\n" + RULES)

K_SCHEMA = {"type": "integer", "minimum": 1, "maximum": 6, "description": "concept number 1..6"}
TOOLS = [
    dict(name="explain", description="Teach concept K (1 turn).", params={"K": K_SCHEMA}),
    dict(name="quiz", description="Give one exercise on concept K and see the child's answer (1 turn).", params={"K": K_SCHEMA}),
    dict(name="correct", description="Work through a named misconception on concept K (1 turn).",
         params={"K": K_SCHEMA, "BUG": {"type": "string", "enum": list(t2.BUG_DESC)}}),
    dict(name="ask", description="Ask the child whether they understand concept K (free, no turn).", params={"K": K_SCHEMA}),
    dict(name="finish", description="End the course and run the exam.", params={}),
]


class Game:
    """Same behaviour and wording as the tutor2 CLI, kept in memory."""
    def __init__(self, seed):
        self.s = t2.Sim(str(seed))
        self.done, self.exam = False, None

    def status(self):
        if self.s.turn >= t2.T:
            return "All 30 turns used. Call finish."
        sess = self.s.turn // t2.TURNS_PER
        return (f"Now: session {sess+1} of 3 ({DAYS[sess]}), turn {self.s.turn % t2.TURNS_PER + 1} of {t2.TURNS_PER} "
                f"(turn {self.s.turn+1} of {t2.T} overall).")

    def call(self, name, args):
        if self.done:
            return "The course is over."
        if name == "finish":
            self.exam = self.s.exam(); self.done = True
            return f"FINISHED after {self.s.turn} turns. Exam score: {round(self.exam*100)}%."
        try:
            k = int(args["K"]) - 1
            assert 0 <= k < t2.K
        except (KeyError, ValueError, AssertionError):
            return "Invalid K. Use 1..6. " + self.status()
        if name == "ask":
            return f'Child: "{self.s.ask(k)}" (free). ' + self.status()
        if self.s.turn >= t2.T:
            return "No turns left. Call finish."
        if name == "explain":
            self.s.explain(k); msg = f"You explain concept {k+1}. The child listens."
        elif name == "quiz":
            ok, q, ans, right = self.s.quiz(k)
            msg = f'Quiz {k+1}: "{q}" Child answers "{ans}" ({"correct" if ok else "wrong, the right answer is " + right}).'
        elif name == "correct":
            bug = args.get("BUG")
            if bug not in t2.BUG_DESC:
                return "Invalid BUG. " + self.status()
            self.s.correct(k, bug); msg = f"You work through {bug} on concept {k+1}."
        else:
            return "Unknown tool. " + self.status()
        return msg + " " + self.status()


# ---------------- backends ----------------
def anthropic_loop(game, args, log):
    import anthropic
    client = anthropic.Anthropic()
    tools = [{"name": t["name"], "description": t["description"], "strict": True,
              "input_schema": {"type": "object", "properties": t["params"], "required": list(t["params"]),
                               "additionalProperties": False}} for t in TOOLS]
    messages = [{"role": "user", "content": "Start the game. " + game.status()}]
    usage = {"input": 0, "output": 0, "cache_read": 0}
    for step in range(args.max_steps):
        kw = dict(model=args.model, max_tokens=16000, system=SYSTEM, tools=tools, messages=messages,
                  output_config={"effort": args.effort}, cache_control={"type": "ephemeral"})
        if args.fallbacks:
            resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
        else:
            resp = client.messages.create(**kw)
        u = resp.usage
        usage["input"] += u.input_tokens; usage["output"] += u.output_tokens
        usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
        if resp.stop_reason == "refusal":
            log({"event": "refusal", "step": step}); break
        messages.append({"role": "assistant", "content": resp.content})
        uses = [b for b in resp.content if b.type == "tool_use"]
        text = " ".join(b.text for b in resp.content if b.type == "text")
        if not uses:
            log({"event": "no_tool", "step": step, "text": text})
            if game.done:
                break
            messages.append({"role": "user", "content": "Please continue with a tool call. " + game.status()})
            continue
        results = []
        for b in uses:
            out = game.call(b.name, dict(b.input))
            log({"event": "tool", "step": step, "turn": game.s.turn, "name": b.name, "input": dict(b.input), "result": out, "text": text})
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": out})
        messages.append({"role": "user", "content": results})
        if game.done:
            break
    return usage


def openai_loop(game, args, log):
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "none"))
    tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
              "parameters": {"type": "object", "properties": t["params"], "required": list(t["params"])}}} for t in TOOLS]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Start the game. " + game.status()}]
    usage = {"input": 0, "output": 0, "cache_read": 0}
    for step in range(args.max_steps):
        resp = client.chat.completions.create(model=args.model, messages=messages, tools=tools,
                                              max_completion_tokens=16000)
        if resp.usage:
            usage["input"] += resp.usage.prompt_tokens; usage["output"] += resp.usage.completion_tokens
        m = resp.choices[0].message
        messages.append(m.model_dump(exclude_none=True))
        if not m.tool_calls:
            log({"event": "no_tool", "step": step, "text": m.content})
            if game.done:
                break
            messages.append({"role": "user", "content": "Please continue with a tool call. " + game.status()})
            continue
        for tc in m.tool_calls:
            try:
                a = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                a = {}
            out = game.call(tc.function.name, a)
            log({"event": "tool", "step": step, "turn": game.s.turn, "name": tc.function.name, "input": a, "result": out, "text": m.content})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
        if game.done:
            break
    return usage


def responses_loop(game, args, log):
    """OpenAI Responses API (needed for reasoning + function tools on newer OpenAI models)."""
    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "none"))
    tools = [{"type": "function", "name": t["name"], "description": t["description"],
              "parameters": {"type": "object", "properties": t["params"], "required": list(t["params"]),
                             "additionalProperties": False}, "strict": True} for t in TOOLS]
    usage = {"input": 0, "output": 0, "cache_read": 0}
    nxt, prev = [{"role": "user", "content": "Start the game. " + game.status()}], None
    for step in range(args.max_steps):
        kw = dict(model=args.model, instructions=SYSTEM, input=nxt, tools=tools)
        if args.reasoning:
            kw["reasoning"] = {"effort": args.reasoning}
        if prev:
            kw["previous_response_id"] = prev
        resp = client.responses.create(**kw)
        prev = resp.id
        if resp.usage:
            usage["input"] += resp.usage.input_tokens; usage["output"] += resp.usage.output_tokens
            det = getattr(resp.usage, "input_tokens_details", None)
            usage["cache_read"] += (getattr(det, "cached_tokens", 0) or 0) if det else 0
        calls = [o for o in resp.output if o.type == "function_call"]
        text = resp.output_text or ""
        if not calls:
            log({"event": "no_tool", "step": step, "text": text})
            if game.done:
                break
            nxt = [{"role": "user", "content": "Please continue with a tool call. " + game.status()}]
            continue
        nxt = []
        for c in calls:
            try:
                a = json.loads(c.arguments or "{}")
            except json.JSONDecodeError:
                a = {}
            out = game.call(c.name, a)
            log({"event": "tool", "step": step, "turn": game.s.turn, "name": c.name, "input": a, "result": out, "text": text})
            nxt.append({"type": "function_call_output", "call_id": c.call_id, "output": out})
        if game.done:
            break
    return usage


# ---------------- scoring ----------------
def normalise(seed, exam):
    import lit_ref, sweep_tutor2 as w
    base, orc = t2.Sim(str(seed)).exam(), w.oracle_exam(str(seed))
    ref = w.norm(str(seed), t2.run_policy(str(seed), lit_ref.lit_step, obs=True).exam())
    return (exam - base) / (orc - base), ref


def play(seed, args):
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"seed_{seed}.jsonl")
    game = Game(seed)
    t0 = time.time()
    with open(path, "w") as fh:
        log = lambda d: fh.write(json.dumps(d) + "\n")
        log({"event": "start", "seed": seed, "model": args.model, "backend": args.backend})
        loop = {"anthropic": anthropic_loop, "openai": openai_loop, "responses": responses_loop}[args.backend]
        usage = loop(game, args, log)
        if not game.done:
            game.call("finish", {})
        norm, ref = normalise(seed, game.exam)
        log({"event": "end", "exam": game.exam, "norm": norm, "reference": ref, "usage": usage, "secs": round(time.time() - t0)})
    return dict(seed=seed, exam=round(game.exam, 3), norm=round(norm, 2), reference=round(ref, 2),
                turns=game.s.turn, input_tok=usage["input"], output_tok=usage["output"], cache_read=usage["cache_read"])


def parse_seeds(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["anthropic", "openai", "responses"], default="anthropic")
    ap.add_argument("--reasoning", default="medium", help="OpenAI reasoning effort for --backend responses ('' to omit)")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--no-fallbacks", dest="fallbacks", action="store_false",
                    help="disable server-side refusal fallbacks (on by default for the Anthropic backend)")
    ap.add_argument("--seeds", default="2001")
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"{args.model.replace('/', '_')}_{time.strftime('%m%d_%H%M')}"
    args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", tag)
    seeds = parse_seeds(args.seeds)
    with ThreadPoolExecutor(args.parallel) as ex:
        rows = list(ex.map(lambda sd: play(sd, args), seeds))
    cols = ["seed", "exam", "norm", "reference", "turns", "input_tok", "output_tok", "cache_read"]
    lines = ["\t".join(cols)] + ["\t".join(str(r[c]) for c in cols) for r in rows]
    mean = lambda c: sum(r[c] for r in rows) / len(rows)
    lines.append(f"mean\t{mean('exam'):.3f}\t{mean('norm'):.2f}\t{mean('reference'):.2f}")
    open(os.path.join(args.out, "summary.tsv"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("trajectories:", args.out)


if __name__ == "__main__":
    main()
