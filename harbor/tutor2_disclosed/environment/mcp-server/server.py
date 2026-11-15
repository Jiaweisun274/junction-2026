"""MCP facade for the hidden-state tutor2 simulator.

One run = several children (AFTERQUERY_CHILDREN, default 6), tutored one after another, each for one week
(30 turns, exam next Monday). The reward pools all children:
    (sum agent exams - sum no-teaching exams) / (sum full-information exams - sum no-teaching exams)
Pooling several children per run makes the comparison with the reference far less noisy across 3 judge runs.
"""
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse

import lit_ref
import ppo_policy
import sweep_tutor2
import tutor2 as sim

N_CHILDREN = int(os.environ.get("AFTERQUERY_CHILDREN", "6"))


def _child_seed(i):
    """Public seed per child, optionally mixed with a secret salt that lives only in this container's environment.
    With a salt, an agent that downloads the public simulator still cannot reproduce the hidden children."""
    seed = os.environ.get("AFTERQUERY_SEED", "2001")
    salt = os.environ.get("AFTERQUERY_SEED_SALT", "")
    if not salt and N_CHILDREN == 1:
        return seed
    return str(int(hashlib.sha256(f"{salt}:{seed}:{i}".encode()).hexdigest(), 16) % 1_000_000_000)


SEEDS = [_child_seed(i) for i in range(N_CHILDREN)]
DAY_NAMES = ["Monday", "Wednesday", "Friday"]
CHILD = 0
GAME = sim.Sim(SEEDS[0])
EXAMS = []
TURNS_USED = []
LOCK = threading.RLock()
FINISHED = False
RESULT_PATH = Path(os.environ.get("AFTERQUERY_RESULT_PATH", "/grader/result.json"))
RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULT_PATH.write_text(json.dumps({"completed": False}), encoding="utf-8")

mcp = FastMCP("AfterQuery Fractions Tutor", host="0.0.0.0", port=8000)


def _status():
    turn = GAME.turn
    session = min(turn // sim.TURNS_PER, len(DAY_NAMES) - 1)
    return {
        "child": CHILD + 1,
        "children_total": N_CHILDREN,
        "turn": turn,
        "turns_total": sim.T,
        "turns_remaining": max(0, sim.T - turn),
        "session": session + 1,
        "session_name": DAY_NAMES[session],
        "turn_in_session": min(turn % sim.TURNS_PER + 1, sim.TURNS_PER),
        "finished": FINISHED,
    }


def _check_k(k):
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= sim.K:
        raise ValueError("K must be an integer from 1 to 6")
    return k - 1


def _end_child():
    """Run this child's exam, then move to the next child (or end the run after the last one)."""
    global CHILD, GAME, FINISHED
    exam = GAME.exam()
    EXAMS.append(exam)
    TURNS_USED.append(GAME.turn)
    msg = f"Child {CHILD + 1} of {N_CHILDREN}: course over after {GAME.turn} turns, exam score {round(exam * 100)}%."
    if CHILD + 1 < N_CHILDREN:
        CHILD += 1
        GAME = sim.Sim(SEEDS[CHILD])
        msg += f" Now starting child {CHILD + 1} of {N_CHILDREN}: a new child, Monday, turn 1."
    else:
        FINISHED = True
        _persist_result(_result())
        msg += " That was the last child: the run is over."
    return msg


def _after_turn(payload):
    """A child's course ends by itself after the 30th turn, so forgetting to call finish costs nothing."""
    if GAME.turn >= sim.T and not FINISHED:
        payload["message"] += " " + _end_child()
        payload.update(_status())
    return payload


def _closed():
    return {"message": "The run is over.", **_status()} if FINISHED else None


@mcp.tool()
def status() -> dict:
    """Get the current child, session, turn, and remaining turns."""
    with LOCK:
        return _status()


@mcp.tool()
def explain(K: int) -> dict:
    """Explain concept K (1-6) to the current child. This uses one turn."""
    with LOCK:
        k = _check_k(K)
        if _closed():
            return _closed()
        GAME.explain(k)
        return _after_turn({"message": f"You explain concept {K}. The child listens.", **_status()})


@mcp.tool()
def quiz(K: int) -> dict:
    """Quiz the current child on concept K (1-6), observe the answer, and give feedback. This uses one turn."""
    with LOCK:
        k = _check_k(K)
        if _closed():
            return _closed()
        ok, question, answer, right = GAME.quiz(k)
        return _after_turn({
            "question": question,
            "answer": answer,
            "correct": bool(ok),
            "right_answer": right,
            "message": f'Quiz {K}: "{question}" Child answers "{answer}" ({"correct" if ok else "wrong; the right answer is " + right}).',
            **_status(),
        })


@mcp.tool()
def correct(K: int, BUG: Literal["bigger_denominator", "add_to_top_and_bottom", "add_across"]) -> dict:
    """Address a named misconception on concept K (1-6) with the current child. This uses one turn."""
    with LOCK:
        k = _check_k(K)
        if BUG not in sim.BUG_DESC:
            raise ValueError("BUG must be one of the listed misconception names")
        if _closed():
            return _closed()
        GAME.correct(k, BUG)
        return _after_turn({"message": f"You work through {BUG} on concept {K}.", **_status()})


@mcp.tool()
def ask(K: int) -> dict:
    """Ask whether the current child understands concept K (1-6). This is free."""
    with LOCK:
        k = _check_k(K)
        if _closed():
            return _closed()
        answer = GAME.ask(k)
        return {"answer": answer, "message": f'Child: "{answer}" (free).', **_status()}


@mcp.tool()
def finish() -> dict:
    """End the current child's course now and run their exam; unused turns are lost. Moves on to the next child."""
    with LOCK:
        if _closed():
            return _closed()
        return {"message": _end_child(), **_status()}


def _result():
    base = [sim.Sim(s).exam() for s in SEEDS]
    full = [sweep_tutor2.oracle_exam(s) for s in SEEDS]
    ref = [ppo_policy.run(s).exam() for s in SEEDS]             # reference = trained PPO policy (observation-only)
    lit = [sim.run_policy(s, lit_ref.lit_step, obs=True).exam() for s in SEEDS]
    denom = sum(full) - sum(base)
    pooled = lambda ex: (sum(ex) - sum(base)) / denom if denom else 0.0
    return {
        "completed": True,
        "children": N_CHILDREN,
        "reward": float(pooled(EXAMS)),
        "exam_score": float(sum(EXAMS) / N_CHILDREN),
        "no_teaching_score": float(sum(base) / N_CHILDREN),
        "reference_exam_score": float(sum(ref) / N_CHILDREN),
        "reference_reward": float(pooled(ref)),
        "literature_exam_score": float(sum(lit) / N_CHILDREN),
        "literature_reward": float(pooled(lit)),
        "full_info_score": float(sum(full) / N_CHILDREN),
        "per_child_exam": [float(e) for e in EXAMS],
        "turns_used": int(sum(TURNS_USED)),
    }


def _persist_result(payload):
    temporary = RESULT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(RESULT_PATH)


@mcp.custom_route("/health", methods=["GET"])
async def health(_request):
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
