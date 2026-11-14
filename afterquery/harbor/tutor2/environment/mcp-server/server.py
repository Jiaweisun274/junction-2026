"""MCP facade for the hidden-state tutor2 simulator."""
import os
import threading
from typing import Literal
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse

import lit_ref
import sweep_tutor2
import tutor2 as sim


def _instance_seed():
    """Public seed, optionally mixed with a secret salt that lives only in this container's environment.
    With a salt, an agent that downloads the public simulator still cannot reproduce the hidden child."""
    import hashlib
    seed = os.environ.get("AFTERQUERY_SEED", "2001")
    salt = os.environ.get("AFTERQUERY_SEED_SALT", "")
    if not salt:
        return seed
    return str(int(hashlib.sha256(f"{salt}:{seed}".encode()).hexdigest(), 16) % 1_000_000_000)


SEED = _instance_seed()
DAY_NAMES = ["Monday", "Wednesday", "Friday"]
GAME = sim.Sim(SEED)
LOCK = threading.RLock()
FINISHED = False
EXAM = None
RESULT_PATH = Path(os.environ.get("AFTERQUERY_RESULT_PATH", "/grader/result.json"))
RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULT_PATH.write_text(json.dumps({"completed": False}), encoding="utf-8")

mcp = FastMCP("AfterQuery Fractions Tutor", host="0.0.0.0", port=8000)


def _status():
    turn = GAME.turn
    session = min(turn // sim.TURNS_PER, len(DAY_NAMES) - 1)
    return {
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


@mcp.tool()
def status() -> dict:
    """Get the current session, turn, and remaining turns."""
    with LOCK:
        return _status()


@mcp.tool()
def explain(K: int) -> dict:
    """Explain concept K (1-6). This uses one turn."""
    with LOCK:
        k = _check_k(K)
        if FINISHED:
            return {"message": "The course is over.", **_status()}
        if GAME.turn >= sim.T:
            return {"message": "All turns are used. Call finish.", **_status()}
        GAME.explain(k)
        return {"message": f"You explain concept {K}. The child listens.", **_status()}


@mcp.tool()
def quiz(K: int) -> dict:
    """Quiz concept K (1-6), observe the answer, and give feedback. This uses one turn."""
    with LOCK:
        k = _check_k(K)
        if FINISHED:
            return {"message": "The course is over.", **_status()}
        if GAME.turn >= sim.T:
            return {"message": "No turns left. Call finish.", **_status()}
        ok, question, answer, right = GAME.quiz(k)
        return {
            "question": question,
            "answer": answer,
            "correct": bool(ok),
            "right_answer": right,
            "message": f'Quiz {K}: "{question}" Child answers "{answer}" ({"correct" if ok else "wrong; the right answer is " + right}).',
            **_status(),
        }


@mcp.tool()
def correct(K: int, BUG: Literal["bigger_denominator", "add_to_top_and_bottom", "add_across"]) -> dict:
    """Address a named misconception on concept K (1-6). This uses one turn."""
    with LOCK:
        k = _check_k(K)
        if BUG not in sim.BUG_DESC:
            raise ValueError("BUG must be one of the listed misconception names")
        if FINISHED:
            return {"message": "The course is over.", **_status()}
        if GAME.turn >= sim.T:
            return {"message": "No turns left. Call finish.", **_status()}
        GAME.correct(k, BUG)
        return {"message": f"You work through {BUG} on concept {K}.", **_status()}


@mcp.tool()
def ask(K: int) -> dict:
    """Ask whether the child understands concept K (1-6). This is free."""
    with LOCK:
        k = _check_k(K)
        if FINISHED:
            return {"message": "The course is over.", **_status()}
        answer = GAME.ask(k)
        return {"answer": answer, "message": f'Child: "{answer}" (free).', **_status()}


@mcp.tool()
def finish() -> dict:
    """End the course and run the exam. Unused turns are lost."""
    global FINISHED, EXAM
    with LOCK:
        if not FINISHED:
            EXAM = GAME.exam()
            FINISHED = True
            _persist_result(_result())
        return {
            "message": f"FINISHED after {GAME.turn} turns. Exam score: {round(EXAM * 100)}%.",
            "exam_score": EXAM,
            **_status(),
        }


def _result():
    if not FINISHED:
        return None
    baseline = sim.Sim(SEED).exam()
    oracle = sweep_tutor2.oracle_exam(SEED)
    reference_exam = sim.run_policy(SEED, lit_ref.lit_step, obs=True).exam()
    reward = (EXAM - baseline) / (oracle - baseline) if oracle != baseline else 0.0
    reference_reward = (reference_exam - baseline) / (oracle - baseline) if oracle != baseline else 0.0
    return {
        "completed": True,
        "reward": float(reward),
        "exam_score": float(EXAM),
        "no_teaching_score": float(baseline),
        "reference_exam_score": float(reference_exam),
        "reference_reward": float(reference_reward),
        "oracle_score": float(oracle),
        "turns_used": int(GAME.turn),
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
