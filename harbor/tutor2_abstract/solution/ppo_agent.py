"""Reference run (abstract skin): the trained PPO policy playing through the same MCP tools as the agent.
Tool names and outputs are mapped back to the policy's own vocabulary with skin.py."""
import asyncio
import json
import os
import sys

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppo_policy as P
import tutor2 as sim
import skin


def decode(result):
    structured = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    for item in getattr(result, "content", []):
        if getattr(item, "type", None) == "text":
            try:
                return json.loads(item.text)
            except (TypeError, json.JSONDecodeError):
                pass
    raise RuntimeError("MCP tool returned no structured result")


async def main():
    url = os.environ.get("AFTERQUERY_MCP_URL", "http://127.0.0.1:8000/mcp")
    policy, tracker = P.Policy(), P.Tracker()
    async with streamablehttp_client(url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as client:
            await client.initialize()
            child = None
            while True:
                st = decode(await client.call_tool("status", {}))
                if st.get("finished"):
                    break
                if st.get("board") != child:            # a new board: start a fresh observation record
                    child, tracker = st.get("board"), P.Tracker()
                turn = int(st["step"])
                kind, k, bug = a = P.decode(policy.act(tracker.obs(turn)))
                if kind == "explain":
                    await client.call_tool("charge", {"K": k + 1}); tracker.update(turn, a)
                elif kind == "correct":
                    await client.call_tool("repair", {"K": k + 1, "FAULT": skin.FAULTS[bug]}); tracker.update(turn, a)
                else:
                    r = decode(await client.call_tool("probe", {"K": k + 1}))
                    tracker.update(turn, a, (bool(r["correct"]), skin.ANS_BACK[k][r["output"]]))
            print(json.dumps(decode(await client.call_tool("status", {})), sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
