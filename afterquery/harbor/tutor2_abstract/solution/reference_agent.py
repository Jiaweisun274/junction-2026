"""Play through the MCP interface using the fixed literature reference policy."""
import asyncio
import json
import os
import sys

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lit_ref
import tutor2 as sim


def decode(result):
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    for item in getattr(result, "content", []):
        if getattr(item, "type", None) == "text":
            try:
                return json.loads(item.text)
            except (TypeError, json.JSONDecodeError):
                pass
    raise RuntimeError("MCP tool returned no structured result")


class ObservationClock:
    def __init__(self, turn):
        self.turn = turn

    def now(self):
        session = min(self.turn // sim.TURNS_PER, len(sim.SESSIONS) - 1)
        return sim.SESSIONS[session] + 0.01 * (self.turn % sim.TURNS_PER)


async def main():
    url = os.environ.get("AFTERQUERY_MCP_URL", "http://127.0.0.1:8000/mcp")
    async with streamablehttp_client(url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as client:
            await client.initialize()
            memory = {}
            last_observation = None
            while True:
                status = decode(await client.call_tool("status", {}))
                turn = int(status["turn"])
                if turn >= sim.T:
                    break
                state = ObservationClock(turn)
                action, k, bug = lit_ref.lit_step(state, memory, last_observation)
                if action == "explain":
                    await client.call_tool("explain", {"K": k + 1})
                    last_observation = ("explain", k, None)
                elif action == "quiz":
                    result = decode(await client.call_tool("quiz", {"K": k + 1}))
                    last_observation = ("quiz", k, (bool(result["correct"]), result["answer"]))
                elif action == "correct":
                    await client.call_tool("correct", {"K": k + 1, "BUG": bug})
                    last_observation = ("correct", k, None)
                else:
                    raise RuntimeError(f"Unsupported reference action: {action}")
            final = decode(await client.call_tool("finish", {}))
            print(json.dumps(final, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
