"""Baseline agents for the judges' checks (no model calls).

  -a nop                                         Harbor's built-in do-nothing agent: never calls a tool.
  -a baseline_agents:StopEarlyAgent              calls `finish` for every child straight away (stop early).
  -a baseline_agents:StopAfterMondayAgent        teaches only Monday (explain/quiz loop), then finishes each child.

Run with PYTHONPATH=harbor so Harbor can import this module.
"""
from typing import override

from harbor.agents.base import BaseAgent
from harbor.agents.capabilities import AgentCapabilities
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

_CLIENT = r'''
import asyncio, json, os
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
MODE = os.environ.get("BASELINE_MODE", "stop")
def dec(r):
    s = getattr(r, "structuredContent", None)
    if s is not None: return s
    for c in r.content:
        if getattr(c, "type", "") == "text": return json.loads(c.text)
async def main():
    async with streamablehttp_client(os.environ.get("AFTERQUERY_MCP_URL", "http://127.0.0.1:8000/mcp")) as (r, w, _):
        async with ClientSession(r, w) as c:
            await c.initialize()
            while True:
                st = dec(await c.call_tool("status", {}))
                if st.get("finished"): break
                if MODE == "monday" and st["turn"] < 10:
                    k = 1 + (st["turn"] // 2) % 6
                    await c.call_tool("explain" if st["turn"] % 2 == 0 else "quiz", {"K": k})
                else:
                    await c.call_tool("finish", {})
            print(json.dumps(dec(await c.call_tool("status", {}))))
asyncio.run(main())
'''


class _ScriptAgent(BaseAgent):
    MODE = "stop"
    capabilities = AgentCapabilities(mcp_servers=True)

    @staticmethod
    @override
    def name() -> str:
        return "baseline"

    @override
    def version(self) -> str:
        return "1.0.0"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        return

    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        cmd = f"BASELINE_MODE={self.MODE} python - <<'PYEOF'\n{_CLIENT}\nPYEOF"
        await environment.exec(command=cmd)


class StopEarlyAgent(_ScriptAgent):
    MODE = "stop"


class StopAfterMondayAgent(_ScriptAgent):
    MODE = "monday"
