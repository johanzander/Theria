"""Copy the running add-on's settings and state into a local data directory (used by dev-start.sh).

Reads the add-on through Home Assistant: its ingress address over the websocket API, then
`/api/data-files` behind ingress. Writes nothing to Home Assistant or the add-on.

    python scripts/pull_addon_data.py .dev-data
"""

import asyncio
import json
import os
import sys
from typing import Any

import requests
import websockets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.theria.data_dir import write_data_files

ADDON = os.environ.get("THERIA_ADDON_SLUG", "local_theria")
HEADERS = {
    "User-Agent": "curl/8.4.0"
}  # the proxy in front of Home Assistant blocks python's


async def ingress(url: str, token: str) -> tuple[str, str]:
    """The add-on's ingress path and a session for it."""
    socket = url.replace("http", "ws", 1) + "/api/websocket"
    async with websockets.connect(
        socket, user_agent_header=HEADERS["User-Agent"]
    ) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        if json.loads(await ws.recv())["type"] != "auth_ok":
            raise SystemExit("Home Assistant refused the token in .env")

        async def supervisor(call: int, endpoint: str, method: str) -> dict[str, Any]:
            await ws.send(
                json.dumps(
                    {
                        "id": call,
                        "type": "supervisor/api",
                        "endpoint": endpoint,
                        "method": method,
                    }
                )
            )
            answer = json.loads(await ws.recv())
            if not answer["success"]:
                raise SystemExit(f"supervisor {endpoint}: {answer['error']['message']}")
            result: dict[str, Any] = answer["result"]
            return result

        info = await supervisor(1, f"/addons/{ADDON}/info", "get")
        if info["state"] != "started":
            raise SystemExit(f"the add-on {ADDON} is {info['state']}, not started")
        session = await supervisor(2, "/ingress/session", "post")
        return info["ingress_entry"], session["session"]


def main(target: str) -> None:
    url, token = os.environ["HA_URL"].rstrip("/"), os.environ["HA_TOKEN"]
    entry, session = asyncio.run(ingress(url, token))
    response = requests.get(
        f"{url}{entry}/api/data-files",
        cookies={"ingress_session": session},
        headers={**HEADERS, "Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if response.status_code == 404:
        raise SystemExit("the add-on has no /api/data-files yet: deploy it first")
    response.raise_for_status()
    files = response.json()
    write_data_files(target, files)
    print(f"copied {', '.join(sorted(files))} from {ADDON} to {target}")


if __name__ == "__main__":
    main(sys.argv[1])
