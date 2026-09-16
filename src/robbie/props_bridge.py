"""Report the review lifecycle to the props board — best-effort, never load-bearing.

The board lives on the box; the daemon reaches it through the Mac's local tunnel
(host.docker.internal:4021 → box:8090), so there is no credential to hold. A dead
tunnel costs nothing but board freshness: every call swallows its failure after a
debug line, and a review never waits on the board hearing about it.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

# what the board's /api/reviewer accepts; anything else it rejects with a 400
STATUSES = ("requested", "queued", "reviewing", "drafted", "posted", "skipped", "clear")


async def report(
    url: str, pr: int, head: str, status: str, verdict: str | None = None
) -> None:
    """One lifecycle event to the board. No URL configured means no bridge.

    `verdict` rides along on `posted` so the board can tell an approval from a
    needs-work — "robbie ok'd this at the current head" is the operator's
    look-at-next list."""
    if not url:
        return
    assert status in STATUSES, status  # a caller bug, not a runtime condition

    def _post() -> None:
        body: dict = {"pr": pr, "head": head, "status": status}
        if verdict:
            body["verdict"] = verdict
        httpx.post(
            f"{url.rstrip('/')}/api/reviewer", json=body, timeout=5
        ).raise_for_status()

    try:
        await asyncio.to_thread(_post)
    except Exception as ex:  # noqa: BLE001 — freshness lost, nothing else
        logger.debug("props bridge: #%s %s not reported: %s", pr, status, ex)


async def heartbeat(url: str, payload: dict) -> None:
    """The panel's one-glance summary, pushed once per tick. Same contract as
    `report`: best-effort, a dead tunnel costs freshness and nothing else."""
    if not url:
        return

    def _post() -> None:
        httpx.post(
            f"{url.rstrip('/')}/api/reviewer/heartbeat", json=payload, timeout=5
        ).raise_for_status()

    try:
        await asyncio.to_thread(_post)
    except Exception as ex:  # noqa: BLE001 — freshness lost, nothing else
        logger.debug("props bridge: heartbeat not delivered: %s", ex)
