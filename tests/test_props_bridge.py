"""The props bridge: best-effort lifecycle reports, never load-bearing."""

from __future__ import annotations

import pytest

from robbie import props_bridge


async def test_no_url_means_no_bridge(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("nothing should be posted without a URL")

    monkeypatch.setattr(props_bridge.httpx, "post", boom)
    await props_bridge.report("", 7, "abc123", "reviewing")


async def test_a_lifecycle_event_reaches_the_board(monkeypatch):
    sent = []

    def fake_post(url, *, json, timeout):
        sent.append((url, json))

        class R:
            def raise_for_status(self):
                pass

        return R()

    monkeypatch.setattr(props_bridge.httpx, "post", fake_post)
    await props_bridge.report("http://host.docker.internal:4021/", 7, "abc123", "posted")
    assert sent == [
        (
            "http://host.docker.internal:4021/api/reviewer",
            {"pr": 7, "head": "abc123", "status": "posted"},
        )
    ]


async def test_a_dead_tunnel_costs_only_freshness(monkeypatch):
    def down(*a, **kw):
        raise OSError("connection refused")

    monkeypatch.setattr(props_bridge.httpx, "post", down)
    await props_bridge.report("http://host.docker.internal:4021", 7, "abc123", "skipped")


async def test_a_made_up_status_is_a_caller_bug():
    with pytest.raises(AssertionError):
        await props_bridge.report("http://x", 7, "abc123", "pondering")


async def test_a_heartbeat_reaches_the_board(monkeypatch):
    sent = []

    def fake_post(url, *, json, timeout):
        sent.append((url, json))

        class R:
            def raise_for_status(self):
                pass

        return R()

    monkeypatch.setattr(props_bridge.httpx, "post", fake_post)
    await props_bridge.heartbeat("http://host.docker.internal:4021", {"at": 1})
    assert sent == [("http://host.docker.internal:4021/api/reviewer/heartbeat", {"at": 1})]


async def test_a_heartbeat_without_a_board_is_a_no_op(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("nothing should be posted without a URL")

    monkeypatch.setattr(props_bridge.httpx, "post", boom)
    await props_bridge.heartbeat("", {"at": 1})


async def test_a_dead_tunnel_drops_the_heartbeat_quietly(monkeypatch):
    def down(*a, **kw):
        raise OSError("connection refused")

    monkeypatch.setattr(props_bridge.httpx, "post", down)
    await props_bridge.heartbeat("http://host.docker.internal:4021", {"at": 1})
