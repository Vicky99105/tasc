"""One POST to an incoming webhook, after approval only. The webhook URL is bound
to one channel at creation, so there is no channel-selection code and no way to
post somewhere unintended. Never logged, never printed.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Protocol


class SlackClient(Protocol):
    def post(self, text: str) -> str: ...


class DefaultSlackClient:
    def __init__(self, webhook_url: str):
        self._webhook_url = webhook_url

    def post(self, text: str) -> str:
        body = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            self._webhook_url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            status = resp.status
            reply = resp.read().decode()
        if status != 200:
            raise RuntimeError(f"slack post failed: {status} {reply}")
        return reply  # incoming webhooks return "ok", not a real message ts


class FakeSlackClient:
    def __init__(self):
        self.posts: list[str] = []

    def post(self, text: str) -> str:
        self.posts.append(text)
        return "ok"
