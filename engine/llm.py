"""OpenRouter chat-completions client. The only caller of urllib in this codebase.

DefaultLLMClient is the adapter that touches the network; it is proven by using it
for real (P2's taxonomy build, P3's linking), not by mocking urllib. FakeLLMClient
is for tests of code that calls an LLMClient — it records calls and returns
scripted responses, never touches the network.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def cost_usd(self, input_per_million: float, output_per_million: float) -> float:
        return (
            self.prompt_tokens / 1e6 * input_per_million
            + self.completion_tokens / 1e6 * output_per_million
        )


class LLMClient(Protocol):
    def call(self, system: str, user: str, schema: dict, model: str | None = None) -> dict: ...

    @property
    def usage(self) -> Usage: ...


class DefaultLLMClient:
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.0):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._usage = Usage()

    @property
    def usage(self) -> Usage:
        return self._usage

    def call(
        self,
        system: str,
        user: str,
        schema: dict,
        model: str | None = None,
        retries: int = 3,
        max_tokens: int = 32000,
    ) -> dict:
        body = {
            "model": model or self._model,
            "temperature": self._temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "out", "strict": True, "schema": schema},
            },
        }
        last: Exception | None = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    self._base_url + "/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(req, timeout=600) as resp:
                    data = json.loads(resp.read())
                if "error" in data and not data.get("choices"):
                    raise RuntimeError(str(data["error"])[:300])
                usage = data.get("usage") or {}
                self._usage.prompt_tokens += usage.get("prompt_tokens", 0)
                self._usage.completion_tokens += usage.get("completion_tokens", 0)
                self._usage.calls += 1
                return json.loads(data["choices"][0]["message"]["content"])
            except Exception as e:  # retry on any failure: network, timeout, malformed response
                last = e
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"OpenRouter call failed after {retries} attempts: {last}")


class FakeLLMClient:
    def __init__(self, responses: list[dict] | None = None):
        self._responses = list(responses or [])
        self.calls: list[dict] = []
        self._usage = Usage()

    @property
    def usage(self) -> Usage:
        return self._usage

    def call(self, system: str, user: str, schema: dict, model: str | None = None) -> dict:
        self.calls.append({"system": system, "user": user, "schema": schema, "model": model})
        self._usage.calls += 1
        if not self._responses:
            raise RuntimeError("FakeLLMClient: no scripted response left")
        return self._responses.pop(0)
