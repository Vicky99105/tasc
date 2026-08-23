"""Langfuse instrumentation. Optional — the whole system runs without it; traces
are for understanding a run, not for producing it. LangGraph is LangChain-
compatible, so one CallbackHandler instruments every node in the graph as a
span automatically. Model calls do NOT come for free from that handler, though
— confirmed live: engine/llm.py's DefaultLLMClient makes a raw urllib call, not
a LangChain Runnable, so the callback handler has no visibility inside a node
and every trace showed $0.00 model cost despite real calls happening. wrap_llm()
below closes that gap explicitly, one generation observation per call, nested
under whichever node is currently executing.

What the traces are for here: proving the architecture claim. A trace of a
session should show ~1 call per candidate (already made, once, before the
session), 1 per rubric compile, 1 per brief render, and nothing at all under
run_match. That absence is the whole argument this system makes, and a trace is
how it's shown rather than asserted.
"""
from __future__ import annotations

import os

from engine.config import Config
from engine.llm import LLMClient, Usage


def is_configured(cfg: Config) -> bool:
    return bool(cfg.langfuse_host and cfg.langfuse_public_key and cfg.langfuse_secret_key)


def build_run_config(
    cfg: Config,
    session_id: str,
    role_id: str | None = None,
    taxonomy_version: str | None = None,
    assessment_key: str | None = None,
) -> dict:
    """Extra keys to merge into a graph.invoke()/stream() config dict. Empty
    when Langfuse isn't configured, so callers can always merge it in — no
    branching required at the call site."""
    if not is_configured(cfg):
        return {}

    os.environ["LANGFUSE_HOST"] = cfg.langfuse_host
    os.environ["LANGFUSE_PUBLIC_KEY"] = cfg.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = cfg.langfuse_secret_key

    from langfuse.langchain import CallbackHandler

    tags = []
    if role_id:
        tags.append(f"role:{role_id}")
    if taxonomy_version:
        tags.append(f"taxonomy:{taxonomy_version}")
    if assessment_key:
        tags.append(f"assessment:{assessment_key}")

    return {
        "callbacks": [CallbackHandler()],
        "tags": tags,
        # Langfuse groups every trace sharing a session id in one timeline in
        # the UI, which is what makes the gap between "rubric shown" and
        # "rubric approved" visible across the pause/resume boundary — each
        # interrupt's resume is a separate graph.invoke() call, so they can
        # only be tied together by session, not by one continuous span.
        "metadata": {"langfuse_session_id": session_id},
    }


class _TracedLLMClient:
    """Wraps a real LLMClient so each .call() reports one generation
    observation to Langfuse, nested under whatever node span is currently
    executing (LangGraph's CallbackHandler and this client share the same
    OTEL context, so nesting happens automatically). Delegates everything
    else, including the exact Usage object, to the wrapped client."""

    def __init__(self, inner: LLMClient, model_name: str):
        self._inner = inner
        self._model_name = model_name

    @property
    def usage(self) -> Usage:
        return self._inner.usage

    def call(self, system: str, user: str, schema: dict, model: str | None = None) -> dict:
        from langfuse import get_client

        client = get_client()
        before = self._inner.usage
        prompt_before, completion_before = before.prompt_tokens, before.completion_tokens

        with client.start_as_current_observation(
            name="openrouter_call", as_type="generation",
            model=model or self._model_name,
            input={"system": system, "user": user},
        ) as gen:
            try:
                result = self._inner.call(system, user, schema, model=model)
            except Exception as e:
                gen.update(level="ERROR", status_message=str(e))
                raise
            after = self._inner.usage
            gen.update(
                output=result,
                usage_details={
                    "input": after.prompt_tokens - prompt_before,
                    "output": after.completion_tokens - completion_before,
                },
                cost_details={"total": round(after.cost_usd(0.375, 1.875) - before.cost_usd(0.375, 1.875), 6)},
            )
            return result


def wrap_llm(cfg: Config, llm: LLMClient) -> LLMClient:
    """Returns llm unchanged when Langfuse isn't configured — same
    always-safe-to-call shape as build_run_config()."""
    if not is_configured(cfg):
        return llm
    return _TracedLLMClient(llm, cfg.model_link)
