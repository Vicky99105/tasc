"""Langfuse instrumentation. Optional — the whole system runs without it; traces
are for understanding a run, not for producing it. LangGraph is LangChain-
compatible, so one CallbackHandler instruments every node in the graph as a
span, and every model call inside a node as a generation with its prompt,
response, token counts and cost.

What the traces are for here: proving the architecture claim. A trace of a
session should show ~1 call per candidate (already made, once, before the
session), 1 per rubric compile, 1 per brief render, and nothing at all under
run_match. That absence is the whole argument this system makes, and a trace is
how it's shown rather than asserted.
"""
from __future__ import annotations

import os

from engine.config import Config


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
