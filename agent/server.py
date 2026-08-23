"""Minimal FastAPI + SSE server. One page, one chat, streamed tool calls and
permission gates — no conversation/results split, per the product direction: the
recruiter types free text at every step, including the two approval gates, and
intent classification (agent/intent.py) routes it. Nothing is permitted before a
role is resolved.

create_app() takes its dependencies as arguments so it can be built with fakes in
tests — the module-level `app` (built by build_default_app(), called only when
this file runs as __main__) is the one real, network-touching instance.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langgraph.types import Command
from pydantic import BaseModel

from agent.intent import classify_decision, classify_role, resume_payload
from agent.observability import build_run_config, wrap_llm
from engine.config import Config
from engine.ingest import Role

GATE_DECISIONS = {
    "approve_rubric": ["approve", "amend", "ask", "reject"],
    "approve_summary": ["approve", "edit", "change_rubric", "ask", "reject"],
}

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    session_id: str
    message: str


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


def _summarize_node(node: str, update: dict) -> dict:
    if node == "compile_rubric":
        return {"diff_en": update.get("diff_en"), "est_cost": update.get("est_cost")}
    if node == "run_match":
        return {
            "finalists": len(update.get("results", [])),
            "excluded": len(update.get("excluded", [])),
            "fallback": update.get("fallback", False),
        }
    if node == "render_brief":
        return {"candidates_briefed": len(update.get("briefs", []))}
    if node == "answer_question":
        return {"answer": update.get("answer")}
    if node == "send_to_slack":
        return {"slack_response": update.get("slack_ts")}
    return {}


def create_app(llm, graph, roles: list[Role], app_cfg: Config | None = None) -> FastAPI:
    app = FastAPI()
    sessions: dict[str, dict] = {}  # session_id -> {"role_id": str|None, "awaiting_gate": str|None}

    def stream_graph(session_id: str, session: dict, stream_input):
        cfg = {"configurable": {"thread_id": f"web-{session_id}-{session['role_id']}"}}
        if app_cfg is not None:
            # assessment_key isn't known until compile_rubric runs inside this
            # very call, so it can't be a config-time tag here; role_id and
            # taxonomy_version are enough to find a session in the Langfuse UI.
            cfg = {**cfg, **build_run_config(app_cfg, session_id, role_id=session.get("role_id"))}
        for chunk in graph.stream(stream_input, config=cfg, stream_mode="updates"):
            if "__interrupt__" in chunk:
                payload = chunk["__interrupt__"][0].value
                session["awaiting_gate"] = payload["gate"]
                yield _sse({"type": "interrupt", "gate": payload["gate"], "payload": payload})
                return
            for node, update in chunk.items():
                summary = _summarize_node(node, update)
                yield _sse({"type": "tool_call", "node": node, "summary": summary})
                if node == "answer_question" and summary.get("answer"):
                    yield _sse({"type": "assistant_text", "text": summary["answer"]})
                if node == "send_to_slack":
                    yield _sse({"type": "assistant_text", "text": "Posted to Slack."})
        session["awaiting_gate"] = None
        yield _sse({"type": "done"})

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/session")
    def new_session():
        sid = uuid.uuid4().hex
        sessions[sid] = {"role_id": None, "awaiting_gate": None}
        return {"session_id": sid}

    @app.post("/chat")
    def chat(req: ChatRequest):
        session = sessions.setdefault(req.session_id, {"role_id": None, "awaiting_gate": None})

        def gen():
            if session["role_id"] is None:
                role_id = classify_role(llm, req.message, roles)
                if role_id is None:
                    names = ", ".join(f"{r.role_id} ({r.title})" for r in roles)
                    yield _sse({"type": "assistant_text", "text": f"I couldn't tell which role you mean. Pick one: {names}"})
                    yield _sse({"type": "done"})
                    return
                session["role_id"] = role_id
                role = next(r for r in roles if r.role_id == role_id)
                yield _sse({"type": "assistant_text", "text": f"Starting a match for {role_id} — {role.title}."})
                yield from stream_graph(req.session_id, session, {"role_id": role_id})
                return

            if session["awaiting_gate"] is None:
                yield _sse({"type": "assistant_text", "text": "Nothing is waiting for a decision right now."})
                yield _sse({"type": "done"})
                return

            allowed = GATE_DECISIONS[session["awaiting_gate"]]
            decision = classify_decision(llm, req.message, allowed)
            resume = resume_payload(decision)
            yield from stream_graph(req.session_id, session, Command(resume=resume))

        return StreamingResponse(gen(), media_type="text/event-stream")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    return app


def build_default_app() -> FastAPI:
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    from agent.graph import build_graph
    from agent.slack import DefaultSlackClient
    from engine.config import load_config
    from engine.extract import load_links
    from engine.ingest import load_candidates, load_roles
    from engine.llm import DefaultLLMClient
    from engine.taxonomy import load_taxonomy

    cfg = load_config(".env")
    result = load_candidates("data/candidate_profiles.csv", cfg.reference_date)
    roles = load_roles("data/open_roles.csv")
    taxonomy = load_taxonomy("data/taxonomy.json")
    links = {r.candidate_id: r for r in load_links("data/links.json")}
    llm = wrap_llm(cfg, DefaultLLMClient(cfg.openrouter_api_key, cfg.openrouter_base_url, cfg.model_link))
    slack = DefaultSlackClient(cfg.slack_webhook_url or "")
    conn = sqlite3.connect("data/app.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph(llm, taxonomy, result.candidates, links, slack, checkpointer)
    return create_app(llm, graph, roles, app_cfg=cfg)


if __name__ == "__main__":
    # Run as `python -m agent.server`, not `uvicorn agent.server:app` — the real
    # app (real API key, real Slack client, real SQLite checkpointer) is built
    # here rather than at import time, so importing this module for tests never
    # touches the network.
    import uvicorn

    uvicorn.run(build_default_app(), host="127.0.0.1", port=8000)
