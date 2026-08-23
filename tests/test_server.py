import json

from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from agent.graph import build_graph
from agent.server import create_app
from agent.slack import FakeSlackClient
from engine.extract import LinkResult, Mention
from engine.ingest import Candidate, Role
from engine.llm import FakeLLMClient
from engine.taxonomy import Taxonomy, Term


def _roles() -> list[Role]:
    return [Role("R008", "DevOps Engineer", "Eng", ("Docker",), (), 1, 3, "Senior", "Riyadh")]


def _tax() -> Taxonomy:
    terms = (Term(id="DOCKER", pillar="skill", label="Docker", surfaces=("Docker",), from_requirements=("Docker",)),)
    return Taxonomy(version="t", terms=terms, edges=())


def _candidate() -> Candidate:
    return Candidate(
        candidate_id="C001", headline="DevOps engineer", skills_raw="Docker",
        experience_years_stated="2", experience_years_stated_numeric=2.0,
        past_roles=(), certifications="-", education="-", projects="-",
        extra_curriculars="-", city="Riyadh", country="Saudi Arabia",
        notice_period_days=0, notice_period_raw="Immediate",
    )


_COMPILE = {
    "weights": {"required": 80, "preferred": 10, "availability": 5, "location": 5},
    "threshold": 56.0, "retier": [], "diff_en": "base rubric", "unsupported": [],
}
_BRIEF = {"briefs": [{"candidate_id": "C001", "summary": "s", "differentiator": "d", "questions": []}]}


def _client(llm_responses):
    llm = FakeLLMClient(llm_responses)
    slack = FakeSlackClient()
    graph = build_graph(llm, _tax(), [_candidate()], {"C001": LinkResult("C001", (Mention("C001", "DOCKER", "skills", "Docker", "met"),), ())}, slack, MemorySaver())
    app = create_app(llm, graph, _roles())
    return TestClient(app), llm, slack


def _read_events(resp):
    events = []
    for line in resp.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


class TestSession:
    def test_creates_a_session_id(self):
        client, _, _ = _client([])
        r = client.post("/session")
        assert "session_id" in r.json()


class TestRoleGatekeeping:
    def test_unclear_message_before_role_asks_for_clarification_not_graph_run(self):
        client, llm, _ = _client([{"role_id": "unclear"}])
        sid = client.post("/session").json()["session_id"]
        resp = client.post("/chat", json={"session_id": sid, "message": "hello"})
        events = _read_events(resp)
        assert any("couldn't tell" in e.get("text", "") for e in events if e["type"] == "assistant_text")
        assert llm.usage.calls == 1  # only the role classification, nothing else ran

    def test_resolved_role_starts_the_graph_and_reaches_first_interrupt(self):
        client, llm, slack = _client([{"role_id": "R008"}, dict(_COMPILE)])
        sid = client.post("/session").json()["session_id"]
        resp = client.post("/chat", json={"session_id": sid, "message": "devops role please"})
        events = _read_events(resp)
        assert any(e["type"] == "interrupt" and e["gate"] == "approve_rubric" for e in events)
        assert slack.posts == []


class TestFullFlowThroughChat:
    def test_approve_approve_reaches_slack(self):
        client, llm, slack = _client([
            {"role_id": "R008"}, dict(_COMPILE),
            {"decision": "approve", "detail": ""}, dict(_BRIEF),
            {"decision": "approve", "detail": ""},
        ])
        sid = client.post("/session").json()["session_id"]
        _read_events(client.post("/chat", json={"session_id": sid, "message": "devops"}))
        events2 = _read_events(client.post("/chat", json={"session_id": sid, "message": "looks good, approve"}))
        assert any(e["type"] == "interrupt" and e["gate"] == "approve_summary" for e in events2)
        events3 = _read_events(client.post("/chat", json={"session_id": sid, "message": "send it"}))
        assert any(e["type"] == "assistant_text" and "Posted to Slack" in e["text"] for e in events3)
        assert len(slack.posts) == 1

    def test_reject_never_reaches_slack(self):
        client, llm, slack = _client([
            {"role_id": "R008"}, dict(_COMPILE), {"decision": "reject", "detail": ""},
        ])
        sid = client.post("/session").json()["session_id"]
        _read_events(client.post("/chat", json={"session_id": sid, "message": "devops"}))
        events = _read_events(client.post("/chat", json={"session_id": sid, "message": "no, not this rubric"}))
        assert any(e["type"] == "done" for e in events)
        assert not any(e["type"] == "interrupt" for e in events)
        assert slack.posts == []
