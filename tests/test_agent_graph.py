import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent.graph import build_graph
from agent.slack import FakeSlackClient
from engine.extract import LinkResult, Mention
from engine.ingest import Candidate, Role
from engine.llm import FakeLLMClient
from engine.taxonomy import Taxonomy, Term


def _role() -> Role:
    return Role(role_id="R008", title="DevOps Engineer", department="Eng",
                required_skills=("Docker",), nice_to_have_skills=(),
                experience_min=1, experience_max=3, seniority="Senior", location="Riyadh")


def _tax() -> Taxonomy:
    terms = (Term(id="DOCKER", pillar="skill", label="Docker", surfaces=("Docker",), from_requirements=("Docker",)),)
    return Taxonomy(version="t", terms=terms, edges=())


def _candidate(cid="C001") -> Candidate:
    return Candidate(
        candidate_id=cid, headline="DevOps engineer", skills_raw="Docker",
        experience_years_stated="2", experience_years_stated_numeric=2.0,
        past_roles=(), certifications="-", education="-", projects="-",
        extra_curriculars="-", city="Riyadh", country="Saudi Arabia",
        notice_period_days=0, notice_period_raw="Immediate",
    )


def _links() -> dict:
    return {"C001": LinkResult("C001", (Mention("C001", "DOCKER", "skills", "Docker", "met"),), ())}


_COMPILE_RESPONSE = {
    "weights": {"required": 80, "preferred": 10, "availability": 5, "location": 5},
    "threshold": 56.0, "retier": [], "diff_en": "base rubric: Docker required", "unsupported": [],
}
_BRIEF_RESPONSE = {"briefs": [
    {"candidate_id": "C001", "summary": "Strong on Docker.", "differentiator": "Only finalist.", "questions": []}
]}


def _build(llm=None, slack=None, checkpointer=None, roles_path="data/open_roles.csv"):
    llm = llm or FakeLLMClient([dict(_COMPILE_RESPONSE), dict(_BRIEF_RESPONSE)])
    slack = slack or FakeSlackClient()
    checkpointer = checkpointer or MemorySaver()
    graph = build_graph(llm, _tax(), [_candidate()], _links(), slack, checkpointer)
    return graph, llm, slack


class TestApproveRubricInterrupt:
    def test_pauses_at_approve_rubric_with_rubric_payload(self):
        graph, llm, slack = _build()
        cfg = {"configurable": {"thread_id": "t1"}}
        result = graph.invoke({"role_id": "R008"}, config=cfg)
        assert "__interrupt__" in result
        payload = result["__interrupt__"][0].value
        assert payload["gate"] == "approve_rubric"
        assert payload["diff_en"] == "base rubric: Docker required"
        assert slack.posts == []  # nothing sent before either interrupt

    def test_reject_at_rubric_gate_ends_without_scoring_or_sending(self):
        graph, llm, slack = _build()
        cfg = {"configurable": {"thread_id": "t2"}}
        graph.invoke({"role_id": "R008"}, config=cfg)
        result = graph.invoke(Command(resume={"decision": "reject"}), config=cfg)
        assert "__interrupt__" not in result
        assert result.get("results") in (None, [])
        assert slack.posts == []

    def test_amend_reenters_compile_rubric_with_guidance_appended(self):
        llm = FakeLLMClient([
            dict(_COMPILE_RESPONSE),
            {**_COMPILE_RESPONSE, "diff_en": "reweighted per guidance"},
            dict(_BRIEF_RESPONSE),
        ])
        graph, llm, slack = _build(llm=llm)
        cfg = {"configurable": {"thread_id": "t3"}}
        graph.invoke({"role_id": "R008"}, config=cfg)
        result = graph.invoke(Command(resume={"decision": "amend", "guidance": "weight availability higher"}), config=cfg)
        assert result["__interrupt__"][0].value["diff_en"] == "reweighted per guidance"
        assert "weight availability higher" in llm.calls[1]["user"]

    def test_ask_at_rubric_gate_routes_to_answer_question_and_back(self):
        llm = FakeLLMClient([dict(_COMPILE_RESPONSE), {"answer": "Docker is required at full weight."}])
        graph, llm, slack = _build(llm=llm)
        cfg = {"configurable": {"thread_id": "t4"}}
        graph.invoke({"role_id": "R008"}, config=cfg)
        result = graph.invoke(Command(resume={"decision": "ask", "question": "why is Docker required?"}), config=cfg)
        # answer_question runs (no score change), then routes back to approve_rubric, pausing again
        assert result["__interrupt__"][0].value["gate"] == "approve_rubric"


class TestApproveSummaryInterrupt:
    def test_approve_then_approve_sends_to_slack(self):
        graph, llm, slack = _build()
        cfg = {"configurable": {"thread_id": "t5"}}
        graph.invoke({"role_id": "R008"}, config=cfg)
        result = graph.invoke(Command(resume={"decision": "approve"}), config=cfg)
        assert result["__interrupt__"][0].value["gate"] == "approve_summary"
        assert slack.posts == []
        final = graph.invoke(Command(resume={"decision": "approve"}), config=cfg)
        assert "__interrupt__" not in final
        assert len(slack.posts) == 1
        assert final["slack_ts"] == "ok"

    def test_send_to_slack_unreachable_without_final_approval(self):
        graph, llm, slack = _build()
        cfg = {"configurable": {"thread_id": "t6"}}
        graph.invoke({"role_id": "R008"}, config=cfg)
        graph.invoke(Command(resume={"decision": "approve"}), config=cfg)
        result = graph.invoke(Command(resume={"decision": "reject"}), config=cfg)
        assert "__interrupt__" not in result
        assert slack.posts == []

    def test_edit_at_summary_gate_reruns_render_brief_only(self):
        llm = FakeLLMClient([
            dict(_COMPILE_RESPONSE), dict(_BRIEF_RESPONSE),
            {"briefs": [{"candidate_id": "C001", "summary": "Edited.", "differentiator": "x", "questions": []}]},
        ])
        graph, llm, slack = _build(llm=llm)
        cfg = {"configurable": {"thread_id": "t7"}}
        graph.invoke({"role_id": "R008"}, config=cfg)
        graph.invoke(Command(resume={"decision": "approve"}), config=cfg)
        result = graph.invoke(Command(resume={"decision": "edit", "note": "shorter please"}), config=cfg)
        assert "Edited." in result["__interrupt__"][0].value["markdown"]
        assert llm.usage.calls == 3  # compile + brief + re-brief, not a re-compile


class TestCrossProcessResume:
    def test_kill_and_restart_resumes_at_the_same_interrupt(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "checkpoints.db")

        conn1 = sqlite3.connect(db_path, check_same_thread=False)
        cp1 = SqliteSaver(conn1)
        graph1, llm1, slack1 = _build(checkpointer=cp1)
        cfg = {"configurable": {"thread_id": "t8"}}
        result1 = graph1.invoke({"role_id": "R008"}, config=cfg)
        assert result1["__interrupt__"][0].value["gate"] == "approve_rubric"
        conn1.close()

        conn2 = sqlite3.connect(db_path, check_same_thread=False)
        cp2 = SqliteSaver(conn2)
        graph2, llm2, slack2 = _build(llm=FakeLLMClient([dict(_BRIEF_RESPONSE)]), checkpointer=cp2)
        state = graph2.get_state(cfg)
        assert state.next == ("approve_rubric",)
        result2 = graph2.invoke(Command(resume={"decision": "approve"}), config=cfg)
        assert result2["__interrupt__"][0].value["gate"] == "approve_summary"
        conn2.close()


class TestAgainstRealData:
    def test_builds_with_real_taxonomy_candidates_and_role(self):
        from engine.config import load_config
        from engine.ingest import load_candidates, load_roles
        from engine.taxonomy import load_taxonomy
        from engine.extract import load_links

        cfg = load_config(".env")
        result = load_candidates("data/candidate_profiles.csv", cfg.reference_date)
        roles = load_roles("data/open_roles.csv")
        tax = load_taxonomy("data/taxonomy.json")
        links = {r.candidate_id: r for r in load_links("data/links.json")}

        llm = FakeLLMClient([{
            "weights": {"required": 80, "preferred": 10, "availability": 5, "location": 5},
            "threshold": 56.0, "retier": [], "diff_en": "base rubric", "unsupported": [],
        }])
        graph = build_graph(llm, tax, result.candidates, links, FakeSlackClient(), MemorySaver())
        cfg2 = {"configurable": {"thread_id": "real1"}}
        out = graph.invoke({"role_id": "R008"}, config=cfg2)
        assert out["__interrupt__"][0].value["role_id"] == "R008"
