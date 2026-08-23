import pytest

from agent.slack import FakeSlackClient
from agent.tools import (
    CallBudgetExceeded,
    answer_question_tool,
    check_call_budget,
    compile_rubric_tool,
    get_assessments_tool,
    list_roles_tool,
    render_brief_tool,
    run_match_tool,
    send_to_slack_tool,
)
from engine.extract import LinkResult, Mention
from engine.ingest import Candidate, Role
from engine.llm import FakeLLMClient
from engine.match import MatchScore, RequirementScore
from engine.rubric import compile_rubric
from engine.taxonomy import Taxonomy, Term


def _role(**kw) -> Role:
    defaults = dict(
        role_id="R008", title="DevOps Engineer", department="Eng",
        required_skills=("Docker",), nice_to_have_skills=(),
        experience_min=4, experience_max=7, seniority="Senior", location="Riyadh",
    )
    defaults.update(kw)
    return Role(**defaults)


def _tax() -> Taxonomy:
    terms = (Term(id="DOCKER", pillar="skill", label="Docker", surfaces=("Docker",), from_requirements=("Docker",)),)
    return Taxonomy(version="t", terms=terms, edges=())


def _candidate(**kw) -> Candidate:
    defaults = dict(
        candidate_id="C001", headline="h", skills_raw="Docker",
        experience_years_stated="5", experience_years_stated_numeric=5.0,
        past_roles=(), certifications="-", education="-", projects="-",
        extra_curriculars="-", city="Riyadh", country="Saudi Arabia",
        notice_period_days=0, notice_period_raw="Immediate",
    )
    defaults.update(kw)
    return Candidate(**defaults)


class TestListRolesTool:
    def test_loads_real_roles(self):
        roles = list_roles_tool("data/open_roles.csv")
        assert len(roles) == 10


class TestCompileRubricTool:
    def test_first_compile_with_empty_guidance(self):
        fake = FakeLLMClient([{
            "weights": {"required": 80, "preferred": 10, "availability": 5, "location": 5},
            "threshold": 56.0, "retier": [], "diff_en": "base rubric", "unsupported": [],
        }])
        rubric, diff_en, unsupported = compile_rubric_tool(fake, _role(), _tax(), "")
        assert diff_en == "base rubric"
        assert unsupported == []
        assert rubric.role_id == "R008"

    def test_amend_reuses_base_rubric_and_appends_guidance(self):
        fake = FakeLLMClient([{
            "weights": {"required": 70, "preferred": 20, "availability": 5, "location": 5},
            "threshold": 56.0, "retier": [], "diff_en": "reweighted", "unsupported": [],
        }])
        rubric, diff_en, _ = compile_rubric_tool(fake, _role(), _tax(), "weight preferred higher")
        assert rubric.weights["preferred"] == 20
        assert "weight preferred higher" in fake.calls[0]["user"]


class TestRunMatchTool:
    def test_splits_finalists_and_excluded(self):
        role = _role(required_skills=(), experience_min=0, experience_max=1)
        from engine.rubric import compile_rubric
        rubric = compile_rubric(role, _tax())
        candidates = [_candidate(candidate_id="C001"), _candidate(candidate_id="C002")]
        links = {"C001": LinkResult("C001", (), ()), "C002": LinkResult("C002", (), ())}
        finalists, excluded, fallback = run_match_tool(role, rubric, candidates, links, _tax())
        assert len(finalists) + len(excluded) == 2


class TestGetAssessmentsTool:
    def test_filters_by_candidate_id(self):
        results = [MatchScore("R008", "C001", 10, 0, 0, 0, 0, (), 0, 0),
                   MatchScore("R008", "C002", 20, 0, 0, 0, 0, (), 0, 0)]
        filtered = get_assessments_tool(results, ["C002"])
        assert [r.candidate_id for r in filtered] == ["C002"]

    def test_no_filter_returns_all(self):
        results = [MatchScore("R008", "C001", 10, 0, 0, 0, 0, (), 0, 0)]
        assert get_assessments_tool(results) == results


class TestAnswerQuestionTool:
    def test_grounded_answer_returned(self):
        fake = FakeLLMClient([{"answer": "C076 has CI/CD from skills, 'Jenkins'."}])
        results = [MatchScore("R008", "C076", 58.1, 0, 0, 0, 0, (
            RequirementScore("CI/CD", "skill", "required", ("CI_CD",), 1.0, "direct", "skills", "Jenkins"),
        ), 0, 0)]
        answer = answer_question_tool(fake, "Why is C076 above C079?", results)
        assert "Jenkins" in answer

    def test_question_and_results_both_in_prompt(self):
        fake = FakeLLMClient([{"answer": "x"}])
        results = [MatchScore("R008", "C076", 58.1, 0, 0, 0, 0, (), 0, 0)]
        answer_question_tool(fake, "why C076?", results)
        assert "why C076?" in fake.calls[0]["user"]
        assert "C076" in fake.calls[0]["user"]

    def test_empty_results_does_not_crash_asked_at_rubric_gate(self):
        fake = FakeLLMClient([{"answer": "Docker is weighted at 80% via the required tier."}])
        rubric = compile_rubric(_role(), _tax())
        answer = answer_question_tool(fake, "why is Docker required?", [], rubric)
        assert answer
        assert "no rubric compiled yet" not in fake.calls[0]["user"]


class TestSendToSlackTool:
    def test_posts_the_markdown(self):
        fake_slack = FakeSlackClient()
        send_to_slack_tool(fake_slack, "# hello")
        assert fake_slack.posts == ["# hello"]


class TestCheckCallBudget:
    def test_under_budget_does_not_raise(self):
        check_call_budget(5, 40)

    def test_at_budget_raises(self):
        with pytest.raises(CallBudgetExceeded):
            check_call_budget(40, 40)
