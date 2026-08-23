import pytest

from engine.brief import (
    CandidateBrief,
    CandidateBriefInput,
    Gap,
    StrengthItem,
    build_brief_input,
    detect_gaps,
    detect_strengths,
    generate_briefs,
    render_brief_prompt,
    select_finalists,
    select_top_gaps,
)
from engine.ingest import Candidate, PastRole, Role
from engine.llm import FakeLLMClient
from engine.match import MatchScore, RequirementScore
from engine.rubric import Requirement, Rubric, compile_rubric
from engine.taxonomy import Taxonomy, Term


def _role(**kw) -> Role:
    defaults = dict(
        role_id="R008", title="DevOps Engineer", department="Eng",
        required_skills=("CI/CD", "Docker", "Kubernetes"), nice_to_have_skills=("Terraform",),
        experience_min=4, experience_max=7, seniority="Senior", location="Riyadh",
    )
    defaults.update(kw)
    return Role(**defaults)


def _candidate(**kw) -> Candidate:
    defaults = dict(
        candidate_id="C001", headline="DevOps engineer", skills_raw="Docker",
        experience_years_stated="5", experience_years_stated_numeric=5.0,
        past_roles=(), certifications="-", education="-", projects="-",
        extra_curriculars="-", city="Riyadh", country="Saudi Arabia",
        notice_period_days=0, notice_period_raw="Immediate",
    )
    defaults.update(kw)
    return Candidate(**defaults)


def _score(**kw) -> MatchScore:
    defaults = dict(
        role_id="R008", candidate_id="C001", total=58.1,
        required_component=40.0, preferred_component=8.0,
        availability_component=5.0, location_component=5.0,
        requirement_scores=(
            RequirementScore("CI/CD", "skill", "required", ("CI_CD",), 1.0, "direct", "skills", "Jenkins"),
            RequirementScore("Docker", "skill", "required", ("DOCKER",), 0.0, "absent", "", ""),
            RequirementScore("Kubernetes", "skill", "required", ("KUBERNETES",), 0.6, "adjacent", "skills", "Docker"),
            RequirementScore("Terraform", "skill", "preferred", ("TERRAFORM",), 0.75, "direct", "certifications", "Terraform"),
            RequirementScore("4-7 years...", "experience", "required", (), 1.0, "band", "past_roles", "",
                              stated_years=10.0, implied_years=5.0, conflict=True),
        ),
        availability_score=1.0, location_score=1.0,
    )
    defaults.update(kw)
    return MatchScore(**defaults)


class TestDetectStrengths:
    def test_only_positive_skill_scores_kept_sorted_desc(self):
        strengths = detect_strengths(_score())
        assert [s.source for s in strengths] == ["CI/CD", "Terraform", "Kubernetes"]

    def test_experience_never_a_strength_item(self):
        strengths = detect_strengths(_score())
        assert all(s.source != "4-7 years..." for s in strengths)


class TestDetectGaps:
    def test_absent_required_comes_first(self):
        rubric = Rubric(role_id="R008", requirements=(
            Requirement("CI/CD", "skill", ("CI_CD",), "required", True, True),
            Requirement("Docker", "skill", ("DOCKER",), "required", True, True),
            Requirement("Kubernetes", "skill", ("KUBERNETES",), "required", True, True),
            Requirement("Terraform", "skill", ("TERRAFORM",), "preferred", True, True),
        ), weights={"required": 80, "preferred": 10, "availability": 5, "location": 5},
           shortlist_threshold=56.0, max_return=50, assessment_key="x")
        gaps = detect_gaps(_candidate(), rubric, _score())
        assert gaps[0].kind == "absent_required"
        assert "Docker" in gaps[0].detail

    def test_adjacent_substitution_detected(self):
        rubric = Rubric(role_id="R008", requirements=(
            Requirement("CI/CD", "skill", ("CI_CD",), "required", True, True),
            Requirement("Docker", "skill", ("DOCKER",), "required", True, True),
            Requirement("Kubernetes", "skill", ("KUBERNETES",), "required", True, True),
            Requirement("Terraform", "skill", ("TERRAFORM",), "preferred", True, True),
        ), weights={"required": 80, "preferred": 10, "availability": 5, "location": 5},
           shortlist_threshold=56.0, max_return=50, assessment_key="x")
        gaps = detect_gaps(_candidate(), rubric, _score())
        assert any(g.kind == "adjacent_substitution" and "Kubernetes" in g.detail for g in gaps)

    def test_empty_prose_fields_detected(self):
        rubric = Rubric(role_id="R008", requirements=(), weights={"required": 80, "preferred": 10, "availability": 5, "location": 5},
                         shortlist_threshold=56.0, max_return=50, assessment_key="x")
        gaps = detect_gaps(_candidate(projects="-", extra_curriculars="-", education="-", certifications="-"),
                            rubric, _score(requirement_scores=()))
        assert any(g.kind == "empty_field" for g in gaps)

    def test_logistics_gap_when_not_available_immediately(self):
        rubric = Rubric(role_id="R008", requirements=(), weights={"required": 80, "preferred": 10, "availability": 5, "location": 5},
                         shortlist_threshold=56.0, max_return=50, assessment_key="x")
        score = _score(requirement_scores=(), availability_score=0.5)
        gaps = detect_gaps(_candidate(), rubric, score)
        assert any(g.kind == "logistics" and "notice" in g.detail for g in gaps)


class TestSelectTopGaps:
    def test_caps_at_n(self):
        gaps = tuple(Gap("absent_required", str(i)) for i in range(5))
        assert select_top_gaps(gaps, 3) == gaps[:3]


class TestSelectFinalists:
    def test_selects_those_clearing_threshold(self):
        rubric = compile_rubric(_role(), _fixture_taxonomy())
        scores = [_score(candidate_id="C001", total=70.0), _score(candidate_id="C002", total=40.0)]
        finalists, fallback = select_finalists(scores, rubric)
        assert [s.candidate_id for s in finalists] == ["C001"]
        assert fallback is False

    def test_falls_back_to_highest_scores_when_nobody_clears(self):
        rubric = compile_rubric(_role(), _fixture_taxonomy())
        scores = [_score(candidate_id="C001", total=40.0), _score(candidate_id="C002", total=30.0)]
        finalists, fallback = select_finalists(scores, rubric, fallback_n=2)
        assert len(finalists) == 2
        assert fallback is True
        assert finalists[0].candidate_id == "C001"

    def test_never_returns_empty(self):
        rubric = compile_rubric(_role(), _fixture_taxonomy())
        finalists, fallback = select_finalists([_score(total=0.0)], rubric)
        assert len(finalists) == 1


def _fixture_taxonomy() -> Taxonomy:
    terms = (
        Term(id="CI_CD", pillar="skill", label="CI/CD", surfaces=("CI/CD",), from_requirements=("CI/CD",)),
        Term(id="DOCKER", pillar="skill", label="Docker", surfaces=("Docker",), from_requirements=("Docker",)),
        Term(id="KUBERNETES", pillar="skill", label="Kubernetes", surfaces=("Kubernetes",), from_requirements=("Kubernetes",)),
        Term(id="TERRAFORM", pillar="skill", label="Terraform", surfaces=("Terraform",), from_requirements=("Terraform",)),
        Term(id="DEVOPS_ENGINEER", pillar="occupation", label="DevOps Engineer", surfaces=("DevOps Engineer",)),
    )
    return Taxonomy(version="t", terms=terms, edges=())


class TestBuildBriefInput:
    def test_pulls_experience_from_score(self):
        bi = build_brief_input(_candidate(), compile_rubric(_role(), _fixture_taxonomy()), _score())
        assert bi.experience_stated == 10.0
        assert bi.experience_implied == 5.0
        assert bi.experience_conflict is True


class TestGenerateBriefs:
    def test_model_output_mapped_to_candidate_briefs(self):
        rubric = compile_rubric(_role(), _fixture_taxonomy())
        bi = build_brief_input(_candidate(), rubric, _score())
        fake = FakeLLMClient([{"briefs": [
            {"candidate_id": "C001", "summary": "Strong on CI/CD.", "differentiator": "Only one with Jenkins.",
             "questions": ["Have you used Docker in production?"]}
        ]}])
        briefs = generate_briefs(fake, _role(), rubric, [bi], top_n_gaps=1)
        assert len(briefs) == 1
        assert briefs[0].candidate_id == "C001"
        assert len(briefs[0].questions) == len(briefs[0].gaps_asked)

    def test_no_finalists_returns_empty_without_calling_llm(self):
        fake = FakeLLMClient([])
        rubric = compile_rubric(_role(), _fixture_taxonomy())
        assert generate_briefs(fake, _role(), rubric, []) == []
        assert fake.usage.calls == 0

    def test_missing_questions_fall_back_to_the_gap_text_not_invention(self):
        rubric = compile_rubric(_role(), _fixture_taxonomy())
        bi = build_brief_input(_candidate(), rubric, _score())
        fake = FakeLLMClient([{"briefs": [
            {"candidate_id": "C001", "summary": "s", "differentiator": "d", "questions": []}
        ]}])
        briefs = generate_briefs(fake, _role(), rubric, [bi], top_n_gaps=2)
        assert len(briefs[0].questions) == len(briefs[0].gaps_asked) == 2
        assert briefs[0].questions[0] == briefs[0].gaps_asked[0].detail


class TestRenderBriefPrompt:
    def test_enum_constrained_to_finalist_ids(self):
        rubric = compile_rubric(_role(), _fixture_taxonomy())
        bi = build_brief_input(_candidate(), rubric, _score())
        _, _, schema = render_brief_prompt(_role(), rubric, [(bi, bi.gaps[:2])])
        assert schema["properties"]["briefs"]["items"]["properties"]["candidate_id"]["enum"] == ["C001"]

    def test_finalist_block_contains_strengths_and_gaps(self):
        rubric = compile_rubric(_role(), _fixture_taxonomy())
        bi = build_brief_input(_candidate(), rubric, _score())
        _, user, _ = render_brief_prompt(_role(), rubric, [(bi, bi.gaps[:2])])
        assert "C001" in user
        assert "CI/CD" in user
