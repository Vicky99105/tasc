import pytest

from engine.extract import LinkResult, Mention
from engine.ingest import Candidate, PastRole, Role
from engine.match import (
    avail_score,
    have_terms,
    loc_score,
    relevant_years,
    role_occupation,
    score_experience_requirement,
    score_pair,
    score_skill_requirement,
    single_term_lookup,
)
from engine.rubric import Requirement, compile_rubric
from engine.taxonomy import Edge, Taxonomy, Term


def _tax() -> Taxonomy:
    terms = (
        Term(id="DOCKER", pillar="skill", label="Docker", surfaces=("Docker",), from_requirements=("Docker",),
             related=(("wrote a Dockerfile", 0.7),)),
        Term(id="KUBERNETES", pillar="skill", label="Kubernetes", surfaces=("Kubernetes",),
             from_requirements=("Kubernetes",)),
        Term(id="CI_CD", pillar="skill", label="CI/CD", surfaces=("CI/CD", "Jenkins"), from_requirements=("CI/CD",)),
        Term(id="TERRAFORM", pillar="skill", label="Terraform", surfaces=("Terraform",),
             from_requirements=("Terraform",)),
        Term(id="BACKEND_ENGINEER", pillar="occupation", label="Backend Engineer",
             surfaces=("Backend Engineer",)),
        Term(id="DEVOPS_ENGINEER", pillar="occupation", label="DevOps Engineer",
             surfaces=("DevOps Engineer",)),
    )
    edges = (
        Edge(pillar="skill", src="DOCKER", dst="KUBERNETES", kind="adjacent", weight=0.6),
        Edge(pillar="occupation", src="BACKEND_ENGINEER", dst="DEVOPS_ENGINEER", kind="adjacent", weight=0.6),
    )
    return Taxonomy(version="t", terms=terms, edges=edges)


def _role(**kw) -> Role:
    defaults = dict(
        role_id="R008", title="DevOps Engineer", department="Eng",
        required_skills=("CI/CD", "Docker", "Kubernetes"),
        nice_to_have_skills=("Terraform",),
        experience_min=4, experience_max=7, seniority="Senior", location="Riyadh",
    )
    defaults.update(kw)
    return Role(**defaults)


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


class TestHaveTerms:
    def test_met_relation_full_weight_times_field_strength(self):
        link = LinkResult("C001", (Mention("C001", "DOCKER", "skills", "Docker", "met"),), ())
        have = have_terms(link, _tax())
        assert have["DOCKER"] == (1.0, "skills", "Docker")

    def test_adjacent_relation_defaults_to_point_six(self):
        link = LinkResult("C001", (Mention("C001", "DOCKER", "certifications", "container work", "adjacent"),), ())
        have = have_terms(link, _tax())
        assert have["DOCKER"][0] == round(0.6 * 0.75, 3)

    def test_related_phrase_caps_the_score_even_when_extraction_said_met(self):
        link = LinkResult("C001", (Mention("C001", "DOCKER", "skills", "wrote a Dockerfile", "met"),), ())
        have = have_terms(link, _tax())
        assert have["DOCKER"][0] == 0.7  # capped by the term's own related weight, not 1.0

    def test_best_mention_wins_when_term_mentioned_twice(self):
        link = LinkResult("C001", (
            Mention("C001", "DOCKER", "certifications", "x", "adjacent"),
            Mention("C001", "DOCKER", "skills", "Docker", "met"),
        ), ())
        have = have_terms(link, _tax())
        assert have["DOCKER"] == (1.0, "skills", "Docker")


class TestSingleTermLookup:
    def test_direct_match(self):
        have = {"DOCKER": (1.0, "skills", "Docker")}
        assert single_term_lookup("DOCKER", have, (), _tax()) == (1.0, "direct", "skills", "Docker")

    def test_adjacent_match_discounted(self):
        have = {"DOCKER": (1.0, "skills", "Docker")}
        v, via, field, phrase = single_term_lookup("KUBERNETES", have, (), _tax())
        assert via == "adjacent" and v == 0.6

    def test_blocked_term_does_not_substitute(self):
        have = {"DOCKER": (1.0, "skills", "Docker")}
        v, via, field, phrase = single_term_lookup("KUBERNETES", have, ("DOCKER",), _tax())
        assert via == "absent" and v == 0.0

    def test_absent_when_nothing_matches(self):
        assert single_term_lookup("TERRAFORM", {}, (), _tax()) == (0.0, "absent", "", "")


class TestScoreSkillRequirement:
    def test_uses_best_of_any_of_terms(self):
        req = Requirement(source="s", kind="skill", term_ids=("DOCKER", "KUBERNETES"), tier="required",
                           substitutable=True, verifiable=True)
        have = {"DOCKER": (1.0, "skills", "Docker")}
        result = score_skill_requirement(req, have, _tax())
        assert result.score == 1.0 and result.via == "direct"

    def test_empty_term_ids_scores_absent(self):
        req = Requirement(source="s", kind="skill", term_ids=(), tier="required", substitutable=True, verifiable=False)
        result = score_skill_requirement(req, {}, _tax())
        assert result.score == 0.0 and result.via == "absent"


class TestRoleOccupationAndRelevantYears:
    def test_role_occupation_resolves(self):
        assert role_occupation(_role(), _tax()) == "DEVOPS_ENGINEER"

    def test_relevant_years_same_occupation_full_weight(self):
        from engine.extract import CandidateRole
        roles = (CandidateRole("DevOps Engineer", "DEVOPS_ENGINEER", 1.0, 3.0, True),)
        assert relevant_years(roles, "DEVOPS_ENGINEER", _tax()) == 3.0

    def test_relevant_years_adjacent_occupation_discounted(self):
        from engine.extract import CandidateRole
        roles = (CandidateRole("Backend Engineer", "BACKEND_ENGINEER", 1.0, 3.0, True),)
        assert relevant_years(roles, "DEVOPS_ENGINEER", _tax()) == round(3.0 * 0.6, 2)

    def test_unrelated_occupation_contributes_zero(self):
        from engine.extract import CandidateRole
        roles = (CandidateRole("Recruiter", None, 0.0, 3.0, True),)
        assert relevant_years(roles, "DEVOPS_ENGINEER", _tax()) == 0.0


class TestScoreExperienceRequirement:
    def test_within_band_scores_full_credit(self):
        req = Requirement(source="s", kind="experience", term_ids=(), tier="required",
                           substitutable=False, verifiable=True, experience_min=4, experience_max=7)
        past_roles = (PastRole("DevOps Engineer", "X", "Y", True, 2022, 5.0, ""),)
        c = _candidate(past_roles=past_roles, experience_years_stated_numeric=10.0)
        result = score_experience_requirement(req, _role(), c, _tax())
        assert result.score == 1.0
        assert result.conflict is True  # stated 10 vs implied 5, >2 gap

    def test_zero_relevant_years_scores_zero(self):
        req = Requirement(source="s", kind="experience", term_ids=(), tier="required",
                           substitutable=False, verifiable=True, experience_min=4, experience_max=7)
        c = _candidate(past_roles=())
        result = score_experience_requirement(req, _role(), c, _tax())
        assert result.score == 0.0

    def test_stated_years_carried_never_scored(self):
        req = Requirement(source="s", kind="experience", term_ids=(), tier="required",
                           substitutable=False, verifiable=True, experience_min=4, experience_max=7)
        c = _candidate(past_roles=(), experience_years_stated_numeric=99.0)
        result = score_experience_requirement(req, _role(), c, _tax())
        assert result.stated_years == 99.0
        assert result.score == 0.0  # stated=99 never influences the band score


class TestAvailScore:
    def test_zero_based_worst_case_is_zero(self):
        assert avail_score(None) == 0.0
        assert avail_score(90) == 0.0

    def test_immediate_is_full_credit(self):
        assert avail_score(0) == 1.0

    def test_tiers(self):
        assert avail_score(30) == 0.75
        assert avail_score(45) == 0.5
        assert avail_score(60) == 0.25


class TestLocScore:
    def test_exact_city_match(self):
        assert loc_score("Riyadh", "Riyadh", "Saudi Arabia") == 1.0

    def test_same_country_different_city(self):
        assert loc_score("Riyadh", "Jeddah", "Saudi Arabia") == 0.5

    def test_different_country_scores_zero(self):
        assert loc_score("Riyadh", "Cairo", "Egypt") == 0.0

    def test_no_city_on_file_scores_zero_not_partial(self):
        assert loc_score("Riyadh", "", "") == 0.0


class TestScorePair:
    def test_full_pipeline_produces_bounded_total(self):
        role = _role()
        tax = _tax()
        rubric = compile_rubric(role, tax)
        c = _candidate(city="Riyadh", country="Saudi Arabia", notice_period_days=0,
                        past_roles=(PastRole("DevOps Engineer", "X", "Y", True, 2022, 5.0, ""),))
        link = LinkResult("C001", (
            Mention("C001", "DOCKER", "skills", "Docker", "met"),
            Mention("C001", "CI_CD", "skills", "Jenkins", "met"),
        ), ())
        score = score_pair(role, rubric, c, link, tax)
        assert 0.0 <= score.total <= 100.0
        assert score.availability_component == round(rubric.weights["availability"] * 1.0, 3)
        assert score.location_component == round(rubric.weights["location"] * 1.0, 3)

    def test_recompute_is_deterministic(self):
        role = _role()
        tax = _tax()
        rubric = compile_rubric(role, tax)
        c = _candidate()
        link = LinkResult("C001", (Mention("C001", "DOCKER", "skills", "Docker", "met"),), ())
        s1 = score_pair(role, rubric, c, link, tax)
        s2 = score_pair(role, rubric, c, link, tax)
        assert s1 == s2
