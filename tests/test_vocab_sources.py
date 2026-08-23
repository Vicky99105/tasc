from engine.ingest import Candidate, Role
from engine.vocab_sources import (
    candidate_skill_strings,
    observed_job_titles,
    role_requirement_strings,
    role_titles,
)


def _role(role_id, required, nice):
    return Role(
        role_id=role_id, title="T", department="D",
        required_skills=tuple(required), nice_to_have_skills=tuple(nice),
        experience_min=1, experience_max=3, seniority="Junior", location="X",
    )


def _candidate(cid, skills_raw, past_role_titles=()):
    from engine.ingest import PastRole
    roles = tuple(
        PastRole(title=t, company="C", city="X", is_current=False, start_year=None, duration_years=1.0, raw="")
        for t in past_role_titles
    )
    return Candidate(
        candidate_id=cid, headline="h", skills_raw=skills_raw,
        experience_years_stated="1", experience_years_stated_numeric=1.0,
        past_roles=roles, certifications="-", education="-", projects="-",
        extra_curriculars="-", city="X", country="Y",
        notice_period_days=0, notice_period_raw="Immediate",
    )


class TestRoleRequirementStrings:
    def test_dedupes_across_roles_preserving_first_occurrence_order(self):
        roles = [_role("R1", ["Python", "SQL"], ["AWS"]), _role("R2", ["SQL"], ["Python"])]
        assert role_requirement_strings(roles) == ["Python", "SQL", "AWS"]

    def test_empty_roles_yield_empty(self):
        assert role_requirement_strings([]) == []


class TestCandidateSkillStrings:
    def test_splits_and_dedupes(self):
        cands = [_candidate("C1", "Python, SQL"), _candidate("C2", "SQL, AWS")]
        assert candidate_skill_strings(cands) == ["Python", "SQL", "AWS"]

    def test_blank_and_dash_skip(self):
        cands = [_candidate("C1", ""), _candidate("C2", "-"), _candidate("C3", "Python")]
        assert candidate_skill_strings(cands) == ["Python"]


class TestRoleTitles:
    def test_preserves_order(self):
        roles = [_role("R1", [], []), _role("R2", [], [])]
        roles[0] = _role("R1", [], [])
        assert role_titles(roles) == ["T", "T"]


class TestObservedJobTitles:
    def test_dedupes_across_candidates(self):
        cands = [
            _candidate("C1", "-", past_role_titles=["Backend Engineer", "Intern"]),
            _candidate("C2", "-", past_role_titles=["Backend Engineer"]),
        ]
        assert observed_job_titles(cands) == ["Backend Engineer", "Intern"]

    def test_no_past_roles_yields_empty(self):
        cands = [_candidate("C1", "-", past_role_titles=())]
        assert observed_job_titles(cands) == []
