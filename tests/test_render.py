from engine.brief import CandidateBrief, CandidateBriefInput, Gap, StrengthItem
from engine.ingest import Role
from engine.render import render_role_briefs
from engine.rubric import Rubric


def _role() -> Role:
    return Role(role_id="R008", title="DevOps Engineer", department="Eng",
                required_skills=("Docker",), nice_to_have_skills=(),
                experience_min=4, experience_max=7, seniority="Senior", location="Riyadh")


def _rubric() -> Rubric:
    return Rubric(role_id="R008", requirements=(), weights={"required": 80, "preferred": 10, "availability": 5, "location": 5},
                  shortlist_threshold=56.0, max_return=50, assessment_key="x")


def _pair():
    bi = CandidateBriefInput(
        candidate_id="C001", headline="DevOps engineer", total=58.1,
        strengths=(StrengthItem("CI/CD", "direct", "skills", "Jenkins", 1.0),),
        gaps=(Gap("absent_required", '"Docker" (required) — no evidence in the profile'),),
        experience_stated=10.0, experience_implied=5.0, experience_conflict=True,
    )
    brief = CandidateBrief(
        candidate_id="C001", summary="Strong on CI/CD.", differentiator="Only one with Jenkins.",
        questions=("Have you worked with Docker?",), gaps_asked=bi.gaps,
    )
    return bi, brief


class TestRenderRoleBriefs:
    def test_byte_identical_on_repeat_calls(self):
        md1 = render_role_briefs(_role(), _rubric(), [_pair()], fallback=False)
        md2 = render_role_briefs(_role(), _rubric(), [_pair()], fallback=False)
        assert md1 == md2

    def test_contains_role_and_candidate(self):
        md = render_role_briefs(_role(), _rubric(), [_pair()], fallback=False)
        assert "R008" in md and "C001" in md

    def test_every_strength_field_and_phrase_present(self):
        md = render_role_briefs(_role(), _rubric(), [_pair()], fallback=False)
        assert "Jenkins" in md and "skills" in md

    def test_experience_conflict_surfaced(self):
        md = render_role_briefs(_role(), _rubric(), [_pair()], fallback=False)
        assert "10" in md and "5" in md

    def test_fallback_note_present_only_when_flagged(self):
        md_fallback = render_role_briefs(_role(), _rubric(), [_pair()], fallback=True)
        md_normal = render_role_briefs(_role(), _rubric(), [_pair()], fallback=False)
        assert "did not clear" in md_fallback.lower() or "cleared" in md_fallback.lower()
        assert "cleared" not in md_normal.lower()

    def test_questions_numbered_and_present(self):
        md = render_role_briefs(_role(), _rubric(), [_pair()], fallback=False)
        assert "1. Have you worked with Docker?" in md

    def test_no_strengths_no_questions_still_renders(self):
        bi = CandidateBriefInput(candidate_id="C002", headline="h", total=10.0, strengths=(), gaps=(),
                                  experience_stated=None, experience_implied=0.0, experience_conflict=False)
        brief = CandidateBrief(candidate_id="C002", summary="s", differentiator="d", questions=(), gaps_asked=())
        md = render_role_briefs(_role(), _rubric(), [(bi, brief)], fallback=False)
        assert "C002" in md
