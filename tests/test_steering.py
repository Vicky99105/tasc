import pytest

from engine.ingest import Role
from engine.llm import FakeLLMClient
from engine.rubric import Requirement, Rubric, compile_rubric
from engine.steering import apply_steering, render_steering_prompt, steer
from engine.taxonomy import Taxonomy, Term


def _role(**kw) -> Role:
    defaults = dict(
        role_id="R008", title="DevOps Engineer", department="Eng",
        required_skills=("CI/CD", "Docker", "Kubernetes"), nice_to_have_skills=("Terraform",),
        experience_min=4, experience_max=7, seniority="Senior", location="Riyadh",
    )
    defaults.update(kw)
    return Role(**defaults)


def _tax() -> Taxonomy:
    terms = (
        Term(id="CI_CD", pillar="skill", label="CI/CD", surfaces=("CI/CD",), from_requirements=("CI/CD",)),
        Term(id="DOCKER", pillar="skill", label="Docker", surfaces=("Docker",), from_requirements=("Docker",)),
        Term(id="KUBERNETES", pillar="skill", label="Kubernetes", surfaces=("Kubernetes",), from_requirements=("Kubernetes",)),
        Term(id="TERRAFORM", pillar="skill", label="Terraform", surfaces=("Terraform",), from_requirements=("Terraform",)),
    )
    return Taxonomy(version="t", terms=terms, edges=())


def _base_rubric() -> Rubric:
    return compile_rubric(_role(), _tax())


def _echo_response(rubric, retier=None, threshold=None, unsupported=None):
    return {
        "weights": dict(rubric.weights),
        "threshold": threshold if threshold is not None else rubric.shortlist_threshold,
        "retier": retier or [],
        "diff_en": "no change",
        "unsupported": unsupported or [],
    }


class TestRenderSteeringPrompt:
    def test_requirement_source_enum_only_skill_requirements(self):
        rubric = _base_rubric()
        _, _, schema = render_steering_prompt(_role(), rubric, "")
        enum = schema["properties"]["retier"]["items"]["properties"]["requirement_source"]["enum"]
        assert "Docker" in enum
        assert not any("years of relevant" in e for e in enum)

    def test_empty_guidance_marked_as_first_compile(self):
        rubric = _base_rubric()
        _, user, _ = render_steering_prompt(_role(), rubric, "")
        assert "first compile" in user


class TestApplySteering:
    def test_reweight_applied(self):
        rubric = _base_rubric()
        raw = _echo_response(rubric)
        raw["weights"] = {"required": 70, "preferred": 20, "availability": 5, "location": 5}
        new_rubric = apply_steering(rubric, raw)
        assert new_rubric.weights["required"] == 70

    def test_weights_not_summing_to_100_raises(self):
        rubric = _base_rubric()
        raw = _echo_response(rubric)
        raw["weights"] = {"required": 50, "preferred": 20, "availability": 5, "location": 5}
        with pytest.raises(ValueError):
            apply_steering(rubric, raw)

    def test_threshold_applied(self):
        rubric = _base_rubric()
        raw = _echo_response(rubric, threshold=60.0)
        new_rubric = apply_steering(rubric, raw)
        assert new_rubric.shortlist_threshold == 60.0

    def test_retier_moves_requirement(self):
        rubric = _base_rubric()
        raw = _echo_response(rubric, retier=[{"requirement_source": "Docker", "new_tier": "preferred"}])
        new_rubric = apply_steering(rubric, raw)
        docker = next(r for r in new_rubric.requirements if r.source == "Docker")
        assert docker.tier == "preferred"

    def test_retier_to_dropped(self):
        rubric = _base_rubric()
        raw = _echo_response(rubric, retier=[{"requirement_source": "Docker", "new_tier": "dropped"}])
        new_rubric = apply_steering(rubric, raw)
        docker = next(r for r in new_rubric.requirements if r.source == "Docker")
        assert docker.tier == "dropped"

    def test_unaffected_requirements_keep_their_tier(self):
        rubric = _base_rubric()
        raw = _echo_response(rubric, retier=[{"requirement_source": "Docker", "new_tier": "preferred"}])
        new_rubric = apply_steering(rubric, raw)
        cicd = next(r for r in new_rubric.requirements if r.source == "CI/CD")
        assert cicd.tier == "required"

    def test_assessment_key_changes_when_tier_changes(self):
        rubric = _base_rubric()
        raw = _echo_response(rubric, retier=[{"requirement_source": "Docker", "new_tier": "preferred"}])
        new_rubric = apply_steering(rubric, raw)
        assert new_rubric.assessment_key != rubric.assessment_key

    def test_no_change_echo_keeps_hash_stable(self):
        rubric = _base_rubric()
        raw = _echo_response(rubric)
        new_rubric = apply_steering(rubric, raw)
        assert new_rubric.assessment_key == rubric.assessment_key


class TestSteer:
    def test_end_to_end_with_fake_llm(self):
        rubric = _base_rubric()
        fake = FakeLLMClient([_echo_response(
            rubric, retier=[{"requirement_source": "Docker", "new_tier": "preferred"}]
        ) | {"diff_en": "containers softened", "unsupported": [
            {"guidance": "weight CI/CD twice as much", "reason": "no per-requirement weight exists"}
        ]}])
        result = steer(fake, _role(), rubric, "containers aren't essential; weight CI/CD twice as much")
        assert result.diff_en == "containers softened"
        assert len(result.unsupported) == 1
        docker = next(r for r in result.rubric.requirements if r.source == "Docker")
        assert docker.tier == "preferred"
