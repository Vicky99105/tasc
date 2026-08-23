from engine.ingest import Role
from engine.llm import FakeLLMClient
from engine.taxonomy import Taxonomy, Term
from eval.steering import SteeringCase, build_golden_cases, run


def _role() -> Role:
    return Role(role_id="R008", title="DevOps Engineer", department="Eng",
                required_skills=("CI/CD", "Docker", "Kubernetes", "AWS/Azure"),
                nice_to_have_skills=("Terraform",),
                experience_min=4, experience_max=7, seniority="Senior", location="Riyadh")


def _tax() -> Taxonomy:
    terms = (
        Term(id="CI_CD", pillar="skill", label="CI/CD", surfaces=("CI/CD",), from_requirements=("CI/CD",)),
        Term(id="DOCKER", pillar="skill", label="Docker", surfaces=("Docker",), from_requirements=("Docker",)),
        Term(id="KUBERNETES", pillar="skill", label="Kubernetes", surfaces=("Kubernetes",), from_requirements=("Kubernetes",)),
        Term(id="AWS", pillar="skill", label="AWS", surfaces=("AWS",), from_requirements=("AWS/Azure",)),
        Term(id="AZURE", pillar="skill", label="Azure", surfaces=("Azure",), from_requirements=("AWS/Azure",)),
        Term(id="TERRAFORM", pillar="skill", label="Terraform", surfaces=("Terraform",), from_requirements=("Terraform",)),
    )
    return Taxonomy(version="t", terms=terms, edges=())


def _echo(weights=None, threshold=56.0, retier=None, unsupported=None):
    return {
        "weights": weights or {"required": 80, "preferred": 10, "availability": 5, "location": 5},
        "threshold": threshold, "retier": retier or [], "diff_en": "x", "unsupported": unsupported or [],
    }


class TestBuildGoldenCases:
    def test_returns_a_nonempty_fixed_set(self):
        cases = build_golden_cases(_role())
        assert len(cases) >= 5


class TestRunCorrectRetier:
    def test_correct_scoped_retier_passes(self):
        fake = FakeLLMClient([_echo(retier=[
            {"requirement_source": "Docker", "new_tier": "preferred"},
            {"requirement_source": "Kubernetes", "new_tier": "preferred"},
        ])])
        case = SteeringCase("containers aren't a big deal", expected_retier=(("Docker", "preferred"), ("Kubernetes", "preferred")))
        results = run(fake, _role(), _tax(), [case])
        assert results[0].passed, results[0].reason

    def test_missing_expected_retier_fails(self):
        fake = FakeLLMClient([_echo(retier=[{"requirement_source": "Docker", "new_tier": "preferred"}])])
        case = SteeringCase("containers aren't a big deal", expected_retier=(("Docker", "preferred"), ("Kubernetes", "preferred")))
        results = run(fake, _role(), _tax(), [case])
        assert not results[0].passed
        assert "Kubernetes" in results[0].reason

    def test_the_p8_bug_reproduced_and_caught(self):
        # exactly what the live browser run produced: Docker moved, Kubernetes not,
        # AWS moved unprompted
        fake = FakeLLMClient([_echo(retier=[
            {"requirement_source": "Docker", "new_tier": "preferred"},
            {"requirement_source": "AWS/Azure", "new_tier": "preferred"},
        ])])
        case = SteeringCase("containers aren't a big deal", expected_retier=(("Docker", "preferred"), ("Kubernetes", "preferred")))
        results = run(fake, _role(), _tax(), [case])
        assert not results[0].passed

    def test_unprompted_extra_retier_fails_even_if_named_ones_correct(self):
        fake = FakeLLMClient([_echo(retier=[
            {"requirement_source": "Docker", "new_tier": "preferred"},
            {"requirement_source": "Kubernetes", "new_tier": "preferred"},
            {"requirement_source": "AWS/Azure", "new_tier": "preferred"},
        ])])
        case = SteeringCase("containers aren't a big deal", expected_retier=(("Docker", "preferred"), ("Kubernetes", "preferred")))
        results = run(fake, _role(), _tax(), [case])
        assert not results[0].passed
        assert "AWS/Azure" in results[0].reason


class TestWeightDirection:
    def test_correct_direction_passes(self):
        fake = FakeLLMClient([_echo(weights={"required": 75, "preferred": 10, "availability": 10, "location": 5})])
        case = SteeringCase("weight availability higher", expected_weight_direction=("availability", "up"))
        results = run(fake, _role(), _tax(), [case])
        assert results[0].passed

    def test_wrong_direction_fails(self):
        fake = FakeLLMClient([_echo(weights={"required": 85, "preferred": 10, "availability": 0, "location": 5})])
        case = SteeringCase("weight availability higher", expected_weight_direction=("availability", "up"))
        results = run(fake, _role(), _tax(), [case])
        assert not results[0].passed


class TestUnsupported:
    def test_expected_unsupported_and_present_passes(self):
        fake = FakeLLMClient([_echo(unsupported=[{"guidance": "interview well", "reason": "no source in data"}])])
        case = SteeringCase("prioritise candidates who interview well", expect_unsupported=True)
        results = run(fake, _role(), _tax(), [case])
        assert results[0].passed

    def test_expected_unsupported_but_silently_applied_fails(self):
        fake = FakeLLMClient([_echo()])
        case = SteeringCase("prioritise candidates who interview well", expect_unsupported=True)
        results = run(fake, _role(), _tax(), [case])
        assert not results[0].passed
