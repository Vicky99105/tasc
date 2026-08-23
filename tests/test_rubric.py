import pytest

from engine.ingest import Role, load_roles
from engine.rubric import (
    DEFAULT_WEIGHTS,
    Requirement,
    compile_rubric,
    resolve_requirement_terms,
)
from engine.taxonomy import Edge, Taxonomy, Term, load_taxonomy


def _tax() -> Taxonomy:
    terms = (
        Term(id="AWS", pillar="skill", label="AWS", surfaces=("AWS",), from_requirements=("AWS/Azure",)),
        Term(id="AZURE", pillar="skill", label="Azure", surfaces=("Azure",), from_requirements=("AWS/Azure",)),
        Term(id="DOCKER", pillar="skill", label="Docker", surfaces=("Docker",), from_requirements=("Docker",)),
        Term(id="KUBERNETES", pillar="skill", label="Kubernetes", surfaces=("Kubernetes",),
             from_requirements=("Kubernetes",)),
        Term(id="CI_CD", pillar="skill", label="CI/CD", surfaces=("CI/CD",), from_requirements=("CI/CD",)),
        Term(id="TERRAFORM", pillar="skill", label="Terraform", surfaces=("Terraform",),
             from_requirements=("Terraform",)),
    )
    edges = (Edge(pillar="skill", src="DOCKER", dst="KUBERNETES", kind="adjacent", weight=0.6),)
    return Taxonomy(version="t", terms=terms, edges=edges)


def _role(**kw) -> Role:
    defaults = dict(
        role_id="R008", title="DevOps Engineer", department="Eng",
        required_skills=("CI/CD", "Docker", "Kubernetes", "AWS/Azure"),
        nice_to_have_skills=("Terraform",),
        experience_min=4, experience_max=7, seniority="Senior", location="Riyadh",
    )
    defaults.update(kw)
    return Role(**defaults)


class TestResolveRequirementTerms:
    def test_exact_match(self):
        assert resolve_requirement_terms(_tax(), "Docker") == ("DOCKER",)

    def test_slash_string_resolves_to_both_terms(self):
        assert resolve_requirement_terms(_tax(), "AWS/Azure") == ("AWS", "AZURE")

    def test_trailing_space_and_case_do_not_block_resolution(self):
        assert resolve_requirement_terms(_tax(), "  docker  ") == ("DOCKER",)
        assert resolve_requirement_terms(_tax(), "AWS/Azure ") == ("AWS", "AZURE")

    def test_unknown_string_resolves_to_empty(self):
        assert resolve_requirement_terms(_tax(), "Rust") == ()


class TestCompileRubric:
    def test_all_requirements_present_including_experience(self):
        rubric = compile_rubric(_role(), _tax())
        sources = [r.source for r in rubric.requirements]
        assert "CI/CD" in sources and "Docker" in sources and "AWS/Azure" in sources
        assert any(r.kind == "experience" for r in rubric.requirements)

    def test_aws_azure_becomes_any_of_two_terms(self):
        rubric = compile_rubric(_role(), _tax())
        req = next(r for r in rubric.requirements if r.source == "AWS/Azure")
        assert set(req.term_ids) == {"AWS", "AZURE"}

    def test_weights_default_and_sum_to_100(self):
        rubric = compile_rubric(_role(), _tax())
        assert rubric.weights == DEFAULT_WEIGHTS
        assert sum(rubric.weights.values()) == 100

    def test_weights_not_summing_to_100_raises(self):
        with pytest.raises(ValueError):
            compile_rubric(_role(), _tax(), weights={"required": 50, "preferred": 10, "availability": 5, "location": 5})

    def test_docker_and_kubernetes_block_each_other(self):
        rubric = compile_rubric(_role(), _tax())
        docker_req = next(r for r in rubric.requirements if r.source == "Docker")
        kubernetes_req = next(r for r in rubric.requirements if r.source == "Kubernetes")
        assert "KUBERNETES" in docker_req.blocked_terms
        assert "DOCKER" in kubernetes_req.blocked_terms

    def test_terms_inside_one_any_of_are_not_mutually_blocked(self):
        rubric = compile_rubric(_role(), _tax())
        aws_azure_req = next(r for r in rubric.requirements if r.source == "AWS/Azure")
        assert "AWS" not in aws_azure_req.blocked_terms
        assert "AZURE" not in aws_azure_req.blocked_terms

    def test_tiers_come_from_the_right_csv_column(self):
        rubric = compile_rubric(_role(), _tax())
        assert next(r for r in rubric.requirements if r.source == "Docker").tier == "required"
        assert next(r for r in rubric.requirements if r.source == "Terraform").tier == "preferred"

    def test_unresolvable_requirement_is_marked_unverifiable_not_dropped(self):
        role = _role(required_skills=("Docker", "Rust"))
        rubric = compile_rubric(role, _tax())
        rust_req = next(r for r in rubric.requirements if r.source == "Rust")
        assert rust_req.verifiable is False
        assert rust_req.term_ids == ()

    def test_assessment_key_stable_across_weight_and_threshold_changes(self):
        r1 = compile_rubric(_role(), _tax(), shortlist_threshold=56.0)
        r2 = compile_rubric(_role(), _tax(), shortlist_threshold=60.0,
                             weights={"required": 70, "preferred": 20, "availability": 5, "location": 5})
        assert r1.assessment_key == r2.assessment_key

    def test_assessment_key_changes_when_requirements_change(self):
        r1 = compile_rubric(_role(), _tax())
        r2 = compile_rubric(_role(required_skills=("Docker",)), _tax())
        assert r1.assessment_key != r2.assessment_key


class TestAllRealRolesCompile:
    def test_all_ten_roles_compile(self):
        tax = load_taxonomy("data/taxonomy.json")
        roles = load_roles("data/open_roles.csv")
        for role in roles:
            rubric = compile_rubric(role, tax)
            assert rubric.role_id == role.role_id
            assert sum(rubric.weights.values()) == 100

    def test_r008_suppression_set_contains_docker_and_kubernetes(self):
        tax = load_taxonomy("data/taxonomy.json")
        roles = load_roles("data/open_roles.csv")
        r008 = next(r for r in roles if r.role_id == "R008")
        rubric = compile_rubric(r008, tax)
        docker_req = next(r for r in rubric.requirements if r.source == "Docker")
        kubernetes_req = next(r for r in rubric.requirements if r.source == "Kubernetes")
        assert "KUBERNETES" in docker_req.blocked_terms
        assert "DOCKER" in kubernetes_req.blocked_terms

    def test_r001_aws_or_r008_any_of_on_real_taxonomy(self):
        tax = load_taxonomy("data/taxonomy.json")
        roles = load_roles("data/open_roles.csv")
        r008 = next(r for r in roles if r.role_id == "R008")
        rubric = compile_rubric(r008, tax)
        aws_azure = next((r for r in rubric.requirements if "AWS" in r.source or "Azure" in r.source), None)
        assert aws_azure is not None
        assert set(aws_azure.term_ids) == {"AWS", "AZURE"}
