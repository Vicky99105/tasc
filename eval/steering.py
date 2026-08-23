"""Does the guidance get heard and applied to the right knob?

Direct motivation: the P8 live-browser run of "containers aren't a big deal,
make those less important" against R008 produced a diff_en claiming Docker AND
Kubernetes moved to preferred, while the actual retier only moved Docker and,
unprompted, also moved AWS/Azure. No scripted unit test caught it — every one of
them scripts the model's own response. This harness runs the real model against
a fixed set of utterances and checks the STRUCTURED delta exactly, not the
prose, which is exactly the class of error that slipped through before.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.ingest import Role
from engine.llm import LLMClient
from engine.rubric import Rubric, compile_rubric
from engine.steering import steer
from engine.taxonomy import Taxonomy


@dataclass(frozen=True)
class SteeringCase:
    utterance: str
    expected_retier: tuple[tuple[str, str], ...] = ()  # (requirement_source, new_tier) pairs, order-independent
    expected_weight_direction: tuple[str, str] | None = None  # (bucket, "up"|"down") — direction only, not an exact number
    expect_unsupported: bool = False


@dataclass(frozen=True)
class SteeringResult:
    case: SteeringCase
    passed: bool
    reason: str
    actual_retier: tuple[tuple[str, str], ...]
    actual_weights: dict
    unsupported: tuple[str, ...]


def build_golden_cases(role: Role) -> list[SteeringCase]:
    """Cases are written against R008's own requirements — the fixture this
    harness is meant to run against. Written before looking at any model output."""
    return [
        SteeringCase("containers aren't a big deal for us, make those less important",
                     expected_retier=(("Docker", "preferred"), ("Kubernetes", "preferred"))),
        SteeringCase("Docker and Kubernetes experience is not essential, drop them entirely",
                     expected_retier=(("Docker", "dropped"), ("Kubernetes", "dropped"))),
        SteeringCase("weight availability much higher, it matters most for this hire",
                     expected_weight_direction=("availability", "up")),
        SteeringCase("we care more about location than availability",
                     expected_weight_direction=("location", "up")),
        SteeringCase("lower the bar a bit, we're struggling to fill this role",
                     expected_weight_direction=None),  # threshold-only case, checked separately
        SteeringCase("prioritise candidates who interview well", expect_unsupported=True),
        SteeringCase("weight CI/CD twice as much as the other required skills", expect_unsupported=True),
        SteeringCase("only consider candidates who went to a top-tier university", expect_unsupported=True),
    ]


def _check_case(case: SteeringCase, base_rubric: Rubric, rubric: Rubric, unsupported: tuple[str, ...]) -> tuple[bool, str]:
    actual_retier = {r.source: r.tier for r in rubric.requirements}
    if case.expected_retier:
        for source, tier in case.expected_retier:
            if actual_retier.get(source) != tier:
                return False, f"expected {source} -> {tier}, got {actual_retier.get(source)}"
        # nothing else should have moved
        base_tiers = {r.source: r.tier for r in base_rubric.requirements}
        touched = {s for s, _ in case.expected_retier}
        for source, tier in actual_retier.items():
            if source not in touched and tier != base_tiers.get(source):
                return False, f"unexpectedly retiered {source}: {base_tiers.get(source)} -> {tier}"

    if case.expected_weight_direction:
        bucket, direction = case.expected_weight_direction
        before, after = base_rubric.weights[bucket], rubric.weights[bucket]
        if direction == "up" and not (after > before):
            return False, f"expected {bucket} weight to increase from {before}, got {after}"
        if direction == "down" and not (after < before):
            return False, f"expected {bucket} weight to decrease from {before}, got {after}"

    if case.expect_unsupported and not unsupported:
        return False, "expected this guidance to be refused as unsupported, but nothing was"
    if not case.expect_unsupported and unsupported and not case.expected_retier and not case.expected_weight_direction:
        return False, f"unexpectedly marked unsupported: {unsupported}"
    return True, "ok"


def run(llm: LLMClient, role: Role, taxonomy: Taxonomy, cases: list[SteeringCase]) -> list[SteeringResult]:
    base_rubric = compile_rubric(role, taxonomy)
    results = []
    for case in cases:
        result = steer(llm, role, base_rubric, case.utterance)
        unsupported = tuple(u.guidance for u in result.unsupported)
        passed, reason = _check_case(case, base_rubric, result.rubric, unsupported)
        results.append(SteeringResult(
            case=case, passed=passed, reason=reason,
            actual_retier=tuple((r.source, r.tier) for r in result.rubric.requirements if r.kind == "skill"),
            actual_weights=result.rubric.weights, unsupported=unsupported,
        ))
    return results


if __name__ == "__main__":
    from engine.config import load_config
    from engine.ingest import load_roles
    from engine.llm import DefaultLLMClient
    from engine.taxonomy import load_taxonomy

    cfg = load_config(".env")
    roles = load_roles("data/open_roles.csv")
    tax = load_taxonomy("data/taxonomy.json")
    role = next(r for r in roles if r.role_id == "R008")
    cases = build_golden_cases(role)

    llm = DefaultLLMClient(cfg.openrouter_api_key, cfg.openrouter_base_url, cfg.model_link)
    results = run(llm, role, tax, cases)

    passed = sum(1 for r in results if r.passed)
    print(f"passed: {passed}/{len(results)}  cost: ${llm.usage.cost_usd(0.375, 1.875):.4f}")
    for r in results:
        print(f"[{'PASS' if r.passed else 'FAIL'}] {r.case.utterance!r} — {r.reason}")
