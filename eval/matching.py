"""precision@5 and nDCG@10 against a hand-graded sample, taken across the WHOLE
ranking, not just the top — a top-only sample cannot see the wrongly-excluded
candidate, which is the failure this harness exists to catch (the other one,
a wrong shortlist, precision@5 catches on its own).

GRADING RUBRIC — written before any candidate was read, so the golden set can't
become a description of what the system already does:

  3 — excellent fit. Meets nearly every required skill directly (not via
      adjacency), occupation matches or is a clear specialisation of the role.
  2 — good fit. Meets most required skills, at most one real gap or one
      adjacency-credited substitution; occupation matches or is genuinely
      transferable.
  1 — weak fit. Meets some required skills but has two or more real gaps, or
      the occupation is unrelated to the role even if a skill or two overlaps.
  0 — not a fit. Meets few or no required skills, or the profile is
      essentially unrelated to the role's function.

A grade is assigned by reading the candidate's raw profile fields against the
role's raw requirement strings — never by reading this system's own score or
rubric breakdown first, which would just be grading the system against itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from engine.match import MatchScore


@dataclass(frozen=True)
class GoldenMatchGrade:
    role_id: str
    candidate_id: str
    grade: int  # 0-3
    notes: str = ""


def sample_across_ranking(scores: list[MatchScore], n: int) -> list[MatchScore]:
    """Stratified sample by rank position, evenly spaced across the FULL sorted
    list — not the top n. Deterministic given the same scores."""
    ranked = sorted(scores, key=lambda s: (-s.total, s.candidate_id))
    if n >= len(ranked):
        return ranked
    step = (len(ranked) - 1) / (n - 1) if n > 1 else 0
    indices = sorted({round(i * step) for i in range(n)})
    return [ranked[i] for i in indices]


def precision_at_k(scores: list[MatchScore], grades: dict[str, int], k: int, relevant_threshold: int = 2) -> float:
    ranked = sorted(scores, key=lambda s: (-s.total, s.candidate_id))[:k]
    graded = [s for s in ranked if s.candidate_id in grades]
    if not graded:
        return 0.0
    relevant = sum(1 for s in graded if grades[s.candidate_id] >= relevant_threshold)
    return relevant / len(graded)


def ndcg_at_k(scores: list[MatchScore], grades: dict[str, int], k: int) -> float:
    ranked = sorted(scores, key=lambda s: (-s.total, s.candidate_id))[:k]
    dcg = 0.0
    for i, s in enumerate(ranked):
        rel = grades.get(s.candidate_id, 0)
        dcg += (2**rel - 1) / math.log2(i + 2)

    ideal_grades = sorted(grades.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal_grades))
    return dcg / idcg if idcg > 0 else 1.0


def evaluate(scores: list[MatchScore], grades: list[GoldenMatchGrade]) -> dict:
    grades_by_id = {g.candidate_id: g.grade for g in grades}
    return {
        "n_graded": len(grades),
        "precision_at_5": round(precision_at_k(scores, grades_by_id, 5), 4),
        "ndcg_at_10": round(ndcg_at_k(scores, grades_by_id, 10), 4),
    }


if __name__ == "__main__":
    import json

    from engine.config import load_config
    from engine.extract import load_links
    from engine.ingest import load_candidates, load_roles
    from engine.match import score_all
    from engine.rubric import compile_rubric
    from engine.taxonomy import load_taxonomy

    cfg = load_config(".env")
    result = load_candidates("data/candidate_profiles.csv", cfg.reference_date)
    roles = load_roles("data/open_roles.csv")
    tax = load_taxonomy("data/taxonomy.json")
    links = {r.candidate_id: r for r in load_links("data/links.json")}
    rubrics = {r.role_id: compile_rubric(r, tax) for r in roles}
    scores = score_all(roles, rubrics, result.candidates, links, tax)

    try:
        raw_grades = json.load(open("data/golden_matching.json"))
    except FileNotFoundError:
        print("no golden grades on disk — see data/golden_matching.json for the format")
        raise SystemExit(1)

    for rid in sorted({g["role_id"] for g in raw_grades}):
        role_scores = [s for s in scores if s.role_id == rid]
        grades = [GoldenMatchGrade(**g) for g in raw_grades if g["role_id"] == rid]
        print(rid, evaluate(role_scores, grades))
