"""Does "Jenkins" reach CI_CD, and do the discards stay discarded?

The golden set is built from data/taxonomy.json — the P2 reviewed artifact — and
the real candidate corpus, WITHOUT ever running extraction. That ordering is the
point: the grading rubric (what each candidate string or phrase should resolve
to) exists before any P3 output is read, so the golden set can't quietly become
a description of what the linker already does. A candidate skill string's
correct answer is exact-surface resolution against the taxonomy (deterministic);
a prose phrase's correct answer is a literal occurrence of one of the taxonomy's
own `related` phrases. Both were fixed at P2 review time.

The unmatched list is half of this set on purpose — P0 found 8 wrongly-dropped
strings that only an audit of the DISCARD pile could see, and a harness that
only grades claimed links inherits the same blind spot.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.extract import FIELD_GROUPS, LinkResult, field_text
from engine.ingest import Candidate
from engine.taxonomy import Taxonomy, resolve_surface


@dataclass(frozen=True)
class GoldenLinkCase:
    candidate_id: str
    text: str
    field: str
    expected_term: str | None  # None means: this text must NOT produce a mention


def _related_term_for(taxonomy: Taxonomy, text: str) -> str | None:
    """A related phrase matches a short string (a list item) by full equality,
    not substring — a comma-split skills entry IS the phrase, not a sentence
    containing it."""
    key = text.strip().lower()
    for t in taxonomy.by_pillar("skill"):
        for phrase, _w in t.related:
            if phrase.strip().lower() == key:
                return t.id
    return None


def build_golden_set(candidates: list[Candidate], taxonomy: Taxonomy) -> list[GoldenLinkCase]:
    cases: list[GoldenLinkCase] = []
    seen: set[tuple[str, str, str]] = set()

    for c in candidates:
        if c.skills_raw and c.skills_raw.strip() not in ("", "-"):
            for s in [x.strip() for x in c.skills_raw.split(",") if x.strip()]:
                key = (c.candidate_id, "skills", s.lower())
                if key in seen:
                    continue
                seen.add(key)
                term = resolve_surface(taxonomy, "skill", s)
                expected = term.id if term else _related_term_for(taxonomy, s)
                cases.append(GoldenLinkCase(c.candidate_id, s, "skills", expected))

        for field in FIELD_GROUPS["prose"]:
            text = field_text(c, field)
            if not text.strip() or text.strip() == "-":
                continue
            for t in taxonomy.by_pillar("skill"):
                for phrase, _w in t.related:
                    if phrase.lower() in text.lower():
                        key = (c.candidate_id, field, phrase.lower())
                        if key in seen:
                            continue
                        seen.add(key)
                        cases.append(GoldenLinkCase(c.candidate_id, phrase, field, t.id))

    return cases


@dataclass(frozen=True)
class LinkingMetrics:
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    precision: float
    recall: float
    n: int

    def as_dict(self) -> dict:
        return {
            "true_positive": self.true_positive, "false_positive": self.false_positive,
            "false_negative": self.false_negative, "true_negative": self.true_negative,
            "precision": self.precision, "recall": self.recall, "n": self.n,
        }


def evaluate(golden: list[GoldenLinkCase], links_by_id: dict[str, LinkResult]) -> LinkingMetrics:
    tp = fp = fn = tn = 0
    for case in golden:
        link = links_by_id.get(case.candidate_id)
        mentions = link.mentions if link else ()
        found = next((m for m in mentions if m.field == case.field and m.phrase.lower() == case.text.lower()), None)
        if case.expected_term is None:
            if found is None:
                tn += 1
            else:
                fp += 1
        else:
            if found is not None and found.term_id == case.expected_term:
                tp += 1
            elif found is not None:
                fp += 1
            else:
                fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return LinkingMetrics(tp, fp, fn, tn, precision, recall, len(golden))


def consistency_across_duplicates(candidates: list[Candidate], links_by_id: dict[str, LinkResult]) -> tuple[int, int]:
    """(checked, inconsistent) duplicate-content groups (same headline+skills,
    different people) whose skills-field linking should be byte-identical."""
    from collections import defaultdict

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for c in candidates:
        groups[(c.headline, c.skills_raw)].append(c.candidate_id)

    checked = inconsistent = 0
    for ids in groups.values():
        present = [i for i in ids if i in links_by_id]
        if len(present) < 2:
            continue
        checked += 1
        term_sets = {frozenset(m.term_id for m in links_by_id[i].mentions if m.field == "skills") for i in present}
        if len(term_sets) > 1:
            inconsistent += 1
    return checked, inconsistent


def check_regression(current: dict, baseline: dict) -> list[str]:
    violations = []
    if current["precision"] < baseline["precision"] - 1e-9:
        violations.append(f"precision dropped: {current['precision']:.4f} < baseline {baseline['precision']:.4f}")
    if current["recall"] < baseline["recall"] - 1e-9:
        violations.append(f"recall dropped: {current['recall']:.4f} < baseline {baseline['recall']:.4f}")
    if current["consistency_inconsistent"] > baseline["consistency_inconsistent"]:
        violations.append(
            f"consistency worsened: {current['consistency_inconsistent']} inconsistent "
            f"> baseline {baseline['consistency_inconsistent']}"
        )
    return violations


if __name__ == "__main__":
    import json

    from engine.config import load_config
    from engine.extract import load_links
    from engine.ingest import load_candidates
    from engine.taxonomy import load_taxonomy

    cfg = load_config(".env")
    result = load_candidates("data/candidate_profiles.csv", cfg.reference_date)
    tax = load_taxonomy("data/taxonomy.json")
    links = {r.candidate_id: r for r in load_links("data/links.json")}

    golden = build_golden_set(result.candidates, tax)
    metrics = evaluate(golden, links)
    checked, inconsistent = consistency_across_duplicates(result.candidates, links)
    current = {**metrics.as_dict(), "consistency_inconsistent": inconsistent}
    print(f"golden set: {len(golden)} cases")
    print(metrics.as_dict())
    print(f"consistency: checked={checked} inconsistent={inconsistent}")

    try:
        baseline = json.load(open("data/eval_linking_baseline.json"))
        violations = check_regression(current, baseline)
        print("regression gate:", "PASS" if not violations else f"FAIL — {violations}")
    except FileNotFoundError:
        print("no baseline on disk yet — run once and save data/eval_linking_baseline.json to set one")
