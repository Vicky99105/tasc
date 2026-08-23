"""Guards the checked-in data/links.json — the P3 output artifact — against the
same audits that were run by hand after generation. If this fails after a re-link,
the new links need review before P4 builds on them."""
from engine.config import load_config
from engine.extract import FIELD_GROUPS, field_text, load_links, resolve_candidate_roles
from engine.ingest import load_candidates
from engine.taxonomy import load_taxonomy, resolve_surface


def _load_all():
    cfg = load_config(".env")
    result = load_candidates("data/candidate_profiles.csv", cfg.reference_date)
    tax = load_taxonomy("data/taxonomy.json")
    links = {r.candidate_id: r for r in load_links("data/links.json")}
    return result, tax, links


class TestLinksArtifact:
    def test_every_candidate_has_a_link_result(self):
        result, _, links = _load_all()
        assert {c.candidate_id for c in result.candidates} <= set(links)

    def test_no_partial_results(self):
        _, _, links = _load_all()
        partial = [cid for cid, r in links.items() if r.partial]
        assert partial == [], f"partial link results (a call failed): {partial}"

    def test_groundedness_is_100_percent(self):
        _, _, links = _load_all()
        rejected = sum(len(r.rejected) for r in links.values())
        assert rejected == 0, f"{rejected} mentions failed the groundedness check"

    def test_every_mention_term_is_a_real_skill_term(self):
        _, tax, links = _load_all()
        known = {t.id for t in tax.by_pillar("skill")}
        for r in links.values():
            for m in r.mentions:
                assert m.term_id in known

    def test_c008_resolves_kubernetes_from_certification(self):
        _, _, links = _load_all()
        c008 = links["C008"]
        assert any(m.term_id == "KUBERNETES" and m.field == "certifications" for m in c008.mentions)

    def test_c076_resolves_jenkins_to_ci_cd_and_has_no_ansible_mention(self):
        _, _, links = _load_all()
        c076 = links["C076"]
        assert any(m.term_id == "CI_CD" and "jenkins" in m.phrase.lower() for m in c076.mentions)
        assert not any("ansible" in m.phrase.lower() for m in c076.mentions)

    def test_discard_pile_list_field_no_known_string_left_unlinked(self):
        result, tax, links = _load_all()
        misses = []
        for c in result.candidates:
            if not c.skills_raw or c.skills_raw.strip() in ("", "-"):
                continue
            r = links.get(c.candidate_id)
            linked_terms = {m.term_id for m in r.mentions} if r else set()
            for s in [x.strip() for x in c.skills_raw.split(",") if x.strip()]:
                t = resolve_surface(tax, "skill", s)
                if t is not None and t.id not in linked_terms:
                    misses.append((c.candidate_id, s, t.id))
        assert misses == []

    def test_discard_pile_prose_related_phrase_no_known_phrase_left_unlinked(self):
        result, tax, links = _load_all()
        misses = []
        for c in result.candidates:
            r = links.get(c.candidate_id)
            linked_terms = {m.term_id for m in r.mentions} if r else set()
            for f in FIELD_GROUPS["prose"]:
                text = field_text(c, f)
                if not text.strip() or text.strip() == "-":
                    continue
                for t in tax.by_pillar("skill"):
                    for phrase, _w in t.related:
                        if phrase.lower() in text.lower() and t.id not in linked_terms:
                            misses.append((c.candidate_id, f, phrase, t.id))
        assert misses == []


class TestOccupationResolution:
    def test_every_real_past_role_title_resolves_deterministically(self):
        result, tax, _ = _load_all()
        unmapped = []
        for c in result.candidates:
            for r in resolve_candidate_roles(c, tax):
                if r.occupation_id is None:
                    unmapped.append((c.candidate_id, r.title))
        assert unmapped == []
