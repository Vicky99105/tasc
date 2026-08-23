"""Guards the checked-in artifact, not the generation code. If this fails after a
taxonomy rebuild, the new draft needs review before it replaces data/taxonomy.json."""
from engine.taxonomy import load_db, load_taxonomy, validate_taxonomy
from engine.vocab_sources import candidate_skill_strings, role_requirement_strings, role_titles
from engine.ingest import load_candidates, load_roles
from engine.config import load_config


class TestCheckedInTaxonomy:
    def test_json_is_valid(self):
        tax = load_taxonomy("data/taxonomy.json")
        assert validate_taxonomy(tax) == []

    def test_two_pillars_present(self):
        tax = load_taxonomy("data/taxonomy.json")
        assert len(tax.by_pillar("skill")) > 0
        assert len(tax.by_pillar("occupation")) == 10

    def test_every_real_role_requirement_resolves(self):
        tax = load_taxonomy("data/taxonomy.json")
        roles = load_roles("data/open_roles.csv")
        from engine.taxonomy import audit_requirements
        assert audit_requirements(tax, "skill", role_requirement_strings(roles)) == []
        assert audit_requirements(tax, "occupation", role_titles(roles)) == []

    def test_db_matches_json(self, tmp_path):
        from engine.taxonomy import build_db

        tax = load_taxonomy("data/taxonomy.json")
        db_path = str(tmp_path / "rebuilt.db")
        build_db(tax, db_path)
        back = load_db(db_path)
        assert {t.id for t in back.terms} == {t.id for t in tax.terms}

    def test_no_surface_is_claimed_by_two_terms_in_the_same_pillar(self):
        from engine.taxonomy import normalize

        tax = load_taxonomy("data/taxonomy.json")
        for pillar in ("skill", "occupation"):
            owner: dict[str, str] = {}
            for t in tax.by_pillar(pillar):
                for s in t.surfaces:
                    key = normalize(s)
                    existing = owner.get(key)
                    assert existing is None or existing == t.id, \
                        f"{key!r} claimed by both {existing} and {t.id}"
                    owner[key] = t.id
