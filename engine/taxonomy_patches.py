"""Hand review of the two raw model drafts, applied before they become
data/taxonomy.json. Each patch names what it does and why — this file is the
artifact a human re-reads, same as P0's manual edits were.

Rerun engine/build_taxonomy.py to regenerate the *_raw.json drafts, then rerun this
file to reapply review on top of the new draft. If the model's draft changes shape,
a patch here may no longer apply — that is the point of keeping this explicit
instead of hand-editing the JSON once and losing the reasoning.
"""
from __future__ import annotations

from engine.taxonomy import Edge, Taxonomy, Term
from engine.build_taxonomy import raw_to_terms

# --- skill pillar ------------------------------------------------------------
#
# 1. MySQL is a specific SQL database; no role asks for it by name, so it never
#    became its own term. It belongs as a full-credit surface of SQL directly.
# 2. PostgreSQL DID become its own term (R001 asks for it by name). Folding the
#    string "PostgreSQL" into SQL's surface list would collide with POSTGRESQL's
#    own alias — two terms claiming the identical exact surface. The correct fix
#    is a term-to-term narrower edge (POSTGRESQL is narrower than SQL), so
#    evidence of PostgreSQL gives full credit to a plain "SQL" ask without any
#    surface ambiguity. The draft's own POSTGRESQL~SQL *adjacent* edge (weight
#    0.6) is then redundant and is dropped rather than left to shadow it.
# 3. "Arabic fluency" and "Arabic + English bilingual" / "bilingual Arabic/English"
#    were split into two terms related only at 0.6. From a recruiter's read they
#    are the same underlying signal — profiles are all written in English and the
#    roles are in English-speaking GCC offices, so bilingual is the normal case,
#    not a distinct skill. Splitting them under-credits a candidate who wrote
#    "bilingual Arabic/English" against a role that only asked for "Arabic
#    fluency". Merged into one term, ARABIC_LANGUAGE.


def apply_skill_patches(raw: dict) -> dict:
    raw = {**raw, "terms": [dict(t) for t in raw["terms"]], "unmapped": list(raw["unmapped"])}
    by_id = {t["id"]: t for t in raw["terms"]}

    sql = by_id.get("SQL")
    if sql is not None and "MySQL" not in sql["narrower"]:
        sql["narrower"] = [*sql["narrower"], "MySQL"]
        raw["unmapped"] = [u for u in raw["unmapped"] if u["string"] != "MySQL"]

    postgres = by_id.get("POSTGRESQL")
    if postgres is not None:
        postgres["adjacent"] = [a for a in postgres["adjacent"] if a["term"] != "SQL"]
    if sql is not None:
        sql["adjacent"] = [a for a in sql["adjacent"] if a["term"] != "POSTGRESQL"]

    arabic = by_id.get("ARABIC_LANGUAGE")
    bilingual = by_id.get("BILINGUAL_ARABIC_ENGLISH")
    if arabic is not None and bilingual is not None:
        arabic["aliases"] = list(dict.fromkeys([*arabic["aliases"], *bilingual["aliases"]]))
        arabic["from_requirements"] = list(
            dict.fromkeys([*arabic["from_requirements"], *bilingual["from_requirements"]])
        )
        arabic["adjacent"] = [a for a in arabic["adjacent"] if a["term"] != "BILINGUAL_ARABIC_ENGLISH"]
        raw["terms"] = [t for t in raw["terms"] if t["id"] != "BILINGUAL_ARABIC_ENGLISH"]
        for t in raw["terms"]:
            t["adjacent"] = [a for a in t["adjacent"] if a["term"] != "BILINGUAL_ARABIC_ENGLISH"]

    return raw


def skill_manual_edges() -> list[Edge]:
    return [Edge(pillar="skill", src="POSTGRESQL", dst="SQL", kind="narrower", weight=1.0)]


# --- occupation pillar ---------------------------------------------------------
#
# Titles the draft left unmapped or under-attached. Reasoning follows P0's
# reviewed judgment on the same corpus (spike/p02b_patch.py): a title that
# performs a genuinely different function (closing vs. prospecting, retention
# vs. reactive support) gets partial credit via `related`, never zero and never
# full credit. A title that is simply a broader or narrower phrasing of the same
# job (Recruiter / Talent Acquisition Specialist for a technical recruiter, HR
# Generalist for an HR business partner, Telesales Agent for outbound sales)
# attaches as a full-credit surface — the underlying work is the same.

_NARROWER_ADDITIONS = {
    "SALES_DEVELOPMENT_REPRESENTATIVE": ["Telesales Agent"],
}

_RELATED_ADDITIONS = {
    "BACKEND_ENGINEER": [("Software Engineer", 0.7)],
    "TECHNICAL_RECRUITER": [("Talent Acquisition Specialist", 0.7), ("Recruiter", 0.6)],
    "HR_BUSINESS_PARTNER": [("HR Generalist", 0.7)],
    "SALES_DEVELOPMENT_REPRESENTATIVE": [("Sales Executive", 0.7), ("Account Executive", 0.7)],
    "PRODUCT_MARKETING_MANAGER": [("Marketing Manager", 0.6), ("Growth Marketing Specialist", 0.6)],
    "CUSTOMER_SUPPORT_SPECIALIST": [("Customer Success Associate", 0.7)],
}


def apply_occupation_patches(raw: dict) -> dict:
    raw = {**raw, "terms": [dict(t) for t in raw["terms"]], "unmapped": list(raw["unmapped"])}
    by_id = {t["id"]: t for t in raw["terms"]}
    rescued: set[str] = set()

    for term_id, titles in _NARROWER_ADDITIONS.items():
        t = by_id[term_id]
        for title in titles:
            if title not in t["narrower"]:
                t["narrower"] = [*t["narrower"], title]
            rescued.add(title)

    for term_id, pairs in _RELATED_ADDITIONS.items():
        t = by_id[term_id]
        existing = {r["phrase"] for r in t["related"]}
        for phrase, weight in pairs:
            if phrase not in existing:
                t["related"] = [*t["related"], {"phrase": phrase, "weight": weight}]
            rescued.add(phrase)

    raw["unmapped"] = [u for u in raw["unmapped"] if u["string"] not in rescued]
    return raw


def build_patched_taxonomy(version: str, skill_raw: dict, occupation_raw: dict) -> Taxonomy:
    skill_raw = apply_skill_patches(skill_raw)
    occupation_raw = apply_occupation_patches(occupation_raw)

    skill_terms, skill_edges = raw_to_terms("skill", skill_raw)
    occ_terms, occ_edges = raw_to_terms("occupation", occupation_raw)

    all_terms = tuple(skill_terms) + tuple(occ_terms)
    all_edges = tuple(skill_edges) + tuple(skill_manual_edges()) + tuple(occ_edges)
    return Taxonomy(version=version, terms=all_terms, edges=all_edges)


if __name__ == "__main__":
    import json
    from engine.taxonomy import build_db, save_taxonomy

    skill_raw = json.load(open("data/taxonomy_skill_raw.json"))
    occ_raw = json.load(open("data/taxonomy_occupation_raw.json"))
    tax = build_patched_taxonomy("2026-08-24.1", skill_raw, occ_raw)
    save_taxonomy(tax, "data/taxonomy.json")
    build_db(tax, "data/taxonomy.db")
    print("Rebuilt data/taxonomy.json and data/taxonomy.db successfully!")
