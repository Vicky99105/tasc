# 01_vocabulary — builds one pillar of the taxonomy (skill or occupation)

Used once per pillar during `engine/build_taxonomy.py`. Output is reviewed by hand
before it becomes `data/taxonomy.json` — this prompt produces a draft, not the
artifact. `{PILLAR_NOUN}` is "skill" for the skill pillar, "occupation" for the
occupation pillar. `{CONCEPT}` is what a term denotes: "a skill, tool, language,
domain or knowledge area" vs "a job title / occupation".

## System

You build a controlled vocabulary for matching job requirements to candidate
records. This vocabulary covers **{PILLAR_NOUN}s only** — {CONCEPT}.

MEMBERSHIP RULE (absolute):
- Only REQUIREMENT strings create terms. Every requirement string must resolve to
  exactly one term.
- CANDIDATE strings never create a term. A candidate string either attaches to an
  existing term (as an alias if it means the same thing, as `narrower` if it is a
  strict specialisation, or as a `related` surface if it is partial, discountable
  evidence), or it goes in `unmapped` with a reason. Never invent a term to
  accommodate a candidate string.

TERM GRANULARITY:
- Terms are ATOMIC. If one requirement string names several distinct {PILLAR_NOUN}s
  separated by a slash, emit them as SEPARATE terms. Do not merge them into a
  combined term — the requirement's "either one is fine" meaning is handled outside
  this vocabulary.
- A parenthesised vendor or example list means the same: emit the general term, and
  put the named items in its `narrower` list.

ALIAS vs NARROWER vs ADJACENT vs RELATED:
- `aliases`: the SAME concept, different spelling or word order. Full credit,
  interchangeable without qualification.
- `narrower`: a strict specialisation (full credit toward the parent). Evidence of
  the child is complete evidence of the parent. Example: "Rails" is narrower than
  "Ruby". "Spark SQL" is narrower than "SQL".
- `adjacent`: two DIFFERENT terms where evidence of one is partial, discountable
  evidence of the other. Symmetric, must only reference other terms in this output.
  Weight 0.6 unless you have a specific reason to differ. Example: "Kotlin" and
  "Java" are adjacent (both target the JVM). "Snowflake" and "BigQuery" are
  adjacent.
- `related`: a PHRASE (not a synonym) that, if someone is described as having done
  it, is partial evidence FOR this term without ever naming it — an activity that
  demonstrates the {PILLAR_NOUN} without saying its name. Weight reflects how
  strongly the activity implies the term, never 1.0 (that would make it an alias).
  Example (outside this dataset): "wrote a Dockerfile for the deploy pipeline" is
  `related` to DOCKER at weight 0.7 — it never says "Docker" but a reader would
  credit it. Only add a `related` phrase if it is a plausible way someone would
  describe doing the thing in a resume, not a synonym for the term's name.

HARD CONSTRAINTS:
- Group on MEANING, never on spelling. Two strings that share characters but denote
  different things stay separate. Two strings with zero character overlap that
  denote the same thing group together.
- Never make a term a child of another merely because the words look similar.
- A term id is UPPER_SNAKE_CASE, derived from the meaning.
- Output only {PILLAR_NOUN}s. {OUT_OF_SCOPE}

## User

REQUIREMENT STRINGS (these define membership; all {n_requirements} must resolve to
a term):
{requirement_strings}

CANDIDATE STRINGS (each one either attaches to a term above, contributes a
`related` phrase, or is `unmapped` with a reason):
{candidate_strings}

Emit the vocabulary.

## Schema

```json
{
  "type": "object", "additionalProperties": false,
  "required": ["terms", "unmapped"],
  "properties": {
    "terms": {"type": "array", "items": {
      "type": "object", "additionalProperties": false,
      "required": ["id", "canonical", "aliases", "narrower", "adjacent", "related", "from_requirements"],
      "properties": {
        "id": {"type": "string"},
        "canonical": {"type": "string"},
        "aliases": {"type": "array", "items": {"type": "string"}},
        "narrower": {"type": "array", "items": {"type": "string"}},
        "adjacent": {"type": "array", "items": {
          "type": "object", "additionalProperties": false,
          "required": ["term", "weight"],
          "properties": {"term": {"type": "string"}, "weight": {"type": "number"}}
        }},
        "related": {"type": "array", "items": {
          "type": "object", "additionalProperties": false,
          "required": ["phrase", "weight"],
          "properties": {"phrase": {"type": "string"}, "weight": {"type": "number"}}
        }},
        "from_requirements": {"type": "array", "items": {"type": "string"}}
      }
    }},
    "unmapped": {"type": "array", "items": {
      "type": "object", "additionalProperties": false,
      "required": ["string", "reason"],
      "properties": {"string": {"type": "string"}, "reason": {"type": "string"}}
    }}
  }
}
```

`{OUT_OF_SCOPE}` — skill pillar: "Job titles are handled in a separate pillar."
Occupation pillar: "Skills, tools and certifications are handled in a separate
pillar; a job title must never imply a skill."
