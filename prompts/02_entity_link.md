# 02_entity_link — links one candidate's fields to the skill vocabulary

Run twice per candidate, never once over the whole profile: once for `{LIST_FIELDS}`
(the explicit, enumerated fields), once for `{PROSE_FIELDS}` (free text). Putting an
enumerated list and free prose in the same call was tested and measured worse — the
explicit list wins the model's attention and prose gets skimmed. A call is skipped
entirely when every field in its group is blank or `-` for this candidate.

Job titles are never read here. A job title implies an occupation, never a skill —
that resolution is separate, deterministic, and costs no model call.

## System

You are given text from one candidate's profile and a fixed skill vocabulary.
Decide what the text shows the person can do, then map each capability to one
vocabulary term.

- A named tool, language or standard is a capability. So is an ACTIVITY described
  without naming a skill — read for what the person DID, not only for words that
  match the vocabulary. Outside this dataset: "wrote the onboarding guide for new
  hires" implies technical writing; "ran the quarterly forecast review" implies
  forecasting; "shipped the payments integration end to end" implies API
  development.
- `relation: "met"` — the text names the term itself, an alias of it, or a strict
  specialisation (a vocabulary entry's `includes` list). Full credit.
- `relation: "adjacent"` — the text is partial, discountable evidence: an activity
  that plausibly demonstrates the term without naming it, or a vocabulary entry's
  own `related` phrase. Never invent a weight; the system already has one for known
  `related` phrases, and applies a default for anything else you judge adjacent.
- `phrase` must be an exact, minimal substring of the field text you read it from —
  quote only what supports the capability, not the whole line. This is checked
  afterward: a phrase that isn't a real substring is a rejected mention, not
  a smaller violation.
- `field` is which of the given fields the phrase came from.
- Return nothing for text the vocabulary has no term for. That is often correct —
  most candidate skill strings in this corpus map to no requirement term at all.
- Calibration, entirely outside this dataset: "Rails" against a vocabulary
  containing RUBY is `met` (Rails is a Ruby framework, strict specialisation).
  "Kotlin" against a vocabulary containing JAVA is `adjacent` (both target the JVM,
  neither implies the other). "Snowflake" against a vocabulary containing BIGQUERY
  is `adjacent` (comparable data warehouses, not the same product).
- Account for every line you are given — read it, decide, move on. Silence is not
  evidence: if nothing in the text supports a term, do not emit a mention for it.

## User

CANDIDATE {candidate_id} — {group_label}
{profile_lines}

VOCABULARY (id | aka: aliases | includes: full-credit specialisations | related: phrase(weight) pairs)
{vocabulary_block}

Emit every mention.

## Schema

```json
{
  "type": "object", "additionalProperties": false,
  "required": ["mentions"],
  "properties": {
    "mentions": {"type": "array", "items": {
      "type": "object", "additionalProperties": false,
      "required": ["term", "relation", "phrase", "field"],
      "properties": {
        "term": {"type": "string", "enum": []},
        "relation": {"type": "string", "enum": ["met", "adjacent"]},
        "phrase": {"type": "string"},
        "field": {"type": "string", "enum": []}
      }
    }}
  }
}
```

`term.enum` is filled in with every skill term id at call time. `field.enum` is
filled in with the field names present in this call's group.
