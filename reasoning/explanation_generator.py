"""
reasoning/explanation_generator.py

Generates the `reasoning` column required by submission_spec.docx section
3, and the fuller strengths/weaknesses/missing_skills breakdown for
internal review (outputs/ranked_candidates_full.csv).

Hard constraints, straight from the spec's Stage 4 manual-review checklist:
  - No hallucination: every claim must trace to an actual field value.
  - No templated reasoning that just swaps in the name.
  - Reasoning's tone must match the rank (a rank-95 candidate shouldn't
    read like a glowing rank-3 candidate).
  - Honest concerns should surface, not just praise.

This is a deliberately non-LLM component: composing grounded sentences
from structured facts, with randomized-but-deterministic (seeded per
candidate_id) sentence-template selection for lexical variety. Using a
hosted LLM here would violate the "no network calls during ranking"
compute constraint anyway -- this is the correct engineering choice, not
a compromise.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _seed_for(candidate_id: str) -> int:
    return int(hashlib.sha256(candidate_id.encode()).hexdigest(), 16) % (2**32)


STRENGTH_OPENERS = [
    "{title} with {yoe:.1f} yrs experience",
    "{title}, {yoe:.1f} years in",
    "{yoe:.1f}-year {title}",
]

MATCH_CLAUSES = [
    "matches {n} of the JD's core required skills ({skills})",
    "covers {n} required skills directly ({skills})",
    "hits {n} of the mandatory skills ({skills})",
]

CONCERN_TEMPLATES = {
    "title_mismatch": "current title reads as a non-AI role, which weighs against fit despite the skills listed",
    "missing_skills": "missing {n} of the JD's core required skills ({skills})",
    "consulting_only": "entire career history is at consulting/IT-services firms with no product-company experience",
    "inactive": "hasn't been active on the platform in {days} days",
    "low_response": "recruiter response rate is only {rate:.0%}",
    "long_notice": "{notice}-day notice period is above the JD's sub-30-day preference",
    "over_experience": "experience ({yoe:.1f} yrs) runs above the JD's stated 5-9 year band",
    "under_experience": "experience ({yoe:.1f} yrs) is below the JD's stated 5-9 year band",
}

TAIL_TEMPLATES_STRONG = [
    "Strong overall fit for this role.",
    "One of the stronger matches in the pool for this JD.",
    "Solid alignment between profile and what this role needs.",
]
TAIL_TEMPLATES_MID = [
    "Reasonable fit with some gaps worth a recruiter screen.",
    "Plausible fit; a few open questions to confirm in screening.",
    "Worth a look, with caveats noted above.",
]
TAIL_TEMPLATES_WEAK = [
    "Included near the tail of the top 100 mainly on adjacent signal.",
    "Marginal fit; ranked here on partial overlap rather than a strong match.",
    "Weakest of the included set -- borderline relative to the JD.",
]


def _pick(rng: np.random.RandomState, options: list[str]) -> str:
    return options[rng.randint(len(options))]


def generate_reasoning(row: dict, rank: int, total: int) -> str:
    rng = np.random.RandomState(_seed_for(row["candidate_id"]))

    title = row.get("current_title") or "Unknown title"
    yoe = row.get("years_of_experience") or 0.0
    matched_required = row.get("matched_required_skills") or []
    missing_required = row.get("missing_required_skills") or []
    title_reasons = row.get("title_career_reasons") or []

    opener = _pick(rng, STRENGTH_OPENERS).format(title=title, yoe=yoe)

    clauses = []
    if matched_required:
        shown = matched_required[:3]
        clauses.append(
            _pick(rng, MATCH_CLAUSES).format(n=len(matched_required), skills=", ".join(shown))
        )
    elif title_reasons:
        clauses.append(title_reasons[0])

    concerns = []
    if row.get("title_career_score", 1.0) < 0.4:
        concerns.append(CONCERN_TEMPLATES["title_mismatch"])
    if len(missing_required) >= 3:
        concerns.append(
            CONCERN_TEMPLATES["missing_skills"].format(
                n=len(missing_required), skills=", ".join(missing_required[:3])
            )
        )
    if row.get("consulting_firm_only"):
        concerns.append(CONCERN_TEMPLATES["consulting_only"])
    days_inactive = row.get("days_since_active", 0) or 0
    if days_inactive > 90:
        concerns.append(CONCERN_TEMPLATES["inactive"].format(days=int(days_inactive)))
    rr = row.get("recruiter_response_rate")
    if rr is not None and rr < 0.2:
        concerns.append(CONCERN_TEMPLATES["low_response"].format(rate=rr))
    notice = row.get("notice_period_days")
    if notice is not None and notice > 90:
        concerns.append(CONCERN_TEMPLATES["long_notice"].format(notice=int(notice)))
    if yoe and yoe > 9:
        concerns.append(CONCERN_TEMPLATES["over_experience"].format(yoe=yoe))
    elif yoe and yoe < 5:
        concerns.append(CONCERN_TEMPLATES["under_experience"].format(yoe=yoe))

    # Tone must match rank: strong ranks get a confident tail only if
    # there are few/no concerns; low ranks always get a hedged tail even
    # if the additive score looks decent, so the language doesn't read as
    # independent of the ranking.
    pctile = rank / max(total, 1)
    if pctile <= 0.15 and len(concerns) <= 1:
        tail = _pick(rng, TAIL_TEMPLATES_STRONG)
    elif pctile >= 0.7 or len(concerns) >= 3:
        tail = _pick(rng, TAIL_TEMPLATES_WEAK)
    else:
        tail = _pick(rng, TAIL_TEMPLATES_MID)

    sentence_parts = [opener]
    if clauses:
        sentence_parts.append("; " + clauses[0])
    lead = "".join(sentence_parts) + "."

    concern_text = ""
    if concerns:
        # cap at two concerns so the string stays a tight 1-2 sentences,
        # per spec ("A 1-2 sentence justification")
        concern_text = " " + " ".join(c[0].upper() + c[1:] + "." for c in concerns[:2])

    reasoning = f"{lead}{concern_text} {tail}".strip()
    return reasoning
