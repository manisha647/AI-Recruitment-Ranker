"""
Honeypot detection: flags candidates with internally-impossible profiles.

Per redrob_signals_doc.docx, the dataset seeds ~80 honeypot candidates with
profiles like "8 years of experience at a company founded 3 years ago" or
"'expert' proficiency in 10 skills with 0 years used". We don't have ground
truth on when companies were founded, so these rules detect *internal*
inconsistency in the fields we do have. This runs during precompute (cheap,
vectorizable, one pass over the candidate pool) and produces a boolean flag
+ a human-readable reason, both persisted to the parquet artifact so the
ranking step can penalize honeypots without re-deriving anything.
"""

from __future__ import annotations

from utils.config import HONEYPOT_RULES

EXPERT_LEVELS = {"expert", "advanced"}


def detect_honeypot(candidate: dict) -> tuple[bool, list[str]]:
    """Return (is_honeypot, reasons) for a single parsed candidate dict."""
    reasons: list[str] = []

    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    education = candidate.get("education", []) or []

    years_of_experience = profile.get("years_of_experience") or 0

    # Rule 1: expert/advanced proficiency claimed with ~zero hands-on time,
    # repeated across multiple skills (one occurrence can be legitimate --
    # e.g. a fast learner, or a formal certification -- so we require a
    # cluster of them before calling it a honeypot).
    zero_duration_expert_skills = [
        s.get("name", "?")
        for s in skills
        if str(s.get("proficiency", "")).lower() in EXPERT_LEVELS
        and (s.get("duration_months") or 0) <= HONEYPOT_RULES["expert_zero_duration_months_max"]
    ]
    if len(zero_duration_expert_skills) >= HONEYPOT_RULES["expert_zero_duration_skill_count"]:
        reasons.append(
            f"{len(zero_duration_expert_skills)} skills claimed at expert/advanced "
            f"level with ~0 months hands-on use: {zero_duration_expert_skills[:5]}"
        )

    # Rule 2: sum of career_history duration_months wildly exceeds the
    # candidate's stated total years_of_experience -- i.e. the roles listed
    # could not have all been worked sequentially given the stated total.
    total_role_months = sum(ch.get("duration_months") or 0 for ch in career_history)
    stated_months = years_of_experience * 12
    if stated_months > 0 and total_role_months > stated_months * HONEYPOT_RULES["experience_overstatement_ratio"]:
        reasons.append(
            f"career_history sums to {total_role_months} months but "
            f"years_of_experience implies ~{stated_months:.0f} months "
            f"(ratio {total_role_months / stated_months:.2f}x)"
        )

    # Rule 3: more than one role simultaneously marked is_current -- a
    # candidate cannot hold two current full-time jobs in this dataset's
    # semantics.
    current_roles = sum(1 for ch in career_history if ch.get("is_current"))
    if current_roles > HONEYPOT_RULES["max_concurrent_current_roles"]:
        reasons.append(f"{current_roles} career_history entries marked is_current simultaneously")

    # Rule 4: education end_year before start_year, or end_year in the
    # future beyond signup (basic timeline sanity).
    for edu in education:
        sy, ey = edu.get("start_year"), edu.get("end_year")
        if sy and ey and ey < sy:
            reasons.append(f"education end_year ({ey}) before start_year ({sy}) at {edu.get('institution', '?')}")

    # Rule 5: a single role's duration_months alone exceeds total stated
    # years_of_experience by a wide margin.
    for ch in career_history:
        dm = ch.get("duration_months") or 0
        if stated_months > 0 and dm > stated_months * HONEYPOT_RULES["experience_overstatement_ratio"]:
            reasons.append(
                f"single role at {ch.get('company', '?')} lists {dm} months, "
                f"exceeding stated total experience (~{stated_months:.0f} months)"
            )
            break

    return (len(reasons) > 0, reasons)
