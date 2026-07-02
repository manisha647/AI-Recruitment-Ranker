"""
candidate_intelligence.py

Stage 1 of the offline precompute pipeline.

Reads data/candidates.jsonl (100,000 profiles), extracts structured,
JD-agnostic features per candidate, builds the text blob that will be
embedded, runs honeypot detection, and writes everything to
artifacts/candidate_features.parquet.

This module knows nothing about any specific job description -- that
separation matters because the same candidate pool should be re-usable
for the *next* JD Redrob throws at this ranker without re-parsing 100k
JSON records. JD-specific matching happens later, in ranking/hybrid_ranker.py.

Usage:
    python -m precompute.candidate_intelligence
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime

import pandas as pd

from utils.config import (
    CANDIDATES_JSONL,
    CANDIDATE_FEATURES_PARQUET,
    ARTIFACTS_DIR,
    REFERENCE_DATE,
)
from utils.honeypot import detect_honeypot

EXPERT_LEVELS = {"expert", "advanced"}
REFERENCE_DATE_OBJ = datetime.strptime(REFERENCE_DATE, "%Y-%m-%d").date()


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _build_embedding_text(candidate: dict) -> str:
    """
    Build the text blob to embed. Per the spec we want semantic signal
    only: headline, summary, career descriptions, skills, education.
    Explicitly EXCLUDED: redrob_signals (behavioral data), which stays
    structured/numeric and is applied later as a scoring modifier, not
    smeared into the embedding where it would just add noise.
    """
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    education = candidate.get("education", []) or []

    parts = []

    headline = profile.get("headline", "") or ""
    summary = profile.get("summary", "") or ""
    current_title = profile.get("current_title", "") or ""
    current_industry = profile.get("current_industry", "") or ""
    if headline:
        parts.append(headline)
    if current_title:
        parts.append(f"Current role: {current_title} in {current_industry}.")
    if summary:
        parts.append(summary)

    # Career history: title + description per role, most recent first
    # (career_history in this dataset is not guaranteed sorted, so we sort
    # by start_date descending). Cap at 6 roles to keep embedding text
    # bounded for very long careers.
    sorted_roles = sorted(
        career_history,
        key=lambda ch: ch.get("start_date") or "",
        reverse=True,
    )
    for role in sorted_roles[:6]:
        title = role.get("title", "")
        company_industry = role.get("industry", "")
        desc = role.get("description", "")
        parts.append(f"Role: {title} ({company_industry}). {desc}")

    # Skills: names only, comma-joined, deduped preserving order
    seen = set()
    skill_names = []
    for s in skills:
        name = (s.get("name") or "").strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            skill_names.append(name)
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names) + ".")

    # Education: degree + field + institution
    for edu in education:
        degree = edu.get("degree", "")
        field = edu.get("field_of_study", "")
        institution = edu.get("institution", "")
        if degree or field:
            parts.append(f"Education: {degree} in {field}, {institution}.")

    return " ".join(p.strip() for p in parts if p and p.strip())


def _extract_features(candidate: dict) -> dict:
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    education = candidate.get("education", []) or []
    languages = candidate.get("languages", []) or []
    signals = candidate.get("redrob_signals", {}) or {}

    # --- skills ---
    skill_names = [(s.get("name") or "").strip().lower() for s in skills if s.get("name")]
    num_expert_skills = sum(
        1 for s in skills if str(s.get("proficiency", "")).lower() in EXPERT_LEVELS
    )
    total_endorsements = sum(s.get("endorsements") or 0 for s in skills)
    total_skill_months = sum(s.get("duration_months") or 0 for s in skills)
    skill_proficiency_map = {
        (s.get("name") or "").strip().lower(): str(s.get("proficiency", "")).lower()
        for s in skills
        if s.get("name")
    }
    skill_duration_map = {
        (s.get("name") or "").strip().lower(): (s.get("duration_months") or 0)
        for s in skills
        if s.get("name")
    }

    # --- career history ---
    career_titles = [(ch.get("title") or "").strip().lower() for ch in career_history]
    career_descriptions = " ".join((ch.get("description") or "") for ch in career_history)
    total_role_months = sum(ch.get("duration_months") or 0 for ch in career_history)
    num_roles = len(career_history)
    consulting_firm_only = bool(career_history) and all(
        any(firm in (ch.get("company") or "").lower() for firm in
            ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"])
        for ch in career_history
    )

    # --- education ---
    tiers = [edu.get("tier") for edu in education if edu.get("tier")]
    best_tier = min(tiers) if tiers else None  # "tier_1" < "tier_2" lexicographically works here
    degree_fields = [
        f"{edu.get('degree', '')} {edu.get('field_of_study', '')}".strip().lower()
        for edu in education
    ]

    # --- behavioral / redrob signals (kept structured, not embedded) ---
    last_active = _parse_date(signals.get("last_active_date"))
    days_since_active = (REFERENCE_DATE_OBJ - last_active).days if last_active else 9999
    signup = _parse_date(signals.get("signup_date"))
    tenure_days = (REFERENCE_DATE_OBJ - signup).days if signup else None

    # --- honeypot detection ---
    is_honeypot, honeypot_reasons = detect_honeypot(candidate)

    embedding_text = _build_embedding_text(candidate)

    return {
        "candidate_id": candidate.get("candidate_id"),
        "anonymized_name": profile.get("anonymized_name"),
        "headline": profile.get("headline"),
        "current_title": (profile.get("current_title") or ""),
        "current_title_lower": (profile.get("current_title") or "").lower(),
        "current_company": profile.get("current_company"),
        "current_company_size": profile.get("current_company_size"),
        "current_industry": profile.get("current_industry"),
        "location": profile.get("location"),
        "country": profile.get("country"),
        "years_of_experience": profile.get("years_of_experience"),

        "skill_names": skill_names,
        "num_skills": len(skill_names),
        "num_expert_skills": num_expert_skills,
        "total_endorsements": total_endorsements,
        "total_skill_months": total_skill_months,
        "skill_proficiency_map": skill_proficiency_map,
        "skill_duration_map": skill_duration_map,

        "career_titles": career_titles,
        "career_descriptions": career_descriptions,
        "total_role_months": total_role_months,
        "num_roles": num_roles,
        "consulting_firm_only": consulting_firm_only,

        "education_best_tier": best_tier,
        "education_degree_fields": degree_fields,

        "languages": [l.get("language") for l in languages if l.get("language")],

        # behavior signals used later as a multiplicative modifier
        "profile_completeness_score": signals.get("profile_completeness_score"),
        "open_to_work_flag": signals.get("open_to_work_flag"),
        "days_since_active": days_since_active,
        "tenure_days": tenure_days,
        "recruiter_response_rate": signals.get("recruiter_response_rate"),
        "avg_response_time_hours": signals.get("avg_response_time_hours"),
        "notice_period_days": signals.get("notice_period_days"),
        "willing_to_relocate": signals.get("willing_to_relocate"),
        "preferred_work_mode": signals.get("preferred_work_mode"),
        "github_activity_score": signals.get("github_activity_score"),
        "interview_completion_rate": signals.get("interview_completion_rate"),
        "offer_acceptance_rate": signals.get("offer_acceptance_rate"),
        "verified_email": signals.get("verified_email"),
        "verified_phone": signals.get("verified_phone"),
        "linkedin_connected": signals.get("linkedin_connected"),
        "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d"),
        "search_appearance_30d": signals.get("search_appearance_30d"),

        "is_honeypot": is_honeypot,
        "honeypot_reasons": "; ".join(honeypot_reasons) if honeypot_reasons else "",

        "embedding_text": embedding_text,
    }


def build_candidate_intelligence(
    input_path=CANDIDATES_JSONL,
    output_path=CANDIDATE_FEATURES_PARQUET,
    log_every: int = 10000,
) -> pd.DataFrame:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = []
    n_bad = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                rows.append(_extract_features(candidate))
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                n_bad += 1
                print(f"[warn] skipping malformed record at line {i}: {e}", file=sys.stderr)
            if i % log_every == 0:
                elapsed = time.time() - t0
                print(f"  parsed {i:,} candidates ({elapsed:.1f}s elapsed)")

    df = pd.DataFrame(rows)
    n_honeypots = int(df["is_honeypot"].sum())
    print(f"Parsed {len(df):,} candidates ({n_bad} malformed/skipped) in {time.time() - t0:.1f}s")
    print(f"Flagged {n_honeypots} candidates ({n_honeypots / len(df) * 100:.2f}%) as likely honeypots")

    df.to_parquet(output_path, index=False)
    print(f"Saved candidate_features.parquet -> {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return df


if __name__ == "__main__":
    build_candidate_intelligence()
