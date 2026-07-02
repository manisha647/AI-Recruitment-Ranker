"""
jd/jd_understanding.py

Reads data/jd.txt and produces a structured jd_features dict, saved to
artifacts/jd_features.json. This is intentionally hybrid: a lightweight
rule/keyword extraction layer for the things the JD is explicit about
(title, experience band, locations, disqualifying backgrounds, consulting-
only exclusion, notice period preference), plus a full-text "semantic_text"
blob that captures everything else -- especially the JD's extensive prose
about *intent* ("we'd rather tilt toward shipper than researcher", "title-
chasers are not a fit") -- for the embedding step to pick up on, since no
keyword list will ever fully capture that.

This module is deliberately readable/auditable rather than clever: at
Stage 5 you may have to explain exactly why the ranker treats a phrase as
a positive or negative signal, and "grep for these words, here's the list"
is a much easier thing to defend than an opaque prompt to an LLM (which
the compute constraints forbid using during ranking anyway).

Usage:
    python -m jd.jd_understanding
"""

from __future__ import annotations

import json
import re

from utils.config import (
    JD_TXT,
    JD_FEATURES_JSON,
    ARTIFACTS_DIR,
    JD_REQUIRED_SKILLS,
    JD_PREFERRED_SKILLS,
    JD_POSITIVE_TITLE_SIGNALS,
    JD_NEGATIVE_TITLE_SIGNALS,
    JD_DISQUALIFYING_BACKGROUNDS,
    JD_CONSULTING_FIRMS,
    JD_PREFERRED_LOCATIONS,
    JD_MIN_YEARS,
    JD_MAX_YEARS,
)


def _read_jd_text(path=JD_TXT) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_jd_features(jd_text: str | None = None) -> dict:
    if jd_text is None:
        jd_text = _read_jd_text()
    lower = jd_text.lower()

    # Experience band: prefer an explicit "X-Y years" style match near the
    # word "experience"; fall back to config defaults (curated from the JD)
    # if the free text doesn't match cleanly -- this module should degrade
    # gracefully on a *different* JD next time it's pointed at one.
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*years?", lower)
    min_years, max_years = (int(m.group(1)), int(m.group(2))) if m else (JD_MIN_YEARS, JD_MAX_YEARS)

    job_title_match = re.search(r"job description:\s*(.+)", jd_text, re.IGNORECASE)
    job_title = job_title_match.group(1).strip() if job_title_match else "Senior AI Engineer"

    work_mode = "hybrid" if "hybrid" in lower else (
        "remote" if "remote" in lower else ("onsite" if "onsite" in lower else "unspecified")
    )

    mandatory_skills = [s for s in JD_REQUIRED_SKILLS if s in lower]
    preferred_skills = [s for s in JD_PREFERRED_SKILLS if s in lower]

    locations = [loc for loc in JD_PREFERRED_LOCATIONS if loc in lower]

    positive_title_signals = list(JD_POSITIVE_TITLE_SIGNALS)
    negative_title_signals = list(JD_NEGATIVE_TITLE_SIGNALS)
    disqualifying_backgrounds = [b for b in JD_DISQUALIFYING_BACKGROUNDS if b in lower]
    consulting_firms = list(JD_CONSULTING_FIRMS)

    # Recruiter intent, in the JD's own words -- kept as short spans for
    # transparency/interview defensibility rather than re-derived.
    positive_recruiter_intent = [
        "production deployment of embeddings/retrieval/ranking systems",
        "shipped an end-to-end ranking, search, or recommendation system at scale",
        "strong Python and rigorous evaluation-framework experience (NDCG, MRR, MAP)",
        "comfortable shipping fast even when the underlying ML is imperfect",
        "6-8 years total experience, 4-5 in applied ML/AI at product companies",
        "based in or willing to relocate to Pune/Noida",
    ]
    negative_recruiter_intent = [
        "pure research background with no production deployment",
        "title-chasing career trajectory (company-hopping every ~1.5 years for titles)",
        "framework-tutorial-only GitHub / blog (LangChain demos, no systems thinking)",
        "entire career at a single consulting firm (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini) with no product-company experience",
        "primary expertise in computer vision, speech, or robotics without NLP/IR exposure",
        "senior engineer who has not written production code in 18+ months (pure architecture/tech-lead track)",
        "recent (<12 months) LangChain-calls-OpenAI experience with no pre-LLM-era ML production background",
    ]
    behavior_preferences = [
        "actively open to work / recently active on platform",
        "reasonable notice period (sub-30-days strongly preferred, up to 30 buyout)",
        "healthy recruiter response rate (signal they're reachable / in-market)",
    ]

    features = {
        "job_title": job_title,
        "min_years": min_years,
        "max_years": max_years,
        "work_mode": work_mode,
        "preferred_locations": locations or JD_PREFERRED_LOCATIONS,
        "mandatory_skills": mandatory_skills or JD_REQUIRED_SKILLS,
        "preferred_skills": preferred_skills or JD_PREFERRED_SKILLS,
        "positive_title_signals": positive_title_signals,
        "negative_title_signals": negative_title_signals,
        "disqualifying_backgrounds": disqualifying_backgrounds or JD_DISQUALIFYING_BACKGROUNDS,
        "consulting_only_exclusion": consulting_firms,
        "positive_recruiter_intent": positive_recruiter_intent,
        "negative_recruiter_intent": negative_recruiter_intent,
        "behavior_preferences": behavior_preferences,
        "max_preferred_notice_days": 30,
        "education_note": "No hard degree requirement stated in JD; weighted lightly.",
        "semantic_text": jd_text,
    }
    return features


def save_jd_features(features: dict, path=JD_FEATURES_JSON) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)
    print(f"Saved JD features -> {path}")


if __name__ == "__main__":
    feats = extract_jd_features()
    save_jd_features(feats)
    print(json.dumps({k: v for k, v in feats.items() if k != "semantic_text"}, indent=2))
