"""
submission/submission_generator.py

Takes the scored + reasoned top-100 DataFrame and writes submission.csv in
the exact format required by submission_spec.docx section 2-3:

  - header exactly: candidate_id,rank,score,reasoning
  - exactly 100 data rows
  - rank 1..100, each used exactly once
  - score non-increasing with rank (ties allowed)
  - ties broken deterministically by candidate_id ascending
  - UTF-8 encoding, .csv extension
"""

from __future__ import annotations

import pandas as pd

from utils.config import SUBMISSION_CSV, FINAL_TOP_N, OUTPUTS_DIR


def build_submission(scored_df: pd.DataFrame, out_path=SUBMISSION_CSV) -> pd.DataFrame:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = scored_df.copy()
    # Deterministic ordering: overall_score descending, candidate_id
    # ascending as the tie-break (matches spec section 3 exactly).
    df = df.sort_values(by=["overall_score", "candidate_id"], ascending=[False, True])
    df = df.head(FINAL_TOP_N).reset_index(drop=True)

    if len(df) < FINAL_TOP_N:
        raise ValueError(
            f"Only {len(df)} candidates available after filtering/retrieval; "
            f"need {FINAL_TOP_N}. Increase RETRIEVAL_TOP_K in utils/config.py."
        )

    df["score"] = df["overall_score"].round(4)

    # Re-sort using the ROUNDED score for the tie-break: rounding to 4
    # decimals can turn near-ties into exact ties that weren't ties in the
    # unrounded overall_score, and the spec's tie-break (candidate_id
    # ascending) is defined on the score column that actually ships in
    # the CSV, not on the internal float.
    df = df.sort_values(by=["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
    df["rank"] = range(1, FINAL_TOP_N + 1)

    # Enforce non-increasing score by rank as a hard invariant (floating
    # point + rounding could in principle produce a tiny inversion at a
    # tie boundary; clip defensively rather than trust the sort alone).
    scores = df["score"].tolist()
    for i in range(1, len(scores)):
        if scores[i] > scores[i - 1]:
            scores[i] = scores[i - 1]
    df["score"] = scores

    out = df[["candidate_id", "rank", "score", "reasoning"]]
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote submission -> {out_path} ({len(out)} rows)")
    return out
