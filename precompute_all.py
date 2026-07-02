#!/usr/bin/env python3
"""
precompute_all.py -- run the full offline precompute pipeline once.

    python precompute_all.py

This is allowed to take longer than 5 minutes and (if you switch
EMBEDDING_BACKEND to "sentence_transformer" in utils/config.py) is allowed
to use a GPU -- none of that is compute-constrained, per submission_spec.docx
section 10.3: "If your system requires pre-computation (e.g., generating
embeddings), document this clearly -- pre-computation may exceed the
5-minute window, but the ranking step that produces the CSV must complete
within it."

Produces (all under artifacts/):
    candidate_features.parquet   (precompute.candidate_intelligence)
    candidate_embeddings.npy     (precompute.embed_candidates)
    candidate_ids.npy            (precompute.embed_candidates)
    embedder.joblib              (precompute.embed_candidates)
    candidate_index.faiss        (precompute.build_index)

Run this once after cloning the repo (or whenever candidates.jsonl
changes); rank.py then reuses these artifacts for every JD without
re-touching the 100k-candidate pool.
"""

import time

from precompute.candidate_intelligence import build_candidate_intelligence
from precompute.embed_candidates import embed_candidates
from precompute.build_index import build_faiss_index


def main():
    t0 = time.time()

    print("=" * 70)
    print("STAGE 1/3: candidate_intelligence -- parsing candidates.jsonl")
    print("=" * 70)
    build_candidate_intelligence()

    print("\n" + "=" * 70)
    print("STAGE 2/3: embed_candidates -- encoding candidate embedding_text")
    print("=" * 70)
    embed_candidates()

    print("\n" + "=" * 70)
    print("STAGE 3/3: build_index -- building FAISS IndexFlatIP")
    print("=" * 70)
    build_faiss_index()

    print(f"\nPrecompute complete in {time.time() - t0:.1f}s. Artifacts ready in artifacts/.")
    print("You can now run: python rank.py --candidates ./data/candidates.jsonl --out ./outputs/submission.csv")


if __name__ == "__main__":
    main()
