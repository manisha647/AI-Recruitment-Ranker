"""
precompute/embed_candidates.py

Stage 2 of the offline precompute pipeline.

Loads artifacts/candidate_features.parquet (from candidate_intelligence.py),
fits/loads the embedder on the embedding_text column, encodes all 100,000
candidates in batches, and saves:

  artifacts/candidate_embeddings.npy   (N x D float32, L2-normalized)
  artifacts/candidate_ids.npy          (N,) string array, same row order
  artifacts/embedder.joblib            (fitted embedder, needed to embed
                                         the JD at rank time with the same
                                         vector space)

Supports checkpointing / resume: if interrupted partway through a
sentence-transformer run, re-running will pick up from the last saved
batch instead of starting over. (The default tfidf_svd backend is fast
enough on 100k rows that this mostly matters for the optional
sentence-transformer backend on constrained hardware.)

Usage:
    python -m precompute.embed_candidates
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from utils.config import (
    CANDIDATE_FEATURES_PARQUET,
    CANDIDATE_EMBEDDINGS_NPY,
    CANDIDATE_IDS_NPY,
    EMBEDDER_ARTIFACT_PATH,
    ARTIFACTS_DIR,
    EMBEDDING_BACKEND,
)
from utils.embedder import get_embedder

CHECKPOINT_EVERY_BATCHES = 20
BATCH_SIZE = 256


def embed_candidates(
    features_path=CANDIDATE_FEATURES_PARQUET,
    out_embeddings_path=CANDIDATE_EMBEDDINGS_NPY,
    out_ids_path=CANDIDATE_IDS_NPY,
    out_embedder_path=EMBEDDER_ARTIFACT_PATH,
    backend: str = EMBEDDING_BACKEND,
):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading {features_path} ...")
    df = pd.read_parquet(features_path, columns=["candidate_id", "embedding_text"])
    texts = df["embedding_text"].fillna("").tolist()
    ids = df["candidate_id"].to_numpy()
    n = len(texts)
    print(f"Embedding {n:,} candidates with backend='{backend}'")

    embedder = get_embedder(backend)
    t0 = time.time()

    checkpoint_path = ARTIFACTS_DIR / f".embed_checkpoint_{backend}.npy"
    start_idx = 0
    chunks = []

    if backend == "tfidf_svd":
        # TF-IDF+SVD must fit on the full corpus at once (it's not an
        # incremental encoder like a bi-encoder is) -- this is a single
        # vectorized pass over 100k short-to-medium text documents, which
        # completes in seconds to low-single-digit minutes on CPU. No
        # per-batch checkpointing needed at this scale.
        embeddings = embedder.fit_transform(texts)
        embedder.save(out_embedder_path)
    else:
        # Sentence-transformer path: batch + checkpoint, since a cold CPU
        # run over 100k rows with a real transformer can take a while and
        # we don't want to lose progress on interruption.
        if checkpoint_path.exists():
            partial = np.load(checkpoint_path)
            start_idx = partial.shape[0]
            chunks = [partial]
            print(f"Resuming from checkpoint at row {start_idx:,}")

        for batch_start in range(start_idx, n, BATCH_SIZE):
            batch_texts = texts[batch_start:batch_start + BATCH_SIZE]
            batch_emb = embedder.transform(batch_texts, batch_size=BATCH_SIZE, is_query=False)
            chunks.append(batch_emb)

            batch_num = batch_start // BATCH_SIZE
            if batch_num % CHECKPOINT_EVERY_BATCHES == 0:
                np.save(checkpoint_path, np.concatenate(chunks, axis=0))
                elapsed = time.time() - t0
                done = batch_start + len(batch_texts)
                rate = done / max(elapsed, 1e-6)
                eta = (n - done) / max(rate, 1e-6)
                print(f"  {done:,}/{n:,} embedded ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")

        embeddings = np.concatenate(chunks, axis=0)
        embedder.save(out_embedder_path)
        if checkpoint_path.exists():
            checkpoint_path.unlink()

    elapsed = time.time() - t0
    print(f"Embedded {n:,} candidates in {elapsed:.1f}s -> shape {embeddings.shape}")

    np.save(out_embeddings_path, embeddings.astype(np.float32))
    np.save(out_ids_path, ids)
    print(f"Saved embeddings -> {out_embeddings_path} ({out_embeddings_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved ids -> {out_ids_path}")
    print(f"Saved embedder -> {out_embedder_path}")
    return embeddings, ids


if __name__ == "__main__":
    embed_candidates()
