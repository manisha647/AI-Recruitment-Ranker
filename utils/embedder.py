"""
utils/embedder.py

A single embedding interface with two interchangeable backends:

  - "tfidf_svd" (default): TF-IDF + TruncatedSVD, trained on the candidate
    corpus itself. Pure scikit-learn, zero network calls, zero GPU. This is
    what makes the *entire* pipeline -- precompute AND rank -- runnable with
    no internet access at all, which is a strictly stronger guarantee than
    the spec requires (spec only forbids network during the ranking step).

  - "sentence_transformer": a real bi-encoder (default BAAI/bge-small-en-v1.5)
    for higher semantic quality. Requires network access ONCE, during
    precompute, to download model weights into the local HuggingFace cache.
    After that, loading the model is a local disk read -- so the ranking
    step still makes zero network calls, satisfying the compute constraint
    in submission_spec.docx section 3 ("Network Off ... during the ranking
    step"). This is the recommended upgrade path if you have GPU access for
    the one-time precompute run (see README "Upgrading the embedding backend").

Both backends expose the same fit / transform / save / load interface so
the rest of the pipeline (rank.py, hybrid_ranker.py) never needs to know
which one is active.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import joblib

from utils.config import EMBEDDING_BACKEND, SENTENCE_TRANSFORMER_MODEL, EMBEDDING_DIM, BGE_QUERY_PREFIX


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class TfidfSvdEmbedder:
    """Offline, no-network, no-GPU embedder. Trained on the candidate corpus."""

    backend = "tfidf_svd"

    def __init__(self, n_components: int = EMBEDDING_DIM):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            max_features=60000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.6,
            sublinear_tf=True,
            stop_words="english",
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfSvdEmbedder":
        tfidf = self.vectorizer.fit_transform(texts)
        self.svd.fit(tfidf)
        self._fitted = True
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        tfidf = self.vectorizer.transform(texts)
        emb = self.svd.transform(tfidf)
        return _l2_normalize(emb.astype(np.float32))

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: Path) -> None:
        joblib.dump({"vectorizer": self.vectorizer, "svd": self.svd, "n_components": self.n_components}, path)

    @classmethod
    def load(cls, path: Path) -> "TfidfSvdEmbedder":
        state = joblib.load(path)
        obj = cls(n_components=state["n_components"])
        obj.vectorizer = state["vectorizer"]
        obj.svd = state["svd"]
        obj._fitted = True
        return obj


class SentenceTransformerEmbedder:
    """
    Wraps a sentence-transformers bi-encoder (e.g. BGE). Nothing here needs
    network access at *ranking* time as long as the model was downloaded
    once during precompute (cached under ~/.cache/huggingface). Only used
    when EMBEDDING_BACKEND == "sentence_transformer".
    """

    backend = "sentence_transformer"

    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def fit(self, texts: list[str]):
        return self  # pretrained, nothing to fit

    def transform(self, texts: list[str], batch_size: int = 128, is_query: bool = False) -> np.ndarray:
        prefixed = [BGE_QUERY_PREFIX + t for t in texts] if is_query else texts
        emb = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 1000,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return emb.astype(np.float32)

    def fit_transform(self, texts: list[str], **kwargs) -> np.ndarray:
        return self.transform(texts, **kwargs)

    def save(self, path: Path) -> None:
        # sentence-transformers models are identified by name + local HF
        # cache; we just persist the name/config, not the weights.
        with open(path, "w") as f:
            json.dump({"backend": self.backend, "model_name": self.model_name}, f)

    @classmethod
    def load(cls, path: Path) -> "SentenceTransformerEmbedder":
        with open(path) as f:
            state = json.load(f)
        return cls(model_name=state["model_name"])


def get_embedder(backend: str = EMBEDDING_BACKEND):
    if backend == "tfidf_svd":
        return TfidfSvdEmbedder()
    elif backend == "sentence_transformer":
        return SentenceTransformerEmbedder()
    else:
        raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend}")


def load_embedder(path: Path, backend: str = EMBEDDING_BACKEND):
    if backend == "tfidf_svd":
        return TfidfSvdEmbedder.load(path)
    elif backend == "sentence_transformer":
        return SentenceTransformerEmbedder.load(path)
    else:
        raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend}")
