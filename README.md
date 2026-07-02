# AI-Recruitment-Ranker

A hybrid semantic + feature-based retrieval and ranking system for the
Redrob Intelligent Candidate Discovery & Ranking Challenge: ranks the
top 100 of 100,000 candidate profiles against a job description.

**Read this before you read the code:** this system is built against the
constraints in `submission_spec.docx`, not just the general goal of
"rank candidates well." In particular:

- Ranking is **CPU-only, no GPU, no network calls, ≤5 minutes, ≤16GB RAM**.
- Output is **exactly** `candidate_id,rank,score,reasoning`, exactly 100 rows.
- The dataset contains **honeypot candidates** with internally impossible
  profiles; a submission with >10% honeypots in the top 100 is
  auto-disqualified at Stage 3, regardless of composite score.
- Reasoning is manually graded at Stage 4 for hallucination, templating,
  and rank-consistency.

Everything below is designed around satisfying all of these simultaneously,
not just producing a plausible-looking ranking.

## Architecture

```
                    ─────────── OFFLINE (precompute_all.py) ───────────
data/candidates.jsonl
        │
        ▼
precompute/candidate_intelligence.py   parse 100K profiles, build
        │                              embedding_text + structured features
        │                              + honeypot flags
        ▼
artifacts/candidate_features.parquet
        │
        ▼
precompute/embed_candidates.py         fit/apply embedder (TF-IDF+SVD by
        │                              default; swappable for a sentence-
        │                              transformer bi-encoder)
        ▼
artifacts/candidate_embeddings.npy, candidate_ids.npy, embedder.joblib
        │
        ▼
precompute/build_index.py              FAISS IndexFlatIP (exact cosine)
        ▼
artifacts/candidate_index.faiss

                    ─────────── ONLINE (rank.py, ~7s, CPU, no network) ───────────
data/jd.txt
        │
        ▼
jd/jd_understanding.py          → structured JD features + semantic text
        │
        ▼
embed JD (1 item, same embedder)  → FAISS search top-K (default 1000)
        │
        ▼
ranking/hybrid_ranker.py        → semantic + skill + title/career +
        │                          experience + location + education,
        │                          × behavior modifier, × honeypot penalty
        ▼
reasoning/explanation_generator.py  → grounded, varied, rank-matched reasoning
        │
        ▼
submission/submission_generator.py  → outputs/submission.csv (top 100,
                                       exact spec format, tie-broken)
```

The split matters: everything expensive (parsing 100K JSON records, fitting
an embedder, encoding 100K vectors) happens once, offline, with no time
limit. The only thing `rank.py` does at "ranking time" is embed **one** job
description and re-rank ~1000 pre-retrieved candidates with vectorized
pandas — a few seconds of work, comfortably inside the 5-minute budget even
on a laptop.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Offline precompute (run once; ~3 minutes on a single CPU core for
#    100K candidates with the default backend; no GPU or network required)
python precompute_all.py

# 2. Ranking step (the command graded at Stage 3; ~7 seconds, CPU-only,
#    no network)
python rank.py --candidates ./data/candidates.jsonl --jd ./data/jd.txt --out ./outputs/submission.csv

# 3. Validate before you submit
python submission/validate_submission.py ./outputs/submission.csv
```

Drop your own `candidates.jsonl` in `data/` and `jd.txt` in `data/` to
re-run against a different pool or role — `precompute_all.py` only needs
to be re-run when `candidates.jsonl` changes; a new JD only requires
re-running `rank.py`.

## Why TF-IDF+SVD by default, not a transformer embedding model

The original brief for this project called for `BAAI/bge-base-en-v1.5` on
a Colab T4 GPU. That directly conflicts with the actual competition rule:
**no GPU during the ranking step, and no network calls during ranking.**
Downloading a HuggingFace model checkpoint mid-ranking would violate both.

So the default embedding backend (`EMBEDDING_BACKEND = "tfidf_svd"` in
`utils/config.py`) is a TF-IDF + TruncatedSVD embedder trained from scratch
on the candidate corpus itself, using only scikit-learn. This makes the
**entire pipeline**, precompute included, runnable with zero network access
and zero GPU — a strictly stronger guarantee than the spec requires, and
one less thing to explain or defend at Stage 5.

### Upgrading to a real sentence-transformer (recommended if you have GPU access for precompute)

Set in `utils/config.py`:
```python
EMBEDDING_BACKEND = "sentence_transformer"
SENTENCE_TRANSFORMER_MODEL = "BAAI/bge-small-en-v1.5"  # or bge-base-en-v1.5
```
Then run `precompute_all.py` once, with network access, on whatever
hardware you like (GPU is fine — precompute has no compute constraint per
spec section 10.3, as long as you document it, which `submission_metadata.yaml`
does). This downloads and caches the model locally and encodes all 100K
candidates. From then on, `rank.py` loads the model from the local
HuggingFace cache — no network call happens during ranking, so this stays
fully compliant. `pip install sentence-transformers torch` first (commented
out in `requirements.txt` by default to keep the base install network/GPU-free).

## Honeypot handling

`utils/honeypot.py` flags candidates with internally-impossible profiles
using rules derived from the fields actually present in the schema (we
don't have ground-truth company founding dates, so these check internal
consistency instead):

- Expert/advanced proficiency claimed on 3+ skills with ~0 months of
  hands-on duration.
- `career_history` duration summing to well beyond what `years_of_experience`
  implies.
- More than one role simultaneously marked `is_current`.
- Education `end_year` before `start_year`.
- A single role's duration alone exceeding total stated experience.

Flagged candidates get their score multiplied by 0.03 in
`ranking/hybrid_ranker.py::score_candidates` — enough to reliably keep
them out of the top 100 without hard-excluding them (a borderline
false-positive flag still gets a chance to rank on its other merits if
the flag turns out to be spurious). On the released bundle this flags 45
candidates (~0.05% of the pool) and produces **0 honeypots in the top
100** for the released JD.

## Scoring weights and rationale

All weights live in `utils/config.py::WEIGHTS`, not scattered through the
code, specifically so they're easy to point to and defend:

| Component | Weight | Why |
|---|---|---|
| Semantic similarity | 30% | Catches candidates whose fit is real but not captured by an explicit skill/title list (the JD explicitly wants this: "a Tier 5 candidate may not use the words 'RAG' or 'Pinecone'... but if their career history shows they built a recommendation system, they're a fit"). |
| Title/career alignment | 20% | The JD's own explicit anti-trap: "a candidate who has all the AI keywords listed as skills but whose title is 'Marketing Manager' is not a fit." Second-highest weight for exactly this reason. |
| Required skill match | 20% | Weighted by proficiency + duration, not just presence — an "expert, 3 years" match counts more than a "beginner, 1 month" one. |
| Experience-band fit | 12% | Soft decay outside 5-9 years, per the JD's own framing of the band as a guideline, not a hard cutoff. |
| Preferred skill match | 8% | Nice-to-haves (LoRA/QLoRA, learning-to-rank, distributed systems). |
| Location | 5% | JD is Pune/Noida-preferred but explicitly open to relocation from other Tier-1 Indian cities. |
| Education | 5% | JD states no hard degree requirement; kept intentionally light. |

On top of the additive blend, a **bounded multiplicative behavior
modifier** (0.55x–1.05x, `utils/config.py::BEHAVIOR_MODIFIER_MIN/MAX`)
applies recruiter-response-rate, recency, notice-period, and
availability signals — implemented as a multiplier rather than an
additive term specifically because `redrob_signals_doc.docx` frames it
that way ("incorporate them as a multiplier or modifier on top of
skill-match scoring"), and because a multiplier can meaningfully demote
an unreachable candidate without letting a merely-chatty, under-qualified
one outscore a strong one on behavior alone.

## Reasoning generation

`reasoning/explanation_generator.py` composes the required `reasoning`
column from the same structured features that drove the score — not from
an LLM call (which the compute constraints forbid during ranking anyway).
Each candidate gets a deterministic-but-varied sentence built from:
current title + years of experience, matched required skills (named,
capped at 3), and up to two genuine concerns (missing skills, title
mismatch, consulting-only background, inactivity, low response rate,
long notice period, experience outside the JD's band). The closing
sentence's tone is tied to both rank percentile *and* concern count, so a
rank-95 candidate never reads like a glowing rank-3 one — directly
targeting the Stage 4 "rank consistency" check.

## Repository layout

```
AI-Recruitment-Ranker/
├── data/                       candidates.jsonl, jd.txt (inputs)
├── artifacts/                  precomputed embeddings, index, features (generated)
├── outputs/                    submission.csv + full scored breakdown (generated)
├── utils/
│   ├── config.py                all weights, paths, thresholds in one place
│   ├── embedder.py               TF-IDF+SVD / sentence-transformer abstraction
│   └── honeypot.py               internal-consistency honeypot detector
├── precompute/
│   ├── candidate_intelligence.py parse candidates.jsonl -> features + embedding_text
│   ├── embed_candidates.py       batch-embed all candidates, checkpointed
│   └── build_index.py            build FAISS IndexFlatIP
├── jd/
│   └── jd_understanding.py       parse jd.txt -> structured requirements
├── ranking/
│   └── hybrid_ranker.py          combine all scoring components
├── reasoning/
│   └── explanation_generator.py  grounded, varied, rank-matched reasoning
├── submission/
│   ├── submission_generator.py   exact-spec CSV writer
│   └── validate_submission.py    official format validator (from the bundle)
├── precompute_all.py             offline orchestrator
├── rank.py                       ONLINE entry point (the graded command)
├── requirements.txt
└── submission_metadata.yaml
```

## Sandbox setup (required per spec section 10.5)

Deploy `rank.py` + `artifacts/` (built from a small candidate sample, ≤100
rows) to any of: HuggingFace Spaces, Streamlit Cloud, Replit, Colab (with a
notebook link), or a public Docker image. To build a small-sample artifact
set for the sandbox:

```bash
head -100 data/candidates.jsonl > data/candidates_sample.jsonl
python -c "from precompute.candidate_intelligence import build_candidate_intelligence; \
           from utils.config import ARTIFACTS_DIR; \
           build_candidate_intelligence(input_path='data/candidates_sample.jsonl', \
                                         output_path=ARTIFACTS_DIR/'candidate_features.parquet')"
python -m precompute.embed_candidates
python -m precompute.build_index
python rank.py --candidates data/candidates_sample.jsonl --jd data/jd.txt --out outputs/submission_sample.csv
```

Wire whichever hosted platform you choose to run that same sequence (or
just the `rank.py` step, if you commit the small-sample artifacts) and
link it in `submission_metadata.yaml::sandbox_link`.

