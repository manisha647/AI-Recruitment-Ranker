"""
Central configuration for the AI-Recruitment-Ranker pipeline.

Design principle: every magic number that affects ranking lives here,
in one place, so it can be reasoned about, tuned, and defended in the
Stage 5 interview without hunting through five files.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
OUTPUTS_DIR = ROOT_DIR / "outputs"

CANDIDATES_JSONL = DATA_DIR / "candidates.jsonl"
JD_TXT = DATA_DIR / "jd.txt"

CANDIDATE_FEATURES_PARQUET = ARTIFACTS_DIR / "candidate_features.parquet"
CANDIDATE_EMBEDDINGS_NPY = ARTIFACTS_DIR / "candidate_embeddings.npy"
CANDIDATE_IDS_NPY = ARTIFACTS_DIR / "candidate_ids.npy"
FAISS_INDEX_PATH = ARTIFACTS_DIR / "candidate_index.faiss"
EMBEDDER_ARTIFACT_PATH = ARTIFACTS_DIR / "embedder.joblib"
JD_FEATURES_JSON = ARTIFACTS_DIR / "jd_features.json"

SUBMISSION_CSV = OUTPUTS_DIR / "submission.csv"

# Reference "today" for computing recency features (days since last active,
# etc.). The dataset's last_active_date tops out around 2026-05-27, so we
# anchor just after that. Kept as a config constant (not datetime.now())
# so feature computation is 100% deterministic and reproducible.
REFERENCE_DATE = "2026-06-01"

# ---------------------------------------------------------------------------
# Embedding backend
# ---------------------------------------------------------------------------
# "tfidf_svd"       -> scikit-learn TF-IDF + Truncated SVD. Zero network,
#                       zero GPU, trains in-repo on the candidate corpus.
#                       Always runnable, always reproducible offline.
# "sentence_transformer" -> e.g. BAAI/bge-base-en-v1.5. Higher semantic
#                       quality, but requires a one-time internet-connected
#                       precompute step to download the model. Once cached
#                       locally, the ranking step itself still makes zero
#                       network calls (loads from local HF cache), so this
#                       remains compliant with the "no network during
#                       ranking" rule. Recommended upgrade path if you have
#                       GPU access for the offline precompute step.
EMBEDDING_BACKEND = "tfidf_svd"
SENTENCE_TRANSFORMER_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # SVD components for tfidf_svd; native dim for ST models

# BGE models want an instruction prefix on the *query* (JD) side, not the
# passage (candidate) side. Only used when EMBEDDING_BACKEND == "sentence_transformer".
BGE_QUERY_PREFIX = "Represent this job description for retrieving suitable candidate profiles: "

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K = 1000  # candidates pulled from FAISS before hybrid re-ranking
FINAL_TOP_N = 100  # required by submission_spec section 2

# ---------------------------------------------------------------------------
# Hybrid score weights (must sum to 1.0 across the primary components;
# the behavior score is applied as a multiplicative modifier on top,
# per redrob_signals_doc.docx: "incorporate them as a multiplier or
# modifier on top of skill-match scoring", not as an additive term that
# could let a chatty-but-unqualified candidate outscore a strong one).
# ---------------------------------------------------------------------------
WEIGHTS = {
    "semantic": 0.30,
    "skill_required": 0.20,
    "skill_preferred": 0.08,
    "title_career_alignment": 0.20,  # decisive signal against keyword-stuffer trap
    "experience": 0.12,
    "location": 0.05,
    "education": 0.05,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6

# Behavior modifier: multiplies the additive score above.
# Bounded so a bad behavior profile can meaningfully demote a candidate
# but a perfect one can't single-handedly rescue a poor skills/title fit.
BEHAVIOR_MODIFIER_MIN = 0.55
BEHAVIOR_MODIFIER_MAX = 1.05

# ---------------------------------------------------------------------------
# JD requirements for this specific challenge (Senior AI Engineer, Redrob).
# Hand-curated from job_description.docx. In jd_understanding.py these seed
# an extraction step; they are not the whole story (semantic score covers
# what isn't captured by explicit keyword lists), but they encode the
# things the JD is explicit and insistent about, including its disqualifiers.
# ---------------------------------------------------------------------------
JD_REQUIRED_SKILLS = [
    "embeddings", "retrieval", "ranking", "vector search", "hybrid search",
    "sentence-transformers", "faiss", "elasticsearch", "opensearch",
    "pinecone", "weaviate", "qdrant", "milvus", "bge", "e5",
]
JD_PREFERRED_SKILLS = [
    "lora", "qlora", "peft", "fine-tuning", "xgboost", "learning to rank",
    "ndcg", "mrr", "map", "a/b testing", "distributed systems", "llm",
    "rag", "nlp", "python",
]
# Roles/titles that historically signal a *product-engineering* fit for this
# JD (used for the title/career alignment component).
JD_POSITIVE_TITLE_SIGNALS = [
    "ml engineer", "machine learning engineer", "ai engineer",
    "applied scientist", "research engineer", "search engineer",
    "recommendation", "ranking engineer", "nlp engineer",
    "data scientist", "software engineer",
]
# Titles that are a hard mismatch regardless of skills listed (the JD's
# explicit trap: "a candidate who has all the AI keywords listed as skills
# but whose title is 'Marketing Manager' is not a fit").
JD_NEGATIVE_TITLE_SIGNALS = [
    "marketing manager", "hr manager", "operations manager",
    "sales executive", "accountant", "graphic designer",
    "content writer", "customer support", "civil engineer",
    "mechanical engineer", "business analyst", "project manager",
]
JD_DISQUALIFYING_BACKGROUNDS = [
    "computer vision", "speech recognition", "robotics",
]
JD_CONSULTING_FIRMS = [
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
]
JD_PREFERRED_LOCATIONS = ["pune", "noida", "hyderabad", "mumbai", "delhi", "ncr", "india"]
JD_MIN_YEARS = 5
JD_MAX_YEARS = 9  # soft band, not a hard cutoff per the JD's own text

# ---------------------------------------------------------------------------
# Honeypot / data-integrity heuristics
# ---------------------------------------------------------------------------
# redrob_signals_doc.docx: "~80 honeypot candidates with subtly impossible
# profiles (e.g., 8 years of experience at a company founded 3 years ago;
# 'expert' proficiency in 10 skills with 0 years used)". We don't have
# ground-truth company founding dates, so we detect *internal* impossibility
# instead -- these are the signals available in the schema.
HONEYPOT_RULES = {
    # A skill claimed at "expert" level with ~0 hands-on duration.
    "expert_zero_duration_months_max": 2,
    # Number of skills at expert/advanced with near-zero duration needed to
    # flag a profile as a honeypot (single occurrence could be data noise).
    "expert_zero_duration_skill_count": 3,
    # Sum of career_history duration_months vs. stated years_of_experience:
    # allow some slack for concurrent/overlapping roles or gaps, but not this.
    "experience_overstatement_ratio": 1.6,
    # Overlapping "is_current" roles beyond this count is impossible.
    "max_concurrent_current_roles": 1,
}
