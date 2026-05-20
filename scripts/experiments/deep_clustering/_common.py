"""
Shared utilities for all deep clustering scripts (metrics, embedding loaders, results writer).
All six deep methods import from here so metrics are computed identically.
"""
from __future__ import annotations

import logging
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

import os as _os

# four parents up to repo root
REPO_ROOT       = pathlib.Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR        = REPO_ROOT / "data"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
MODELS_DIR      = DATA_DIR / "models"
EXPERIMENTS_DIR.mkdir(exist_ok=True)

# DEEP_RESULTS_CSV env var lets you write results to a separate file.
_results_csv_env = _os.environ.get("DEEP_RESULTS_CSV")
RESULTS_CSV = (
    pathlib.Path(_results_csv_env) if _results_csv_env
    else EXPERIMENTS_DIR / "deep_clustering_results_chunks.csv"
)

# default: chunk embeddings from 04_chunk_embeddings.sql; override with --bq-table
_BQ_DEFAULT = (
    "support-analytics-492410.support_analytics"
    ".conversation_chunk_embeddings"
)

# DEEP_NPY_MULTILINGUAL env var lets you swap in a different .npy without editing any script
_npy_multi_env  = _os.environ.get("DEEP_NPY_MULTILINGUAL")

# 768-D base embeddings (text-multilingual-embedding-002, pre-fine-tuning — fair comparison baseline)
NPY_MULTILINGUAL = (
    pathlib.Path(_npy_multi_env) if _npy_multi_env
    else DATA_DIR / "chunk_embeddings_multilingual.npy"
)
# gemini variant was not run; scripts skip missing .npy files
NPY_GEMINI = DATA_DIR / "chunk_embeddings_gemini.npy"   # intentionally absent
BQ_MULTILINGUAL  = "support-analytics-492410.support_analytics.conversation_chunk_embeddings"
BQ_GEMINI        = "support-analytics-492410.support_analytics.conversation_chunk_embeddings_gemini"

SIL_SAMPLE = 10_000
RNG        = 42

RESULTS_COLUMNS = [
    "method", "method_type", "latent_dim", "k", "noise_pct",
    "sil_cos", "sil_latent", "davies_bouldin", "calinski_harabasz",
    "n_clusters", "source_embeddings", "source_model", "computed_at",
]

log = logging.getLogger(__name__)


def load_embeddings(
    npy_path: pathlib.Path | str | None = None,
    bq_table: str = _BQ_DEFAULT,
) -> np.ndarray:
    """Load L2-normalised embeddings: .npy cache first, BigQuery fallback (~5 min). Returns float32 (n, D)."""
    if npy_path is None:
        npy_path = DATA_DIR / "embeddings_multilingual.npy"
    npy_path = pathlib.Path(npy_path)

    if npy_path.exists():
        log.info("Loading embeddings from cache: %s", npy_path)
        embs = np.load(npy_path).astype(np.float32)
        log.info("  shape=%s  dtype=%s", embs.shape, embs.dtype)
        return embs

    log.warning(
        "Cache not found: %s\n"
        "  Falling back to BigQuery: %s\n"
        "  (Run bq_pipeline/04_chunk_embeddings.sql then scripts/chunk_clustering.py to populate the cache.)",
        npy_path,
        bq_table,
    )
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from gcp_config import PROJECT_ID          # noqa: E402
        from google.cloud import bigquery          # noqa: E402

        client = bigquery.Client(project=PROJECT_ID)
        query = f"""
            SELECT
                ml_generate_embedding_result AS embedding
            FROM `{bq_table}`
            WHERE ARRAY_LENGTH(ml_generate_embedding_result) > 0
            ORDER BY chunk_uid
        """
        df = client.query(query).to_dataframe(progress_bar_type="tqdm")
        log.info("Downloaded %d rows", len(df))
        embs = np.asarray(df["embedding"].tolist(), dtype=np.float32)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not load embeddings from {npy_path} or BigQuery.\n"
            "Fix: run  bq_pipeline/04_chunk_embeddings.sql  then  "
            "python scripts/chunk_clustering.py  to populate the cache."
        ) from exc

    # L2-normalise (Vertex AI embeddings should already be unit vectors, but just in case)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs  = embs / np.maximum(norms, 1e-10)

    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, embs)
    log.info("Cached → %s", npy_path)
    return embs


def load_both_embeddings() -> dict[str, np.ndarray | None]:
    """Load both embedding models. Returns {"multilingual": array|None, "gemini": array|None}."""
    result: dict[str, np.ndarray | None] = {}
    for key, npy_path, bq_table in [
        ("multilingual", NPY_MULTILINGUAL, BQ_MULTILINGUAL),
        ("gemini",       NPY_GEMINI,       BQ_GEMINI),
    ]:
        try:
            arr = load_embeddings(npy_path, bq_table=bq_table)
            log.info("load_both_embeddings [%s]: shape=%s dtype=%s", key, arr.shape, arr.dtype)
            result[key] = arr
        except FileNotFoundError as exc:
            log.warning("load_both_embeddings [%s]: MISSING — %s", key, exc)
            result[key] = None
    return result


def sil_cos(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_size: int = SIL_SAMPLE,
) -> float | None:
    """Silhouette in raw cosine space — primary comparison metric, same space for all methods."""
    valid = labels != -1
    X, y  = embeddings[valid], labels[valid]
    n_cls = len(np.unique(y))
    if n_cls < 2 or len(y) < 2:
        return None
    n = min(sample_size, len(y))
    try:
        return round(
            float(silhouette_score(X, y, metric="cosine",
                                   sample_size=n, random_state=RNG)),
            4,
        )
    except Exception as exc:
        log.warning("sil_cos failed: %s", exc)
        return None


def sil_latent(
    latent: np.ndarray,
    labels: np.ndarray,
    sample_size: int = SIL_SAMPLE,
) -> float | None:
    """Silhouette in the method's own latent space — not comparable across methods, reported for completeness."""
    valid = labels != -1
    X, y  = latent[valid], labels[valid]
    if len(np.unique(y)) < 2 or len(y) < 2:
        return None
    n = min(sample_size, len(y))
    try:
        return round(
            float(silhouette_score(X, y, metric="euclidean",
                                   sample_size=n, random_state=RNG)),
            4,
        )
    except Exception as exc:
        log.warning("sil_latent failed: %s", exc)
        return None


def db_ch(
    latent: np.ndarray,
    labels: np.ndarray,
) -> tuple[float | None, float | None]:
    """Davies-Bouldin ↓ and Calinski-Harabasz ↑ in latent space."""
    valid = labels != -1
    X, y  = latent[valid], labels[valid]
    if len(np.unique(y)) < 2 or len(y) < 2:
        return None, None
    try:
        return (
            round(float(davies_bouldin_score(X, y)), 4),
            round(float(calinski_harabasz_score(X, y)), 2),
        )
    except Exception as exc:
        log.warning("DB/CH failed: %s", exc)
        return None, None


def save_result(row: dict) -> None:
    """Upsert a result row into the results CSV (keyed on method, latent_dim, k, source_model)."""
    df_new = pd.DataFrame([{**row, "computed_at": pd.Timestamp.now().isoformat()}])

    if RESULTS_CSV.exists():
        df_old = pd.read_csv(RESULTS_CSV)
        # backward-compat: older CSVs may not have source_model
        if "source_model" not in df_old.columns:
            df_old["source_model"] = "unknown"
        keep = ~(
            (df_old["method"]       == row["method"])
            & (df_old["latent_dim"] == row.get("latent_dim", -1))
            & (df_old["k"]          == row["k"])
            & (df_old["source_model"] == row.get("source_model", "unknown"))
        )
        df_out = pd.concat([df_old[keep], df_new], ignore_index=True)
    else:
        df_out = df_new
        # ensure all columns exist on first write
        for col in RESULTS_COLUMNS:
            if col not in df_out.columns:
                df_out[col] = None

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(RESULTS_CSV, index=False)
    log.info("Saved → %s  (method=%s k=%s sil_cos=%s source_model=%s)",
             RESULTS_CSV, row["method"], row["k"],
             row.get("sil_cos"), row.get("source_model", "unknown"))
