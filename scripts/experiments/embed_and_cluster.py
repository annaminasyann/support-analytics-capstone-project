"""
UMAP + HDBSCAN clustering on Vertex AI conversation-level embeddings.

Legacy experiment (superseded by chunk_clustering.py). Supports multilingual 768-D
and gemini 3072-D embeddings across Track A (original text) and Track B (translated).
Silhouette reported in raw cosine and UMAP Euclidean spaces.

Usage:
  python scripts/experiments/embed_and_cluster.py
  python scripts/experiments/embed_and_cluster.py --track a --skip-embed
  python scripts/experiments/embed_and_cluster.py --embed-suffix gemini
  python scripts/experiments/embed_and_cluster.py --sweep
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import umap
import hdbscan
from hdbscan import validity_index as dbcv_index
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from google.api_core.exceptions import NotFound
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from gcp_config import PROJECT_ID, bq_table  # noqa: E402

OUT_DIR         = REPO_ROOT / "data"
EXPERIMENTS_DIR = REPO_ROOT / "data" / "experiments"
OUT_DIR.mkdir(exist_ok=True)
EXPERIMENTS_DIR.mkdir(exist_ok=True)

MODELS_DIR = REPO_ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEST_LABELS             = bq_table("cluster_labels")
DEST_METRICS            = bq_table("evaluation_metrics")
DEST_CLUSTER_PERSIST    = bq_table("cluster_persistence")
DEST_STABILITY_ARI      = bq_table("cluster_stability_ari")
DEST_SWEEP              = bq_table("hdbscan_sweep_results")
DEST_FOCUSED_SWEEP      = bq_table("hdbscan_focused_sweep")
DEST_CROSS_TRACK_ARI    = bq_table("cross_track_ari")

# canonical paper: nc=20, mcs=150, ms=15; below are the 8D experimental defaults
UMAP_N_COMPONENTS_CLUSTER = 8     # 8D experimental default; paper canonical = 20D
UMAP_N_COMPONENTS_VIZ     = 2
UMAP_N_NEIGHBORS          = 30    # balances local / global structure at n ≈ 47K
UMAP_MIN_DIST_CLUSTER     = 0.0   # tight clusters for density-based clustering
UMAP_MIN_DIST_VIZ         = 0.1   # slightly spread for readable scatter plots
UMAP_METRIC               = "cosine"
UMAP_SEED                 = 42    # primary seed; stability check re-runs with others

HDBSCAN_MIN_CLUSTER_SIZE  = 250   # 8D experimental default; paper canonical = 150
HDBSCAN_MIN_SAMPLES       = 5     # 8D experimental default; paper canonical = 15
HDBSCAN_METRIC            = "euclidean"  # Euclidean in UMAP space ≈ geodesic in embedding space
HDBSCAN_SELECTION         = "eom"  # eom merges leaf sub-clusters → fewer macro-topics
HDBSCAN_EPSILON           = 0.0   # cluster_selection_epsilon; >0 merges nearby clusters

SIL_SAMPLE_SIZE           = 10_000  # silhouette on raw 768-D is O(n²) — cap memory
DBCV_SAMPLE_SIZE          = 20_000  # DBCV is also O(n²); sample for tractability
RNG_SEED                  = 42

SWEEP_N_COMPONENTS        = (5, 6, 8, 10, 12, 15)
SWEEP_EPSILON             = (0.0, 0.1, 0.2, 0.3)
SWEEP_MIN_CLUSTER_SIZES   = (75, 100, 150, 200, 250, 300)
SWEEP_MIN_SAMPLES         = (5, 10)

FOCUSED_SWEEP_NC          = 8
FOCUSED_SWEEP_MCS         = (150, 200, 250, 300, 400)
FOCUSED_SWEEP_MS          = (5, 10, 15)
FOCUSED_SWEEP_SELECTION   = ("eom", "leaf")
FOCUSED_SWEEP_EPSILON     = (0.0,)

# explicitly enumerated (not np.random) so the seed set is auditable and reproducible
DEFAULT_STABILITY_SEEDS   = (1, 7, 13, 21, 42, 99, 123, 314, 1000, 2024)


TRACKS = {
    "a": {
        "name":     "multilingual",
        "bq_table": bq_table("conversation_embeddings_multilingual"),
        "label":    "Track A — Multilingual",
    },
    "b": {
        "name":     "translated",
        "bq_table": bq_table("conversation_embeddings_translated"),
        "label":    "Track B — Translated",
    },
}


def bq_client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID)


def _hyperparam_fingerprint(n_neighbors: int | None = None) -> str:
    """Short MD5 fingerprint of UMAP hyperparams for cache filenames — changes force a re-fit."""
    nn = n_neighbors if n_neighbors is not None else UMAP_N_NEIGHBORS
    key = (
        f"nn{nn}_md{UMAP_MIN_DIST_CLUSTER:.2f}"
        f"_metric{UMAP_METRIC}_seed{UMAP_SEED}"
    )
    return hashlib.md5(key.encode()).hexdigest()[:10]


def _table_last_modified(client: bigquery.Client, table_ref: str) -> str:
    """ISO timestamp of the source table's last_modified_time; used to invalidate stale .npy caches."""
    t = client.get_table(table_ref)
    return t.modified.isoformat()


def _write_cache_sidecar(cache: pathlib.Path, meta: dict) -> None:
    (cache.with_suffix(".json")).write_text(json.dumps(meta, indent=2, default=str))


def _read_cache_sidecar(cache: pathlib.Path) -> dict | None:
    side = cache.with_suffix(".json")
    if not side.exists():
        return None
    try:
        return json.loads(side.read_text())
    except Exception:
        return None


@dataclass
class PulledEmbeddings:
    ids:             np.ndarray        # dtype=object (strings)
    embeddings:      np.ndarray        # shape (n, D) float32  — D=768 or 3072
    content_lens:    np.ndarray        # per-row LENGTH(content), dtype int32
    n_rows:          int
    content_len_p50: float
    content_len_p95: float
    content_len_max: float
    n_near_limit:    int               # rows with LENGTH(content) > 6000 (possible truncation)
    source_modified: str               # ISO timestamp from BQ
    source_table:    str


def pull_embeddings(track_key: str, client: bigquery.Client,
                    skip_embed: bool = False) -> PulledEmbeddings:
    """Pull embeddings for one track; validates cache against source table modified time and hyperparams."""
    cfg           = TRACKS[track_key]
    name          = cfg["name"]
    cache_emb     = OUT_DIR / f"embeddings_{name}.npy"
    cache_ids     = OUT_DIR / f"ids_{name}.npy"
    cache_cl      = OUT_DIR / f"content_len_{name}.npy"

    current_modified = _table_last_modified(client, cfg["bq_table"])

    if cache_emb.exists() and cache_ids.exists():
        meta = _read_cache_sidecar(cache_emb) or {}
        cache_ok = (
            meta.get("source_table")    == cfg["bq_table"]
            and meta.get("source_modified") == current_modified
        )
        if cache_ok:
            log.info("[%s] Cache HIT  (source unchanged since %s)",
                     cfg["label"], current_modified)
            embeddings = np.load(cache_emb)
            ids        = np.load(cache_ids, allow_pickle=True)
            if embeddings.shape[0] != len(ids):
                raise ValueError(
                    f"[{cfg['label']}] Corrupt cache: "
                    f"{embeddings.shape[0]} embeddings vs {len(ids)} IDs"
                )
            if cache_cl.exists():
                content_lens = np.load(cache_cl)
            else:
                log.info("[%s] content_len cache missing — fetching from BQ (metadata only)…",
                         cfg["label"])
                try:
                    meta_df = client.query(f"""
                        SELECT conversation_id,
                               LENGTH(IFNULL(content, '')) AS cl
                        FROM `{cfg['bq_table']}`
                        WHERE ARRAY_LENGTH(ml_generate_embedding_result) > 0
                        ORDER BY conversation_id
                    """).to_dataframe()
                    id_to_len = dict(zip(meta_df["conversation_id"], meta_df["cl"]))
                    content_lens = np.array(
                        [id_to_len.get(i, 0) for i in ids], dtype=np.int32
                    )
                    np.save(cache_cl, content_lens)
                except Exception as e:
                    log.warning("[%s] Could not fetch content lengths: %s — filtering disabled.",
                                cfg["label"], e)
                    content_lens = np.zeros(len(ids), dtype=np.int32)
            return PulledEmbeddings(
                ids=ids,
                embeddings=embeddings,
                content_lens=content_lens,
                n_rows=int(embeddings.shape[0]),
                content_len_p50=meta.get("content_len_p50", float("nan")),
                content_len_p95=meta.get("content_len_p95", float("nan")),
                content_len_max=meta.get("content_len_max", float("nan")),
                n_near_limit=int(meta.get("n_near_limit", 0)),
                source_modified=current_modified,
                source_table=cfg["bq_table"],
            )
        else:
            log.warning("[%s] Cache STALE — source modified=%s, cached=%s. "
                        "Re-downloading.",
                        cfg["label"], current_modified,
                        meta.get("source_modified", "<missing>"))

    if skip_embed:
        raise FileNotFoundError(
            f"[{cfg['label']}] --skip-embed was set but no fresh cache exists. "
            "Run once without --skip-embed first."
        )

    log.info("[%s] Downloading embeddings from %s", cfg["label"], cfg["bq_table"])
    query = f"""
        SELECT
            conversation_id,
            ml_generate_embedding_result  AS embedding,
            LENGTH(IFNULL(content, ''))   AS content_len
        FROM `{cfg['bq_table']}`
        WHERE ARRAY_LENGTH(ml_generate_embedding_result) > 0
        ORDER BY conversation_id
    """
    df = client.query(query).to_dataframe(progress_bar_type="tqdm")
    log.info("[%s] Downloaded %d rows", cfg["label"], len(df))

    embeddings = np.asarray(df["embedding"].tolist(), dtype=np.float32)
    log.info("[%s] Embedding shape: %s", cfg["label"], embeddings.shape)

    # gemini at truncated dims may deviate from unit norm — normalise automatically
    norms = np.linalg.norm(embeddings, axis=1)
    max_norm_dev = float(np.max(np.abs(norms - 1.0)))
    if max_norm_dev > 1e-2:
        log.warning(
            "[%s] ℓ2-norms deviate from 1 by up to %.4f — applying L2 normalisation.",
            cfg["label"], max_norm_dev,
        )
        embeddings = embeddings / norms[:, np.newaxis]
    else:
        log.info("[%s] ℓ2-norm max deviation from 1.0: %.2e", cfg["label"], max_norm_dev)

    ids = df["conversation_id"].to_numpy()

    content_lens_arr = df["content_len"].to_numpy(dtype=np.int32)
    p50  = float(np.percentile(content_lens_arr, 50))
    p95  = float(np.percentile(content_lens_arr, 95))
    cmax = float(np.max(content_lens_arr))
    n_near_limit = int((content_lens_arr > 6000).sum())
    log.info("[%s] content length: median=%.0f  p95=%.0f  max=%.0f  "
             "near-limit(>6000 chars)=%d (%.1f%%)",
             cfg["label"], p50, p95, cmax, n_near_limit,
             100.0 * n_near_limit / len(df) if len(df) else 0.0)

    np.save(cache_emb, embeddings)
    np.save(cache_ids, ids)
    np.save(cache_cl, content_lens_arr)
    _write_cache_sidecar(cache_emb, {
        "source_table":     cfg["bq_table"],
        "source_modified":  current_modified,
        "n_rows":           int(embeddings.shape[0]),
        "dim":              int(embeddings.shape[1]),
        "content_len_p50":  p50,
        "content_len_p95":  p95,
        "content_len_max":  cmax,
        "n_near_limit":     n_near_limit,
    })
    log.info("[%s] Cached embeddings → %s", cfg["label"], cache_emb)

    return PulledEmbeddings(
        ids=ids,
        embeddings=embeddings,
        content_lens=content_lens_arr,
        n_rows=int(embeddings.shape[0]),
        content_len_p50=p50,
        content_len_p95=p95,
        content_len_max=cmax,
        n_near_limit=n_near_limit,
        source_modified=current_modified,
        source_table=cfg["bq_table"],
    )


def reduce_umap(
    embeddings: np.ndarray,
    n_components: int,
    min_dist: float,
    track_name: str,
    suffix: str,
    random_state: int = UMAP_SEED,
    n_neighbors: int | None = None,
) -> tuple[np.ndarray, umap.UMAP]:
    """Fit UMAP or load fingerprinted cache. Cache filename encodes n_components + hyperparams."""
    nn = n_neighbors if n_neighbors is not None else UMAP_N_NEIGHBORS
    fp = _hyperparam_fingerprint(n_neighbors=nn)
    cache_arr   = OUT_DIR    / f"umap_{suffix}_{track_name}_{fp}_s{random_state}.npy"
    cache_model = MODELS_DIR / f"umap_{suffix}_{track_name}_{fp}_s{random_state}.joblib"

    if cache_arr.exists() and cache_model.exists():
        log.info("Loading cached UMAP-%dD (n_neighbors=%d, fp=%s, seed=%d) from %s",
                 n_components, nn, fp, random_state, cache_arr)
        return np.load(cache_arr), joblib.load(cache_model)

    log.info("Running UMAP → %dD  (n_neighbors=%d, min_dist=%.2f, seed=%d)…",
             n_components, nn, min_dist, random_state)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=nn,
        min_dist=min_dist,
        metric=UMAP_METRIC,
        random_state=random_state,
        low_memory=False,
    )
    reduced = reducer.fit_transform(embeddings)

    np.save(cache_arr, reduced)
    joblib.dump(reducer, cache_model)
    log.info("UMAP cached → %s  (model → %s)  shape=%s",
             cache_arr, cache_model, reduced.shape)
    return reduced, reducer


def run_hdbscan(
    umap_nd: np.ndarray,
    min_cluster_size: int,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    epsilon: float = HDBSCAN_EPSILON,
    selection_method: str = HDBSCAN_SELECTION,
):
    """Fit HDBSCAN on the UMAP-reduced embeddings."""
    log.info(
        "Running HDBSCAN  (min_cluster_size=%d, min_samples=%d, "
        "selection=%s, epsilon=%.3f)…",
        min_cluster_size, min_samples, selection_method, epsilon,
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=selection_method,
        cluster_selection_epsilon=epsilon,
        prediction_data=True,
    )
    labels      = clusterer.fit_predict(umap_nd)
    membership  = clusterer.probabilities_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    log.info("Clusters: %d  |  Noise: %d (%.1f%%)",
             n_clusters, n_noise, 100 * n_noise / len(labels))

    sizes = pd.Series(labels).value_counts().sort_index()
    for cid, cnt in sizes.items():
        lbl = "NOISE" if cid == -1 else f"Cluster {cid}"
        log.info("  %s: %d  (%.1f%%)", lbl, cnt, 100 * cnt / len(labels))

    return labels, membership, clusterer


def assign_noise_to_nearest(
    clusterer,
    labels: np.ndarray,
    membership: np.ndarray,
    min_prob: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign noise points to their highest-probability cluster via HDBSCAN soft membership vectors."""
    noise_mask = labels == -1
    n_noise = int(noise_mask.sum())
    if n_noise == 0:
        log.info("Soft assignment: no noise points to reassign.")
        return labels, membership

    log.info("Soft assignment (min_prob=%.2f): evaluating %d noise points (%.1f%%) …",
             min_prob, n_noise, 100.0 * n_noise / len(labels))

    soft_probs = hdbscan.all_points_membership_vectors(clusterer)

    if soft_probs.shape[1] == 0:
        log.warning("Soft assignment: no clusters found — skipping.")
        return labels, membership

    new_labels     = labels.copy()
    new_membership = membership.copy()

    noise_soft    = soft_probs[noise_mask]
    best_clusters = np.argmax(noise_soft, axis=1)
    best_probs    = noise_soft[np.arange(n_noise), best_clusters]

    for thr in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
        n_above = int((best_probs >= thr).sum())
        log.info("  Best soft-prob >= %.2f: %d / %d noise points (%.1f%%)",
                 thr, n_above, n_noise, 100.0 * n_above / n_noise)

    assignable = best_probs >= min_prob
    n_assigned = int(assignable.sum())
    n_kept_noise = n_noise - n_assigned

    noise_idx = np.where(noise_mask)[0]
    assign_idx = noise_idx[assignable]

    new_labels[assign_idx]     = best_clusters[assignable]
    new_membership[assign_idx] = best_probs[assignable]

    n_clusters_after = len(np.unique(new_labels[new_labels != -1]))
    log.info(
        "Soft assignment complete: %d assigned (%.1f%%), %d remain noise (%.1f%%), "
        "%d clusters (unchanged).",
        n_assigned, 100.0 * n_assigned / n_noise,
        n_kept_noise, 100.0 * n_kept_noise / n_noise,
        n_clusters_after,
    )

    return new_labels, new_membership


def _log_noise_summary(
    track_name: str,
    labels: np.ndarray,
    content_lens: np.ndarray,
    umap_nd: np.ndarray,
) -> None:
    """Log content-length and UMAP-spread stats for noise vs clustered points."""
    noise_mask     = labels == -1
    clustered_mask = labels != -1
    n_noise     = int(noise_mask.sum())
    n_clustered = int(clustered_mask.sum())
    n_total     = len(labels)

    log.info("\n--- [%s] NOISE ANALYSIS (before soft assignment) ---", track_name)
    log.info("  Noise points : %d / %d  (%.1f%%)",
             n_noise, n_total, 100.0 * n_noise / n_total)

    if n_noise == 0:
        log.info("  No noise points — nothing to analyse.")
        return

    if len(content_lens) == n_total:
        cl_noise = content_lens[noise_mask]
        cl_clust = content_lens[clustered_mask]
        log.info("  Content length — noise    : median=%d  mean=%.0f  p10=%d  p90=%d",
                 int(np.median(cl_noise)), float(cl_noise.mean()),
                 int(np.percentile(cl_noise, 10)), int(np.percentile(cl_noise, 90)))
        log.info("  Content length — clustered: median=%d  mean=%.0f  p10=%d  p90=%d",
                 int(np.median(cl_clust)), float(cl_clust.mean()),
                 int(np.percentile(cl_clust, 10)), int(np.percentile(cl_clust, 90)))
        pct_short_noise = 100.0 * (cl_noise < 50).sum() / n_noise
        pct_short_clust = 100.0 * (cl_clust < 50).sum() / n_clustered
        log.info("  Pct with content < 50 chars — noise: %.1f%%  clustered: %.1f%%",
                 pct_short_noise, pct_short_clust)

    umap_noise = umap_nd[noise_mask]
    umap_clust = umap_nd[clustered_mask]
    log.info("  UMAP spread (std across dims) — noise: %.3f  clustered: %.3f",
             float(umap_noise.std()), float(umap_clust.std()))
    log.info("  → If noise spread ≈ clustered spread and content lengths are similar,")
    log.info("    noise is structural (boundary points) → soft assignment is appropriate.")
    log.info("    If noise is disproportionately short/trivial → use --min-content-len instead.")


def merge_similar_clusters(
    labels: np.ndarray,
    embeddings: np.ndarray,
    cosine_threshold: float = 0.85,
) -> tuple[np.ndarray, int, int]:
    """Merge clusters whose centroid cosine similarity exceeds threshold via average-linkage cut."""
    cluster_ids = sorted(set(labels[labels != -1].tolist()))
    n_before = len(cluster_ids)

    if n_before < 2:
        log.info("Centroid merge: only %d cluster — nothing to merge.", n_before)
        return labels, n_before, n_before

    log.info("Centroid merge: computing centroids for %d clusters …", n_before)

    centroids = np.array(
        [embeddings[labels == c].mean(axis=0) for c in cluster_ids],
        dtype=np.float64,
    )
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_norm = centroids / np.maximum(norms, 1e-10)

    cosine_sim  = centroids_norm @ centroids_norm.T
    cosine_dist = np.clip(1.0 - cosine_sim, 0.0, 2.0)
    np.fill_diagonal(cosine_dist, 0.0)

    dist_condensed = squareform(cosine_dist, checks=False)
    Z = linkage(dist_condensed, method="average")

    cut_distance = 1.0 - cosine_threshold
    merged_assignments = fcluster(Z, t=cut_distance, criterion="distance")

    old_to_group = {old: int(grp) for old, grp in zip(cluster_ids, merged_assignments)}

    unique_groups = sorted(set(old_to_group.values()))
    group_to_new  = {grp: new_id for new_id, grp in enumerate(unique_groups)}
    old_to_new    = {old: group_to_new[grp] for old, grp in old_to_group.items()}

    new_labels = labels.copy()
    for i, lbl in enumerate(labels):
        if lbl != -1:
            new_labels[i] = old_to_new[lbl]

    n_after = len(unique_groups)
    n_merged = n_before - n_after

    if n_merged > 0:
        from collections import defaultdict
        groups: dict[int, list[int]] = defaultdict(list)
        for old, new_id in old_to_new.items():
            groups[new_id].append(old)
        for new_id, members in sorted(groups.items()):
            if len(members) > 1:
                log.info("  Merged → C%d:  original clusters %s  (cosine threshold=%.2f)",
                         new_id, members, cosine_threshold)

    log.info("Centroid merge: %d → %d clusters  (%d merged, threshold=%.2f)",
             n_before, n_after, n_merged, cosine_threshold)

    return new_labels, n_before, n_after


def _silhouette_raw(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_size: int = SIL_SAMPLE_SIZE,
    random_state: int = RNG_SEED,
) -> tuple[float | None, int]:
    """Silhouette on raw embeddings (cosine). Subsampled because silhouette is O(n²). Returns (sil, n_used)."""
    valid = labels != -1
    n = int(valid.sum())
    if n < 2 or len(np.unique(labels[valid])) < 2:
        return None, 0
    n_used = min(sample_size, n)
    try:
        sil = silhouette_score(
            embeddings[valid], labels[valid],
            metric="cosine", sample_size=n_used, random_state=random_state,
        )
        return float(sil), n_used
    except Exception as e:
        log.warning("silhouette_raw failed: %s", e)
        return None, 0


def _dbcv_safe(umap_nd: np.ndarray, labels: np.ndarray,
               sample_size: int = DBCV_SAMPLE_SIZE,
               random_state: int = RNG_SEED) -> float | None:
    """DBCV with a graceful subsample fallback on large datasets."""
    valid = labels != -1
    X = umap_nd[valid].astype(np.float64)
    y = labels[valid]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None
    try:
        if len(y) > sample_size:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(y), size=sample_size, replace=False)
            return float(dbcv_index(X[idx], y[idx], metric="euclidean"))
        return float(dbcv_index(X, y, metric="euclidean"))
    except Exception as e:
        log.warning("DBCV failed: %s", e)
        return None


def compute_metrics(
    track_name: str,
    embeddings: np.ndarray,
    umap_nd: np.ndarray,
    labels: np.ndarray,
    clusterer,
    pulled: PulledEmbeddings,
    min_cluster_size: int,
    min_samples: int,
    epsilon: float = 0.0,
    n_components: int = UMAP_N_COMPONENTS_CLUSTER,
    selection_method: str = HDBSCAN_SELECTION,
    soft_assigned: bool = False,
    n_clusters_before_merge: int | None = None,
    merge_cosine_threshold: float | None = None,
    n_neighbors: int | None = None,
) -> dict:
    """Compute intrinsic quality metrics, input diagnostics, and hyperparameter audit."""
    valid = labels != -1
    n = int(valid.sum())
    n_clusters = len(np.unique(labels[valid]))
    if n < 2 or n_clusters < 2:
        log.warning("[%s] Not enough valid clusters for metrics.", track_name)
        return {}

    umap_valid = umap_nd[valid]
    lbl_valid  = labels[valid]

    sil_raw, sil_raw_n  = _silhouette_raw(embeddings, labels)

    sil_umap_n = min(SIL_SAMPLE_SIZE, len(lbl_valid))
    sil_umap = float(silhouette_score(
        umap_valid, lbl_valid, metric="euclidean",
        sample_size=sil_umap_n, random_state=RNG_SEED,
    ))

    persistences = np.asarray(clusterer.cluster_persistence_, dtype=float)
    relative_validity = float(getattr(clusterer, "relative_validity_", float("nan")))

    metrics = {
        "track":                    track_name,
        "pipeline_track":           track_name,
        "n_rows_total":             int(len(labels)),
        "n_rows_valid":             n,
        "n_clusters":               n_clusters,
        "noise_points":             int((labels == -1).sum()),
        "noise_pct":                round(100.0 * (labels == -1).sum() / len(labels), 2),

        "silhouette_embed_cosine":  None if sil_raw is None else round(sil_raw, 4),
        "silhouette_embed_n":       sil_raw_n,
        "silhouette_umap":          round(sil_umap, 4),
        "davies_bouldin_umap":      round(float(davies_bouldin_score(umap_valid, lbl_valid)), 4),
        "calinski_harabasz_umap":   round(float(calinski_harabasz_score(umap_valid, lbl_valid)), 2),
        "dbcv_umap":                _dbcv_safe(umap_nd, labels),
        "cluster_persistence_mean": round(float(persistences.mean()), 4) if len(persistences) else None,
        "cluster_persistence_min":  round(float(persistences.min()),  4) if len(persistences) else None,
        "cluster_persistence_p25":  round(float(np.percentile(persistences, 25)), 4) if len(persistences) else None,
        "cluster_persistence_p75":  round(float(np.percentile(persistences, 75)), 4) if len(persistences) else None,
        "hdbscan_relative_validity": round(relative_validity, 4) if not np.isnan(relative_validity) else None,

        "umap_n_components":        n_components,
        "umap_n_neighbors":         n_neighbors if n_neighbors is not None else UMAP_N_NEIGHBORS,
        "umap_min_dist":            UMAP_MIN_DIST_CLUSTER,
        "umap_metric":              UMAP_METRIC,
        "umap_seed":                UMAP_SEED,
        "hdbscan_min_cluster_size": min_cluster_size,
        "hdbscan_min_samples":      min_samples,
        "hdbscan_epsilon":          epsilon,
        "hdbscan_metric":           HDBSCAN_METRIC,
        "hdbscan_selection":        selection_method,

        "soft_assigned":            soft_assigned,
        "n_clusters_before_merge":  n_clusters_before_merge,
        "merge_cosine_threshold":   merge_cosine_threshold,

        "content_len_p50":          pulled.content_len_p50,
        "content_len_p95":          pulled.content_len_p95,
        "content_len_max":          pulled.content_len_max,
        "n_inputs_near_token_limit": pulled.n_near_limit,
        "source_table":             pulled.source_table,
        "source_modified":          pulled.source_modified,
    }

    for k, v in metrics.items():
        if k not in ("track", "pipeline_track"):
            log.info("  %s: %s", k, v)
    return metrics


def _delete_track_rows(client: bigquery.Client, table_ref: str,
                       track_name: str, col: str = "pipeline_track") -> None:
    """Delete rows for a given track, tolerating a not-yet-existent table."""
    try:
        client.query(
            f"DELETE FROM `{table_ref}` WHERE {col} = @t",
            job_config=bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("t", "STRING", track_name)]
            ),
        ).result()
        log.info("Cleared previous rows from %s where %s='%s'",
                 table_ref, col, track_name)
    except NotFound:
        log.info("%s does not exist yet — will be created on first write.", table_ref)


def write_labels_to_bq(
    ids: np.ndarray,
    labels: np.ndarray,
    umap_2d: np.ndarray,
    umap_nd: np.ndarray,
    membership: np.ndarray,
    track_name: str,
    client: bigquery.Client,
) -> None:
    df = pd.DataFrame({
        "conversation_id": ids,
        "pipeline_track":  track_name,
        "cluster_id":      labels.astype(int),
        "umap_x":          umap_2d[:, 0].astype(float),
        "umap_y":          umap_2d[:, 1].astype(float),
        "umap_15d_norm":   np.linalg.norm(umap_nd, axis=1).astype(float),
        "membership_prob": membership.astype(float),
    })

    _delete_track_rows(client, DEST_LABELS, track_name)

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=[
            bigquery.SchemaField("conversation_id",  "STRING"),
            bigquery.SchemaField("pipeline_track",   "STRING"),
            bigquery.SchemaField("cluster_id",       "INTEGER"),
            bigquery.SchemaField("umap_x",           "FLOAT"),
            bigquery.SchemaField("umap_y",           "FLOAT"),
            bigquery.SchemaField("umap_15d_norm",    "FLOAT"),
            bigquery.SchemaField("membership_prob",  "FLOAT"),
        ],
    )

    log.info("[%s] Writing %d cluster labels → %s", track_name, len(df), DEST_LABELS)
    client.load_table_from_dataframe(df, DEST_LABELS, job_config=job_config).result()

    local_path = EXPERIMENTS_DIR / f"cluster_labels_{track_name}.csv"
    df.to_csv(local_path, index=False)
    log.info("[%s] Local copy → %s", track_name, local_path)


def write_cluster_persistence_to_bq(clusterer, track_name: str,
                                    client: bigquery.Client) -> None:
    persistences = np.asarray(clusterer.cluster_persistence_, dtype=float)
    if len(persistences) == 0:
        log.warning("[%s] No cluster persistences to write.", track_name)
        return

    df = pd.DataFrame({
        "pipeline_track":       track_name,
        "cluster_id":           np.arange(len(persistences), dtype=int),
        "cluster_persistence":  persistences.astype(float),
        "computed_at":          pd.Timestamp.now(tz="UTC"),
    })

    _delete_track_rows(client, DEST_CLUSTER_PERSIST, track_name)

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=[
            bigquery.SchemaField("pipeline_track",       "STRING"),
            bigquery.SchemaField("cluster_id",           "INTEGER"),
            bigquery.SchemaField("cluster_persistence",  "FLOAT"),
            bigquery.SchemaField("computed_at",          "TIMESTAMP"),
        ],
    )
    log.info("[%s] Writing %d cluster persistences → %s",
             track_name, len(df), DEST_CLUSTER_PERSIST)
    client.load_table_from_dataframe(df, DEST_CLUSTER_PERSIST, job_config=job_config).result()


def write_metrics_to_bq(metrics: dict, client: bigquery.Client) -> None:
    if not metrics:
        return
    metrics = {**metrics, "computed_at": pd.Timestamp.now(tz="UTC")}
    df = pd.DataFrame([metrics])

    schema = [
        bigquery.SchemaField("track",                    "STRING"),
        bigquery.SchemaField("pipeline_track",           "STRING"),
        bigquery.SchemaField("n_rows_total",             "INTEGER"),
        bigquery.SchemaField("n_rows_valid",             "INTEGER"),
        bigquery.SchemaField("n_clusters",               "INTEGER"),
        bigquery.SchemaField("noise_points",             "INTEGER"),
        bigquery.SchemaField("noise_pct",                "FLOAT"),
        bigquery.SchemaField("silhouette_embed_cosine",  "FLOAT"),
        bigquery.SchemaField("silhouette_embed_n",       "INTEGER"),
        bigquery.SchemaField("silhouette_umap",          "FLOAT"),
        bigquery.SchemaField("davies_bouldin_umap",      "FLOAT"),
        bigquery.SchemaField("calinski_harabasz_umap",   "FLOAT"),
        bigquery.SchemaField("dbcv_umap",                "FLOAT"),
        bigquery.SchemaField("cluster_persistence_mean", "FLOAT"),
        bigquery.SchemaField("cluster_persistence_min",  "FLOAT"),
        bigquery.SchemaField("cluster_persistence_p25",  "FLOAT"),
        bigquery.SchemaField("cluster_persistence_p75",  "FLOAT"),
        bigquery.SchemaField("hdbscan_relative_validity", "FLOAT"),
        bigquery.SchemaField("umap_n_components",        "INTEGER"),
        bigquery.SchemaField("umap_n_neighbors",         "INTEGER"),
        bigquery.SchemaField("umap_min_dist",            "FLOAT"),
        bigquery.SchemaField("umap_metric",              "STRING"),
        bigquery.SchemaField("umap_seed",                "INTEGER"),
        bigquery.SchemaField("hdbscan_min_cluster_size", "INTEGER"),
        bigquery.SchemaField("hdbscan_min_samples",      "INTEGER"),
        bigquery.SchemaField("hdbscan_epsilon",          "FLOAT"),
        bigquery.SchemaField("hdbscan_metric",           "STRING"),
        bigquery.SchemaField("hdbscan_selection",        "STRING"),
        bigquery.SchemaField("soft_assigned",            "BOOL"),
        bigquery.SchemaField("n_clusters_before_merge",  "INTEGER"),
        bigquery.SchemaField("merge_cosine_threshold",   "FLOAT"),
        bigquery.SchemaField("content_len_p50",          "FLOAT"),
        bigquery.SchemaField("content_len_p95",          "FLOAT"),
        bigquery.SchemaField("content_len_max",          "FLOAT"),
        bigquery.SchemaField("n_inputs_near_token_limit", "INTEGER"),
        bigquery.SchemaField("source_table",             "STRING"),
        bigquery.SchemaField("source_modified",          "STRING"),
        bigquery.SchemaField("computed_at",              "TIMESTAMP"),
    ]
    for field in schema:
        if field.name not in df.columns:
            df[field.name] = None
    df = df[[f.name for f in schema]]

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=schema,
        schema_update_options=[
            bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION,
        ],
    )
    log.info("Writing evaluation metrics → %s", DEST_METRICS)
    try:
        client.load_table_from_dataframe(df, DEST_METRICS, job_config=job_config).result()
    except Exception as e:
        log.error(
            "Metrics write to %s failed: %s\n"
            "If the schema has changed incompatibly, back up and drop the "
            "table in BigQuery, then re-run:  DROP TABLE `%s`;",
            DEST_METRICS, e, DEST_METRICS,
        )
        raise
    log.info("Metrics written successfully.")


def run_stability_seeds(
    track_name: str,
    embeddings: np.ndarray,
    ids: np.ndarray,
    primary_labels: np.ndarray,
    seeds: Iterable[int],
    min_cluster_size: int,
    min_samples: int,
    n_components: int,
    client: bigquery.Client,
    n_neighbors: int | None = None,
    filter_tag: str = "",
) -> pd.DataFrame:
    """Refit UMAP+HDBSCAN with each seed, compute ARI vs primary labels, write to cluster_stability_ari."""
    rows = []
    primary_n_clusters = int(len(np.unique(primary_labels[primary_labels != -1])))
    suffix = f"{n_components}d{filter_tag}"

    for seed in seeds:
        log.info("[%s] Stability seed %d …", track_name, seed)
        umap_nd, _ = reduce_umap(
            embeddings, n_components,
            UMAP_MIN_DIST_CLUSTER, track_name, suffix,
            random_state=seed,
            n_neighbors=n_neighbors,
        )
        lbl, _, _ = run_hdbscan(umap_nd, min_cluster_size, min_samples, epsilon=0.0)
        n_clusters  = int(len(np.unique(lbl[lbl != -1])))
        n_noise     = int((lbl == -1).sum())
        ari         = float(adjusted_rand_score(primary_labels, lbl))
        rows.append({
            "pipeline_track":  track_name,
            "seed":            int(seed),
            "primary_seed":    int(UMAP_SEED),
            "ari_vs_primary":  round(ari, 4),
            "n_clusters":      n_clusters,
            "primary_n_clusters": primary_n_clusters,
            "noise_pct":       round(100.0 * n_noise / len(lbl), 2),
            "computed_at":     pd.Timestamp.now(tz="UTC"),
        })
        log.info("[%s] seed=%d  ARI=%.4f  clusters=%d",
                 track_name, seed, ari, n_clusters)

    df = pd.DataFrame(rows)

    _delete_track_rows(client, DEST_STABILITY_ARI, track_name)
    schema = [
        bigquery.SchemaField("pipeline_track",    "STRING"),
        bigquery.SchemaField("seed",              "INTEGER"),
        bigquery.SchemaField("primary_seed",      "INTEGER"),
        bigquery.SchemaField("ari_vs_primary",    "FLOAT"),
        bigquery.SchemaField("n_clusters",        "INTEGER"),
        bigquery.SchemaField("primary_n_clusters", "INTEGER"),
        bigquery.SchemaField("noise_pct",         "FLOAT"),
        bigquery.SchemaField("computed_at",       "TIMESTAMP"),
    ]
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=schema,
    )
    client.load_table_from_dataframe(df, DEST_STABILITY_ARI, job_config=job_config).result()

    df.to_csv(EXPERIMENTS_DIR / f"stability_ari_{track_name}.csv", index=False)
    log.info("[%s] Stability: mean ARI=%.4f  min ARI=%.4f",
             track_name, df["ari_vs_primary"].mean(), df["ari_vs_primary"].min())
    return df


def run_sweep(
    track_name: str,
    embeddings: np.ndarray,
    n_components_values: Iterable[int],
    min_cluster_sizes: Iterable[int],
    min_samples_values: Iterable[int],
    epsilon_values: Iterable[float],
    client: bigquery.Client,
    n_neighbors: int | None = None,
    filter_tag: str = "",
) -> pd.DataFrame:
    """Sweep n_components × mcs × min_samples × epsilon; UMAP fit once per nc (cached)."""
    rows = []
    n_components_list   = list(n_components_values)
    mcs_list            = list(min_cluster_sizes)
    ms_list             = list(min_samples_values)
    epsilon_list        = list(epsilon_values)
    total = len(n_components_list) * len(mcs_list) * len(ms_list) * len(epsilon_list)
    log.info("[%s] Sweep: %d combinations  (n_comp×mcs×ms×eps = %d×%d×%d×%d)",
             track_name, total,
             len(n_components_list), len(mcs_list), len(ms_list), len(epsilon_list))

    done = 0
    for nc in n_components_list:
        log.info("[%s] UMAP → %dD …", track_name, nc)
        umap_nd, _ = reduce_umap(
            embeddings, nc, UMAP_MIN_DIST_CLUSTER, track_name, f"{nc}d{filter_tag}",
            n_neighbors=n_neighbors,
        )
        for mcs, ms, eps in itertools.product(mcs_list, ms_list, epsilon_list):
            lbl, _, clusterer = run_hdbscan(umap_nd, mcs, ms, epsilon=eps)
            valid      = lbl != -1
            n_clusters = int(len(np.unique(lbl[valid])))
            n_noise    = int((~valid).sum())

            sil_umap = (
                float(silhouette_score(
                    umap_nd[valid], lbl[valid], metric="euclidean",
                    sample_size=min(SIL_SAMPLE_SIZE, int(valid.sum())),
                    random_state=RNG_SEED,
                ))
                if n_clusters >= 2 else None
            )
            sil_raw, sil_raw_n = _silhouette_raw(embeddings, lbl)
            persistences = np.asarray(clusterer.cluster_persistence_, dtype=float)

            rows.append({
                "pipeline_track":             track_name,
                "n_components":               int(nc),
                "min_cluster_size":           int(mcs),
                "min_samples":                int(ms),
                "epsilon":                    float(eps),
                "n_clusters":                 n_clusters,
                "noise_pct":                  round(100.0 * n_noise / len(lbl), 2),
                "silhouette_embed_cosine":    None if sil_raw is None else round(sil_raw, 4),
                "silhouette_embed_n":         sil_raw_n,
                "silhouette_umap":            None if sil_umap is None else round(sil_umap, 4),
                "cluster_persistence_mean":   round(float(persistences.mean()), 4) if len(persistences) else None,
                "computed_at":                pd.Timestamp.now(tz="UTC"),
            })
            done += 1
            if done % 20 == 0:
                log.info("[%s] Sweep progress: %d/%d", track_name, done, total)

    df = pd.DataFrame(rows)
    _delete_track_rows(client, DEST_SWEEP, track_name)
    schema = [
        bigquery.SchemaField("pipeline_track",             "STRING"),
        bigquery.SchemaField("n_components",               "INTEGER"),
        bigquery.SchemaField("min_cluster_size",           "INTEGER"),
        bigquery.SchemaField("min_samples",                "INTEGER"),
        bigquery.SchemaField("epsilon",                    "FLOAT"),
        bigquery.SchemaField("n_clusters",                 "INTEGER"),
        bigquery.SchemaField("noise_pct",                  "FLOAT"),
        bigquery.SchemaField("silhouette_embed_cosine",    "FLOAT"),
        bigquery.SchemaField("silhouette_embed_n",         "INTEGER"),
        bigquery.SchemaField("silhouette_umap",            "FLOAT"),
        bigquery.SchemaField("cluster_persistence_mean",   "FLOAT"),
        bigquery.SchemaField("computed_at",                "TIMESTAMP"),
    ]

    try:
        remaining = list(client.query(
            f"SELECT COUNT(*) AS n FROM `{DEST_SWEEP}`"
        ).result())[0]["n"]
        disposition = (
            bigquery.WriteDisposition.WRITE_TRUNCATE
            if remaining == 0
            else bigquery.WriteDisposition.WRITE_APPEND
        )
    except Exception:
        disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

    job_config = bigquery.LoadJobConfig(
        write_disposition=disposition,
        schema=schema,
    )
    client.load_table_from_dataframe(df, DEST_SWEEP, job_config=job_config).result()
    df.to_csv(EXPERIMENTS_DIR / f"sweep_{track_name}.csv", index=False)
    log.info("[%s] Sweep complete: %d configurations → %s", track_name, len(df), DEST_SWEEP)
    return df


def run_focused_sweep(
    track_name: str,
    embeddings: np.ndarray,
    client: bigquery.Client,
    n_neighbors: int | None = None,
    filter_tag: str = "",
) -> pd.DataFrame:
    """Sweep selection_method × higher min_samples at fixed nc (cache hit); full sweep never tested these."""
    nc = FOCUSED_SWEEP_NC
    nn = n_neighbors if n_neighbors is not None else UMAP_N_NEIGHBORS
    log.info("[%s] Focused sweep: nc=%d, n_neighbors=%d, "
             "selection=%s, mcs=%s, ms=%s, eps=%s",
             track_name, nc, nn,
             FOCUSED_SWEEP_SELECTION, FOCUSED_SWEEP_MCS,
             FOCUSED_SWEEP_MS, FOCUSED_SWEEP_EPSILON)

    log.info("[%s] Loading UMAP-%dD (cache hit expected)…", track_name, nc)
    umap_nd, _ = reduce_umap(
        embeddings, nc, UMAP_MIN_DIST_CLUSTER, track_name, f"{nc}d{filter_tag}",
        n_neighbors=n_neighbors,
    )

    rows = []
    total = (len(FOCUSED_SWEEP_SELECTION) * len(FOCUSED_SWEEP_MCS)
             * len(FOCUSED_SWEEP_MS) * len(FOCUSED_SWEEP_EPSILON))
    done = 0

    for sel in FOCUSED_SWEEP_SELECTION:
        for mcs, ms, eps in itertools.product(
            FOCUSED_SWEEP_MCS, FOCUSED_SWEEP_MS, FOCUSED_SWEEP_EPSILON
        ):
            lbl, _, clusterer = run_hdbscan(
                umap_nd, mcs, ms, epsilon=eps, selection_method=sel,
            )
            valid      = lbl != -1
            n_clusters = int(len(np.unique(lbl[valid])))
            n_noise    = int((~valid).sum())

            sil_umap = (
                float(silhouette_score(
                    umap_nd[valid], lbl[valid], metric="euclidean",
                    sample_size=min(SIL_SAMPLE_SIZE, int(valid.sum())),
                    random_state=RNG_SEED,
                ))
                if n_clusters >= 2 else None
            )
            sil_raw, sil_raw_n = _silhouette_raw(embeddings, lbl)
            persistences = np.asarray(clusterer.cluster_persistence_, dtype=float)
            dbcv = _dbcv_safe(umap_nd, lbl)

            rows.append({
                "pipeline_track":           track_name,
                "n_components":             int(nc),
                "umap_n_neighbors":         int(nn),
                "selection_method":         sel,
                "min_cluster_size":         int(mcs),
                "min_samples":              int(ms),
                "epsilon":                  float(eps),
                "n_clusters":               n_clusters,
                "noise_pct":                round(100.0 * n_noise / len(lbl), 2),
                "silhouette_embed_cosine":  None if sil_raw is None else round(sil_raw, 4),
                "silhouette_embed_n":       sil_raw_n,
                "silhouette_umap":          None if sil_umap is None else round(sil_umap, 4),
                "dbcv_umap":                None if dbcv is None else round(dbcv, 4),
                "cluster_persistence_mean": (
                    round(float(persistences.mean()), 4) if len(persistences) else None
                ),
                "computed_at":              pd.Timestamp.now(tz="UTC"),
            })
            done += 1
            if done % 8 == 0:
                log.info("[%s] Focused sweep: %d/%d", track_name, done, total)

    df = pd.DataFrame(rows)

    out_csv = EXPERIMENTS_DIR / f"focused_sweep_{track_name}.csv"
    df.to_csv(out_csv, index=False)
    log.info("[%s] Focused sweep → %s", track_name, out_csv)

    _delete_track_rows(client, DEST_FOCUSED_SWEEP, track_name)
    schema = [
        bigquery.SchemaField("pipeline_track",            "STRING"),
        bigquery.SchemaField("n_components",              "INTEGER"),
        bigquery.SchemaField("umap_n_neighbors",          "INTEGER"),
        bigquery.SchemaField("selection_method",          "STRING"),
        bigquery.SchemaField("min_cluster_size",          "INTEGER"),
        bigquery.SchemaField("min_samples",               "INTEGER"),
        bigquery.SchemaField("epsilon",                   "FLOAT"),
        bigquery.SchemaField("n_clusters",                "INTEGER"),
        bigquery.SchemaField("noise_pct",                 "FLOAT"),
        bigquery.SchemaField("silhouette_embed_cosine",   "FLOAT"),
        bigquery.SchemaField("silhouette_embed_n",        "INTEGER"),
        bigquery.SchemaField("silhouette_umap",           "FLOAT"),
        bigquery.SchemaField("dbcv_umap",                 "FLOAT"),
        bigquery.SchemaField("cluster_persistence_mean",  "FLOAT"),
        bigquery.SchemaField("computed_at",               "TIMESTAMP"),
    ]
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        schema=schema,
        schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
    )
    client.load_table_from_dataframe(df, DEST_FOCUSED_SWEEP, job_config=job_config).result()
    log.info("[%s] Focused sweep → %s  (%d rows)", track_name, DEST_FOCUSED_SWEEP, len(df))

    top = (df.dropna(subset=["silhouette_embed_cosine"])
             .sort_values("silhouette_embed_cosine", ascending=False))
    print(f"\n{'='*90}")
    print(f"FOCUSED SWEEP TOP RESULTS — {track_name.upper()}")
    print(f"  sel=selection_method  mcs=min_cluster_size  ms=min_samples")
    print(f"{'='*90}")
    print(f"  {'sel':>4}  {'mcs':>5}  {'ms':>4}  {'k':>6}  "
          f"{'noise':>7}  {'sil_cos':>9}  {'sil_umap':>9}  {'dbcv':>9}  {'persist':>9}")
    print(f"  {'-'*92}")
    for _, r in top.head(16).iterrows():
        dbcv_val = r.get("dbcv_umap", float("nan"))
        print(f"  {r['selection_method']:>4}  {int(r['min_cluster_size']):>5}  "
              f"{int(r['min_samples']):>4}  {int(r['n_clusters']):>6}  "
              f"{r['noise_pct']:>6.1f}%  "
              f"{r['silhouette_embed_cosine']:>9.4f}  "
              f"{r.get('silhouette_umap', float('nan')):>9.4f}  "
              f"{dbcv_val:>9.4f}  "
              f"{r.get('cluster_persistence_mean', float('nan')):>9.4f}")

    return df


@dataclass
class TrackResult:
    track_key:  str
    track_name: str
    ids:        np.ndarray
    labels:     np.ndarray
    metrics:    dict


def run_track(
    track_key: str,
    min_cluster_size: int,
    min_samples: int,
    epsilon: float,
    skip_embed: bool,
    stability_seeds: tuple[int, ...] | None,
    do_sweep: bool,
    n_components: int = UMAP_N_COMPONENTS_CLUSTER,
    selection_method: str = HDBSCAN_SELECTION,
    soft_assign: bool = False,
    soft_assign_min_prob: float = 0.0,
    merge_cosine: float | None = None,
    do_focused_sweep: bool = False,
    n_neighbors: int | None = None,
    min_content_len: int | None = None,
    max_content_len: int | None = None,
) -> TrackResult:
    """Run the full clustering pipeline for one track."""
    cfg    = TRACKS[track_key]
    name   = cfg["name"]
    client = bq_client()

    log.info("\n%s\n%s\n%s", "=" * 60, cfg["label"], "=" * 60)

    pulled = pull_embeddings(track_key, client, skip_embed=skip_embed)
    ids, embeddings = pulled.ids, pulled.embeddings

    if min_content_len is not None or max_content_len is not None:
        mask = np.ones(len(ids), dtype=bool)
        if min_content_len is not None:
            mask &= pulled.content_lens >= min_content_len
        if max_content_len is not None:
            mask &= pulled.content_lens <= max_content_len
        n_before = int(len(ids))
        ids          = ids[mask]
        embeddings   = embeddings[mask]
        content_lens = pulled.content_lens[mask]
        n_kept = int(len(ids))
        log.info(
            "[%s] content-len filter [%s–%s]: %d → %d rows (%.1f%% kept, %.1f%% removed)",
            name,
            f"{min_content_len}" if min_content_len is not None else "*",
            f"{max_content_len}" if max_content_len is not None else "*",
            n_before, n_kept,
            100.0 * n_kept / n_before,
            100.0 * (n_before - n_kept) / n_before,
        )
        if n_kept < 500:
            log.warning("[%s] Only %d rows after filtering — results may be unreliable.", name, n_kept)
    else:
        content_lens = pulled.content_lens

    filter_tag = ""
    if min_content_len is not None or max_content_len is not None:
        lo = min_content_len if min_content_len is not None else 0
        hi = max_content_len if max_content_len is not None else 99999
        filter_tag = f"_L{lo}to{hi}"
    cluster_suffix = f"{n_components}d{filter_tag}"

    umap_nd, _ = reduce_umap(
        embeddings, n_components,
        UMAP_MIN_DIST_CLUSTER, name, cluster_suffix,
        n_neighbors=n_neighbors,
    )
    labels, membership, clusterer = run_hdbscan(
        umap_nd, min_cluster_size, min_samples, epsilon, selection_method,
    )

    umap_2d, _ = reduce_umap(
        embeddings, UMAP_N_COMPONENTS_VIZ,
        UMAP_MIN_DIST_VIZ, name, f"2d{filter_tag}",
        n_neighbors=n_neighbors,
    )

    _log_noise_summary(name, labels, content_lens, umap_nd)

    log.info("\n--- [%s] METRICS BEFORE SOFT ASSIGNMENT ---", name)
    metrics_before = compute_metrics(
        track_name=name,
        embeddings=embeddings,
        umap_nd=umap_nd,
        labels=labels,
        clusterer=clusterer,
        pulled=pulled,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        epsilon=epsilon,
        n_components=n_components,
        selection_method=selection_method,
        soft_assigned=False,
        n_clusters_before_merge=None,
        merge_cosine_threshold=None,
        n_neighbors=n_neighbors,
    )

    if soft_assign:
        labels, membership = assign_noise_to_nearest(
            clusterer, labels, membership, min_prob=soft_assign_min_prob
        )
        log.info("\n--- [%s] METRICS AFTER SOFT ASSIGNMENT ---", name)

    n_clusters_before_merge: int | None = None
    if merge_cosine is not None:
        n_clusters_before_merge = int(len(np.unique(labels[labels != -1])))
        labels, _, n_after = merge_similar_clusters(labels, embeddings, merge_cosine)
        log.info("[%s] After centroid merge: %d → %d clusters",
                 name, n_clusters_before_merge, n_after)

    if soft_assign or merge_cosine is not None:
        metrics = compute_metrics(
            track_name=name,
            embeddings=embeddings,
            umap_nd=umap_nd,
            labels=labels,
            clusterer=clusterer,
            pulled=pulled,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            epsilon=epsilon,
            n_components=n_components,
            selection_method=selection_method,
            soft_assigned=soft_assign,
            n_clusters_before_merge=n_clusters_before_merge,
            merge_cosine_threshold=merge_cosine,
            n_neighbors=n_neighbors,
        )
    else:
        metrics = metrics_before

    clusterer_path = MODELS_DIR / f"hdbscan_{name}_{_hyperparam_fingerprint(n_neighbors)}.joblib"
    joblib.dump(clusterer, clusterer_path)
    log.info("[%s] HDBSCAN clusterer persisted → %s", name, clusterer_path)

    write_labels_to_bq(ids, labels, umap_2d, umap_nd, membership, name, client)
    write_cluster_persistence_to_bq(clusterer, name, client)
    write_metrics_to_bq(metrics, client)

    if stability_seeds:
        run_stability_seeds(
            track_name=name,
            embeddings=embeddings,
            ids=ids,
            primary_labels=labels,
            seeds=stability_seeds,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            n_components=n_components,
            client=client,
            n_neighbors=n_neighbors,
            filter_tag=filter_tag,
        )

    if do_sweep:
        run_sweep(
            track_name=name,
            embeddings=embeddings,
            n_components_values=SWEEP_N_COMPONENTS,
            min_cluster_sizes=SWEEP_MIN_CLUSTER_SIZES,
            min_samples_values=SWEEP_MIN_SAMPLES,
            epsilon_values=SWEEP_EPSILON,
            client=client,
            n_neighbors=n_neighbors,
            filter_tag=filter_tag,
        )

    if do_focused_sweep:
        run_focused_sweep(
            track_name=name,
            embeddings=embeddings,
            client=client,
            n_neighbors=n_neighbors,
            filter_tag=filter_tag,
        )

    return TrackResult(
        track_key=track_key,
        track_name=name,
        ids=ids,
        labels=labels,
        metrics=metrics,
    )


def compute_cross_track_ari(results: dict[str, TrackResult],
                            client: bigquery.Client) -> float | None:
    """ARI(A, B) on the intersection of IDs from both tracks."""
    if "a" not in results or "b" not in results:
        log.info("Cross-track ARI skipped — need both tracks.")
        return None

    ra, rb = results["a"], results["b"]

    df_a = pd.DataFrame({"conversation_id": ra.ids, "label_a": ra.labels})
    df_b = pd.DataFrame({"conversation_id": rb.ids, "label_b": rb.labels})

    merged = df_a.merge(df_b, on="conversation_id", how="inner")
    n_shared = len(merged)
    n_only_a = len(df_a) - n_shared
    n_only_b = len(df_b) - n_shared
    if n_shared < 2:
        log.warning("Cross-track ARI: too few shared IDs (%d) — skipping.", n_shared)
        return None

    ari = float(adjusted_rand_score(merged["label_a"], merged["label_b"]))
    log.info("Cross-track ARI = %.4f  (shared=%d, only-A=%d, only-B=%d)",
             ari, n_shared, n_only_a, n_only_b)

    out = pd.DataFrame([{
        "track_a":           ra.track_name,
        "track_b":           rb.track_name,
        "n_shared":          int(n_shared),
        "n_only_a":          int(n_only_a),
        "n_only_b":          int(n_only_b),
        "ari":               round(ari, 4),
        "n_clusters_a":      int(len(np.unique(ra.labels[ra.labels != -1]))),
        "n_clusters_b":      int(len(np.unique(rb.labels[rb.labels != -1]))),
        "computed_at":       pd.Timestamp.now(tz="UTC"),
    }])

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        schema=[
            bigquery.SchemaField("track_a",        "STRING"),
            bigquery.SchemaField("track_b",        "STRING"),
            bigquery.SchemaField("n_shared",       "INTEGER"),
            bigquery.SchemaField("n_only_a",       "INTEGER"),
            bigquery.SchemaField("n_only_b",       "INTEGER"),
            bigquery.SchemaField("ari",            "FLOAT"),
            bigquery.SchemaField("n_clusters_a",   "INTEGER"),
            bigquery.SchemaField("n_clusters_b",   "INTEGER"),
            bigquery.SchemaField("computed_at",    "TIMESTAMP"),
        ],
    )
    client.load_table_from_dataframe(out, DEST_CROSS_TRACK_ARI, job_config=job_config).result()
    log.info("Cross-track ARI written → %s", DEST_CROSS_TRACK_ARI)
    return ari


def _parse_seeds(arg: str | None) -> tuple[int, ...] | None:
    if arg is None:
        return None
    if arg.strip() == "":
        return tuple(DEFAULT_STABILITY_SEEDS)
    try:
        if "," in arg:
            return tuple(int(x.strip()) for x in arg.split(",") if x.strip())
        k = int(arg)
        if k <= 0:
            return None
        return tuple(DEFAULT_STABILITY_SEEDS[:k])
    except ValueError:
        raise SystemExit(
            f"--stability-seeds must be an integer or comma-separated list, got: {arg!r}"
        )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--track", choices=["a", "b", "both"], default="both",
                        help="Which pipeline track to run (default: both)")
    parser.add_argument("--min-cluster-size", type=int, default=HDBSCAN_MIN_CLUSTER_SIZE)
    parser.add_argument("--min-samples",      type=int, default=HDBSCAN_MIN_SAMPLES)
    parser.add_argument("--epsilon",           type=float, default=HDBSCAN_EPSILON,
                        help="cluster_selection_epsilon: merges clusters closer than this "
                             "in mutual-reachability space (0.0 = no merging, try 0.2–0.4). "
                             "Primary lever for reducing over-split cluster count.")
    parser.add_argument("--umap-components",  type=int, default=UMAP_N_COMPONENTS_CLUSTER,
                        help="UMAP output dimensions for clustering (default %(default)s). "
                             "Try 30 or 50 for better cluster separation at 120+ clusters. "
                             "Changing this value invalidates UMAP caches.")
    parser.add_argument("--selection-method", choices=["eom", "leaf"],
                        default=HDBSCAN_SELECTION,
                        help="HDBSCAN cluster_selection_method. 'eom' (default) merges "
                             "sub-clusters into macro-topics → fewer, coarser clusters. "
                             "'leaf' preserves fine-grained sub-clusters.")
    parser.add_argument("--soft-assign", action="store_true",
                        help="Assign noise points (-1) to their highest-probability cluster "
                             "via HDBSCAN soft membership vectors. Applied before centroid merging. "
                             "Use --soft-assign-min-prob to restrict to high-confidence assignments.")
    parser.add_argument("--soft-assign-min-prob", type=float, default=0.0, metavar="PROB",
                        help="Only soft-assign noise points whose best cluster soft-membership "
                             "probability >= PROB (0.0–1.0). Default 0.0 = assign all noise. "
                             "0.7–0.9 = conservative: only reassign borderline points clearly "
                             "close to a cluster; truly ambiguous points stay as noise (-1). "
                             "The script logs how many noise points exceed each threshold so "
                             "you can choose an appropriate value. Requires --soft-assign.")
    parser.add_argument("--merge-cosine", type=float, default=None, metavar="THRESHOLD",
                        help="After clustering, merge any two clusters whose 768-D centroid "
                             "cosine similarity exceeds THRESHOLD (e.g. 0.85). Reversible "
                             "post-processing step; does not require re-running UMAP/HDBSCAN.")
    parser.add_argument("--skip-embed", action="store_true",
                        help="Reuse cached .npy embeddings (cache freshness is "
                             "validated against the source table's modified_time).")
    parser.add_argument("--stability-seeds", nargs="?", const="",
                        default=None,
                        help="Run UMAP stability analysis. Pass an integer N to use "
                             "the first N of the default 10 seeds, or a comma-separated "
                             "list like '1,7,42'. With no value, uses all 10 default seeds.")
    parser.add_argument("--umap-neighbors", type=int, default=None, metavar="N",
                        help=f"Override UMAP n_neighbors (default {UMAP_N_NEIGHBORS}). "
                             "Changing this from the default triggers a UMAP refit "
                             "(cache miss for new fingerprint). Try 15 for finer local "
                             "structure or 50 for broader global structure.")
    parser.add_argument("--sweep", action="store_true",
                        help=f"Sweep n_components ∈ {SWEEP_N_COMPONENTS}, "
                             f"min_cluster_size ∈ {SWEEP_MIN_CLUSTER_SIZES}, "
                             f"min_samples ∈ {SWEEP_MIN_SAMPLES}, "
                             f"epsilon ∈ {SWEEP_EPSILON}. "
                             f"Prints top-10 configs ranked by silhouette_embed_cosine.")
    parser.add_argument("--focused-sweep", action="store_true",
                        help="Run a focused sweep over the dimensions the original sweep "
                             "never tested: selection_method ∈ {eom, leaf} × "
                             f"min_samples ∈ {FOCUSED_SWEEP_MS} × "
                             f"min_cluster_size ∈ {FOCUSED_SWEEP_MCS}. "
                             "Fixed at nc=20 (cache hit) and eps=0.0. "
                             "Only ~32 configs per track — fast (~5 min). "
                             "Results → hdbscan_focused_sweep + focused_sweep_<track>.csv")
    parser.add_argument("--no-cross-track-ari", action="store_true",
                        help="Skip computing ARI(A,B) after both tracks finish.")
    parser.add_argument("--min-content-len", type=int, default=None, metavar="CHARS",
                        help="Exclude conversations whose embedding input is shorter than "
                             "CHARS characters. Removes trivial/abandoned conversations "
                             "(e.g. single 'hi' or 'ok' messages). "
                             "Changes the UMAP cache suffix so filtered and unfiltered "
                             "projections are stored separately.")
    parser.add_argument("--max-content-len", type=int, default=None, metavar="CHARS",
                        help="Exclude conversations whose embedding input is longer than "
                             "CHARS characters. Removes outlier long conversations that "
                             "may blur topic embeddings due to repetitive back-and-forth. "
                             "Changes the UMAP cache suffix (see --min-content-len).")
    parser.add_argument("--embed-suffix", type=str, default=None, metavar="SUFFIX",
                        help="Append SUFFIX to the source embedding table names and cache "
                             "filenames. --embed-suffix gemini uses "
                             "conversation_embeddings_multilingual_gemini instead of "
                             "conversation_embeddings_multilingual (run 04e_embeddings_gemini.sql "
                             "first to populate the Gemini tables). Does not affect destination "
                             "tables so results remain comparable across runs. "
                             "Example: --embed-suffix gemini")
    args = parser.parse_args()

    if args.soft_assign and args.epsilon > 0.0:
        parser.error(
            "--soft-assign and --epsilon > 0 are mutually exclusive.\n"
            "  --epsilon merges nearby clusters during HDBSCAN (changes the model).\n"
            "  --soft-assign reassigns noise points after HDBSCAN (post-processing).\n"
            "Use one or the other, not both."
        )

    stability_seeds = _parse_seeds(args.stability_seeds)

    if args.embed_suffix:
        suffix = args.embed_suffix.strip("_")
        for cfg in TRACKS.values():
            cfg["bq_table"] = cfg["bq_table"] + f"_{suffix}"
            cfg["name"]     = cfg["name"] + f"_{suffix}"
            log.info("--embed-suffix %s: source table → %s", suffix, cfg["bq_table"])

    tracks_to_run = ["a", "b"] if args.track == "both" else [args.track]

    results: dict[str, TrackResult] = {}
    for tk in tracks_to_run:
        results[tk] = run_track(
            track_key=tk,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            epsilon=args.epsilon,
            skip_embed=args.skip_embed,
            stability_seeds=stability_seeds,
            do_sweep=args.sweep,
            n_components=args.umap_components,
            selection_method=args.selection_method,
            soft_assign=args.soft_assign,
            soft_assign_min_prob=args.soft_assign_min_prob,
            merge_cosine=args.merge_cosine,
            do_focused_sweep=args.focused_sweep,
            n_neighbors=args.umap_neighbors,
            min_content_len=args.min_content_len,
            max_content_len=args.max_content_len,
        )

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    for tk, res in results.items():
        print(f"\n{TRACKS[tk]['label']}")
        for k, v in res.metrics.items():
            if k not in ("track", "pipeline_track"):
                print(f"  {k:32s}: {v}")

    if args.sweep:
        for tk, res in results.items():
            csv_path = EXPERIMENTS_DIR / f"sweep_{res.track_name}.csv"
            if not csv_path.exists():
                continue
            sw = pd.read_csv(csv_path).dropna(subset=["silhouette_embed_cosine"])
            sw = sw.sort_values("silhouette_embed_cosine", ascending=False)
            print(f"\n{'='*95}")
            print(f"TOP 10 BY silhouette_embed_cosine  —  {TRACKS[tk]['label']}")
            print(f"  nc=n_components  mcs=min_cluster_size  ms=min_samples  eps=epsilon")
            print(f"{'='*95}")
            print(f"  {'nc':>4}  {'mcs':>5}  {'ms':>4}  {'eps':>5}  {'k':>6}  "
                  f"{'noise':>7}  {'sil_cos':>9}  {'sil_umap':>9}  {'persist':>9}")
            print(f"  {'-'*88}")
            for _, r in sw.head(10).iterrows():
                print(f"  {int(r['n_components']):>4}  {int(r['min_cluster_size']):>5}  "
                      f"{int(r['min_samples']):>4}  {r['epsilon']:>5.2f}  "
                      f"{int(r['n_clusters']):>6}  {r['noise_pct']:>6.1f}%  "
                      f"{r['silhouette_embed_cosine']:>9.4f}  "
                      f"{r.get('silhouette_umap', float('nan')):>9.4f}  "
                      f"{r.get('cluster_persistence_mean', float('nan')):>9.4f}")
            print(f"\n  Current run: nc={args.umap_components}, mcs={args.min_cluster_size}, "
                  f"ms={args.min_samples}, eps={args.epsilon:.2f}, "
                  f"method={args.selection_method}")
            print(f"  → Re-run with: python scripts/experiments/embed_and_cluster.py "
                  f"--umap-components <nc> --min-cluster-size <mcs> "
                  f"--min-samples <ms> --epsilon <eps> --selection-method <method>")

    if len(results) == 2 and not args.no_cross_track_ari:
        compute_cross_track_ari(results, bq_client())

    print("\nDone.")
    print(f"  cluster_labels           → {DEST_LABELS}")
    print(f"  cluster_persistence      → {DEST_CLUSTER_PERSIST}")
    print(f"  evaluation_metrics       → {DEST_METRICS}")
    if stability_seeds:
        print(f"  cluster_stability_ari    → {DEST_STABILITY_ARI}")
    if args.sweep:
        print(f"  hdbscan_sweep_results    → {DEST_SWEEP}")
    if args.focused_sweep:
        print(f"  hdbscan_focused_sweep    → {DEST_FOCUSED_SWEEP}")
    if len(results) == 2 and not args.no_cross_track_ari:
        print(f"  cross_track_ari          → {DEST_CROSS_TRACK_ARI}")


if __name__ == "__main__":
    main()
