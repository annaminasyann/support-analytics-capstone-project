"""
UMAP + HDBSCAN clustering on semantic chunk embeddings.

Pipeline (canonical):
  semantic_chunker.py → 04_chunk_embeddings.sql → chunk_clustering.py
  → contrastive_finetune.py → chunk_clustering.py --skip-embed --embed-suffix finetuned

  chunk_level_translation.sql (future_work/) was not executed.
  translated_chunk_text = original_chunk_text; embeddings via multilingual model.

Canonical result: nc=10, mcs=2000, ms=20, leaf, embed-suffix=finetuned → 24 clusters, noise=12%, sil_cos=0.804, DBCV=0.547

Usage:
    # Step 1 — initial clustering on multilingual-768D (generates pairs for fine-tuning)
    python chunk_clustering.py

    # Step 2 — re-cluster after contrastive_finetune.py (canonical result: 24 clusters, 12% noise, sil_cos=0.804)
    python chunk_clustering.py --skip-embed --embed-suffix finetuned

    # Other flags
    python chunk_clustering.py --sweep                   # grid search
    python chunk_clustering.py --combine-embeddings      # blend two embedding sets
"""
from __future__ import annotations

import argparse
import itertools
import logging
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import umap
import hdbscan
from hdbscan import validity_index as dbcv_index
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from google.cloud import bigquery

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from gcp_config import PROJECT_ID, bq_table  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

OUT_DIR     = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "data" / "results"
MODELS_DIR  = REPO_ROOT / "data" / "models" / "chunk"
OUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SRC_CHUNK_EMBEDDINGS          = bq_table("conversation_chunk_embeddings")
SRC_CHUNK_EMBEDDINGS_ORIGINAL = bq_table("conversation_chunk_embeddings_original")
DEST_CHUNK_LABELS      = bq_table("chunk_cluster_labels")
DEST_TOPIC_ASSIGNMENTS = bq_table("conversation_chunk_topics")
DEST_METRICS           = bq_table("chunk_cluster_metrics")
DEST_SWEEP             = bq_table("chunk_hdbscan_sweep")

UMAP_COMPONENTS          = 10
UMAP_NEIGHBORS           = 30
UMAP_MIN_DIST            = 0.0
UMAP_METRIC              = "cosine"
UMAP_SEED                = 42

HDBSCAN_MIN_CLUSTER_SIZE = 2000
HDBSCAN_MIN_SAMPLES      = 20
HDBSCAN_SELECTION        = "leaf"
HDBSCAN_EPSILON          = 0.0

CONFIDENCE_THRESHOLD     = 0.70
MERGE_COSINE_THRESHOLD   = 0.0
COMBINE_WEIGHT           = 0.7

SIL_SAMPLE               = 10_000
RNG_SEED                 = 42

SWEEP_UMAP_COMPONENTS   = (10, 15, 20)
SWEEP_MIN_CLUSTER_SIZES = (1000, 1500, 1800, 2000, 2500)  
SWEEP_MIN_SAMPLES       = (10, 15, 20)
SWEEP_SELECTIONS        = ("eom", "leaf")

N_REPRESENTATIVE_CHUNKS = 3


def _emb_query(table_name: str) -> str:
    return f"""
    SELECT
        chunk_uid,
        conversation_id,
        chunk_id,
        detected_language,
        message_start_idx,
        message_end_idx,
        message_count,
        chunk_char_count,
        original_chunk_text,
        -- translated_chunk_text = original_chunk_text (chunk_level_translation.sql not executed)
        content                      AS translated_chunk_text,
        ml_generate_embedding_result AS embedding,
        LENGTH(content)              AS content_len
    FROM `{table_name}`
    WHERE ARRAY_LENGTH(ml_generate_embedding_result) > 0
    ORDER BY conversation_id, chunk_id
    """

def _original_emb_query(table_name: str) -> str:
    """Query for original-text chunk embeddings (10b table).  Minimal columns."""
    return f"""
    SELECT
        chunk_uid,
        ml_generate_embedding_result AS embedding_original
    FROM `{table_name}`
    WHERE ARRAY_LENGTH(ml_generate_embedding_result) > 0
    ORDER BY chunk_uid
    """

def _cache_paths(suffix: str) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    return (
        OUT_DIR / f"chunk_embeddings_{suffix}.npy",
        OUT_DIR / f"chunk_ids_{suffix}.npy",
        OUT_DIR / f"chunk_conv_ids_{suffix}.npy",
    )

def pull_chunk_embeddings(
    client: bigquery.Client,
    embed_suffix: str = "",
    skip_embed: bool = False,
    combine: bool = False,
    combine_weight: float = COMBINE_WEIGHT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Pull chunk embeddings → (embeddings, chunk_ids, conv_ids, metadata_df).
    If combine=True, blends with original-text embeddings (requires combined_chunk_embeddings.sql).
    """
    suffix = embed_suffix if embed_suffix else "multilingual"
    table_name = (
        bq_table(f"conversation_chunk_embeddings_{embed_suffix}")
        if embed_suffix and embed_suffix not in ("", "multilingual")
        else SRC_CHUNK_EMBEDDINGS
    )
    emb_path, ids_path, conv_path = _cache_paths(suffix)

    if skip_embed and emb_path.exists() and ids_path.exists():
        log.info("Loading translated chunk embeddings from cache: %s", emb_path)
        embs = np.load(emb_path).astype(np.float32)
        ids  = np.load(ids_path,  allow_pickle=True)
        cids = np.load(conv_path, allow_pickle=True)
        # metadata is the same across variants; base table works even when --embed-suffix finetuned has no BQ table
        meta_table = SRC_CHUNK_EMBEDDINGS if embed_suffix and embed_suffix not in ("", "multilingual") else table_name
        df_meta = _pull_meta(client, str(meta_table))
    else:
        log.info("Pulling translated chunk embeddings from %s …", table_name)
        df = client.query(_emb_query(table_name)).to_dataframe(progress_bar_type="tqdm")
        log.info("  %d chunks, %d conversations.", len(df), df["conversation_id"].nunique())

        embs = _l2_norm(np.asarray(df["embedding"].tolist(), dtype=np.float32))
        ids  = df["chunk_uid"].to_numpy(dtype=object)
        cids = df["conversation_id"].to_numpy(dtype=object)
        df_meta = df.drop(columns=["embedding"])

        np.save(emb_path, embs)
        np.save(ids_path, ids)
        np.save(conv_path, cids)
        log.info("Cached → %s", emb_path)

    if combine:
        embs = _blend_with_original(client, embs, ids, combine_weight, skip_embed)

    return embs, ids, cids, df_meta

def _blend_with_original(
    client: bigquery.Client,
    embs_translated: np.ndarray,
    ids: np.ndarray,
    weight: float,
    skip_embed: bool,
) -> np.ndarray:
    """Blend translated embeddings with original-text embeddings."""
    orig_path = OUT_DIR / "chunk_embeddings_original.npy"
    orig_ids_path = OUT_DIR / "chunk_ids_original.npy"

    if skip_embed and orig_path.exists():
        log.info("Loading original chunk embeddings from cache: %s", orig_path)
        embs_orig = np.load(orig_path).astype(np.float32)
        orig_ids  = np.load(orig_ids_path, allow_pickle=True)
    else:
        log.info("Pulling original chunk embeddings from %s …", SRC_CHUNK_EMBEDDINGS_ORIGINAL)
        df_orig = client.query(
            _original_emb_query(str(SRC_CHUNK_EMBEDDINGS_ORIGINAL))
        ).to_dataframe(progress_bar_type="tqdm")
        log.info("  %d original-text embeddings.", len(df_orig))
        embs_orig = _l2_norm(np.asarray(df_orig["embedding_original"].tolist(), dtype=np.float32))
        orig_ids  = df_orig["chunk_uid"].to_numpy(dtype=object)
        np.save(orig_path, embs_orig)
        np.save(orig_ids_path, orig_ids)

    orig_lookup = dict(zip(orig_ids.tolist(), range(len(orig_ids))))
    aligned_orig = np.zeros_like(embs_translated)
    n_matched = 0
    for i, uid in enumerate(ids.tolist()):
        j = orig_lookup.get(uid)
        if j is not None:
            aligned_orig[i] = embs_orig[j]
            n_matched += 1

    log.info(
        "Combined embeddings: weight=%.2f×translated + %.2f×original  (%d/%d matched).",
        weight, 1 - weight, n_matched, len(ids),
    )
    combined = weight * embs_translated + (1 - weight) * aligned_orig
    return _l2_norm(combined)

def _pull_meta(client: bigquery.Client, table_name: str) -> pd.DataFrame:
    query = f"""
    SELECT
        chunk_uid, conversation_id, chunk_id, detected_language,
        message_start_idx, message_end_idx, message_count, chunk_char_count,
        original_chunk_text,
        -- translated_chunk_text = original_chunk_text (chunk_level_translation.sql not executed)
        content AS translated_chunk_text,
        LENGTH(content) AS content_len
    FROM `{table_name}`
    ORDER BY conversation_id, chunk_id
    """
    return client.query(query).to_dataframe()

def _l2_norm(embs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-10)

def merge_similar_clusters(
    labels: np.ndarray,
    embeddings: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Merge clusters whose centroid cosine similarity exceeds threshold. Noise (-1) unchanged."""
    if threshold <= 0.0:
        return labels

    cluster_ids = sorted(set(labels[labels != -1]))
    if len(cluster_ids) < 2:
        return labels

    centroids: dict[int, np.ndarray] = {}
    for cid in cluster_ids:
        mask = labels == cid
        c = embeddings[mask].mean(axis=0)
        norm = np.linalg.norm(c)
        centroids[cid] = c / max(norm, 1e-10)

    # Union-Find for cluster merging
    parent = {cid: cid for cid in cluster_ids}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        pa, pb = _find(a), _find(b)
        if pa != pb:
            parent[pb] = pa

    n_merges = 0
    for i, ci in enumerate(cluster_ids):
        for cj in cluster_ids[i + 1:]:
            sim = float(np.dot(centroids[ci], centroids[cj]))
            if sim >= threshold:
                _union(ci, cj)
                n_merges += 1

    if n_merges == 0:
        return labels

    remap = {cid: _find(cid) for cid in cluster_ids}
    unique_roots = sorted(set(remap.values()))
    canonical = {root: idx for idx, root in enumerate(unique_roots)}

    new_labels = labels.copy()
    for old, root in remap.items():
        new_labels[labels == old] = canonical[root]

    n_before = len(cluster_ids)
    n_after  = len(unique_roots)
    log.info(
        "Cluster merge (threshold=%.2f): %d → %d clusters (%d merges).",
        threshold, n_before, n_after, n_merges,
    )
    return new_labels

def reduce_and_cluster(
    embeddings: np.ndarray,
    *,
    umap_components: int = UMAP_COMPONENTS,
    umap_neighbors: int = UMAP_NEIGHBORS,
    umap_min_dist: float = UMAP_MIN_DIST,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    selection_method: str = HDBSCAN_SELECTION,
    epsilon: float = HDBSCAN_EPSILON,
    merge_cosine: float = MERGE_COSINE_THRESHOLD,
    seed: int = UMAP_SEED,
    run_tag: str = "default",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """UMAP → HDBSCAN. Returns (labels, probs, reduced_emb)."""
    n, D = embeddings.shape
    log.info("[%s] Input: n=%d, D=%d", run_tag, n, D)

    nc = min(umap_components, D - 1, n - 2)
    log.info("[%s] UMAP %d → %d  (neighbors=%d, min_dist=%.2f) …",
             run_tag, D, nc, umap_neighbors, umap_min_dist)
    reducer = umap.UMAP(
        n_components=nc,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        metric=UMAP_METRIC,
        random_state=seed,
        low_memory=True,
    )
    embedded = reducer.fit_transform(embeddings)
    joblib.dump(reducer, MODELS_DIR / f"umap_{run_tag}.joblib")

    log.info("[%s] HDBSCAN (mcs=%d, ms=%d, selection=%s, eps=%.2f) …",
             run_tag, min_cluster_size, min_samples, selection_method, epsilon)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method=selection_method,
        cluster_selection_epsilon=epsilon,
        prediction_data=True,
    )
    clusterer.fit(embedded)
    labels = clusterer.labels_
    probs  = clusterer.probabilities_
    joblib.dump(clusterer, MODELS_DIR / f"hdbscan_{run_tag}.joblib")

    if merge_cosine > 0.0:
        labels = merge_similar_clusters(labels, embeddings, merge_cosine)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct  = 100 * (labels == -1).sum() / n
    log.info("[%s] → %d clusters, %.1f%% noise", run_tag, n_clusters, noise_pct)

    return labels, probs, embedded

def compute_metrics(
    raw_embs: np.ndarray,
    reduced_embs: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    run_tag: str = "default",
) -> dict:
    """Silhouette, DBCV, DB, CH in raw cosine and UMAP Euclidean space."""
    n = len(labels)
    mask_all  = labels != -1
    mask_conf = (labels != -1) & (probs >= conf_threshold)

    results: dict = {
        "run_tag":             run_tag,
        "n_total":             n,
        "n_clustered":         int(mask_all.sum()),
        "n_high_conf":         int(mask_conf.sum()),
        "n_noise":             int((labels == -1).sum()),
        "noise_pct":           round(100 * (labels == -1).sum() / n, 2),
        "n_clusters":          int(len(set(labels[mask_all]))),
        "confidence_threshold": conf_threshold,
    }

    for space, metric, embs in [("cos", "cosine", raw_embs), ("umap", "euclidean", reduced_embs)]:
        for tag, mask in [("all", mask_all), ("high_conf", mask_conf)]:
            X, y = embs[mask], labels[mask]
            if len(np.unique(y)) >= 2 and len(y) >= 2:
                k = min(SIL_SAMPLE, len(y))
                try:
                    s = silhouette_score(X, y, metric=metric, sample_size=k, random_state=RNG_SEED)
                    results[f"sil_{space}_{tag}"] = round(float(s), 4)
                except Exception as e:
                    log.warning("sil_%s_%s failed: %s", space, tag, e)
                    results[f"sil_{space}_{tag}"] = None
            else:
                results[f"sil_{space}_{tag}"] = None

    X, y = reduced_embs[mask_all], labels[mask_all]
    if len(np.unique(y)) >= 2 and len(y) >= 2:
        try:
            results["davies_bouldin"]    = round(float(davies_bouldin_score(X, y)), 4)
            results["calinski_harabasz"] = round(float(calinski_harabasz_score(X, y)), 2)
        except Exception as e:
            log.warning("DB/CH failed: %s", e)
            results["davies_bouldin"] = results["calinski_harabasz"] = None

        try:
            # DBCV is O(n²) memory — sample down to 15K points
            DBCV_MAX = 15_000
            if len(X) > DBCV_MAX:
                rng = np.random.default_rng(RNG_SEED)
                idx = rng.choice(len(X), size=DBCV_MAX, replace=False)
                Xs, ys = X[idx], y[idx]
            else:
                Xs, ys = X, y
            X64 = np.ascontiguousarray(Xs.astype(np.float64))
            results["dbcv"] = round(float(dbcv_index(X64, ys, metric="euclidean")), 4)
        except Exception as e:
            log.warning("DBCV failed: %s", e)
            results["dbcv"] = None
    else:
        results["davies_bouldin"] = results["calinski_harabasz"] = results["dbcv"] = None

    log.info(
        "[%s] clusters=%d  noise=%.1f%%  "
        "sil_umap_all=%.4f  sil_umap_high_conf=%.4f  "
        "sil_cos_all=%.4f  dbcv=%.4f  DB=%.3f  CH=%.1f",
        run_tag,
        results["n_clusters"],
        results["noise_pct"],
        results.get("sil_umap_all")       or -9,
        results.get("sil_umap_high_conf") or -9,
        results.get("sil_cos_all")        or -9,
        results.get("dbcv")               or -9,
        results.get("davies_bouldin")     or -9,
        results.get("calinski_harabasz")  or -9,
    )
    return results

def assign_conversation_topics(
    chunk_df: pd.DataFrame,
    labels: np.ndarray,
    probs: np.ndarray,
    embeddings: np.ndarray,
    conf_threshold: float = CONFIDENCE_THRESHOLD,
    n_representative: int = N_REPRESENTATIVE_CHUNKS,
) -> pd.DataFrame:
    """Majority vote on high-confidence chunks to assign each conversation a primary topic."""
    df = chunk_df.copy()
    df["cluster_id"]   = labels
    df["hdbscan_prob"] = probs
    df["is_noise"]     = labels == -1
    df["is_high_conf"] = (labels != -1) & (probs >= conf_threshold)

    # representative chunks per cluster (closest to centroid)
    cluster_reps: dict[int, list[str]] = {}
    text_col = "translated_chunk_text" if "translated_chunk_text" in df.columns else "content"
    uid_col  = "chunk_uid" if "chunk_uid" in df.columns else df.index.name or "index"

    for cid in sorted(set(labels[labels != -1])):
        mask = labels == cid
        cluster_embs = embeddings[mask]
        centroid = _l2_norm(cluster_embs.mean(axis=0, keepdims=True))[0]
        sims = cluster_embs @ centroid
        top_k = np.argsort(sims)[::-1][:n_representative]
        chunk_indices = np.where(mask)[0][top_k]
        if uid_col in df.columns:
            reps = df.iloc[chunk_indices][uid_col].tolist()
        else:
            reps = [str(i) for i in chunk_indices]
        cluster_reps[cid] = [str(r) for r in reps]

    records = []
    for conv_id, grp in df.groupby("conversation_id"):
        n_chunks    = len(grp)
        n_noise     = int(grp["is_noise"].sum())
        n_high_conf = int(grp["is_high_conf"].sum())

        if n_high_conf == 0:
            records.append({
                "conversation_id":         conv_id,
                "primary_topic":           -1,
                "secondary_topics":        "",
                "n_chunks":                n_chunks,
                "n_high_conf_chunks":      n_high_conf,
                "n_noise_chunks":          n_noise,
                "topic_confidence":        0.0,
                "is_mixed_topic":          True,
                "is_noise":                n_noise == n_chunks,
                "representative_chunk_uids": "",
            })
            continue

        conf_chunks = grp[grp["is_high_conf"]]
        counts  = conf_chunks["cluster_id"].value_counts()
        primary = int(counts.index[0])
        secondary = [int(c) for c in counts.index[1:4]]
        primary_conf = round(counts.iloc[0] / n_high_conf, 3)

        records.append({
            "conversation_id":    conv_id,
            "primary_topic":      primary,
            "secondary_topics":   ",".join(str(c) for c in secondary),
            "n_chunks":           n_chunks,
            "n_high_conf_chunks": n_high_conf,
            "n_noise_chunks":     n_noise,
            "topic_confidence":   primary_conf,
            "is_mixed_topic":     len(counts) > 1 and counts.iloc[1] >= counts.iloc[0] * 0.5,
            "is_noise":           False,
            "representative_chunk_uids": ",".join(cluster_reps.get(primary, [])),
        })

    return pd.DataFrame(records)

def _bq_write(client: bigquery.Client, df: pd.DataFrame, dest: str, tag: str = "") -> None:
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        autodetect=True,
    )
    log.info("Writing %d rows → %s%s…", len(df), dest, f" [{tag}]" if tag else "")
    client.load_table_from_dataframe(df, dest, job_config=job_config).result()

def run_sweep(
    embeddings: np.ndarray,
    client: bigquery.Client,
    conf_threshold: float,
    merge_cosine: float,
    skip_bq_write: bool = False,
    embed_suffix: str = "",
) -> pd.DataFrame:
    grid = list(itertools.product(
        SWEEP_UMAP_COMPONENTS, SWEEP_MIN_CLUSTER_SIZES,
        SWEEP_MIN_SAMPLES, SWEEP_SELECTIONS,
    ))
    log.info("Sweep: %d configurations.", len(grid))

    # cache UMAP per nc to avoid recomputing it for every HDBSCAN config (main bottleneck)
    umap_cache: dict[int, np.ndarray] = {}

    rows = []
    for nc, mcs, ms, sel in grid:
        tag = f"nc{nc}_mcs{mcs}_ms{ms}_{sel}"
        try:
            if nc not in umap_cache:
                log.info("Sweep: computing UMAP nc=%d …", nc)
                reducer = umap.UMAP(
                    n_components=nc,
                    n_neighbors=UMAP_NEIGHBORS,
                    min_dist=UMAP_MIN_DIST,
                    metric=UMAP_METRIC,
                    random_state=UMAP_SEED,
                    low_memory=True,
                )
                umap_cache[nc] = reducer.fit_transform(embeddings)
                joblib.dump(reducer, MODELS_DIR / f"umap_sweep_nc{nc}.joblib")

            reduced = umap_cache[nc]
            log.info("Sweep: HDBSCAN tag=%s …", tag)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                metric="euclidean",
                cluster_selection_method=sel,
                cluster_selection_epsilon=HDBSCAN_EPSILON,
                prediction_data=True,
            )
            clusterer.fit(reduced)
            labels = clusterer.labels_
            probs  = clusterer.probabilities_
            if merge_cosine > 0.0:
                labels = merge_similar_clusters(labels, embeddings, merge_cosine)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_pct  = 100 * (labels == -1).sum() / len(labels)
            log.info("  → %d clusters, %.1f%% noise", n_clusters, noise_pct)

            m = compute_metrics(embeddings, reduced, labels, probs,
                                conf_threshold=conf_threshold, run_tag=tag)
            rows.append({**m, "umap_components": nc, "min_cluster_size": mcs,
                         "min_samples": ms, "selection_method": sel})
        except Exception as e:
            log.warning("Sweep config %s failed: %s", tag, e)

    sweep_df = pd.DataFrame(rows)
    suffix_tag = f"_{embed_suffix}" if embed_suffix else "_multilingual"
    csv_path = RESULTS_DIR / f"chunk_sweep_results{suffix_tag}.csv"
    sweep_df.to_csv(csv_path, index=False)
    log.info("Sweep results → %s", csv_path)
    if not skip_bq_write:
        _bq_write(client, sweep_df, DEST_SWEEP, tag="sweep")
    return sweep_df

def run(args: argparse.Namespace) -> None:
    client = bigquery.Client(project=PROJECT_ID)

    embs, ids, cids, df_meta = pull_chunk_embeddings(
        client,
        embed_suffix=args.embed_suffix,
        skip_embed=args.skip_embed,
        combine=args.combine_embeddings,
        combine_weight=args.combine_weight,
    )
    log.info("Embeddings shape: %s", embs.shape)

    if args.sweep:
        run_sweep(embs, client, args.confidence_threshold, args.merge_cosine,
                  skip_bq_write=args.skip_bq_write, embed_suffix=args.embed_suffix)
        return

    run_tag = f"nc{args.umap_components}_mcs{args.min_cluster_size}_ms{args.min_samples}"

    labels, probs, reduced = reduce_and_cluster(
        embs,
        umap_components=args.umap_components,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        selection_method=args.selection_method,
        epsilon=args.epsilon,
        merge_cosine=args.merge_cosine,
        run_tag=run_tag,
    )

    metrics = compute_metrics(embs, reduced, labels, probs,
                              conf_threshold=args.confidence_threshold,
                              run_tag=run_tag)

    chunk_labels_df = df_meta.copy()
    if "chunk_uid" not in chunk_labels_df.columns:
        chunk_labels_df["chunk_uid"] = ids
    chunk_labels_df["cluster_id"]   = labels
    chunk_labels_df["hdbscan_prob"] = probs
    chunk_labels_df["is_noise"]     = labels == -1
    chunk_labels_df["is_high_conf"] = (labels != -1) & (probs >= args.confidence_threshold)
    chunk_labels_df["umap_x"]       = reduced[:, 0]
    chunk_labels_df["umap_y"]       = reduced[:, 1] if reduced.shape[1] > 1 else 0.0

    topic_df = assign_conversation_topics(
        df_meta, labels, probs, embs,
        conf_threshold=args.confidence_threshold,
    )

    metrics_df = pd.DataFrame([{
        **metrics,
        "embed_suffix":       args.embed_suffix or "multilingual",
        "combine_embeddings": args.combine_embeddings,
        "combine_weight":     args.combine_weight if args.combine_embeddings else None,
        "merge_cosine":       args.merge_cosine,
        "umap_components":    args.umap_components,
        "umap_neighbors":     args.umap_neighbors,
        "min_cluster_size":   args.min_cluster_size,
        "min_samples":        args.min_samples,
        "selection_method":   args.selection_method,
        "confidence_threshold": args.confidence_threshold,
        "computed_at":        pd.Timestamp.now().isoformat(),
    }])

    chunk_labels_df.to_csv(RESULTS_DIR / "chunk_cluster_labels.csv", index=False)
    topic_df.to_csv(       RESULTS_DIR / "conversation_topic_assignments.csv", index=False)
    metrics_df.to_csv(     RESULTS_DIR / "chunk_evaluation_metrics.csv", index=False)
    log.info(
        "CSVs saved:\n  chunk_cluster_labels.csv\n"
        "  conversation_topic_assignments.csv\n  chunk_evaluation_metrics.csv"
    )

    mixed    = int(topic_df["is_mixed_topic"].sum())
    no_topic = int((topic_df["primary_topic"] == -1).sum())
    log.info(
        "Topic summary: %d conversations | %d assigned | %d mixed | %d unassigned",
        len(topic_df), int((topic_df["primary_topic"] != -1).sum()), mixed, no_topic,
    )

    if not args.skip_bq_write:
        _bq_write(client, chunk_labels_df, DEST_CHUNK_LABELS)
        _bq_write(client, topic_df,        DEST_TOPIC_ASSIGNMENTS)
        _bq_write(client, metrics_df,      DEST_METRICS)
    else:
        log.info("--skip-bq-write: BigQuery writes skipped.")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Chunk-level UMAP + HDBSCAN clustering for topic assignment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--embed-suffix", default="",
                   help="Embedding table suffix: '' → conversation_chunk_embeddings, "
                        "'gemini' → conversation_chunk_embeddings_gemini, "
                        "'finetuned' → conversation_chunk_embeddings_finetuned.")
    p.add_argument("--combine-embeddings", action="store_true",
                   help="Blend translated + original embeddings: "
                        "final = combine_weight × translated + (1-w) × original. "
                        "Requires future_work/combined_chunk_embeddings.sql to be run first.")
    p.add_argument("--combine-weight", type=float, default=COMBINE_WEIGHT,
                   help="Weight for translated embeddings when --combine-embeddings is set "
                        "(default: 0.7 → 70%% translated, 30%% original).")
    p.add_argument("--merge-cosine", type=float, default=MERGE_COSINE_THRESHOLD,
                   help="Merge clusters whose centroid cosine similarity exceeds this value. "
                        "0 = disabled (default). Recommended: 0.85.")
    p.add_argument("--skip-embed",   action="store_true",
                   help="Reuse cached .npy embedding files.")
    p.add_argument("--umap-components", type=int,   default=UMAP_COMPONENTS)
    p.add_argument("--umap-neighbors",  type=int,   default=UMAP_NEIGHBORS)
    p.add_argument("--umap-min-dist",   type=float, default=UMAP_MIN_DIST)
    p.add_argument("--min-cluster-size",type=int,   default=HDBSCAN_MIN_CLUSTER_SIZE)
    p.add_argument("--min-samples",     type=int,   default=HDBSCAN_MIN_SAMPLES)
    p.add_argument("--selection-method",default=HDBSCAN_SELECTION, choices=["eom", "leaf"])
    p.add_argument("--epsilon",         type=float, default=HDBSCAN_EPSILON)
    p.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD,
                   help="HDBSCAN membership probability threshold for 'high-confidence' metrics "
                        "(silhouette high_conf, conversation topic vote). Does not change HDBSCAN "
                        "noise labels (-1); noise is determined only by HDBSCAN.")
    p.add_argument("--sweep",          action="store_true",
                   help="Grid search over UMAP + HDBSCAN parameters.")
    p.add_argument("--skip-bq-write",  action="store_true",
                   help="Do not write results to BigQuery.")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
