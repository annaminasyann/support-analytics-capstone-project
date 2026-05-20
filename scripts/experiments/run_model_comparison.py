"""
SentenceTransformer model comparison across both pipeline tracks.

Compares MPNet (768-D), MiniLM (384-D), DistilUSE (512-D), LaBSE (768-D).
For each model × track: UMAP 20-D → HDBSCAN + K-Means → Silhouette / DB / CH / DBCV / ARI.

Usage:
    python scripts/experiments/run_model_comparison.py
    python scripts/experiments/run_model_comparison.py --track a
    python scripts/experiments/run_model_comparison.py --skip-embed
"""

import argparse
import logging
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import hdbscan
from hdbscan import validity_index as dbcv_index
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR         = REPO_ROOT / "data"
EXPERIMENTS_DIR = REPO_ROOT / "data" / "experiments"
OUT_DIR.mkdir(exist_ok=True)
EXPERIMENTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))
from gcp_config import PROJECT_ID, bq_table  

BQ_SOURCE_TABLE  = bq_table("conversation_docs")
BQ_RESULTS_TABLE = bq_table("model_comparison_results")

MODELS = [
    {
        "name": "MPNet-multilingual",
        "model_id": "paraphrase-multilingual-mpnet-base-v2",
        "short": "mpnet",
        "dims": 768,
        "is_primary": True,
        # Baseline SBERT model (Reimers & Gurevych 2020).
    },
    {
        "name": "MiniLM-multilingual",
        "model_id": "paraphrase-multilingual-MiniLM-L12-v2",
        "short": "minilm",
        "dims": 384,
        "is_primary": False,
    },
    {
        "name": "DistilUSE-multilingual",
        "model_id": "distiluse-base-multilingual-cased-v2",
        "short": "distiluse",
        "dims": 512,
        "is_primary": False,
    },
    {
        "name": "LaBSE",
        "model_id": "sentence-transformers/LaBSE",
        "short": "labse",
        "dims": 768,
        "is_primary": False,
        # Google's Language-agnostic BERT Sentence Encoder.
    },
    # E5-multilingual-large (intfloat/multilingual-e5-large-instruct, 1024D) is excluded
    # The four models above cover the full range of architectures and embedding sizes.
]

TRACK_COLUMNS = {
    "a": "conversation_text_multilingual",
    "b": "conversation_text_translated",
}
TRACK_LABELS = {
    "a": "multilingual",
    "b": "translated",
}

UMAP_N_COMPONENTS = 15
UMAP_N_NEIGHBORS  = 30   # matched to embed_and_cluster.py
UMAP_MIN_DIST     = 0.0
RANDOM_STATE      = 42

HDBSCAN_MIN_SAMPLES = 10  # matched to embed_and_cluster.py
HDBSCAN_METRIC      = "euclidean"

def bq_client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID)


def load_texts(track_key: str, client: bigquery.Client) -> tuple[pd.DataFrame, list[str]]:
    """Pull conversation texts from BigQuery for the given track."""
    text_col = TRACK_COLUMNS[track_key]
    cache_path = OUT_DIR / f"texts_{TRACK_LABELS[track_key]}.csv"

    if cache_path.exists():
        log.info("[Track %s] Loading cached texts from %s", track_key.upper(), cache_path)
        df = pd.read_csv(cache_path)
    else:
        log.info("[Track %s] Pulling texts from BigQuery: %s", track_key.upper(), BQ_SOURCE_TABLE)
        query = f"""
            SELECT
                conversation_id,
                {text_col} AS text,
                detected_language,
                client_message_count,
                rating_score
            FROM `{BQ_SOURCE_TABLE}`
            WHERE {text_col} IS NOT NULL
              AND TRIM({text_col}) != ''
            ORDER BY conversation_id
        """
        df = client.query(query).to_dataframe(progress_bar_type="tqdm")
        df.to_csv(cache_path, index=False)
        log.info("[Track %s] Downloaded %d rows, cached to %s", track_key.upper(), len(df), cache_path)

    texts = df["text"].fillna("").tolist()
    mask = [bool(t.strip()) for t in texts]
    df    = df[mask].reset_index(drop=True)
    texts = [t for t, m in zip(texts, mask) if m]
    log.info("[Track %s] Usable texts: %d", track_key.upper(), len(texts))
    return df, texts


def embed(model_id: str, short: str, texts: list[str], skip: bool,
          encode_kwargs: dict | None = None) -> tuple[np.ndarray, float]:
    cache_file = OUT_DIR / f"embeddings_{short}.npy"
    if skip and cache_file.exists():
        log.info("[%s] Loading cached embeddings.", short)
        return np.load(cache_file), 0.0

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("[%s] Encoding %d texts on %s ...", short, len(texts), device.upper())
    model = SentenceTransformer(model_id, device=device)
    t0 = time.perf_counter()
    batch_size = 256 if device == "cuda" else 64
    kwargs = dict(batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    if encode_kwargs:
        kwargs.update(encode_kwargs)
    embs = model.encode(texts, **kwargs)
    encode_sec = time.perf_counter() - t0
    np.save(cache_file, embs)
    log.info("[%s] Encoded in %.1f sec (%.1f sec/1000 texts)", short, encode_sec, 1000 * encode_sec / len(texts))
    return embs, encode_sec


def reduce_umap(embs: np.ndarray, short: str, skip: bool) -> np.ndarray:
    cache_file = OUT_DIR / f"umap15d_{short}.npy"
    if skip and cache_file.exists():
        log.info("[%s] Loading cached UMAP-15D.", short)
        return np.load(cache_file)

    log.info("[%s] UMAP reduction to %dD ...", short, UMAP_N_COMPONENTS)
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    umap_15d = reducer.fit_transform(embs)
    np.save(cache_file, umap_15d)
    return umap_15d


def run_hdbscan(umap_15d: np.ndarray, min_cluster_size: int) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric=HDBSCAN_METRIC,
        cluster_selection_method="eom",
        prediction_data=True,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(umap_15d)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct  = (labels == -1).mean() * 100
    log.info("HDBSCAN → %d clusters | noise=%.1f%%", n_clusters, noise_pct)
    return labels


def run_kmeans(umap_15d: np.ndarray, k: int) -> np.ndarray:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(umap_15d)
    log.info("K-Means (k=%d) → done", k)
    return labels


def compute_metrics(X: np.ndarray, labels: np.ndarray, method: str, model_short: str,
                    primary_labels: np.ndarray | None = None,
                    noise_pct: float = 0.0,
                    encode_sec: float = 0.0,
                    n_texts: int = 0,
                    dims: int = 0) -> dict:
    mask = labels != -1
    X_clean   = X[mask]
    lbl_clean = labels[mask]

    n_clusters = len(set(lbl_clean))
    result = {
        "model": model_short, "method": method, "dims": dims,
        "n_clusters": n_clusters, "noise_pct": round(noise_pct, 1),
        "encoding_time_s": round(encode_sec, 1),
        "encode_sec_per_1k": round(1000 * encode_sec / max(n_texts, 1), 1),
    }

    if n_clusters < 2:
        result.update({"silhouette": np.nan, "davies_bouldin": np.nan,
                        "calinski_harabasz": np.nan, "dbcv": np.nan,
                        "ari_vs_primary": np.nan, "nmi_vs_primary": np.nan})
        return result

    result["silhouette"]        = round(silhouette_score(X_clean, lbl_clean, sample_size=2000, random_state=RANDOM_STATE), 4)
    result["davies_bouldin"]    = round(davies_bouldin_score(X_clean, lbl_clean), 4)
    result["calinski_harabasz"] = round(calinski_harabasz_score(X_clean, lbl_clean), 2)

    if "HDBSCAN" in method and n_clusters >= 2:
        try:
            result["dbcv"] = round(dbcv_index(X_clean, lbl_clean), 4)
        except Exception:
            result["dbcv"] = np.nan
    else:
        result["dbcv"] = np.nan

    if primary_labels is not None:
        both_valid = (labels != -1) & (primary_labels != -1)
        if both_valid.sum() > 10:
            result["ari_vs_primary"] = round(
                adjusted_rand_score(primary_labels[both_valid], labels[both_valid]), 4
            )
            result["nmi_vs_primary"] = round(
                normalized_mutual_info_score(primary_labels[both_valid], labels[both_valid]), 4
            )
        else:
            result["ari_vs_primary"] = np.nan
            result["nmi_vs_primary"] = np.nan
    else:
        result["ari_vs_primary"] = np.nan
        result["nmi_vs_primary"] = np.nan

    return result


def plot_comparison(df_results: pd.DataFrame, out_path: Path):
    metrics = ["silhouette", "davies_bouldin", "calinski_harabasz", "dbcv", "ari_vs_primary", "encode_sec_per_1k"]
    titles  = ["Silhouette ↑", "Davies-Bouldin ↓", "Calinski-Harabasz ↑",
               "DBCV ↑\n(density-based)", "ARI vs Primary ↑", "Encoding Latency\n(sec / 1k texts)"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Multi-Model Clustering Comparison\n(Capstone — Anna Minasyan)", fontsize=14, fontweight="bold")

    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        sub = df_results.dropna(subset=[metric])
        if sub.empty:
            ax.set_visible(False)
            continue
        sns.barplot(data=sub, x="model", y=metric, hue="method", ax=ax, palette="Set2")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        ax.legend(title="Method", fontsize=7)
        ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    log.info("Comparison plot saved: %s", out_path)


def write_results_to_bq(results_df: pd.DataFrame, client: bigquery.Client) -> None:
    results_df["computed_at"] = pd.Timestamp.utcnow()

    tracks = results_df["pipeline_track"].unique().tolist()
    for track in tracks:
        try:
            client.query(
                f"DELETE FROM `{BQ_RESULTS_TABLE}` WHERE pipeline_track = '{track}'"
            ).result()
            log.info("Cleared previous results for track '%s' from %s.", track, BQ_RESULTS_TABLE)
        except Exception:
            log.info("Table %s not found yet — will be created on first write.", BQ_RESULTS_TABLE)

    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        autodetect=True,
    )
    log.info("Writing comparison results → %s", BQ_RESULTS_TABLE)
    job = client.load_table_from_dataframe(results_df, BQ_RESULTS_TABLE, job_config=job_config)
    job.result()
    log.info("Results written to BigQuery.")


def run_track(track_key: str, args, client: bigquery.Client) -> list[dict]:
    track_name = TRACK_LABELS[track_key]
    log.info("TRACK %s — %s", track_key.upper(), track_name.upper())

    df, texts = load_texts(track_key, client)
    n = len(texts)

    primary_labels = None
    primary_labels_path = EXPERIMENTS_DIR / f"cluster_labels_{track_name}.csv"
    if primary_labels_path.exists():
        prim_df = pd.read_csv(primary_labels_path)
        if "conversation_id" in prim_df.columns and "conversation_id" in df.columns:
            merged = df[["conversation_id"]].merge(
                prim_df[["conversation_id", "cluster_id"]],
                on="conversation_id",
                how="left",
            )
            primary_labels = merged["cluster_id"].fillna(-1).values.astype(int)
            log.info("Primary labels loaded via conversation_id join → k=%d",
                     len(set(primary_labels)) - (1 if -1 in primary_labels else 0))
        elif len(prim_df) == n:
            # Fallback: positional alignment (assumes identical sort order)
            primary_labels = prim_df["cluster_id"].values
            log.warning(
                "Primary labels aligned positionally (no conversation_id column). "
                "ARI may be incorrect if row order differs."
            )
        if primary_labels is not None:
            primary_k = len(set(primary_labels)) - (1 if -1 in primary_labels else 0)
            log.info("Primary labels: k=%d", primary_k)

    k_for_kmeans = max(2, len(set(primary_labels)) - (1 if -1 in primary_labels else 0)) \
        if primary_labels is not None else 15
    log.info("K-Means k = %d", k_for_kmeans)

    all_results = []

    for model_cfg in MODELS:
        short    = model_cfg["short"]
        model_id = model_cfg["model_id"]
        is_prim  = model_cfg["is_primary"]
        dims     = model_cfg["dims"]

        log.info("--- Model: %s (%dD) ---", model_cfg["name"], dims)

        embs, encode_sec = embed(model_id, f"{short}_{track_name}", texts,
                                   skip=args.skip_embed,
                                   encode_kwargs=model_cfg.get("encode_kwargs"))
        umap_15d = reduce_umap(embs, f"{short}_{track_name}", skip=args.skip_embed)

        hdb_labels = run_hdbscan(umap_15d, args.min_cluster_size)
        np.save(OUT_DIR / f"labels_hdbscan_{short}_{track_name}.npy", hdb_labels)
        # Also save without track suffix so notebooks can find it by short name alone.
        np.save(OUT_DIR / f"labels_hdbscan_{short}.npy", hdb_labels)
        noise_pct = (hdb_labels == -1).mean() * 100
        result_hdb = compute_metrics(
            umap_15d, hdb_labels, "HDBSCAN", short,
            primary_labels=(None if is_prim else primary_labels),
            noise_pct=noise_pct, encode_sec=encode_sec, n_texts=n, dims=dims,
        )
        result_hdb["pipeline_track"] = track_name
        all_results.append(result_hdb)

        km_labels = run_kmeans(umap_15d, k_for_kmeans)
        np.save(OUT_DIR / f"labels_kmeans_{short}_{track_name}.npy", km_labels)
        result_km = compute_metrics(
            umap_15d, km_labels, f"K-Means(k={k_for_kmeans})", short,
            primary_labels=(None if is_prim else primary_labels),
            noise_pct=0.0, encode_sec=encode_sec, n_texts=n, dims=dims,
        )
        result_km["pipeline_track"] = track_name
        all_results.append(result_km)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Multi-model clustering comparison")
    parser.add_argument("--track", choices=["a", "b", "both"], default="both")
    parser.add_argument("--min-cluster-size", type=int, default=50)
    parser.add_argument("--skip-embed", action="store_true")
    args = parser.parse_args()

    client = bq_client()
    tracks_to_run = ["a", "b"] if args.track == "both" else [args.track]

    all_results = []
    for track_key in tracks_to_run:
        all_results.extend(run_track(track_key, args, client))

    results_df = pd.DataFrame(all_results)
    out_csv = EXPERIMENTS_DIR / "model_comparison_results.csv"
    results_df.to_csv(out_csv, index=False)
    log.info("Results saved locally: %s", out_csv)

    print("\n" + "="*60)
    print("MULTI-MODEL COMPARISON RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))

    plot_comparison(results_df, EXPERIMENTS_DIR / "model_comparison_plot.png")
    write_results_to_bq(results_df, client)

    log.info("Done. Outputs in %s and BigQuery: %s", OUT_DIR, BQ_RESULTS_TABLE)


if __name__ == "__main__":
    main()
