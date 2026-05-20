"""
BERTopic on pre-computed embeddings: UMAP + HDBSCAN + c-TF-IDF. Optional noise reassignment and hierarchical merging.

Usage:
  python scripts/experiments/bertopic_cluster.py --quick
  python scripts/experiments/bertopic_cluster.py --sweep
  python scripts/experiments/bertopic_cluster.py --nc 5 --mcs 50 --save-bq
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.deep_clustering._common import (   # noqa: E402
    DATA_DIR,
    EXPERIMENTS_DIR,
    RNG,
    SIL_SAMPLE,
    sil_cos,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_CSV   = EXPERIMENTS_DIR / "bertopic_results.csv"
PROJECT_ID    = "support-analytics-492410"
DATASET_ID    = "support_analytics"

# nc  = UMAP n_components (latent dims before HDBSCAN)
# mcs = HDBSCAN min_cluster_size
# ms  = HDBSCAN min_samples
# nr  = BERTopic nr_topics; "auto" = hierarchical agglomerative merging
SWEEP_NC  = [5, 8, 10]
SWEEP_MCS = [50, 75, 100, 150]
SWEEP_MS  = [5, 10]
SWEEP_NR  = ["auto"]

QUICK_NC  = 5
QUICK_MCS = 50
QUICK_MS  = 5


def _bq_pull_text(embed_suffix: str | None = None) -> pd.DataFrame:
    """Pull (conversation_id, content) from BigQuery conversation_docs."""
    from google.cloud import bigquery  # noqa: E402

    client = bigquery.Client(project=PROJECT_ID)

    # embed_suffix accepted for API compatibility; all ablation variants removed, multilingual only
    content_col = "conversation_text_multilingual"

    query = f"""
        SELECT
            conversation_id,
            {content_col} AS content
        FROM `{PROJECT_ID}.{DATASET_ID}.conversation_docs`
        WHERE conversation_text_multilingual IS NOT NULL
          AND TRIM(conversation_text_multilingual) != ''
        ORDER BY conversation_id
    """
    log.info("Pulling text from BigQuery …")
    df = client.query(query).to_dataframe()
    log.info("  Pulled %d rows", len(df))
    return df


def _bq_pull_embeddings(embed_suffix: str | None, embedding: str) -> tuple[np.ndarray, np.ndarray]:
    """Pull (ids, embeddings): .npy cache first, then BigQuery. Returns (str array, float32 (n, D))."""
    from google.cloud import bigquery  # noqa: E402

    suffix = f"_{embed_suffix}" if embed_suffix else ""
    model_suffix = "_gemini" if embedding == "gemini" else ""
    npy_emb  = DATA_DIR / f"embeddings_multilingual{model_suffix}{suffix}.npy"
    npy_ids  = DATA_DIR / f"ids_multilingual{model_suffix}{suffix}.npy"

    if npy_emb.exists() and npy_ids.exists():
        log.info("Loading embeddings from cache: %s", npy_emb)
        embs = np.load(npy_emb).astype(np.float32)
        ids  = np.load(npy_ids, allow_pickle=True)
        log.info("  shape=%s", embs.shape)
        return ids, embs

    # Embeddings cache exists but IDs cache doesn't (older pipeline version)
    if npy_emb.exists() and not npy_ids.exists():
        log.warning(
            "Embeddings cache found but IDs cache missing (%s). "
            "Will pull IDs from BigQuery and align by row order.", npy_ids
        )

    if embedding == "gemini":
        base_table = f"{PROJECT_ID}.{DATASET_ID}.conversation_embeddings_multilingual_gemini"
    else:
        base_table = f"{PROJECT_ID}.{DATASET_ID}.conversation_embeddings_multilingual"

    if embed_suffix:
        table = f"{base_table}_{embed_suffix}"
    else:
        table = base_table

    log.info("Pulling embeddings from BigQuery table: %s", table)
    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
        SELECT
            conversation_id,
            ml_generate_embedding_result AS embedding
        FROM `{table}`
        WHERE ARRAY_LENGTH(ml_generate_embedding_result) > 0
        ORDER BY conversation_id
    """
    df = client.query(query).to_dataframe()
    log.info("  Downloaded %d rows, D=%d", len(df), len(df["embedding"].iloc[0]))

    ids  = df["conversation_id"].values
    embs = np.asarray(df["embedding"].tolist(), dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs  = embs / np.maximum(norms, 1e-10)

    np.save(npy_emb, embs)
    np.save(npy_ids, ids)
    log.info("  Cached embeddings → %s", npy_emb)
    log.info("  Cached IDs        → %s", npy_ids)
    return ids, embs


def run_bertopic(
    docs: list[str],
    ids: np.ndarray,
    embeddings: np.ndarray,
    nc: int,
    mcs: int,
    ms: int,
    nr_topics: str | int,
    outlier_strategy: str = "embeddings",
    save_bq: bool = False,
    embed_suffix: str | None = None,
    embedding_name: str = "multilingual",
) -> dict:
    """Run one BERTopic config; returns metrics dict with k, noise_pct, sil_cos, and outlier stats."""
    import hdbscan as hdbscan_lib
    from bertopic import BERTopic
    from umap import UMAP

    config_str = f"BERTopic_nc{nc}_mcs{mcs}_ms{ms}_nr{nr_topics}_{embedding_name}"
    if embed_suffix:
        config_str += f"_{embed_suffix}"

    log.info("=" * 60)
    log.info("BERTopic run: %s", config_str)
    log.info("  n_docs=%d  D=%d", len(docs), embeddings.shape[1])

    umap_model = UMAP(
        n_components=nc,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=RNG,
        low_memory=True,
    )

    hdbscan_model = hdbscan_lib.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,   # required for soft membership / reduce_outliers
    )

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=None,   # we pass pre-computed embeddings
        nr_topics=nr_topics,
        calculate_probabilities=True,
        verbose=True,
    )

    t0 = time.time()
    topics_raw, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    elapsed_fit = time.time() - t0
    log.info("  fit_transform done in %.1fs", elapsed_fit)

    labels_raw = np.array(topics_raw)
    n_total    = len(labels_raw)
    n_noise    = int((labels_raw == -1).sum())
    k_before   = int(labels_raw[labels_raw != -1].max()) + 1 if n_noise < n_total else 0
    noise_before = round(100 * n_noise / n_total, 1)

    log.info("  BEFORE reduce_outliers: k=%d  noise=%.1f%%", k_before, noise_before)

    sc_before = sil_cos(embeddings, labels_raw)
    log.info("  sil_cos BEFORE = %s", sc_before)

    log.info("  Applying reduce_outliers(strategy=%s) …", outlier_strategy)
    try:
        if outlier_strategy == "embeddings":
            topics_reduced = topic_model.reduce_outliers(
                docs, topics_raw, strategy="embeddings", embeddings=embeddings,
                threshold=0.0,
            )
        elif outlier_strategy == "c-tf-idf":
            topics_reduced = topic_model.reduce_outliers(
                docs, topics_raw, strategy="c-tf-idf",
                threshold=0.0,
            )
        elif outlier_strategy == "distributions":
            topics_reduced = topic_model.reduce_outliers(
                docs, topics_raw, probabilities=probs, strategy="distributions",
                threshold=0.0,
            )
        else:
            topics_reduced = topics_raw
    except Exception as exc:
        log.warning("reduce_outliers failed: %s — using raw topics", exc)
        topics_reduced = topics_raw

    labels_reduced = np.array(topics_reduced)
    n_noise_after  = int((labels_reduced == -1).sum())
    k_after = (
        int(labels_reduced[labels_reduced != -1].max()) + 1
        if n_noise_after < n_total else 0
    )
    noise_after = round(100 * n_noise_after / n_total, 1)

    log.info("  AFTER  reduce_outliers: k=%d  noise=%.1f%%", k_after, noise_after)

    sc_after = sil_cos(embeddings, labels_reduced)
    log.info("  sil_cos AFTER  = %s", sc_after)

    topic_info = topic_model.get_topic_info()
    top_topics = topic_info[topic_info["Topic"] != -1].head(10)
    log.info("\n  Top 10 topics:\n%s", top_topics[["Topic", "Count", "Name"]].to_string(index=False))

    labels_path = EXPERIMENTS_DIR / f"bertopic_labels_{config_str}.csv"
    df_labels = pd.DataFrame({
        "conversation_id": ids,
        "topic_raw":       labels_raw,
        "topic_reduced":   labels_reduced,
        "config":          config_str,
    })
    df_labels.to_csv(labels_path, index=False)
    log.info("  Cluster labels → %s", labels_path)

    if save_bq:
        _save_labels_bq(df_labels, config_str)

    result = {
        "method":           config_str,
        "nc":               nc,
        "mcs":              mcs,
        "ms":               ms,
        "nr_topics":        str(nr_topics),
        "k_before":         k_before,
        "k_after":          k_after,
        "noise_pct_before": noise_before,
        "noise_pct_after":  noise_after,
        "sil_cos_before":   sc_before,
        "sil_cos_after":    sc_after,
        "outlier_strategy": outlier_strategy,
        "embedding":        embedding_name,
        "embed_suffix":     embed_suffix or "full_conv",
        "elapsed_fit_s":    round(elapsed_fit, 1),
        "computed_at":      datetime.now().isoformat(),
    }

    _save_result(result)
    return result


def _save_result(row: dict) -> None:
    df_new = pd.DataFrame([row])
    if RESULTS_CSV.exists():
        df_old = pd.read_csv(RESULTS_CSV)
        keep   = df_old["method"] != row["method"]
        df_out = pd.concat([df_old[keep], df_new], ignore_index=True)
    else:
        df_out = df_new
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(RESULTS_CSV, index=False)
    log.info(
        "Saved → %s  k_after=%s  noise_after=%.1f%%  sil_cos_after=%s",
        RESULTS_CSV, row["k_after"], row["noise_pct_after"], row["sil_cos_after"],
    )


def _save_labels_bq(df: pd.DataFrame, config_str: str) -> None:
    try:
        from google.cloud import bigquery  # noqa: E402
        client = bigquery.Client(project=PROJECT_ID)
        table  = f"{PROJECT_ID}.{DATASET_ID}.bertopic_labels"
        df_bq  = df.copy()
        df_bq["config"] = config_str
        job    = client.load_table_from_dataframe(
            df_bq,
            table,
            job_config=bigquery.LoadJobConfig(
                write_disposition="WRITE_TRUNCATE",
                autodetect=True,
            ),
        )
        job.result()
        log.info("  Labels written to BigQuery: %s", table)
    except Exception as exc:
        log.warning("BQ write failed: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BERTopic clustering using pre-computed embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sweep", action="store_true",
                        help="Run full parameter sweep (nc × mcs × ms grid).")
    parser.add_argument("--quick", action="store_true",
                        help=f"Single quick run: nc={QUICK_NC} mcs={QUICK_MCS} ms={QUICK_MS}.")
    parser.add_argument("--nc",  type=int, default=5, metavar="N",
                        help="UMAP n_components (default 5).")
    parser.add_argument("--mcs", type=int, default=50, metavar="N",
                        help="HDBSCAN min_cluster_size (default 50).")
    parser.add_argument("--ms",  type=int, default=5, metavar="N",
                        help="HDBSCAN min_samples (default 5).")
    parser.add_argument("--nr-topics", default="auto",
                        help="BERTopic nr_topics: 'auto' or integer (default auto).")
    parser.add_argument("--outlier-strategy", default="embeddings",
                        choices=["embeddings", "c-tf-idf", "distributions"],
                        help="reduce_outliers strategy (default: embeddings).")
    parser.add_argument("--embedding", default="multilingual",
                        choices=["multilingual", "gemini"],
                        help="Which embedding model to use (default: multilingual).")
    parser.add_argument("--embed-suffix", default=None, metavar="SUFFIX",
                        help="Accepted for API compatibility; currently unused (all embedding "
                             "ablation variants were removed). Pass gemini to load Gemini "
                             "embeddings from the _gemini BQ table if available.")
    parser.add_argument("--save-bq", action="store_true",
                        help="Write cluster labels to BigQuery (bertopic_labels table).")
    args = parser.parse_args()

    try:
        import bertopic  # noqa: F401
    except ImportError as _e:
        sys.exit(
            f"\nBERTopic import failed: {_e}\n\n"
            "Fix for PyTorch 1.13 environments:\n"
            "  pip install 'transformers==4.35.2' 'sentence-transformers==2.3.1' bertopic --quiet\n\n"
            "Or strip sentence-transformers entirely:\n"
            "  pip install 'transformers==4.35.2' --quiet\n"
            "  pip install bertopic --no-deps --quiet\n"
            "  pip install hdbscan umap-learn scikit-learn numpy pandas tqdm --quiet\n"
        )

    ids, embeddings = _bq_pull_embeddings(args.embed_suffix, args.embedding)
    df_text         = _bq_pull_text(args.embed_suffix)

    df_emb = pd.DataFrame({"conversation_id": ids})
    df_merged = df_emb.merge(df_text, on="conversation_id", how="left")
    df_merged["content"] = df_merged["content"].fillna("")
    docs = df_merged["content"].tolist()

    log.info("Aligned %d docs with embeddings", len(docs))
    assert len(docs) == len(embeddings), (
        f"Doc count ({len(docs)}) != embedding count ({len(embeddings)})"
    )

    nr_topics = args.nr_topics if args.nr_topics == "auto" else int(args.nr_topics)

    if args.quick:
        run_bertopic(
            docs=docs, ids=ids, embeddings=embeddings,
            nc=QUICK_NC, mcs=QUICK_MCS, ms=QUICK_MS, nr_topics="auto",
            outlier_strategy=args.outlier_strategy,
            save_bq=args.save_bq,
            embed_suffix=args.embed_suffix,
            embedding_name=args.embedding,
        )
        return

    if args.sweep:
        log.info(
            "Starting sweep: nc=%s  mcs=%s  ms=%s  nr=%s",
            SWEEP_NC, SWEEP_MCS, SWEEP_MS, SWEEP_NR,
        )
        results = []
        for nc, mcs, ms, nr in product(SWEEP_NC, SWEEP_MCS, SWEEP_MS, SWEEP_NR):
            try:
                r = run_bertopic(
                    docs=docs, ids=ids, embeddings=embeddings,
                    nc=nc, mcs=mcs, ms=ms, nr_topics=nr,
                    outlier_strategy=args.outlier_strategy,
                    save_bq=False,
                    embed_suffix=args.embed_suffix,
                    embedding_name=args.embedding,
                )
                results.append(r)
            except Exception as exc:
                log.error("Config nc=%d mcs=%d ms=%d failed: %s", nc, mcs, ms, exc)

        log.info("\n\n=== SWEEP SUMMARY ===")
        df_res = pd.DataFrame(results)
        df_res = df_res.sort_values("sil_cos_after", ascending=False)
        cols   = ["nc", "mcs", "ms", "k_after", "noise_pct_after", "sil_cos_after",
                  "sil_cos_before", "noise_pct_before"]
        log.info("\n%s", df_res[cols].to_string(index=False))
        log.info("\nBest config: %s", df_res.iloc[0]["method"])
        return

    run_bertopic(
        docs=docs, ids=ids, embeddings=embeddings,
        nc=args.nc, mcs=args.mcs, ms=args.ms, nr_topics=nr_topics,
        outlier_strategy=args.outlier_strategy,
        save_bq=args.save_bq,
        embed_suffix=args.embed_suffix,
        embedding_name=args.embedding,
    )


if __name__ == "__main__":
    main()
