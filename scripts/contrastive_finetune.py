"""
Contrastive fine-tuning of a SentenceTransformer on cluster-derived pairs.

Uses chunk cluster labels from chunk_clustering.py to build positive/negative pairs,
then trains with MultipleNegativesRankingLoss (MNRL).

Pipeline:
  chunk_clustering.py → contrastive_finetune.py → chunk_clustering.py --embed-suffix finetuned

Output:
  data/models/finetuned_embedder/  — saved model
  data/finetune_pairs.csv          — training pairs (QA only)
  data/results/chunk_cluster_labels.csv  — read as input

Usage:
    python contrastive_finetune.py
    python contrastive_finetune.py --epochs 3 --batch-size 64
    python contrastive_finetune.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import random
import sys
from typing import NamedTuple

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

OUT_DIR     = REPO_ROOT / "data"
RESULTS_DIR = OUT_DIR / "results"
MODELS_DIR  = OUT_DIR / "models"
FINETUNED_DIR = MODELS_DIR / "finetuned_embedder"

CHUNK_LABELS_CSV = RESULTS_DIR / "chunk_cluster_labels.csv"

BASE_MODEL_DEFAULT = "paraphrase-multilingual-MiniLM-L12-v2"
POSITIVES_PER_CHUNK = 50
HARD_NEGATIVES_PER_CHUNK = 10
EASY_NEGATIVES_PER_CHUNK = 10
MIN_CLUSTER_PROB = 0.70       # only use chunks above this confidence for pairing
MIN_CLUSTER_SIZE = 10         # skip clusters too small to generate reliable pairs
EPOCHS = 3
BATCH_SIZE = 32
WARMUP_RATIO = 0.1
MAX_SEQ_LENGTH = 256          # truncate chunks to this many tokens
RNG_SEED = 42

class TrainingPair(NamedTuple):
    anchor:   str
    positive: str
    # negatives are implicit in MNRL (batch) or explicit for CosineSimilarityLoss
    label: int  # 1 = positive, 0 = negative (used for CosineSimilarityLoss only)

def build_pairs(
    df: pd.DataFrame,
    positives_per_chunk: int,
    hard_negatives_per_chunk: int,
    easy_negatives_per_chunk: int,
    min_cluster_prob: float,
    min_cluster_size: int,
    rng: random.Random,
    embeddings: np.ndarray | None = None,
) -> tuple[list[TrainingPair], list[TrainingPair]]:
    """Build positive and hard/easy negative pairs from high-confidence chunks."""
    high_conf = df[
        (~df["is_noise"]) & (df["hdbscan_prob"] >= min_cluster_prob)
    ].copy()

    cluster_sizes = high_conf["cluster_id"].value_counts()
    valid_clusters = cluster_sizes[cluster_sizes >= min_cluster_size].index
    high_conf = high_conf[high_conf["cluster_id"].isin(valid_clusters)]

    log.info(
        "Pair generation pool: %d chunks across %d clusters "
        "(filtered from %d total).",
        len(high_conf), high_conf["cluster_id"].nunique(), len(df),
    )

    if len(high_conf) < 2:
        raise ValueError("Not enough high-confidence chunks to build pairs. "
                         "Lower --min-cluster-prob or run chunk_clustering.py first.")

    by_cluster: dict[int, list[int]] = {
        int(cid): list(grp.index)
        for cid, grp in high_conf.groupby("cluster_id")
    }
    cluster_ids = list(by_cluster.keys())

    hard_neg_map: dict[int, list[int]] | None = None
    if embeddings is not None and embeddings.shape[0] == len(df):
        log.info("Computing hard negatives via nearest-neighbor search…")
        hard_neg_map = _compute_hard_negatives(
            df=high_conf,
            embeddings=embeddings[high_conf.index],
            k=hard_negatives_per_chunk + 5,
        )

    pos_pairs: list[TrainingPair] = []
    neg_pairs: list[TrainingPair] = []
    # translated_chunk_text = original_chunk_text (chunk_level_translation.sql not executed)
    text_col = "translated_chunk_text"

    for cid, indices in by_cluster.items():
        if len(indices) < 2:
            continue
        for anchor_idx in rng.sample(indices, min(positives_per_chunk * 2, len(indices))):
            anchor_text = df.at[anchor_idx, text_col]

            pos_pool = [i for i in indices if i != anchor_idx]
            for pos_idx in rng.sample(pos_pool, min(positives_per_chunk, len(pos_pool))):
                pos_pairs.append(TrainingPair(anchor_text, df.at[pos_idx, text_col], 1))

            if hard_neg_map and anchor_idx in hard_neg_map:
                for neg_idx in hard_neg_map[anchor_idx][:hard_negatives_per_chunk]:
                    neg_text = df.at[neg_idx, text_col]
                    neg_pairs.append(TrainingPair(anchor_text, neg_text, 0))

            other_clusters = [c for c in cluster_ids if c != cid]
            for _ in range(easy_negatives_per_chunk):
                other_cid = rng.choice(other_clusters)
                neg_idx   = rng.choice(by_cluster[other_cid])
                neg_text  = df.at[neg_idx, text_col]
                neg_pairs.append(TrainingPair(anchor_text, neg_text, 0))

    log.info("Generated %d positive pairs and %d negative pairs.",
             len(pos_pairs), len(neg_pairs))
    return pos_pairs, neg_pairs

def _compute_hard_negatives(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    k: int = 15,
) -> dict[int, list[int]]:
    """k nearest neighbors from a different cluster. Embeddings must be L2-normalised."""
    # batched — full N×N similarity matrix exceeds RAM for >50k chunks
    BATCH = 2000
    n = len(df)
    emb = embeddings.astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-10)

    idx_array = df.index.to_numpy()
    cid_array = df["cluster_id"].to_numpy()

    hard_neg_map: dict[int, list[int]] = {}

    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        batch_emb = emb[start:end]
        sims = batch_emb @ emb.T
        for local_i, global_i in enumerate(range(start, end)):
            anchor_cid = cid_array[global_i]
            row_sims = sims[local_i]
            mask_diff = cid_array != anchor_cid
            diff_idxs = np.where(mask_diff)[0]
            if len(diff_idxs) == 0:
                continue
            sorted_diff = diff_idxs[np.argsort(row_sims[diff_idxs])[::-1]][:k]
            hard_neg_map[int(idx_array[global_i])] = [int(idx_array[j]) for j in sorted_diff]

    return hard_neg_map

def finetune(
    pos_pairs: list[TrainingPair],
    neg_pairs: list[TrainingPair],
    base_model: str,
    epochs: int,
    batch_size: int,
    output_dir: pathlib.Path,
    max_seq_length: int,
    warmup_ratio: float,
    rng_seed: int,
) -> None:
    """Fine-tune a SentenceTransformer using MultipleNegativesRankingLoss."""
    try:
        from sentence_transformers import (
            SentenceTransformer,
            InputExample,
            losses,
        )
        from torch.utils.data import DataLoader
    except ImportError as e:
        raise ImportError(
            "sentence-transformers and torch are required for fine-tuning.\n"
            "Install with: pip install sentence-transformers torch"
        ) from e

    log.info("Loading base model: %s", base_model)
    model = SentenceTransformer(base_model)
    model.max_seq_length = max_seq_length

    # MNRL only needs (anchor, positive); the batch provides implicit negatives
    train_examples = [
        InputExample(texts=[p.anchor, p.positive])
        for p in pos_pairs
    ]

    rng_local = random.Random(rng_seed)
    rng_local.shuffle(train_examples)

    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=batch_size,
    )
    loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)
    log.info(
        "Training: %d anchor–positive examples | %d epochs | batch=%d | warmup=%d steps",
        len(train_examples), epochs, batch_size, warmup_steps,
    )
    log.info("Loss = MNRL (in-batch negatives). Hard/easy pairs in finetune_pairs.csv are for QA only.")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        show_progress_bar=True,
    )
    log.info("Fine-tuned model saved → %s", output_dir)

def embed_chunks_with_finetuned(
    df: pd.DataFrame,
    model_dir: pathlib.Path,
    batch_size: int = 64,
) -> np.ndarray:
    """Re-encode all chunks with the fine-tuned model. Returns L2-normalised (n, D) array."""
    from sentence_transformers import SentenceTransformer

    log.info("Loading fine-tuned model from %s …", model_dir)
    model = SentenceTransformer(str(model_dir))
    # translated_chunk_text = original_chunk_text (chunk_level_translation.sql not executed)
    texts = df["translated_chunk_text"].fillna("").tolist()
    log.info("Encoding %d chunks …", len(texts))
    embs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embs.astype(np.float32)

def run(args: argparse.Namespace) -> None:
    if not CHUNK_LABELS_CSV.exists():
        raise FileNotFoundError(
            f"Chunk labels CSV not found: {CHUNK_LABELS_CSV}\n"
            "Run chunk_clustering.py first to generate chunk_cluster_labels.csv."
        )

    log.info("Loading chunk labels from %s …", CHUNK_LABELS_CSV)
    df = pd.read_csv(CHUNK_LABELS_CSV)
    log.info("  %d chunks loaded.", len(df))

    emb_path = pathlib.Path(args.pair_embedding_npy).expanduser()
    embeddings: np.ndarray | None = None
    if emb_path.exists() and not args.no_hard_negatives:
        log.info("Loading embeddings for hard-negative mining: %s", emb_path)
        embeddings = np.load(emb_path).astype(np.float32)
        if embeddings.shape[0] != len(df):
            log.warning(
                "Embedding shape %s does not match df length %d — "
                "skipping hard negatives.",
                embeddings.shape, len(df),
            )
            embeddings = None
    elif args.no_hard_negatives:
        log.info("--no-hard-negatives: skipping hard-negative mining.")

    rng = random.Random(RNG_SEED)
    pos_pairs, neg_pairs = build_pairs(
        df,
        positives_per_chunk=args.positives_per_chunk,
        hard_negatives_per_chunk=args.hard_negatives if not args.no_hard_negatives else 0,
        easy_negatives_per_chunk=args.easy_negatives,
        min_cluster_prob=args.min_cluster_prob,
        min_cluster_size=args.min_cluster_size,
        rng=rng,
        embeddings=embeddings,
    )

    pairs_df = pd.DataFrame(
        [{"anchor": p.anchor, "positive": p.positive, "label": p.label}
         for p in pos_pairs + neg_pairs]
    )
    pairs_csv = OUT_DIR / "finetune_pairs.csv"
    pairs_df.to_csv(pairs_csv, index=False)
    log.info("Training pairs saved → %s  (%d rows)", pairs_csv, len(pairs_df))

    if args.dry_run:
        log.info("--dry-run: skipping training.")
        print(f"\nPair stats:\n  Positive pairs: {len(pos_pairs)}\n"
              f"  Negative pairs: {len(neg_pairs)}\n"
              f"  Total: {len(pos_pairs) + len(neg_pairs)}")
        return

    if args.skip_train:
        if not FINETUNED_DIR.exists():
            raise FileNotFoundError(
                f"No fine-tuned model found at {FINETUNED_DIR}. "
                "Run without --skip-train first."
            )
        log.info("--skip-train: reusing existing model at %s", FINETUNED_DIR)
    else:
        finetune(
            pos_pairs=pos_pairs,
            neg_pairs=neg_pairs,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            output_dir=FINETUNED_DIR,
            max_seq_length=args.max_seq_length,
            warmup_ratio=WARMUP_RATIO,
            rng_seed=RNG_SEED,
        )

    if args.re_embed:
        log.info("Re-encoding all chunks with fine-tuned model…")
        new_embs = embed_chunks_with_finetuned(df, FINETUNED_DIR, batch_size=args.batch_size)
        finetuned_npy = OUT_DIR / "chunk_embeddings_finetuned.npy"
        np.save(finetuned_npy, new_embs)
        # save IDs so chunk_clustering.py --embed-suffix finetuned --skip-embed can find the cache
        np.save(OUT_DIR / "chunk_ids_finetuned.npy",      df["chunk_uid"].to_numpy(dtype=object))
        np.save(OUT_DIR / "chunk_conv_ids_finetuned.npy", df["conversation_id"].to_numpy(dtype=object))
        log.info("Fine-tuned embeddings saved → %s  shape=%s", finetuned_npy, new_embs.shape)
        log.info(
            "\nNext step: run chunk clustering with fine-tuned embeddings:\n"
            "  python scripts/chunk_clustering.py --embed-suffix finetuned --skip-embed"
        )
    else:
        log.info(
            "\nTo re-embed chunks with the fine-tuned model:\n"
            "  python scripts/contrastive_finetune.py --re-embed\n"
            "Then re-cluster:\n"
            "  python scripts/chunk_clustering.py --embed-suffix finetuned --skip-embed"
        )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Contrastive fine-tuning of a SentenceTransformer on cluster-derived pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-model", default=BASE_MODEL_DEFAULT,
                   help="Hugging Face model name for the base SentenceTransformer.")
    p.add_argument("--epochs",     type=int,   default=EPOCHS)
    p.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    p.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    p.add_argument("--positives-per-chunk", type=int, default=POSITIVES_PER_CHUNK,
                   help="Number of positive pairs to generate per anchor chunk.")
    p.add_argument("--hard-negatives", type=int, default=HARD_NEGATIVES_PER_CHUNK,
                   help="Hard negatives per anchor (requires cached embeddings).")
    p.add_argument("--easy-negatives", type=int, default=EASY_NEGATIVES_PER_CHUNK)
    p.add_argument("--no-hard-negatives", action="store_true",
                   help="Skip hard-negative mining (faster but weaker training signal).")
    p.add_argument(
        "--pair-embedding-npy",
        default=str(OUT_DIR / "chunk_embeddings_multilingual.npy"),
        help="Numpy cache of chunk embeddings used ONLY for hard-negative mining "
             "(must have one row per chunk, same order as chunk_cluster_labels.csv). "
             "After clustering with --embed-suffix gemini, point this to "
             "data/chunk_embeddings_gemini.npy (or copy that file over the default path).",
    )
    p.add_argument("--min-cluster-prob", type=float, default=MIN_CLUSTER_PROB,
                   help="Only use chunks with HDBSCAN probability >= this for pairing.")
    p.add_argument("--min-cluster-size", type=int, default=MIN_CLUSTER_SIZE,
                   help="Skip clusters smaller than this for pairing.")
    p.add_argument("--re-embed", action="store_true",
                   help="Re-encode all chunks with the fine-tuned model after training.")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training and go straight to --re-embed using the existing model.")
    p.add_argument("--dry-run",  action="store_true",
                   help="Generate pairs and print stats but do not train.")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
