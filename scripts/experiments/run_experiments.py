"""
Hypothesis-driven HDBSCAN parameter sweep (G1–G11).

Tests eleven groups covering leaf vs eom selection, min_cluster_size ranges,
merge-cosine post-processing, and soft-assignment thresholds. Does not write
to BigQuery. Results go to data/results/experiment_grid_<track>.csv.

Usage:
  python scripts/experiments/run_experiments.py --track a --skip-embed
  python scripts/experiments/run_experiments.py --track both --skip-embed
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.embed_and_cluster import (  # noqa: E402
    pull_embeddings,
    reduce_umap,
    run_hdbscan,
    merge_similar_clusters,
    assign_noise_to_nearest,
    _silhouette_raw,
    _dbcv_safe,
    bq_client,
    UMAP_MIN_DIST_CLUSTER,
    UMAP_N_NEIGHBORS,
    SIL_SAMPLE_SIZE,
    RNG_SEED,
    TRACKS,
)
from sklearn.metrics import silhouette_score  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "results"

# Quality thresholds
THRESH = {
    "dbcv":    (0.30, 0.50),   # (acceptable, strong)
    "sil_u":   (0.25, 0.50),
    "noise":   (50.0, 40.0),   # (acceptable, good) — lower is better
    "persist": (0.10, 0.12),
}

def _flag(metric: str, val: float) -> str:
    lo, hi = THRESH[metric]
    if metric == "noise":
        if val < hi:   return "✓"
        if val < lo:   return "~"
        return "✗"
    else:
        if val > hi:   return "✓✓"
        if val > lo:   return "✓"
        return "✗"

def _composite(row: dict) -> float:
    """Weighted composite — same formula as find_best_params.py."""
    sil_cos  = row.get("sil_cos", 0.0) or 0.0
    sil_u    = row.get("sil_umap", 0.0) or 0.0
    dbcv     = row.get("dbcv", float("nan"))
    pers     = row.get("persist", 0.0) or 0.0
    noise    = row.get("noise_pct", 50.0) or 50.0

    dbcv_norm   = ((dbcv + 1) / 2) if not np.isnan(dbcv) else 0.0
    noise_bonus = max(0.0, (50.0 - noise) / 50.0)

    return (0.20 * sil_cos
            + 0.25 * sil_u
            + 0.30 * dbcv_norm
            + 0.20 * pers
            + 0.05 * noise_bonus)

# each dict: nc, nn (n_neighbors), sel, mcs, ms, merge (cosine threshold or None)
EXPERIMENTS = [
    # G1: leaf + high mcs — can noise drop below 40%?
    # From focused sweep: leaf mcs=100/ms=25 → DBCV=0.530, noise=42.8% (just over threshold)
    {"group": "G1", "nc": 20, "nn": 30, "sel": "leaf", "mcs": 125, "ms": 20,  "merge": None},
    {"group": "G1", "nc": 20, "nn": 30, "sel": "leaf", "mcs": 125, "ms": 25,  "merge": None},
    {"group": "G1", "nc": 20, "nn": 30, "sel": "leaf", "mcs": 125, "ms": 30,  "merge": None},
    {"group": "G1", "nc": 20, "nn": 30, "sel": "leaf", "mcs": 150, "ms": 15,  "merge": None},
    {"group": "G1", "nc": 20, "nn": 30, "sel": "leaf", "mcs": 150, "ms": 20,  "merge": None},
    {"group": "G1", "nc": 20, "nn": 30, "sel": "leaf", "mcs": 150, "ms": 25,  "merge": None},
    {"group": "G1", "nc": 20, "nn": 30, "sel": "leaf", "mcs": 200, "ms": 15,  "merge": None},
    {"group": "G1", "nc": 20, "nn": 30, "sel": "leaf", "mcs": 200, "ms": 20,  "merge": None},

    # G2: merge-cosine after eom — does merging raise DBCV?
    # Base eom gives DBCV≈0.45, noise≈34%. Merging similar clusters might tighten density.
    {"group": "G2", "nc": 20, "nn": 30, "sel": "eom",  "mcs":  75, "ms": 20,  "merge": 0.95},
    {"group": "G2", "nc": 20, "nn": 30, "sel": "eom",  "mcs":  75, "ms": 20,  "merge": 0.92},
    {"group": "G2", "nc": 20, "nn": 30, "sel": "eom",  "mcs":  75, "ms": 20,  "merge": 0.90},
    {"group": "G2", "nc": 20, "nn": 30, "sel": "eom",  "mcs":  75, "ms": 20,  "merge": 0.85},
    {"group": "G2", "nc": 20, "nn": 30, "sel": "eom",  "mcs": 150, "ms": 15,  "merge": 0.92},
    {"group": "G2", "nc": 20, "nn": 30, "sel": "eom",  "mcs": 150, "ms": 15,  "merge": 0.90},
    {"group": "G2", "nc": 20, "nn": 30, "sel": "eom",  "mcs": 150, "ms": 15,  "merge": 0.85},

    # G3: merge-cosine after leaf — keep high DBCV, reduce k further
    # Leaf starts with DBCV≈0.58 and k≈140. Merging reduces k while preserving density.
    {"group": "G3", "nc": 20, "nn": 30, "sel": "leaf", "mcs":  75, "ms": 20,  "merge": 0.95},
    {"group": "G3", "nc": 20, "nn": 30, "sel": "leaf", "mcs":  75, "ms": 20,  "merge": 0.92},
    {"group": "G3", "nc": 20, "nn": 30, "sel": "leaf", "mcs":  75, "ms": 20,  "merge": 0.90},
    {"group": "G3", "nc": 20, "nn": 30, "sel": "leaf", "mcs":  75, "ms": 20,  "merge": 0.85},
    {"group": "G3", "nc": 20, "nn": 30, "sel": "leaf", "mcs": 100, "ms": 20,  "merge": 0.92},
    {"group": "G3", "nc": 20, "nn": 30, "sel": "leaf", "mcs": 100, "ms": 20,  "merge": 0.90},
    {"group": "G3", "nc": 20, "nn": 30, "sel": "leaf", "mcs": 100, "ms": 20,  "merge": 0.85},
    {"group": "G3", "nc": 20, "nn": 30, "sel": "leaf", "mcs": 150, "ms": 15,  "merge": 0.90},
    {"group": "G3", "nc": 20, "nn": 30, "sel": "leaf", "mcs": 150, "ms": 15,  "merge": 0.85},

    # G4: higher nc — does more UMAP geometry raise eom DBCV above 0.50?
    {"group": "G4", "nc": 25, "nn": 30, "sel": "eom",  "mcs":  75, "ms": 20,  "merge": None},
    {"group": "G4", "nc": 25, "nn": 30, "sel": "leaf", "mcs":  75, "ms": 20,  "merge": None},
    {"group": "G4", "nc": 25, "nn": 30, "sel": "leaf", "mcs": 100, "ms": 20,  "merge": None},
    {"group": "G4", "nc": 30, "nn": 30, "sel": "eom",  "mcs":  75, "ms": 20,  "merge": None},
    {"group": "G4", "nc": 30, "nn": 30, "sel": "leaf", "mcs":  75, "ms": 20,  "merge": None},
    {"group": "G4", "nc": 30, "nn": 30, "sel": "leaf", "mcs": 100, "ms": 20,  "merge": None},

    # G5: n_neighbors — finer vs broader local structure
    {"group": "G5", "nc": 20, "nn": 15, "sel": "eom",  "mcs":  75, "ms": 20,  "merge": None},
    {"group": "G5", "nc": 20, "nn": 15, "sel": "leaf", "mcs":  75, "ms": 20,  "merge": None},
    {"group": "G5", "nc": 20, "nn": 50, "sel": "eom",  "mcs":  75, "ms": 20,  "merge": None},
    {"group": "G5", "nc": 20, "nn": 50, "sel": "leaf", "mcs":  75, "ms": 20,  "merge": None},

    # G6: known-good baselines (anchors)
    {"group": "G6", "nc": 20, "nn": 30, "sel": "eom",  "mcs":  75, "ms": 20,  "merge": None},  # best eom
    {"group": "G6", "nc": 20, "nn": 30, "sel": "leaf", "mcs":  75, "ms": 20,  "merge": None},  # best leaf
    {"group": "G6", "nc": 20, "nn": 30, "sel": "eom",  "mcs": 150, "ms": 15,  "merge": None},  # canonical

    # G7: nn=15 + higher mcs (ROUND 2)
    # nn=15 gave DBCV=0.527 (strong!) at mcs=75/k=125. Goal: keep strong DBCV while
    # pushing k down to ~60-80 by raising mcs. nn=15 UMAP cache already exists.
    {"group": "G7", "nc": 20, "nn": 15, "sel": "eom",  "mcs": 100, "ms": 15,  "merge": None},
    {"group": "G7", "nc": 20, "nn": 15, "sel": "eom",  "mcs": 100, "ms": 20,  "merge": None},
    {"group": "G7", "nc": 20, "nn": 15, "sel": "eom",  "mcs": 125, "ms": 15,  "merge": None},
    {"group": "G7", "nc": 20, "nn": 15, "sel": "eom",  "mcs": 125, "ms": 20,  "merge": None},
    {"group": "G7", "nc": 20, "nn": 15, "sel": "eom",  "mcs": 150, "ms": 15,  "merge": None},
    {"group": "G7", "nc": 20, "nn": 15, "sel": "eom",  "mcs": 150, "ms": 20,  "merge": None},
    {"group": "G7", "nc": 20, "nn": 15, "sel": "eom",  "mcs": 200, "ms": 15,  "merge": None},
    {"group": "G7", "nc": 20, "nn": 15, "sel": "eom",  "mcs": 200, "ms": 20,  "merge": None},

    # G8: nc=25 + leaf + higher mcs (ROUND 2)
    # nc=25/leaf/mcs=100 gave DBCV=0.538 (strong!), noise=42.9% (just over 40%).
    # Raising mcs forces HDBSCAN to only pick denser peaks → noise might finally drop.
    {"group": "G8", "nc": 25, "nn": 30, "sel": "leaf", "mcs": 125, "ms": 15,  "merge": None},
    {"group": "G8", "nc": 25, "nn": 30, "sel": "leaf", "mcs": 125, "ms": 20,  "merge": None},
    {"group": "G8", "nc": 25, "nn": 30, "sel": "leaf", "mcs": 150, "ms": 15,  "merge": None},
    {"group": "G8", "nc": 25, "nn": 30, "sel": "leaf", "mcs": 150, "ms": 20,  "merge": None},
    {"group": "G8", "nc": 25, "nn": 30, "sel": "leaf", "mcs": 200, "ms": 15,  "merge": None},

    # G9: nn=15 + leaf (unexplored combination)
    # nn=15 dramatically improved eom DBCV. Does it also help leaf while reducing noise?
    {"group": "G9", "nc": 20, "nn": 15, "sel": "leaf", "mcs": 100, "ms": 20,  "merge": None},
    {"group": "G9", "nc": 20, "nn": 15, "sel": "leaf", "mcs": 125, "ms": 20,  "merge": None},
    {"group": "G9", "nc": 20, "nn": 15, "sel": "leaf", "mcs": 150, "ms": 15,  "merge": None},
    {"group": "G9", "nc": 20, "nn": 15, "sel": "leaf", "mcs": 150, "ms": 20,  "merge": None},
    {"group": "G9", "nc": 20, "nn": 15, "sel": "leaf", "mcs": 200, "ms": 15,  "merge": None},

    # G10: nn=50 + higher mcs (ROUND 2, Track B focus)
    # Track B's best nn is 50 (DBCV=0.540✓✓ at mcs=75/k=106). Goal: raise mcs to
    # push k down to ~60-80 while preserving strong DBCV. nn=50 UMAP cache exists.
    {"group": "G10", "nc": 20, "nn": 50, "sel": "eom",  "mcs": 100, "ms": 15,  "merge": None},
    {"group": "G10", "nc": 20, "nn": 50, "sel": "eom",  "mcs": 100, "ms": 20,  "merge": None},
    {"group": "G10", "nc": 20, "nn": 50, "sel": "eom",  "mcs": 125, "ms": 15,  "merge": None},
    {"group": "G10", "nc": 20, "nn": 50, "sel": "eom",  "mcs": 125, "ms": 20,  "merge": None},
    {"group": "G10", "nc": 20, "nn": 50, "sel": "eom",  "mcs": 150, "ms": 15,  "merge": None},
    {"group": "G10", "nc": 20, "nn": 50, "sel": "eom",  "mcs": 150, "ms": 20,  "merge": None},
    {"group": "G10", "nc": 20, "nn": 50, "sel": "eom",  "mcs": 200, "ms": 15,  "merge": None},
    {"group": "G10", "nc": 20, "nn": 50, "sel": "eom",  "mcs": 200, "ms": 20,  "merge": None},

    # G11: very high mcs — target k=20–40 (lower granularity)
    # Feedback: k=70+ may be too fine-grained; fewer coarser clusters could be
    # more interpretable. Test mcs=250–500 with the three best nn values.
    {"group": "G11", "nc": 20, "nn": 30, "sel": "eom",  "mcs": 250, "ms": 20,  "merge": None},
    {"group": "G11", "nc": 20, "nn": 30, "sel": "eom",  "mcs": 300, "ms": 25,  "merge": None},
    {"group": "G11", "nc": 20, "nn": 30, "sel": "eom",  "mcs": 400, "ms": 25,  "merge": None},
    {"group": "G11", "nc": 20, "nn": 30, "sel": "eom",  "mcs": 500, "ms": 30,  "merge": None},
    {"group": "G11", "nc": 20, "nn": 15, "sel": "eom",  "mcs": 250, "ms": 20,  "merge": None},
    {"group": "G11", "nc": 20, "nn": 15, "sel": "eom",  "mcs": 300, "ms": 25,  "merge": None},
    {"group": "G11", "nc": 20, "nn": 15, "sel": "eom",  "mcs": 400, "ms": 25,  "merge": None},
    {"group": "G11", "nc": 20, "nn": 50, "sel": "eom",  "mcs": 250, "ms": 20,  "merge": None},
    {"group": "G11", "nc": 20, "nn": 50, "sel": "eom",  "mcs": 300, "ms": 25,  "merge": None},
    {"group": "G11", "nc": 20, "nn": 50, "sel": "eom",  "mcs": 400, "ms": 25,  "merge": None},
]

SOFT_ASSIGN_BASE_CONFIGS = [
    {"label": "canonical (nn=30/eom/mcs=150/ms=15)",  "nc": 20, "nn": 30, "sel": "eom",  "mcs": 150, "ms": 15},
    {"label": "G7 best (nn=15/eom/mcs=150/ms=15)",    "nc": 20, "nn": 15, "sel": "eom",  "mcs": 150, "ms": 15},
    {"label": "G9 near-miss (nn=15/leaf/mcs=150/ms=20)", "nc": 20, "nn": 15, "sel": "leaf", "mcs": 150, "ms": 20},
    {"label": "G10 winner (nn=50/eom/mcs=100/ms=20)", "nc": 20, "nn": 50, "sel": "eom",  "mcs": 100, "ms": 20},
]

# None = no soft assignment (hard clustering baseline)
SOFT_ASSIGN_THRESHOLDS = [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

def run_one(exp: dict, embeddings: np.ndarray, track_name: str) -> dict:
    """Run a single experiment config and return a metrics dict."""
    nc      = exp["nc"]
    nn      = exp["nn"]
    sel     = exp["sel"]
    mcs     = exp["mcs"]
    ms      = exp["ms"]
    merge   = exp.get("merge")

    t0 = time.time()

    nn_arg = nn if nn != UMAP_N_NEIGHBORS else None
    umap_nd, _ = reduce_umap(
        embeddings, nc, UMAP_MIN_DIST_CLUSTER,
        track_name, f"{nc}d",
        n_neighbors=nn_arg,
    )

    labels, _, clusterer = run_hdbscan(umap_nd, mcs, ms, epsilon=0.0,
                                       selection_method=sel)

    valid     = labels != -1
    n_clust   = int(len(np.unique(labels[valid])))
    n_noise   = int((~valid).sum())
    noise_pct = round(100.0 * n_noise / len(labels), 1)

    n_before_merge = n_clust
    if merge is not None and n_clust >= 2:
        labels, n_before_merge, n_clust = merge_similar_clusters(
            labels, embeddings, cosine_threshold=merge
        )
        valid = labels != -1

    sil_cos, _ = _silhouette_raw(embeddings, labels)
    sil_umap = (
        float(silhouette_score(
            umap_nd[valid], labels[valid], metric="euclidean",
            sample_size=min(SIL_SAMPLE_SIZE, int(valid.sum())),
            random_state=RNG_SEED,
        ))
        if n_clust >= 2 else None
    )
    dbcv = _dbcv_safe(umap_nd, labels)
    persts = np.asarray(clusterer.cluster_persistence_, dtype=float)
    persist = float(persts.mean()) if len(persts) > 0 else None

    elapsed = round(time.time() - t0, 1)

    row = {
        "group":       exp["group"],
        "nc":          nc,
        "nn":          nn,
        "sel":         sel,
        "mcs":         mcs,
        "ms":          ms,
        "merge":       merge,
        "k":           n_clust,
        "k_pre_merge": n_before_merge if merge is not None else n_clust,
        "noise_pct":   noise_pct,
        "sil_cos":     round(sil_cos, 4) if sil_cos is not None else None,
        "sil_umap":    round(sil_umap, 4) if sil_umap is not None else None,
        "dbcv":        round(dbcv, 4) if dbcv is not None else None,
        "persist":     round(persist, 4) if persist is not None else None,
        "elapsed_s":   elapsed,
    }
    row["score"] = round(_composite(row), 4)
    return row

def print_table(df: pd.DataFrame, track_label: str) -> None:
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    print(f"\n{'='*130}")
    print(f"  {track_label}  —  sorted by composite score")
    print(f"  (sil_cos×0.20 + sil_umap×0.25 + dbcv_norm×0.30 + persist×0.20 + noise_bonus×0.05)")
    print(f"{'='*130}")
    hdr = (f"  {'grp':>3}  {'nc':>3}  {'nn':>3}  {'sel':>4}  {'mcs':>4}  "
           f"{'ms':>3}  {'merge':>5}  {'k':>5}  {'noise':>6}  "
           f"{'sil_cos':>8}  {'sil_umap':>8}  {'dbcv':>7}  {'persist':>8}  "
           f"{'score':>7}  quality")
    print(hdr)
    print(f"  {'-'*124}")

    for _, r in df.iterrows():
        merge_s = f"{r['merge']:.2f}" if r['merge'] is not None else "  —  "
        dbcv_v  = r['dbcv']    if r['dbcv']    is not None else float("nan")
        silu_v  = r['sil_umap'] if r['sil_umap'] is not None else float("nan")
        pers_v  = r['persist']  if r['persist']  is not None else float("nan")

        flags = []
        if not np.isnan(dbcv_v):  flags.append("DBCV" + _flag("dbcv",    dbcv_v))
        if not np.isnan(silu_v):  flags.append("sil"  + _flag("sil_u",   silu_v))
        flags.append("noise" + _flag("noise", r['noise_pct']))
        if not np.isnan(pers_v):  flags.append("pers"  + _flag("persist", pers_v))

        print(
            f"  {r['group']:>3}  {int(r['nc']):>3}  {int(r['nn']):>3}  "
            f"{r['sel']:>4}  {int(r['mcs']):>4}  {int(r['ms']):>3}  "
            f"{merge_s:>5}  {int(r['k']):>5}  {r['noise_pct']:>5.1f}%  "
            f"{r['sil_cos']:>8.4f}  {silu_v:>8.4f}  {dbcv_v:>7.4f}  "
            f"{pers_v:>8.4f}  {r['score']:>7.4f}  "
            + "  ".join(flags)
        )

    best = df.iloc[0]
    print(f"\n  ── Best config ──────────────────────────────────────────────")
    print(f"  group={best['group']}  nc={int(best['nc'])}  nn={int(best['nn'])}  "
          f"sel={best['sel']}  mcs={int(best['mcs'])}  ms={int(best['ms'])}  "
          f"merge={best['merge']}  →  k={int(best['k'])}, "
          f"noise={best['noise_pct']:.1f}%, sil_umap={best['sil_umap']:.4f}, "
          f"dbcv={best['dbcv']:.4f}, persist={best['persist']:.4f}, "
          f"score={best['score']:.4f}")

    print(f"\n  ── Best per group ───────────────────────────────────────────")
    for grp, gdf in df.groupby("group"):
        r = gdf.iloc[0]
        print(f"  {grp}: nc={int(r['nc'])} nn={int(r['nn'])} {r['sel']} "
              f"mcs={int(r['mcs'])} ms={int(r['ms'])} merge={r['merge']} "
              f"→ k={int(r['k'])}, noise={r['noise_pct']:.1f}%, "
              f"DBCV={r['dbcv']:.4f}, sil_umap={r['sil_umap']:.4f}, "
              f"persist={r['persist']:.4f}  score={r['score']:.4f}")

def _metrics_for_labels(
    embeddings: np.ndarray,
    umap_nd: np.ndarray,
    labels: np.ndarray,
    clusterer,
) -> dict:
    """Compute sil_cos, sil_umap, dbcv, persist, k, noise_pct for a label array."""
    valid = labels != -1
    n_clust = int(len(np.unique(labels[valid]))) if valid.any() else 0
    n_noise = int((~valid).sum())
    noise_pct = round(100.0 * n_noise / len(labels), 1)

    sil_cos, _ = _silhouette_raw(embeddings, labels)
    sil_umap = (
        float(silhouette_score(
            umap_nd[valid], labels[valid], metric="euclidean",
            sample_size=min(SIL_SAMPLE_SIZE, int(valid.sum())),
            random_state=RNG_SEED,
        ))
        if n_clust >= 2 and valid.sum() >= 2 else None
    )
    dbcv = _dbcv_safe(umap_nd, labels)
    persts = np.asarray(clusterer.cluster_persistence_, dtype=float)
    persist = float(persts.mean()) if len(persts) > 0 else None

    return {
        "k":         n_clust,
        "noise_pct": noise_pct,
        "sil_cos":   round(sil_cos, 4) if sil_cos is not None else None,
        "sil_umap":  round(sil_umap, 4) if sil_umap is not None else None,
        "dbcv":      round(dbcv, 4) if dbcv is not None else None,
        "persist":   round(persist, 4) if persist is not None else None,
    }

def run_soft_assign_sweep(track_key: str, skip_embed: bool) -> pd.DataFrame:
    """Sweep soft-assignment thresholds over base configs. eps=0 only; outputs soft_assign_sweep_<track>.csv."""
    cfg = TRACKS[track_key]
    name = cfg["name"]
    label = cfg["label"]
    client = bq_client()

    print(f"\n{'#'*70}")
    print(f"  SOFT-ASSIGNMENT SWEEP — {label}")
    print(f"{'#'*70}")
    pulled = pull_embeddings(track_key, client, skip_embed=skip_embed)
    embeddings = pulled.embeddings

    rows = []
    for base_cfg in SOFT_ASSIGN_BASE_CONFIGS:
        nc  = base_cfg["nc"]
        nn  = base_cfg["nn"]
        sel = base_cfg["sel"]
        mcs = base_cfg["mcs"]
        ms  = base_cfg["ms"]
        cfg_label = base_cfg["label"]

        print(f"\n── {cfg_label} ──────────────────────────")

        nn_arg = nn if nn != UMAP_N_NEIGHBORS else None
        umap_nd, _ = reduce_umap(
            embeddings, nc, UMAP_MIN_DIST_CLUSTER,
            name, f"{nc}d",
            n_neighbors=nn_arg,
        )
        labels_orig, membership_orig, clusterer = run_hdbscan(
            umap_nd, mcs, ms, epsilon=0.0, selection_method=sel,
        )

        for thr in SOFT_ASSIGN_THRESHOLDS:
            labels     = labels_orig.copy()
            membership = membership_orig.copy()
            n_reassigned = 0

            if thr is not None:
                before_noise = int((labels == -1).sum())
                labels, membership = assign_noise_to_nearest(
                    clusterer, labels, membership, min_prob=thr,
                )
                n_reassigned = before_noise - int((labels == -1).sum())

            m = _metrics_for_labels(embeddings, umap_nd, labels, clusterer)
            row = {
                "config":       cfg_label,
                "nc":           nc,
                "nn":           nn,
                "sel":          sel,
                "mcs":          mcs,
                "ms":           ms,
                "threshold":    thr if thr is not None else "none",
                "n_reassigned": n_reassigned,
                "track":        name,
                **m,
            }
            row["score"] = round(_composite(row), 4)
            rows.append(row)

            thr_s = f"{thr:.1f}" if thr is not None else "none"
            print(f"  thr={thr_s:>4}  reassigned={n_reassigned:>5}  "
                  f"k={m['k']:>4}  noise={m['noise_pct']:>5.1f}%  "
                  f"sil_u={m['sil_umap']}  dbcv={m['dbcv']}  "
                  f"persist={m['persist']}  score={row['score']}")

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / f"soft_assign_sweep_{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")

    print(f"\n── Best threshold per config (by score) ────────────────────────")
    for cfg_label, gdf in df.groupby("config"):
        best = gdf.sort_values("score", ascending=False).iloc[0]
        print(f"  {cfg_label}")
        print(f"    → thr={best['threshold']}  k={best['k']}  "
              f"noise={best['noise_pct']:.1f}%  dbcv={best['dbcv']}  "
              f"sil_u={best['sil_umap']}  score={best['score']}")

    return df

def run_track_experiments(track_key: str, skip_embed: bool,
                          group_filter: set | None = None) -> pd.DataFrame:
    cfg = TRACKS[track_key]
    name = cfg["name"]
    label = cfg["label"]
    client = bq_client()

    exps = EXPERIMENTS if group_filter is None else [
        e for e in EXPERIMENTS if e["group"] in group_filter
    ]
    if not exps:
        raise ValueError(f"No experiments match groups {group_filter}. "
                         f"Available: {sorted(set(e['group'] for e in EXPERIMENTS))}")

    print(f"\n{'#'*60}")
    print(f"  Loading embeddings: {label}")
    if group_filter:
        print(f"  Groups filter: {sorted(group_filter)}")
    print(f"{'#'*60}")
    pulled = pull_embeddings(track_key, client, skip_embed=skip_embed)
    embeddings = pulled.embeddings

    rows = []
    n = len(exps)
    for i, exp in enumerate(exps, 1):
        tag = (f"G{exp['group'][-1]} nc={exp['nc']} nn={exp['nn']} "
               f"{exp['sel']} mcs={exp['mcs']} ms={exp['ms']} "
               f"merge={exp['merge']}")
        print(f"\n[{i:>2}/{n}] {tag}")
        try:
            row = run_one(exp, embeddings, name)
            row["track"] = name
            rows.append(row)
            print(f"       → k={row['k']}  noise={row['noise_pct']:.1f}%  "
                  f"sil_umap={row['sil_umap']}  dbcv={row['dbcv']}  "
                  f"persist={row['persist']}  score={row['score']}  "
                  f"({row['elapsed_s']}s)")
        except Exception as e:
            print(f"       ✗ FAILED: {e}")

    df = pd.DataFrame(rows)
    df["score"] = df.apply(_composite, axis=1).round(4)
    out_path = OUT_DIR / f"experiment_grid_{name}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")
    print_table(df, label)
    return df

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", choices=["a", "b", "both"], default="a")
    parser.add_argument("--skip-embed", action="store_true",
                        help="Reuse cached .npy embeddings (strongly recommended)")
    parser.add_argument("--groups", default=None,
                        help="Comma-separated group IDs to run (e.g. G7,G8,G9,G10). "
                             "Omit to run all groups.")
    parser.add_argument("--soft-assign-sweep", action="store_true",
                        help="Run soft-assignment threshold sweep (0.0–0.7) on base configs "
                             "instead of the parameter grid. Rule: eps=0 only.")
    parser.add_argument("--embed-suffix", type=str, default=None, metavar="SUFFIX",
                        help="Append SUFFIX to the source embedding table names and cache "
                             "filenames, e.g. --embed-suffix gemini uses "
                             "conversation_embeddings_multilingual_gemini. "
                             "Results saved to experiment_grid_<track>_<suffix>.csv.")
    args = parser.parse_args()

    if args.embed_suffix:
        suffix = args.embed_suffix.strip("_")
        for cfg in TRACKS.values():
            cfg["bq_table"] = cfg["bq_table"] + f"_{suffix}"
            cfg["name"]     = cfg["name"] + f"_{suffix}"

    group_filter = set(args.groups.split(",")) if args.groups else None
    tracks = ["a", "b"] if args.track == "both" else [args.track]

    if args.soft_assign_sweep:
        for tk in tracks:
            run_soft_assign_sweep(tk, skip_embed=args.skip_embed)
        return

    all_dfs = []
    for tk in tracks:
        df = run_track_experiments(tk, skip_embed=args.skip_embed,
                                   group_filter=group_filter)
        all_dfs.append(df)

    if len(all_dfs) == 2:
        print(f"\n\n{'='*80}")
        print("CROSS-TRACK: best config in each group")
        print(f"{'='*80}")
        for grp in sorted(set(e["group"] for e in EXPERIMENTS if group_filter is None or e["group"] in group_filter)):
            for df, tk in zip(all_dfs, tracks):
                sub = df[df["group"] == grp].sort_values("score", ascending=False)
                if len(sub):
                    r = sub.iloc[0]
                    print(f"  {grp} Track {tk.upper()}: k={int(r['k'])}  "
                          f"noise={r['noise_pct']:.1f}%  dbcv={r['dbcv']:.4f}  "
                          f"sil_u={r['sil_umap']:.4f}  persist={r['persist']:.4f}")

if __name__ == "__main__":
    main()
