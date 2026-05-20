"""
Ranks focused-sweep results per track and compares the two best configs for RQ4.

Composite score: 0.25 × sil_umap + 0.30 × dbcv_norm + 0.20 × sil_cos
              + 0.20 × persist + 0.05 × noise_bonus

Usage:
  python scripts/experiments/find_best_params.py
"""
import os
import sys
import numpy as np
import pandas as pd

DATA_DIR = os.path.join("data", "experiments")


def _load(filename: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _composite(df: pd.DataFrame) -> pd.Series:
    """Weighted composite score. All inputs normalised to [0, 1]."""
    sil_cos  = df["silhouette_embed_cosine"].fillna(0.0)
    sil_umap = df["silhouette_umap"].fillna(0.0) if "silhouette_umap" in df.columns else pd.Series(0.0, index=df.index)
    dbcv     = (df["dbcv_umap"].fillna(float("nan")) if "dbcv_umap" in df.columns
                else pd.Series(float("nan"), index=df.index))
    pers     = df["cluster_persistence_mean"].fillna(0.0)
    noise    = df["noise_pct"].fillna(50.0)

    dbcv_norm    = ((dbcv + 1) / 2).fillna(0.0)
    noise_bonus  = ((50.0 - noise) / 50.0).clip(lower=0.0)

    return (0.20 * sil_cos
            + 0.25 * sil_umap
            + 0.30 * dbcv_norm
            + 0.20 * pers
            + 0.05 * noise_bonus)


def _quality_flags(row: pd.Series) -> str:
    """Return a short quality string showing which thresholds are met."""
    flags = []

    dbcv = row.get("dbcv_umap", float("nan"))
    if not np.isnan(dbcv):
        if dbcv > 0.50:   flags.append("DBCV✓✓")
        elif dbcv > 0.30: flags.append("DBCV✓")
        else:             flags.append("DBCV✗")

    sil_u = row.get("silhouette_umap", float("nan"))
    if not np.isnan(sil_u):
        if sil_u > 0.50:   flags.append("sil✓✓")
        elif sil_u > 0.25: flags.append("sil✓")
        else:              flags.append("sil✗")

    noise = row.get("noise_pct", float("nan"))
    if not np.isnan(noise):
        if noise < 40.0:  flags.append("noise✓")
        elif noise < 50.0: flags.append("noise~")
        else:             flags.append("noise✗")

    pers = row.get("cluster_persistence_mean", float("nan"))
    if not np.isnan(pers):
        if pers > 0.12:  flags.append("pers✓✓")
        elif pers > 0.10: flags.append("pers✓")
        else:            flags.append("pers✗")

    return "  " + "  ".join(flags)


def _filter_valid(df: pd.DataFrame,
                  max_noise: float = 55.0,
                  min_k: int = 30,
                  max_k: int = 250) -> pd.DataFrame:
    return df[
        df["n_clusters"].between(min_k, max_k)
        & (df["noise_pct"] <= max_noise)
        & df["silhouette_embed_cosine"].notna()
    ].copy()


def _print_thresholds() -> None:
    print("""
Quality thresholds (from HDBSCAN literature):
  DBCV      > 0.30 acceptable (✓)   > 0.50 strong (✓✓)
  sil_umap  > 0.25 acceptable (✓)   > 0.50 strong (✓✓)
  sil_cos   — shown for TF-IDF baseline comparison (RQ4), not thresholded
  noise%    < 50% acceptable (~)    < 40% good (✓)
  persist   > 0.10 acceptable (✓)   > 0.12 strong (✓✓)
""")


def _print_track_table(df: pd.DataFrame, track_label: str) -> None:
    has_dbcv  = "dbcv_umap" in df.columns and df["dbcv_umap"].notna().any()
    has_umap  = "silhouette_umap" in df.columns and df["silhouette_umap"].notna().any()

    print(f"\n{'='*120}")
    print(f"TRACK {track_label}  —  sorted by composite score")
    print(f"  (sil_cos×0.20 + sil_umap×0.25 + dbcv_norm×0.30 + persist×0.20 + noise_bonus×0.05)")
    print(f"  Valid configs (k 30–250, noise ≤ 55%): {len(df)}")
    print(f"{'='*120}")

    hdr = (f"  {'sel':>4}  {'mcs':>5}  {'ms':>4}  "
           f"{'k':>5}  {'noise':>6}  {'sil_cos':>9}")
    if has_umap:
        hdr += f"  {'sil_umap':>9}"
    if has_dbcv:
        hdr += f"  {'dbcv':>8}"
    hdr += f"  {'persist':>9}  {'score':>7}  quality"
    print(hdr)
    print(f"  {'-'*108}")

    for _, r in df.head(20).iterrows():
        line = (f"  {r.get('selection_method','?'):>4}  "
                f"{int(r['min_cluster_size']):>5}  "
                f"{int(r['min_samples']):>4}  "
                f"{int(r['n_clusters']):>5}  "
                f"{r['noise_pct']:>5.1f}%  "
                f"{r['silhouette_embed_cosine']:>9.4f}")
        if has_umap:
            sil_u = r.get("silhouette_umap", float("nan"))
            line += f"  {sil_u:>9.4f}"
        if has_dbcv:
            dbcv_val = r.get("dbcv_umap", float("nan"))
            line += f"  {dbcv_val:>8.4f}"
        line += (f"  {r.get('cluster_persistence_mean', float('nan')):>9.4f}"
                 f"  {r['score']:>7.4f}"
                 f"{_quality_flags(r)}")
        print(line)


def _print_recommendation(best: pd.Series, track_label: str) -> None:
    print(f"\n{'─'*70}")
    print(f"BEST CONFIG — TRACK {track_label}:")
    print(f"  selection_method : {best.get('selection_method', '?')}")
    print(f"  min_cluster_size : {int(best['min_cluster_size'])}")
    print(f"  min_samples      : {int(best['min_samples'])}")
    nc = int(best.get("n_components", 20))
    nn = int(best.get("umap_n_neighbors", 30))
    print(f"  umap_components  : {nc}")
    print(f"  umap_neighbors   : {nn}")
    print(f"  ── Results ──────────────────────────────────────")
    print(f"  k                : {int(best['n_clusters'])}")
    print(f"  noise%           : {best['noise_pct']:.1f}%  "
          + ("✓" if best["noise_pct"] < 40 else "~" if best["noise_pct"] < 50 else "✗"))
    print(f"  sil_cos (768D)   : {best['silhouette_embed_cosine']:.4f}")
    sil_u = best.get("silhouette_umap", float("nan"))
    if not np.isnan(sil_u):
        flag = "✓✓" if sil_u > 0.50 else "✓" if sil_u > 0.25 else "✗"
        print(f"  sil_umap         : {sil_u:.4f}  {flag}")
    dbcv = best.get("dbcv_umap", float("nan"))
    if not np.isnan(dbcv):
        flag = "✓✓" if dbcv > 0.50 else "✓" if dbcv > 0.30 else "✗"
        print(f"  dbcv             : {dbcv:.4f}  {flag}")
    pers = best.get("cluster_persistence_mean", float("nan"))
    if not np.isnan(pers):
        flag = "✓✓" if pers > 0.12 else "✓" if pers > 0.10 else "✗"
        print(f"  persistence      : {pers:.4f}  {flag}")
    print(f"  composite score  : {best['score']:.4f}")
    sel = best.get("selection_method", "eom")
    eps = best.get("epsilon", 0.0)
    print(f"\n  Run command:")
    print(f"    python scripts/experiments/embed_and_cluster.py \\")
    print(f"      --track a \\  # change to b for Track B")
    print(f"      --skip-embed \\")
    print(f"      --umap-components {nc} \\")
    print(f"      --umap-neighbors  {nn} \\")
    print(f"      --selection-method {sel} \\")
    print(f"      --min-cluster-size {int(best['min_cluster_size'])} \\")
    print(f"      --min-samples {int(best['min_samples'])} \\")
    print(f"      --epsilon {eps:.2f}")



fs_a = _load("focused_sweep_multilingual.csv")
fs_b = _load("focused_sweep_translated.csv")

if fs_a is None or fs_b is None:
    missing = []
    if fs_a is None: missing.append("focused_sweep_multilingual.csv")
    if fs_b is None: missing.append("focused_sweep_translated.csv")
    print(f"\n[ERROR] Missing sweep CSV files: {', '.join(missing)}")
    print("  Run the focused sweep first:")
    print("    python scripts/experiments/embed_and_cluster.py --track both --skip-embed "
          "--umap-components 20 --focused-sweep")
    sys.exit(1)

_print_thresholds()

best_per_track = {}
for track_label, fs in [("A (multilingual)", fs_a), ("B (translated)", fs_b)]:
    valid = _filter_valid(fs)
    if len(valid) == 0:
        print(f"\nTrack {track_label}: no valid configs after filtering.")
        continue
    valid["score"] = _composite(valid)
    valid = valid.sort_values("score", ascending=False)
    _print_track_table(valid, track_label)
    _print_recommendation(valid.iloc[0], track_label)
    best_per_track[track_label] = valid.iloc[0]



if len(best_per_track) == 2:
    print(f"\n\n{'='*110}")
    print("RQ4 COMPARISON — Best Track A config vs Best Track B config")
    print("  (each track independently optimised)")
    print(f"{'='*110}")

    best_a = best_per_track["A (multilingual)"]
    best_b = best_per_track["B (translated)"]

    metrics = [
        ("sil_cos (768D)",    "silhouette_embed_cosine"),
        ("sil_umap",          "silhouette_umap"),
        ("dbcv",              "dbcv_umap"),
        ("persistence",       "cluster_persistence_mean"),
        ("k (clusters)",      "n_clusters"),
        ("noise%",            "noise_pct"),
    ]
    print(f"\n  {'Metric':<24}  {'Track A':>10}  {'Track B':>10}  {'B−A':>10}  thresholds")
    print(f"  {'-'*72}")
    b_wins = 0
    total_comparable = 0
    for label, col in metrics:
        va = best_a.get(col, float("nan"))
        vb = best_b.get(col, float("nan"))
        if isinstance(va, (int, np.integer)):
            print(f"  {label:<24}  {int(va):>10}  {int(vb):>10}")
            continue
        if np.isnan(va) and np.isnan(vb):
            continue
        diff = vb - va if not (np.isnan(va) or np.isnan(vb)) else float("nan")

        thresh = ""
        if col == "dbcv_umap":
            thresh = "> 0.30 acc / > 0.50 strong"
        elif col == "silhouette_umap":
            thresh = "> 0.25 acc / > 0.50 strong"
        elif col == "noise_pct":
            thresh = "< 50% acc / < 40% good"
        elif col == "cluster_persistence_mean":
            thresh = "> 0.10 acc / > 0.12 strong"

        diff_str = f"{diff:>+10.4f}" if not np.isnan(diff) else "       n/a"
        print(f"  {label:<24}  {va:>10.4f}  {vb:>10.4f}  {diff_str}  {thresh}")

        if col not in ("n_clusters", "noise_pct") and not np.isnan(diff):
            total_comparable += 1
            if diff > 0:
                b_wins += 1

    print(f"\n  Track B wins on {b_wins}/{total_comparable} quality metrics.")
    if b_wins >= 3:
        print("  → Translation improves clustering quality (RQ4: B > A on most metrics)")
    elif b_wins >= 2:
        print("  → Mixed: translation shows moderate improvement (RQ4)")
    else:
        print("  → Translation does not consistently improve clustering (RQ4 finding)")

print()
