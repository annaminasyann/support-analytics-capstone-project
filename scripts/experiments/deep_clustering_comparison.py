# %% [markdown]
# # Deep vs Classical Clustering Comparison
# **Project:** Support Analytics Capstone — Anna Minasyan · DS 299 · 10Web Inc.
#
# This notebook compares six deep unsupervised learning clustering methods
# (AE, DEC, RBM, VAE, Parametric UMAP, Parametric t-SNE / TCNE)
# against the chunk pipeline (UMAP+HDBSCAN on semantic chunks).
#
# **Embedding note:**
# - Deep methods: sil_cos evaluated in 768-D multilingual embedding space (chunk_embeddings_multilingual.npy)
# - Chunk HDBSCAN baseline: sil_cos evaluated in 384-D finetuned embedding space (different space)
# - Old conversation-level baselines retained for historical reference only

# %%
import pathlib, sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT  = pathlib.Path("../..").resolve()
DATA_DIR   = REPO_ROOT / "data"
FIGURES    = REPO_ROOT / "data" / "figures"
FIGURES.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))

# %%

RESULTS_CSV = DATA_DIR / "experiments" / "deep_clustering_results_chunks.csv"

if RESULTS_CSV.exists():
    df_deep = pd.read_csv(RESULTS_CSV)
    if "source_model" not in df_deep.columns:
        df_deep["source_model"] = "unknown"
    print(f"Loaded {len(df_deep)} deep clustering results from {RESULTS_CSV.name}")
else:
    print("WARNING: data/experiments/deep_clustering_results_chunks.csv not found.")
    print("Run the deep clustering scripts first (on server, in tmux):")
    print("  python scripts/experiments/deep_clustering/autoencoder_hdbscan.py --model multilingual")
    print("  python scripts/experiments/deep_clustering/dec_clustering.py --model multilingual")
    print("  python scripts/experiments/deep_clustering/rbm_hdbscan.py --model multilingual")
    print("  python scripts/experiments/deep_clustering/parametric_umap_hdbscan.py --model multilingual")
    print("  python scripts/experiments/deep_clustering/vae_hdbscan.py --model multilingual")
    print("  python scripts/experiments/deep_clustering/parametric_tsne_hdbscan.py --model multilingual")
    df_deep = pd.DataFrame()

# chunk pipeline sil_cos is in 384-D finetuned space, not comparable to deep methods (768-D)
CLASSICAL_BASELINES = [
    {
        # Chunk pipeline — production result. sil_cos in finetuned-384D space.
        "method":            "UMAP(10D)+HDBSCAN — Chunk Pipeline",
        "method_type":       "chunk_pipeline",
        "latent_dim":        10,
        "k":                 23,
        "noise_pct":         26.54,
        "sil_cos":           0.6348,
        "sil_latent":        0.6794,
        "davies_bouldin":    0.5352,
        "calinski_harabasz": 301756.0,
        "n_clusters":        23,
        "source_embeddings": "multilingual-e5-small finetuned (384-D)",
        "source_model":      "finetuned-384D",
    },
    {
        "method":            "TF-IDF + K-Means",
        "method_type":       "classical_ml",
        "latent_dim":        None,
        "k":                 73,
        "noise_pct":         0.0,
        "sil_cos":           0.059,
        "sil_latent":        None,
        "davies_bouldin":    3.367,
        "calinski_harabasz": 272.0,
        "n_clusters":        73,
        "source_embeddings": "TF-IDF (bag-of-words)",
        "source_model":      "classical",
    },
    {
        # Canonical conversation-level Track A (historical reference)
        "method":            "UMAP(20D)+HDBSCAN — Conv. Track A",
        "method_type":       "classical_ml",
        "latent_dim":        20,
        "k":                 70,
        "noise_pct":         34.1,
        "sil_cos":           0.077,
        "sil_latent":        0.558,
        "davies_bouldin":    None,
        "calinski_harabasz": None,
        "n_clusters":        70,
        "source_embeddings": "text-multilingual-embedding-002 (768-D)",
        "source_model":      "multilingual-768D",
    },
    {
        # Best conversation-level Track B (historical reference)
        "method":            "UMAP(20D)+HDBSCAN — Conv. Track B best",
        "method_type":       "classical_ml",
        "latent_dim":        20,
        "k":                 90,
        "noise_pct":         37.4,
        "sil_cos":           0.118,
        "sil_latent":        0.618,
        "davies_bouldin":    None,
        "calinski_harabasz": None,
        "n_clusters":        90,
        "source_embeddings": "text-multilingual-embedding-002 (768-D) nn=50",
        "source_model":      "multilingual-768D",
    },
]

df_classical = pd.DataFrame(CLASSICAL_BASELINES)

if not df_deep.empty:
    best_rows = []
    for (method, source_model), grp in df_deep.groupby(["method", "source_model"], dropna=False):
        sub = grp.dropna(subset=["sil_cos"])
        if sub.empty:
            best_rows.append(grp.iloc[0])
        else:
            best_rows.append(sub.loc[sub["sil_cos"].idxmax()])
    df_deep_best = pd.DataFrame(best_rows).reset_index(drop=True)
else:
    df_deep_best = pd.DataFrame()

df_all = pd.concat([df_classical, df_deep_best], ignore_index=True)

df_all["latent_dim_label"] = df_all["latent_dim"].apply(
    lambda x: f"{int(x)}-D" if pd.notna(x) else "N/A"
)

print("\n" + "=" * 80)
print("COMBINED COMPARISON TABLE")
print("=" * 80)
print(df_all[[
    "method", "method_type", "source_model", "latent_dim_label", "k",
    "sil_cos", "sil_latent", "davies_bouldin", "noise_pct"
]].to_string(index=False))

# %%

HDBSCAN_SIL_COS_BASELINE = 0.118   # best conversation-level HDBSCAN (historical reference)

print("\nComparison Table — sil_cos in raw embedding cosine space")
print("-" * 60)

display_cols = {
    "method":            "Method",
    "method_type":       "Type",
    "source_model":      "Source Embedding",
    "latent_dim_label":  "Latent Dim",
    "k":                 "k",
    "sil_cos":           "sil_cos ↑",
    "sil_latent":        "sil_latent ↑",
    "noise_pct":         "Noise %",
    "davies_bouldin":    "DB ↓",
    "calinski_harabasz": "CH ↑",
}

df_display = df_all[list(display_cols)].copy()
df_display.columns = list(display_cols.values())
df_display = df_display.sort_values("sil_cos ↑", ascending=False, na_position="last")

# Mark k >= 40 rows
def _annotate_k(row):
    k_val = row["k"]
    method_val = row["Method"]
    try:
        if pd.notna(k_val) and int(k_val) >= 40:
            return str(method_val) + " (k≥40 — excluded)"
    except (ValueError, TypeError):
        pass
    return method_val

df_display["Method"] = df_display.apply(_annotate_k, axis=1)

for col in ["sil_cos ↑", "sil_latent ↑", "DB ↓", "Noise %"]:
    df_display[col] = df_display[col].apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else "—"
    )
df_display["CH ↑"] = df_display["CH ↑"].apply(
    lambda x: f"{x:.0f}" if pd.notna(x) else "—"
)
df_display["k"] = df_display["k"].apply(lambda x: f"{int(x)}" if pd.notna(x) else "—")

print(df_display.to_string(index=False))

best_row = df_all.dropna(subset=["sil_cos"]).loc[df_all.dropna(subset=["sil_cos"])["sil_cos"].idxmax()]
print(f"\n** Best sil_cos overall: {best_row['method']} ({best_row.get('source_model','')}) = {best_row['sil_cos']:.4f}")

# %%

df_bar = df_all.dropna(subset=["sil_cos"]).copy()
df_bar["method_short"] = df_bar["method"].str.replace(r" — .*", "", regex=True)

COLOR_MAP = {
    "chunk_pipeline": "#70AD47",
    "classical_ml":   "#4472C4",
    "deep_learning":  "#ED7D31",
}

fig_bar = px.bar(
    df_bar.sort_values("sil_cos", ascending=True),
    x                  = "sil_cos",
    y                  = "method",
    color              = "method_type",
    color_discrete_map = COLOR_MAP,
    orientation        = "h",
    title   = "Silhouette Score — Deep vs Classical vs Chunk Pipeline<br>"
              "<sup>Deep methods evaluated in 768-D space · Chunk pipeline in 384-D finetuned space</sup>",
    labels  = {"sil_cos": "Silhouette (cosine)", "method": "Method",
               "method_type": "Method Type", "source_model": "Source Model"},
    text    = "sil_cos",
    template= "plotly_white",
)
fig_bar.add_vline(
    x=0.059, line_dash="dash", line_color="red", line_width=1.5,
    annotation_text="TF-IDF K-Means baseline (0.059)",
    annotation_position="top right",
)
fig_bar.add_vline(
    x=0.118, line_dash="dot", line_color="blue", line_width=1.5,
    annotation_text="Best conv. HDBSCAN (0.118)",
    annotation_position="top right",
)
fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig_bar.update_layout(
    height=max(400, len(df_bar) * 45),
    xaxis_range=[0, df_bar["sil_cos"].max() * 1.25],
    legend_title="Method Type",
)
fig_bar.write_html(str(FIGURES / "deep_vs_classical_silhouette.html"))
fig_bar.show()
print(f"Saved → {FIGURES / 'deep_vs_classical_silhouette.html'}")

# %%

df_scatter = df_all.dropna(subset=["sil_cos", "latent_dim"]).copy()

SYMBOL_MAP = {
    "finetuned-384D":    "star",
    "multilingual-768D": "circle",
    "classical":         "square",
    "unknown":           "circle",
}
df_scatter["symbol"] = df_scatter["source_model"].map(SYMBOL_MAP).fillna("circle")

fig_scatter = px.scatter(
    df_scatter,
    x                  = "latent_dim",
    y                  = "sil_cos",
    size               = "n_clusters",
    color              = "method_type",
    color_discrete_map = COLOR_MAP,
    symbol             = "source_model",
    symbol_map         = SYMBOL_MAP,
    hover_name         = "method",
    hover_data         = {"k": True, "noise_pct": True, "latent_dim": True, "source_model": True},
    title   = "Latent Dimensionality vs Cluster Quality (sil_cos)<br>"
              "<sup>Point size = number of clusters · circle=multilingual · diamond=gemini</sup>",
    labels  = {
        "latent_dim":  "Latent Dimensions Used for Clustering",
        "sil_cos":     "Silhouette (embedding cosine)",
        "method_type": "Method Type",
        "source_model":"Source Model",
    },
    template= "plotly_white",
)
for _, row in df_scatter.iterrows():
    fig_scatter.add_annotation(
        x        = row["latent_dim"] + 0.3,
        y        = row["sil_cos"],
        text     = row["method"].split("(")[0].strip(),
        showarrow= False,
        font     = dict(size=9),
        xanchor  = "left",
    )
fig_scatter.add_hline(
    y=0.059, line_dash="dot", line_color="red",
    annotation_text="TF-IDF baseline",
)
fig_scatter.update_layout(height=550)
fig_scatter.write_html(str(FIGURES / "dimensionality_vs_quality.html"))
fig_scatter.show()
print(f"Saved → {FIGURES / 'dimensionality_vs_quality.html'}")

# %%

conv_baseline_trackA = 0.077    # conversation-level HDBSCAN Track A (historical)
conv_baseline_best   = 0.118    # conversation-level HDBSCAN Track B best (historical)

print("\nSummary — deep methods vs chunk pipeline baselines")
print("-" * 55)
print("Note: deep methods evaluated in 768-D multilingual space.")
print("      Chunk pipeline sil_cos=0.6348 is in 384-D finetuned space — not directly comparable.")
print(f"      Conversation-level baselines (Track A={conv_baseline_trackA}, best={conv_baseline_best}) for historical ref.\n")

deep_sub = df_deep_best[df_deep_best["source_model"] == "multilingual-768D"].dropna(subset=["sil_cos"]) if not df_deep_best.empty else pd.DataFrame()
if not deep_sub.empty:
    best_deep_row    = deep_sub.loc[deep_sub["sil_cos"].idxmax()]
    best_deep_method = best_deep_row["method"]
    best_deep_sil    = float(best_deep_row["sil_cos"])
    delta_a = best_deep_sil - conv_baseline_trackA
    delta_b = best_deep_sil - conv_baseline_best
    print(f"Best deep method (768-D): {best_deep_method}")
    print(f"  sil_cos = {best_deep_sil:.4f}")
    print(f"  vs conv Track A (0.077): {delta_a:+.4f}")
    print(f"  vs conv Track B best (0.118): {delta_b:+.4f}")
else:
    print("No deep results for multilingual-768D yet.")

if not df_deep_best.empty and "sil_cos" in df_deep_best.columns:
    valid_deep = df_deep_best.dropna(subset=["sil_cos"])
    if not valid_deep.empty:
        best_deep_overall = valid_deep.loc[valid_deep["sil_cos"].idxmax()]
        best_sil_overall  = float(best_deep_overall["sil_cos"])
        delta_overall     = best_sil_overall - conv_baseline_trackA
        direction = "outperform" if delta_overall > 0.005 else ("match" if abs(delta_overall) <= 0.005 else "underperform vs")
        print(f"\nConclusion: Deep methods on chunks {direction} "
              f"conversation-level UMAP+HDBSCAN (sil_cos={conv_baseline_trackA}).")
        print(f"Best overall: {best_deep_overall['method']} "
              f"[{best_deep_overall.get('source_model','?')}] = {best_sil_overall:.4f}")
    else:
        print("No deep results with sil_cos available yet.")
else:
    print("No deep learning results found. Run scripts/experiments/deep_clustering/*.py first.")

# %%

if not df_deep.empty:
    dec_sweep = df_deep[df_deep["method"].str.startswith("DEC")].dropna(subset=["sil_cos"])

    if not dec_sweep.empty:
        fig_sweep = px.line(
            dec_sweep,
            x        = "k",
            y        = "sil_cos",
            color    = "source_model",
            markers  = True,
            title    = "DEC Soft-Assignment K-Sweep: sil_cos vs k (initialisation sweep)",
            labels   = {"sil_cos": "sil_cos (embedding cosine)", "k": "Number of clusters k",
                        "source_model": "Source Model"},
            template = "plotly_white",
        )
        fig_sweep.add_hline(y=0.059, line_dash="dot", line_color="red",
                            annotation_text="TF-IDF baseline")
        fig_sweep.add_hline(y=0.077, line_dash="dash", line_color="blue",
                            annotation_text="Conv. HDBSCAN Track A (0.077)")
        fig_sweep.update_layout(height=450)
        fig_sweep.write_html(str(FIGURES / "k_sweep_dec.html"))
        fig_sweep.show()
        print(f"Saved → {FIGURES / 'k_sweep_dec.html'}")
    else:
        print("No DEC results found in deep_clustering_results_chunks.csv yet.")
