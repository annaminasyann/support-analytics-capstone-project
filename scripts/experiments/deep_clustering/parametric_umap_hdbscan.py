"""
Parametric UMAP + HDBSCAN deep clustering (Method 4).
PyTorch encoder trained on UMAP cross-entropy loss (fuzzy graph) — avoids TF dependency for Python 3.12.

Usage:
  python scripts/experiments/deep_clustering/parametric_umap_hdbscan.py
  python scripts/experiments/deep_clustering/parametric_umap_hdbscan.py --model multilingual --components 8
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import hdbscan
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import umap
from torch.utils.data import DataLoader, TensorDataset

_HERE     = pathlib.Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(_HERE))

from _common import (        # noqa: E402
    DATA_DIR,
    MODELS_DIR,
    NPY_GEMINI,
    NPY_MULTILINGUAL,
    RNG,
    db_ch,
    load_both_embeddings,
    load_embeddings,
    save_result,
    sil_cos,
    sil_latent,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


class ParametricEncoder(nn.Module):
    """Encoder trained with UMAP cross-entropy loss (McInnes & Healy 2021). D → hidden1 → hidden2 → n_components."""

    def __init__(self, input_dim: int, n_components: int) -> None:
        super().__init__()
        hidden1 = 512 if input_dim >= 2048 else 256
        hidden2 = 128 if input_dim >= 2048 else 64
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden2, n_components),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_umap_graph(
    embeddings:  np.ndarray,
    n_neighbors: int,
    n_epochs:    int,
) -> sp.coo_matrix:
    """Run UMAP (minimal epochs) to get the fuzzy graph — we need graph_, not the embedding."""
    log.info("Building UMAP fuzzy graph  n_neighbors=%d  n_epochs=%d …",
             n_neighbors, n_epochs)
    reducer = umap.UMAP(
        n_components  = 2,            # irrelevant — we use graph_ not the embedding
        n_neighbors   = n_neighbors,
        min_dist      = 0.0,
        metric        = "cosine",
        n_epochs      = n_epochs,     # low epochs — we only need the graph
        random_state  = RNG,
        low_memory    = False,
    )
    reducer.fit(embeddings)
    graph = reducer.graph_
    log.info("UMAP graph: %d positive edges  density=%.4f",
             graph.nnz, graph.nnz / (embeddings.shape[0] ** 2))
    return graph.tocoo()


def umap_loss(
    z_i:      torch.Tensor,
    z_j:      torch.Tensor,
    weights:  torch.Tensor,
    neg_z_i:  torch.Tensor,
    neg_z_j:  torch.Tensor,
    gamma:    float = 1.0,
    a:        float = 1.0,
    b:        float = 1.0,
) -> torch.Tensor:
    """UMAP cross-entropy loss (McInnes et al. 2018). a=1, b=1 (Cauchy kernel, min_dist=0 optimals)."""
    def prob(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        d2  = ((z1 - z2) ** 2).sum(dim=1)
        return (1.0 + a * d2 ** b) ** (-1.0)

    p_pos  = prob(z_i, z_j)
    p_neg  = prob(neg_z_i, neg_z_j)

    pos_loss = -(weights * torch.log(p_pos.clamp(1e-6, 1 - 1e-6))).mean()
    neg_loss = -(gamma   * torch.log((1 - p_neg).clamp(1e-6, 1 - 1e-6))).mean()
    return pos_loss + neg_loss


def train_encoder(
    encoder:     ParametricEncoder,
    embeddings:  np.ndarray,
    graph:       sp.coo_matrix,
    epochs:      int,
    batch_size:  int,
    device:      torch.device,
) -> ParametricEncoder:
    """Train encoder on UMAP cross-entropy loss: positive pairs from fuzzy graph, random negative pairs."""
    X_t      = torch.from_numpy(embeddings).to(device)
    n        = len(embeddings)

    pos_i    = torch.from_numpy(graph.row.astype(np.int64))
    pos_j    = torch.from_numpy(graph.col.astype(np.int64))
    pos_w    = torch.from_numpy(graph.data.astype(np.float32))

    ds       = TensorDataset(pos_i, pos_j, pos_w)
    dl       = DataLoader(ds, batch_size=batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(RNG))

    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log.info("Training parametric encoder  edges=%d  epochs=%d  batch=%d",
             len(pos_i), epochs, batch_size)

    encoder.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (bi, bj, bw) in dl:
            bi, bj, bw = bi.to(device), bj.to(device), bw.to(device)

            neg_i = torch.randint(0, n, (len(bi),), device=device)
            neg_j = torch.randint(0, n, (len(bi),), device=device)

            z_i    = encoder(X_t[bi])
            z_j    = encoder(X_t[bj])
            z_ni   = encoder(X_t[neg_i])
            z_nj   = encoder(X_t[neg_j])

            loss = umap_loss(z_i, z_j, bw, z_ni, z_nj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bi)

        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            log.info("  Epoch %3d/%d  loss=%.6f  lr=%.2e",
                     epoch, epochs, total_loss / len(pos_i),
                     scheduler.get_last_lr()[0])

    return encoder


@torch.no_grad()
def project(
    encoder:     ParametricEncoder,
    embeddings:  np.ndarray,
    batch_size:  int,
    device:      torch.device,
) -> np.ndarray:
    encoder.eval()
    X_t    = torch.from_numpy(embeddings).to(device)
    ds     = TensorDataset(X_t)
    dl     = DataLoader(ds, batch_size=batch_size, shuffle=False)
    parts  = []
    for (batch,) in dl:
        parts.append(encoder(batch).cpu().numpy())
    return np.concatenate(parts, axis=0)


def run_for_model(
    embeddings:   np.ndarray,
    suffix:       str,
    source_model: str,
    components:   int,
    n_neighbors:  int,
    graph_epochs: int,
    enc_epochs:   int,
    batch_size:   int,
    mcs:          int,
    min_samples:  int,
    device:       torch.device,
) -> None:
    """Full Parametric UMAP + HDBSCAN pipeline for one embedding model."""
    source_embeddings = str(NPY_GEMINI if suffix == "gemini" else NPY_MULTILINGUAL)
    input_dim = embeddings.shape[1]

    encoder_dir  = MODELS_DIR / f"parametric_umap_{components}d_{suffix}"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = encoder_dir / "encoder.pt"
    latent_path  = DATA_DIR / f"latent_parametric_umap_{components}d_{suffix}.npy"
    labels_path  = DATA_DIR / f"labels_pumap_hdbscan_{suffix}.npy"
    graph_path   = DATA_DIR / f"pumap_graph_{suffix}.npz"

    encoder = ParametricEncoder(input_dim, components).to(device)

    if encoder_path.exists():
        log.info("Loading existing encoder from %s", encoder_path)
        state = torch.load(encoder_path, map_location=device, weights_only=True)
        encoder.load_state_dict(state)
    else:
        graph = build_umap_graph(embeddings, n_neighbors, graph_epochs)
        sp.save_npz(str(graph_path), graph.tocsr())
        log.info("Graph saved → %s", graph_path)

        encoder = train_encoder(
            encoder    = encoder,
            embeddings = embeddings,
            graph      = graph,
            epochs     = enc_epochs,
            batch_size = batch_size,
            device     = device,
        )
        torch.save(encoder.state_dict(), encoder_path)
        log.info("Encoder saved → %s", encoder_path)

    log.info("Projecting %d conversations to %d-D [%s] …", len(embeddings), components, suffix)
    latent = project(encoder, embeddings, batch_size, device)
    np.save(latent_path, latent)
    log.info("Latent saved → %s  shape=%s", latent_path, latent.shape)

    log.info("HDBSCAN  min_cluster_size=%d  min_samples=%d …", mcs, min_samples)
    mcs_values = [10, 15, 25, 40, 60, 100, 150]
    best_labels = None
    best_sil    = -2.0
    best_mcs    = mcs_values[0]

    for mcs_val in mcs_values:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size        = mcs_val,
            min_samples             = 5,
            metric                  = "euclidean",
            cluster_selection_method= "leaf",
            prediction_data         = True,
        )
        labels_tmp  = clusterer.fit_predict(latent).astype(np.int32)
        n_clusters  = int(len(set(labels_tmp)) - (1 if -1 in labels_tmp else 0))
        noise_pct_t = round(100.0 * (labels_tmp == -1).sum() / len(labels_tmp), 2)
        log.info("  HDBSCAN mcs=%d → k=%d  noise=%.1f%%", mcs_val, n_clusters, noise_pct_t)
        if n_clusters < 2:
            continue
        sc_t      = sil_cos(embeddings, labels_tmp)
        sl_t      = sil_latent(latent, labels_tmp)
        dbi_t, ch_t = db_ch(latent, labels_tmp)
        log.info("  mcs=%d  k=%d  sil_cos=%.4f  noise=%.1f%%",
                 mcs_val, n_clusters, sc_t or -1, noise_pct_t)
        if n_clusters < 40:
            save_result({
                "method":            f"Param-UMAP({components}D)+HDBSCAN",
                "method_type":       "deep_learning",
                "latent_dim":        components,
                "k":                 n_clusters,
                "noise_pct":         noise_pct_t,
                "sil_cos":           sc_t,
                "sil_latent":        sl_t,
                "davies_bouldin":    dbi_t,
                "calinski_harabasz": ch_t,
                "n_clusters":        n_clusters,
                "source_embeddings": source_embeddings,
                "source_model":      source_model,
            })
        if sc_t is not None and sc_t > best_sil and n_clusters < 40:
            best_sil, best_mcs, best_labels = sc_t, mcs_val, labels_tmp.copy()

    if best_labels is None:
        best_labels = labels_tmp
    labels    = best_labels
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    noise_pct  = round(100.0 * (labels == -1).sum() / len(labels), 2)
    sc         = sil_cos(embeddings, labels)
    sl         = sil_latent(latent, labels)
    dbi, ch    = db_ch(latent, labels)
    log.info("Best mcs=%d  k=%d  sil_cos=%.4f", best_mcs, n_clusters, best_sil)

    np.save(labels_path, labels)
    log.info("Labels saved → %s", labels_path)

    print("\n" + "=" * 60)
    print(f"PARAMETRIC UMAP ({components}D) + HDBSCAN  [{suffix}]")
    print("=" * 60)
    print(f"  n_clusters: {n_clusters}")
    print(f"  noise_pct:  {noise_pct:.1f}%")
    print(f"  sil_cos:    {sc}")
    print(f"  sil_latent: {sl}")
    print(f"  Results:    {DATA_DIR / 'deep_clustering_results.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",            choices=["multilingual", "gemini", "both"],
                        default="both",
                        help="Which embedding model to run (default: both)")
    parser.add_argument("--components",       type=int, default=8,
                        help="Latent dimensionality (default: 8, must be ≤10)")
    parser.add_argument("--n-neighbors",      type=int, default=15,
                        help="UMAP n_neighbors for graph construction (default: 15)")
    parser.add_argument("--graph-epochs",     type=int, default=50,
                        help="UMAP graph epochs — low values suffice (default: 50)")
    parser.add_argument("--encoder-epochs",   type=int, default=50,
                        help="Encoder training epochs (default: 50)")
    parser.add_argument("--batch-size",       type=int, default=1024)
    parser.add_argument("--min-cluster-size", type=int, default=50,
                        help="HDBSCAN min_cluster_size (default: 50)")
    parser.add_argument("--min-samples",      type=int, default=10)
    args = parser.parse_args()

    if args.components > 10:
        parser.error("--components must be ≤10")

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()         else
        "cpu"
    )
    log.info("Device: %s", device)

    if args.model == "both":
        emb_dict = load_both_embeddings()
        entries = [
            ("multilingual", emb_dict.get("multilingual"), "multilingual-768D"),
            ("gemini",       emb_dict.get("gemini"),       "gemini-3072D"),
        ]
    elif args.model == "multilingual":
        emb = load_embeddings(NPY_MULTILINGUAL, bq_table="support-analytics-492410.support_analytics.conversation_embeddings_multilingual")
        entries = [("multilingual", emb, "multilingual-768D")]
    else:
        emb = load_embeddings(NPY_GEMINI, bq_table="support-analytics-492410.support_analytics.conversation_embeddings_multilingual_gemini")
        entries = [("gemini", emb, "gemini-3072D")]

    for suffix, embeddings, source_model in entries:
        if embeddings is None:
            log.warning("Skipping [%s] — embeddings not available.", suffix)
            continue
        log.info("=" * 60)
        log.info("Running Parametric UMAP+HDBSCAN for [%s]  shape=%s", suffix, embeddings.shape)
        log.info("=" * 60)
        run_for_model(
            embeddings   = embeddings,
            suffix       = suffix,
            source_model = source_model,
            components   = args.components,
            n_neighbors  = args.n_neighbors,
            graph_epochs = args.graph_epochs,
            enc_epochs   = args.encoder_epochs,
            batch_size   = args.batch_size,
            mcs          = args.min_cluster_size,
            min_samples  = args.min_samples,
            device       = device,
        )


if __name__ == "__main__":
    main()
