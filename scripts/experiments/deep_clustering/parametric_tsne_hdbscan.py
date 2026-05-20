"""
Parametric t-SNE (TCNE) + HDBSCAN deep clustering (Method 6).
KL divergence with Student-t kernel instead of UMAP cross-entropy — tighter clusters at low dim.

Usage:
  python scripts/experiments/deep_clustering/parametric_tsne_hdbscan.py
  python scripts/experiments/deep_clustering/parametric_tsne_hdbscan.py --model multilingual --latent-dim 5
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

import hdbscan
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

_HERE     = pathlib.Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(_HERE))

from _common import (            # noqa: E402
    DATA_DIR,
    MODELS_DIR,
    RNG,
    db_ch,
    load_both_embeddings,
    load_embeddings,
    NPY_MULTILINGUAL,
    NPY_GEMINI,
    save_result,
    sil_cos,
    sil_latent,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


class TCNEEncoder(nn.Module):
    """Feed-forward encoder for t-SNE / TCNE embedding."""

    def __init__(self, input_dim: int, latent_dim: int = 5) -> None:
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
            nn.Linear(hidden2, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_knn_graph(
    embeddings: np.ndarray,
    n_neighbors: int,
    perplexity: float,
    cache_path: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build kNN graph with Gaussian-kernel P_ij probabilities. Returns (src, dst, p). Caches to .npz."""
    if cache_path.exists():
        log.info("Loading cached kNN graph from %s", cache_path)
        data = np.load(cache_path)
        return data["src"], data["dst"], data["p"]

    log.info("Building kNN graph  n=%d  k=%d  perplexity=%.1f …",
             len(embeddings), n_neighbors, perplexity)
    nn_model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine",
                                algorithm="auto", n_jobs=-1)
    nn_model.fit(embeddings)
    distances, indices = nn_model.kneighbors(embeddings)

    distances = distances[:, 1:]   # drop self
    indices   = indices[:, 1:]

    # sigma per point chosen so Shannon entropy = log(perplexity)
    N, k = distances.shape
    p_conditional = np.zeros((N, k), dtype=np.float32)
    for i in range(N):
        d = distances[i].astype(np.float64)
        # Binary search for sigma that gives target perplexity
        target = np.log(perplexity)
        lo, hi = 1e-10, 1e10
        for _ in range(50):
            sigma = (lo + hi) / 2.0
            w = np.exp(-d ** 2 / (2 * sigma ** 2))
            s = w.sum()
            if s == 0:
                s = 1e-30
            p = w / s
            p = np.clip(p, 1e-30, 1.0)
            h = -np.sum(p * np.log(p))
            if h < target:
                lo = sigma
            else:
                hi = sigma
        p_conditional[i] = w / w.sum()

    # p_ij = (p_j|i + p_i|j) / (2N)
    src_list, dst_list, p_list = [], [], []
    for i in range(N):
        for ki in range(k):
            j    = indices[i, ki]
            p_ij = (p_conditional[i, ki] + p_conditional[j, np.where(indices[j] == i)[0][0]]
                    if i in indices[j] else p_conditional[i, ki]) / (2 * N)
            src_list.append(i)
            dst_list.append(j)
            p_list.append(max(p_ij, 1e-30))

    src = np.array(src_list, dtype=np.int32)
    dst = np.array(dst_list, dtype=np.int32)
    p   = np.array(p_list,   dtype=np.float32)

    np.savez_compressed(cache_path, src=src, dst=dst, p=p)
    log.info("kNN graph saved → %s  edges=%d", cache_path, len(src))
    return src, dst, p


def tsne_loss(
    z: torch.Tensor,
    edge_src: torch.Tensor,
    edge_dst: torch.Tensor,
    p_ij:     torch.Tensor,
    n_neg:    int = 5,
    lam:      float = 1.0,
) -> torch.Tensor:
    """Mini-batch KL(P||Q) loss. q_ij = (1+||zi-zj||²)^(-1). Attractive + λ×repulsive terms."""
    zi = z[edge_src]
    zj = z[edge_dst]
    dist_sq_pos = ((zi - zj) ** 2).sum(dim=1)
    q_pos       = 1.0 / (1.0 + dist_sq_pos)
    loss_attract = -(p_ij * torch.log(q_pos + 1e-10)).mean()

    n = z.shape[0]
    neg_i   = torch.randint(0, n, (n * n_neg,), device=z.device)
    neg_j   = torch.randint(0, n, (n * n_neg,), device=z.device)
    dist_sq_neg = ((z[neg_i] - z[neg_j]) ** 2).sum(dim=1)
    q_neg       = 1.0 / (1.0 + dist_sq_neg)
    loss_repel  = -torch.log(1.0 - q_neg + 1e-10).mean()

    return loss_attract + lam * loss_repel


def train_tcne(
    embeddings:   np.ndarray,
    edge_src:     np.ndarray,
    edge_dst:     np.ndarray,
    p_ij:         np.ndarray,
    latent_dim:   int,
    epochs:       int,
    batch_size:   int,
    lr:           float,
    lam:          float,
    device:       torch.device,
    weights_path: pathlib.Path,
) -> TCNEEncoder:
    """Train TCNE encoder. Loads existing weights if available (idempotent)."""

    encoder = TCNEEncoder(embeddings.shape[1], latent_dim).to(device)

    if weights_path.exists():
        log.info("Loading existing weights from %s — skipping training.", weights_path)
        encoder.load_state_dict(torch.load(weights_path, map_location=device))
        encoder.eval()
        return encoder

    X_t   = torch.tensor(embeddings, dtype=torch.float32, device=device)
    src_t = torch.tensor(edge_src, dtype=torch.long)
    dst_t = torch.tensor(edge_dst, dtype=torch.long)
    p_t   = torch.tensor(p_ij,     dtype=torch.float32)

    edge_ds  = TensorDataset(src_t, dst_t, p_t)
    edge_dl  = DataLoader(edge_ds, batch_size=batch_size, shuffle=True,
                          drop_last=True, pin_memory=(device.type == "cuda"))

    opt   = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=epochs, eta_min=0.0)

    encoder.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for src_b, dst_b, p_b in edge_dl:
            src_b = src_b.to(device)
            dst_b = dst_b.to(device)
            p_b   = p_b.to(device)

            z  = encoder(X_t)
            loss = tsne_loss(z, src_b, dst_b, p_b, lam=lam)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()

        sched.step()
        if epoch % 10 == 0 or epoch == 1:
            log.info("  Epoch %3d/%d  loss=%.6f  lr=%.2e",
                     epoch, epochs, epoch_loss / len(edge_dl),
                     sched.get_last_lr()[0])

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), weights_path)
    log.info("Encoder saved → %s", weights_path)
    encoder.eval()
    return encoder


def project(
    encoder:    TCNEEncoder,
    embeddings: np.ndarray,
    batch_size: int,
    device:     torch.device,
) -> np.ndarray:
    """Project all embeddings to latent space in batches."""
    encoder.eval()
    X_t  = torch.tensor(embeddings, dtype=torch.float32)
    dl   = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in dl:
            parts.append(encoder(xb.to(device)).cpu().numpy())
    return np.vstack(parts)


def hdbscan_sweep(
    embeddings:        np.ndarray,
    latent:            np.ndarray,
    mcs_values:        list[int],
    latent_dim:        int,
    source_model:      str,
    source_embeddings: str,
) -> tuple[np.ndarray | None, int]:
    """Sweep HDBSCAN mcs values. Returns (best_labels, best_mcs) by sil_cos with k < 40."""

    best_labels: np.ndarray | None = None
    best_sil    = -2.0
    best_mcs    = mcs_values[0]
    labels      = np.zeros(len(latent), dtype=np.int32)

    for mcs in mcs_values:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size        = mcs,
            min_samples             = 5,
            metric                  = "euclidean",
            cluster_selection_method= "leaf",
            prediction_data         = True,
        )
        labels     = clusterer.fit_predict(latent).astype(np.int32)
        n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
        noise_pct  = round(100.0 * (labels == -1).sum() / len(labels), 2)

        log.info("  HDBSCAN mcs=%d → k=%d  noise=%.1f%%", mcs, n_clusters, noise_pct)

        if n_clusters < 2:
            continue

        sc      = sil_cos(embeddings, labels)
        sl      = sil_latent(latent, labels)
        dbi, ch = db_ch(latent, labels)

        log.info("  mcs=%d  k=%d  sil_cos=%.4f  sil_latent=%.4f  noise=%.1f%%",
                 mcs, n_clusters, sc or -1, sl or -1, noise_pct)

        if n_clusters < 40:
            save_result({
                "method":            f"TCNE({latent_dim}D)+HDBSCAN",
                "method_type":       "deep_learning",
                "latent_dim":        latent_dim,
                "k":                 n_clusters,
                "noise_pct":         noise_pct,
                "sil_cos":           sc,
                "sil_latent":        sl,
                "davies_bouldin":    dbi,
                "calinski_harabasz": ch,
                "n_clusters":        n_clusters,
                "source_embeddings": source_embeddings,
                "source_model":      source_model,
            })
            if sc is not None and sc > best_sil:
                best_sil    = sc
                best_labels = labels.copy()
                best_mcs    = mcs

    if best_labels is None:
        log.warning("No mcs gave k < 40 with k >= 2. Returning best available.")
        best_labels = labels
    log.info("Best mcs=%d  sil_cos=%.4f", best_mcs, best_sil)
    return best_labels, best_mcs


def run_for_model(
    embeddings:   np.ndarray,
    suffix:       str,
    source_model: str,
    latent_dim:   int,
    n_neighbors:  int,
    perplexity:   float,
    epochs:       int,
    batch_size:   int,
    lr:           float,
    lam:          float,
    mcs_values:   list[int],
    device:       torch.device,
) -> None:

    source_embeddings = str(NPY_GEMINI if suffix == "gemini" else NPY_MULTILINGUAL)
    model_dir         = MODELS_DIR / f"tcne_{latent_dim}d_{suffix}"
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path  = model_dir    / "encoder.pt"
    latent_path   = DATA_DIR     / f"latent_tcne_{latent_dim}d_{suffix}.npy"
    labels_path   = DATA_DIR     / f"labels_tcne_{suffix}.npy"
    graph_path    = DATA_DIR     / f"tcne_graph_{suffix}.npz"

    log.info("=" * 60)
    log.info("TCNE (%dD) + HDBSCAN  [%s]  n=%d  D=%d",
             latent_dim, suffix, len(embeddings), embeddings.shape[1])
    log.info("=" * 60)

    edge_src, edge_dst, p_ij = build_knn_graph(
        embeddings, n_neighbors, perplexity, graph_path
    )

    encoder = train_tcne(
        embeddings   = embeddings,
        edge_src     = edge_src,
        edge_dst     = edge_dst,
        p_ij         = p_ij,
        latent_dim   = latent_dim,
        epochs       = epochs,
        batch_size   = batch_size,
        lr           = lr,
        lam          = lam,
        device       = device,
        weights_path = weights_path,
    )

    log.info("Projecting %d conversations to %d-D [%s] …", len(embeddings), latent_dim, suffix)
    latent = project(encoder, embeddings, batch_size, device)
    np.save(latent_path, latent)
    log.info("Latent saved → %s  shape=%s", latent_path, latent.shape)

    log.info("HDBSCAN sweep [%s]  mcs=%s …", suffix, mcs_values)
    best_labels, best_mcs = hdbscan_sweep(
        embeddings        = embeddings,
        latent            = latent,
        mcs_values        = mcs_values,
        latent_dim        = latent_dim,
        source_model      = source_model,
        source_embeddings = source_embeddings,
    )

    np.save(labels_path, best_labels)
    log.info("Best labels (mcs=%d) saved → %s", best_mcs, labels_path)

    n_cl  = int(len(set(best_labels)) - (1 if -1 in best_labels else 0))
    noise = round(100.0 * (best_labels == -1).sum() / len(best_labels), 2)
    sc    = sil_cos(embeddings, best_labels)

    print("\n" + "=" * 60)
    print(f"TCNE ({latent_dim}D) + HDBSCAN  [{suffix}]")
    print("=" * 60)
    print(f"  Best mcs:     {best_mcs}")
    print(f"  n_clusters:   {n_cl}")
    print(f"  noise_pct:    {noise:.1f}%")
    print(f"  sil_cos:      {sc}")
    print(f"  Encoder:      {weights_path}")
    print(f"  Latent:       {latent_path}")
    print(f"  Results:      {DATA_DIR / 'deep_clustering_results.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",        choices=["multilingual", "gemini", "both"],
                        default="both",
                        help="Which embedding model to run (default: both)")
    parser.add_argument("--latent-dim",   type=int,   default=5,
                        help="Latent dimensionality (default: 5)")
    parser.add_argument("--n-neighbors",  type=int,   default=15,
                        help="kNN graph neighbors (default: 15)")
    parser.add_argument("--perplexity",   type=float, default=30.0,
                        help="t-SNE perplexity for P_ij (default: 30)")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=1024)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--lam",          type=float, default=1.0,
                        help="Repulsive loss weight λ (default: 1.0)")
    parser.add_argument("--mcs",          type=int, nargs="+",
                        default=[10, 15, 25, 40, 60, 100, 150],
                        help="HDBSCAN min_cluster_size sweep")
    args = parser.parse_args()

    if args.latent_dim > 50:
        parser.error("--latent-dim must be ≤50")

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()         else
        "cpu"
    )
    log.info("Device: %s", device)

    if args.model == "both":
        emb_dict = load_both_embeddings()
        entries  = [
            ("multilingual", emb_dict.get("multilingual"), "multilingual-768D"),
            ("gemini",       emb_dict.get("gemini"),       "gemini-3072D"),
        ]
    elif args.model == "multilingual":
        emb     = load_embeddings(NPY_MULTILINGUAL)
        entries = [("multilingual", emb, "multilingual-768D")]
    else:
        emb     = load_embeddings(NPY_GEMINI)
        entries = [("gemini", emb, "gemini-3072D")]

    for suffix, embeddings, source_model in entries:
        if embeddings is None:
            log.warning("Skipping [%s] — embeddings not available.", suffix)
            continue
        run_for_model(
            embeddings   = embeddings,
            suffix       = suffix,
            source_model = source_model,
            latent_dim   = args.latent_dim,
            n_neighbors  = args.n_neighbors,
            perplexity   = args.perplexity,
            epochs       = args.epochs,
            batch_size   = args.batch_size,
            lr           = args.lr,
            lam          = args.lam,
            mcs_values   = args.mcs,
            device       = device,
        )


if __name__ == "__main__":
    main()
