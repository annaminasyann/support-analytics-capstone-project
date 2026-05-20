"""
Gaussian-Bernoulli RBM + HDBSCAN deep clustering (Method 3).
Trains GBRBM via CD-1, extracts mean-field hidden activations, clusters with HDBSCAN.

Usage:
  python scripts/experiments/deep_clustering/rbm_hdbscan.py
  python scripts/experiments/deep_clustering/rbm_hdbscan.py --model multilingual --hidden-dim 8
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
import torch.nn.functional as F  # noqa: F401 (used in reconstruction_error)
from sklearn.cluster import KMeans
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


class GBRBM(nn.Module):
    """Gaussian-Bernoulli RBM for continuous visible units. Trained with CD-1."""

    def __init__(self, visible_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.W   = nn.Parameter(torch.randn(visible_dim, hidden_dim) * 0.01)
        self.b_v = nn.Parameter(torch.zeros(visible_dim))
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

    def p_h_given_v(self, v: torch.Tensor) -> torch.Tensor:
        """P(h=1|v) = sigmoid(Wᵀv + b_h). Shape: (batch, hidden_dim)."""
        return torch.sigmoid(v @ self.W + self.b_h)

    def sample_h(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample h ~ Bernoulli(P(h=1|v)). Returns (h_mean, h_sample)."""
        p   = self.p_h_given_v(v)
        h_s = torch.bernoulli(p)
        return p, h_s

    def p_v_mean_given_h(self, h: torch.Tensor) -> torch.Tensor:
        """E[v|h] = Wh + b_v (Gaussian mean, unit variance)."""
        return h @ self.W.T + self.b_v

    def sample_v(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample v ~ N(Wh + b_v, I). Returns (v_mean, v_sample)."""
        v_mean = self.p_v_mean_given_h(h)
        v_s    = v_mean + torch.randn_like(v_mean)
        return v_mean, v_s

    def cd1_gradients(self, v_data: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute Contrastive Divergence CD-1 gradients."""
        h_data_mean, h_data_s    = self.sample_h(v_data)
        v_model_mean, v_model_s  = self.sample_v(h_data_s)
        h_model_mean, _          = self.sample_h(v_model_s)

        batch = v_data.size(0)
        dW    = (v_data.T @ h_data_mean - v_model_s.T @ h_model_mean) / batch
        dbv   = (v_data - v_model_s).mean(0)
        dbh   = (h_data_mean - h_model_mean).mean(0)

        return {"W": dW, "b_v": dbv, "b_h": dbh}

    def reconstruction_error(self, v: torch.Tensor) -> torch.Tensor:
        """Mean squared reconstruction error (diagnostic, not used for training)."""
        _, h_s        = self.sample_h(v)
        v_rec_mean, _ = self.sample_v(h_s)
        return ((v - v_rec_mean) ** 2).mean()

    @torch.no_grad()
    def hidden_mean(self, v: torch.Tensor) -> torch.Tensor:
        """Deterministic mean-field hidden activations."""
        return self.p_h_given_v(v)


def train_rbm(
    embeddings:   np.ndarray,
    hidden_dim:   int,
    epochs:       int,
    batch_size:   int,
    lr:           float,
    device:       torch.device,
    weights_path: pathlib.Path,
) -> GBRBM:
    """Train GBRBM with CD-1.  Returns the trained model."""
    visible_dim = embeddings.shape[1]
    rbm         = GBRBM(visible_dim, hidden_dim).to(device)

    if weights_path.exists():
        log.info("Loading existing RBM weights from %s", weights_path)
        state = torch.load(weights_path, map_location=device, weights_only=True)
        rbm.load_state_dict(state)
        return rbm

    X_t = torch.from_numpy(embeddings).to(device)
    ds  = TensorDataset(X_t)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True,
                     generator=torch.Generator().manual_seed(RNG))

    log.info("Training GBRBM: %dD visible → %dD hidden  epochs=%d  lr=%.4f  device=%s",
             visible_dim, hidden_dim, epochs, lr, device)

    for epoch in range(1, epochs + 1):
        total_err = 0.0
        for (batch,) in dl:
            grads = rbm.cd1_gradients(batch)
            with torch.no_grad():
                rbm.W   += lr * grads["W"]
                rbm.b_v += lr * grads["b_v"]
                rbm.b_h += lr * grads["b_h"]
            with torch.no_grad():
                err = rbm.reconstruction_error(batch)
            total_err += err.item() * len(batch)

        if epoch % 10 == 0 or epoch == 1:
            log.info("  Epoch %3d/%d  recon_MSE=%.6f", epoch, epochs, total_err / len(embeddings))

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rbm.state_dict(), weights_path)
    log.info("RBM weights saved → %s", weights_path)
    return rbm


@torch.no_grad()
def extract_hidden(
    rbm:        GBRBM,
    embeddings: np.ndarray,
    batch_size: int,
    device:     torch.device,
) -> np.ndarray:
    rbm.eval()
    X_t   = torch.from_numpy(embeddings).to(device)
    ds    = TensorDataset(X_t)
    dl    = DataLoader(ds, batch_size=batch_size, shuffle=False)
    parts = []
    for (batch,) in dl:
        parts.append(rbm.hidden_mean(batch).cpu().numpy())
    return np.concatenate(parts, axis=0)


def hdbscan_sweep(
    embeddings:        np.ndarray,
    latent:            np.ndarray,
    mcs_values:        list[int],
    latent_dim:        int,
    source_model:      str,
    source_embeddings: str,
) -> tuple[np.ndarray | None, int]:
    """Sweep HDBSCAN mcs values. Returns (best_labels, best_mcs) by sil_cos with k < 40. Doubles mcs and retries if needed."""
    best_labels: np.ndarray | None = None
    best_sil    = -2.0
    best_mcs    = mcs_values[0]
    tried: list[int] = list(mcs_values)
    labels: np.ndarray = np.zeros(len(latent), dtype=np.int32)

    round_count = 0
    current_mcs_values = list(mcs_values)

    while True:
        found_valid = False
        for mcs in current_mcs_values:
            if mcs not in tried:
                tried.append(mcs)

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
                log.warning("  mcs=%d produced only %d cluster(s), skipping.", mcs, n_clusters)
                continue

            sc      = sil_cos(embeddings, labels)
            sl      = sil_latent(latent, labels)
            dbi, ch = db_ch(latent, labels)

            log.info("  mcs=%d  k=%d  sil_cos=%.4f  sil_latent=%.4f  noise=%.1f%%",
                     mcs, n_clusters, sc or -1, sl or -1, noise_pct)

            if n_clusters >= 40:
                log.info("  mcs=%d k=%d ≥ 40 — excluded from primary results (logged only).", mcs, n_clusters)
            else:
                save_result({
                    "method":            f"RBM({latent_dim}D)+HDBSCAN",
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

                found_valid = True
                if sc is not None and sc > best_sil:
                    best_sil    = sc
                    best_labels = labels.copy()
                    best_mcs    = mcs

        if found_valid or round_count >= 3:
            break

        round_count += 1
        next_mcs = tried[-1] * 2
        log.warning("No mcs gave k < 40. Retrying with mcs=%d", next_mcs)
        tried.append(next_mcs)
        current_mcs_values = [next_mcs]

    if best_labels is None:
        log.error("Could not find any HDBSCAN config with k < 40. Returning last labels.")
        best_labels = labels
    log.info("Best mcs=%d  sil_cos=%.4f", best_mcs, best_sil)
    return best_labels, best_mcs


def kmeans_baseline_sweep(
    embeddings:        np.ndarray,
    latent:            np.ndarray,
    k_values:          list[int],
    latent_dim:        int,
    source_model:      str,
    source_embeddings: str,
) -> None:
    """Run K-Means sweep and save results with baseline tag."""
    for k in k_values:
        log.info("K-Means baseline k=%d …", k)
        km     = KMeans(n_clusters=k, n_init=20, random_state=RNG)
        labels = km.fit_predict(latent).astype(np.int32)

        sc      = sil_cos(embeddings, labels)
        sl      = sil_latent(latent, labels)
        dbi, ch = db_ch(latent, labels)

        log.info("  k=%2d  sil_cos=%.4f  sil_latent=%.4f  DB=%.4f  CH=%.1f",
                 k, sc or -1, sl or -1, dbi or -1, ch or -1)

        save_result({
            "method":            f"RBM({latent_dim}D)+K-Means(baseline)",
            "method_type":       "deep_learning",
            "latent_dim":        latent_dim,
            "k":                 k,
            "noise_pct":         0.0,
            "sil_cos":           sc,
            "sil_latent":        sl,
            "davies_bouldin":    dbi,
            "calinski_harabasz": ch,
            "n_clusters":        k,
            "source_embeddings": source_embeddings,
            "source_model":      source_model,
        })


def run_for_model(
    embeddings:   np.ndarray,
    suffix:       str,
    source_model: str,
    hidden_dim:   int,
    epochs:       int,
    batch_size:   int,
    lr:           float,
    device:       torch.device,
    mcs_values:   list[int],
    include_kmeans_baseline: bool,
) -> None:
    """Full RBM+HDBSCAN pipeline for one embedding model."""
    source_embeddings = str(NPY_GEMINI if suffix == "gemini" else NPY_MULTILINGUAL)

    weights_path = MODELS_DIR / f"rbm_{hidden_dim}d_{suffix}.pt"
    latent_path  = DATA_DIR   / f"latent_rbm_{hidden_dim}d_{suffix}.npy"
    labels_path  = DATA_DIR   / f"labels_rbm_hdbscan_{suffix}.npy"

    rbm = train_rbm(
        embeddings   = embeddings,
        hidden_dim   = hidden_dim,
        epochs       = epochs,
        batch_size   = batch_size,
        lr           = lr,
        device       = device,
        weights_path = weights_path,
    )

    log.info("Extracting hidden activations [%s] …", suffix)
    hidden = extract_hidden(rbm, embeddings, batch_size, device)
    np.save(latent_path, hidden)
    log.info("Hidden activations saved → %s  shape=%s", latent_path, hidden.shape)
    log.info("  Hidden mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
             hidden.mean(), hidden.std(), hidden.min(), hidden.max())

    log.info("HDBSCAN sweep [%s]  mcs=%s …", suffix, mcs_values)
    best_labels, best_mcs = hdbscan_sweep(
        embeddings        = embeddings,
        latent            = hidden,
        mcs_values        = mcs_values,
        latent_dim        = hidden_dim,
        source_model      = source_model,
        source_embeddings = source_embeddings,
    )

    np.save(labels_path, best_labels)
    log.info("Best labels (mcs=%d) saved → %s", best_mcs, labels_path)

    if include_kmeans_baseline:
        log.info("K-Means baseline sweep [%s] …", suffix)
        kmeans_baseline_sweep(
            embeddings        = embeddings,
            latent            = hidden,
            k_values          = [15, 20, 25, 30],
            latent_dim        = hidden_dim,
            source_model      = source_model,
            source_embeddings = source_embeddings,
        )

    print("\n" + "=" * 60)
    print(f"RBM (Gaussian-Bernoulli, Hinton & Sejnowski 1986)  [{suffix}]  hidden_dim={hidden_dim}")
    print("=" * 60)
    print(f"  Best mcs:   {best_mcs}")
    print(f"  Latent:     {latent_path}")
    print(f"  Weights:    {weights_path}")
    print(f"  Results:    {DATA_DIR / 'deep_clustering_results.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",      choices=["multilingual", "gemini", "both"],
                        default="both",
                        help="Which embedding model to run (default: both)")
    parser.add_argument("--hidden-dim", type=int, default=10,
                        help="Number of hidden units (≤10, default: 10)")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr",         type=float, default=0.01)
    parser.add_argument("--mcs",        type=int, nargs="+", default=[10, 15, 25, 40, 60, 100, 150],
                        help="HDBSCAN min_cluster_size values to sweep")
    parser.add_argument("--include-kmeans-baseline", action="store_true",
                        help="Also run K-Means sweep {15,20,25,30} and save as baseline")
    args = parser.parse_args()

    if args.hidden_dim > 10:
        parser.error("--hidden-dim must be ≤10 (deep-learning gap requirement)")

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
        log.info("Running RBM+HDBSCAN for [%s]  shape=%s", suffix, embeddings.shape)
        log.info("=" * 60)
        run_for_model(
            embeddings               = embeddings,
            suffix                   = suffix,
            source_model             = source_model,
            hidden_dim               = args.hidden_dim,
            epochs                   = args.epochs,
            batch_size               = args.batch_size,
            lr                       = args.lr,
            device                   = device,
            mcs_values               = args.mcs,
            include_kmeans_baseline  = args.include_kmeans_baseline,
        )


if __name__ == "__main__":
    main()
