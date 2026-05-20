"""
Autoencoder + HDBSCAN deep clustering (Method 1).
SDEC loss: 0.5×MSE + 0.5×(1−cosine). Sweeps HDBSCAN mcs, keeps best sil_cos (k < 40).

Usage:
  python scripts/experiments/deep_clustering/autoencoder_hdbscan.py
  python scripts/experiments/deep_clustering/autoencoder_hdbscan.py --model multilingual --latent-dim 8
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
from sklearn.cluster import KMeans
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


class Autoencoder(nn.Module):
    """Symmetric encoder-decoder: D → hidden1 → hidden2 → latent_dim → hidden2 → hidden1 → D. BatchNorm + LeakyReLU."""

    def __init__(self, input_dim: int, latent_dim: int = 10) -> None:
        super().__init__()
        hidden1 = 512 if input_dim >= 2048 else 256
        hidden2 = 128 if input_dim >= 2048 else 64

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden2, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z    = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def sdec_loss(x: torch.Tensor, xhat: torch.Tensor) -> torch.Tensor:
    """Mixed MSE + cosine loss from Rahman et al. 2025 (SDEC paper)."""
    mse_loss = F.mse_loss(xhat, x)
    cos_sim  = F.cosine_similarity(xhat, x, dim=1).mean()
    return 0.5 * mse_loss + 0.5 * (1.0 - cos_sim)


def train_autoencoder(
    embeddings:   np.ndarray,
    latent_dim:   int,
    epochs:       int,
    batch_size:   int,
    lr:           float,
    device:       torch.device,
    weights_path: pathlib.Path,
) -> Autoencoder:
    """Pre-train autoencoder. Returns the trained model."""
    input_dim = embeddings.shape[1]
    model     = Autoencoder(input_dim, latent_dim).to(device)

    if weights_path.exists():
        log.info("Loading existing weights from %s", weights_path)
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X_t = torch.from_numpy(embeddings).to(device)
    ds  = TensorDataset(X_t)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True,
                     generator=torch.Generator().manual_seed(RNG))

    log.info("Training autoencoder: %dD → %dD  epochs=%d  batch=%d  device=%s",
             input_dim, latent_dim, epochs, batch_size, device)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch,) in dl:
            optimizer.zero_grad()
            xhat, _ = model(batch)
            loss     = sdec_loss(batch, xhat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            avg = total_loss / len(embeddings)
            log.info("  Epoch %3d/%d  loss=%.6f  lr=%.2e",
                     epoch, epochs, avg, scheduler.get_last_lr()[0])

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    log.info("Weights saved → %s", weights_path)
    return model


@torch.no_grad()
def extract_latent(
    model:      Autoencoder,
    embeddings: np.ndarray,
    batch_size: int,
    device:     torch.device,
) -> np.ndarray:
    model.eval()
    X_t   = torch.from_numpy(embeddings).to(device)
    ds    = TensorDataset(X_t)
    dl    = DataLoader(ds, batch_size=batch_size, shuffle=False)
    parts = []
    for (batch,) in dl:
        parts.append(model.encode(batch).cpu().numpy())
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
                    "method":            f"Autoencoder({latent_dim}D)+HDBSCAN",
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
    """K-Means sweep on latent space, saved with baseline tag."""
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
            "method":            f"Autoencoder({latent_dim}D)+K-Means(baseline)",
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
    latent_dim:   int,
    epochs:       int,
    batch_size:   int,
    lr:           float,
    device:       torch.device,
    mcs_values:   list[int],
    include_kmeans_baseline: bool,
) -> None:
    """Full pipeline for one embedding model (multilingual or gemini)."""
    source_embeddings = str(NPY_GEMINI if suffix == "gemini" else NPY_MULTILINGUAL)

    weights_path = MODELS_DIR / f"autoencoder_{latent_dim}d_{suffix}.pt"
    latent_path  = DATA_DIR   / f"latent_autoencoder_{latent_dim}d_{suffix}.npy"
    labels_path  = DATA_DIR   / f"labels_ae_hdbscan_{suffix}.npy"

    model = train_autoencoder(
        embeddings   = embeddings,
        latent_dim   = latent_dim,
        epochs       = epochs,
        batch_size   = batch_size,
        lr           = lr,
        device       = device,
        weights_path = weights_path,
    )

    log.info("Extracting %d-D latent representations [%s] …", latent_dim, suffix)
    latent = extract_latent(model, embeddings, batch_size, device)
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

    if include_kmeans_baseline:
        log.info("K-Means baseline sweep [%s] …", suffix)
        kmeans_baseline_sweep(
            embeddings        = embeddings,
            latent            = latent,
            k_values          = [15, 20, 25, 30],
            latent_dim        = latent_dim,
            source_model      = source_model,
            source_embeddings = source_embeddings,
        )

    print("\n" + "=" * 60)
    print(f"AUTOENCODER({latent_dim}D) + HDBSCAN  [{suffix}]")
    print("=" * 60)
    print(f"  Best mcs:          {best_mcs}")
    print(f"  Latent saved:      {latent_path}")
    print(f"  Weights saved:     {weights_path}")
    print(f"  Results:           {DATA_DIR / 'deep_clustering_results.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",      choices=["multilingual", "gemini", "both"],
                        default="both",
                        help="Which embedding model to run (default: both)")
    parser.add_argument("--latent-dim", type=int, default=10,
                        help="Latent space dimensionality (default: 10, must be ≤10)")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--mcs",        type=int, nargs="+", default=[10, 15, 25, 40, 60, 100, 150],
                        help="HDBSCAN min_cluster_size values to sweep")
    parser.add_argument("--include-kmeans-baseline", action="store_true",
                        help="Also run K-Means sweep {15,20,25,30} and save as baseline")
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
        log.info("Running Autoencoder+HDBSCAN for [%s]  shape=%s", suffix, embeddings.shape)
        log.info("=" * 60)
        run_for_model(
            embeddings               = embeddings,
            suffix                   = suffix,
            source_model             = source_model,
            latent_dim               = args.latent_dim,
            epochs                   = args.epochs,
            batch_size               = args.batch_size,
            lr                       = args.lr,
            device                   = device,
            mcs_values               = args.mcs,
            include_kmeans_baseline  = args.include_kmeans_baseline,
        )


if __name__ == "__main__":
    main()
