"""
Deep Embedded Clustering (DEC/IDEC) — Method 2.
Pre-trains AE, initialises centroids with K-Means, jointly optimises KL(P||Q) + reconstruction (gamma weight).

Usage:
  python scripts/experiments/deep_clustering/dec_clustering.py
  python scripts/experiments/deep_clustering/dec_clustering.py --model multilingual --k 20
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys

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
from autoencoder_hdbscan import Autoencoder, sdec_loss, train_autoencoder, extract_latent  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def soft_assign(z: torch.Tensor, centroids: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Student-t soft assignment Q_ij (α=1 = Cauchy kernel, Xie et al.). Shape: (N, k)."""
    diff    = z.unsqueeze(1) - centroids.unsqueeze(0)   # (N, k, D)
    sq_dist = (diff ** 2).sum(dim=2)                    # (N, k)
    q       = (1.0 + sq_dist / alpha) ** (-(alpha + 1) / 2.0)
    return q / q.sum(dim=1, keepdim=True)


def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """Sharpens Q into target distribution P — high-confidence points pull stronger."""
    freq = q.sum(dim=0, keepdim=True)    # (1, k) = soft cluster sizes
    p    = (q ** 2) / freq               # (N, k)
    return p / p.sum(dim=1, keepdim=True)


def run_dec(
    model:       Autoencoder,
    embeddings:  np.ndarray,
    k:           int,
    latent:      np.ndarray,
    epochs:      int,
    batch_size:  int,
    device:      torch.device,
    gamma:       float = 0.1,
    tol:         float = 0.001,
    update_freq: int   = 140,
) -> tuple[np.ndarray, np.ndarray]:
    """Fine-tune encoder with DEC/IDEC loss. Returns (final_labels, final_latent)."""
    log.info("Initialising %d centroids with K-Means on %d-D latent …", k, latent.shape[1])
    km = KMeans(n_clusters=k, n_init=20, random_state=RNG)
    km.fit(latent)
    centroids = torch.from_numpy(km.cluster_centers_.astype(np.float32)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    X_t       = torch.from_numpy(embeddings).to(device)
    ds        = TensorDataset(X_t)
    dl        = DataLoader(ds, batch_size=batch_size, shuffle=False)

    prev_labels    = km.labels_.copy()
    global_step    = 0
    converged      = False

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_kl = 0.0

        for (batch,) in dl:
            xhat, z = model(batch)
            q        = soft_assign(z, centroids)

            # update target distribution periodically (requires full-pass)
            if global_step % update_freq == 0:
                with torch.no_grad():
                    all_z  = extract_latent(model, embeddings, batch_size, device)
                    all_z_t = torch.from_numpy(all_z).to(device)
                    q_all   = soft_assign(all_z_t, centroids)
                    p_all   = target_distribution(q_all)
                    new_labels = q_all.argmax(dim=1).cpu().numpy()
                    changed    = (new_labels != prev_labels).mean()
                    log.info("  Step %5d  changed=%.4f (tol=%.4f)", global_step, changed, tol)
                    if changed < tol and global_step > 0:
                        log.info("  Converged (changed=%.4f < tol=%.4f).", changed, tol)
                        converged = True
                    prev_labels = new_labels

            if converged:
                break

            start = (global_step % (len(X_t) // batch_size + 1)) * batch_size
            end   = min(start + batch_size, len(X_t))
            p_batch = p_all[start:end] if 'p_all' in dir() else target_distribution(q)

            kl_loss  = F.kl_div(q.log(), p_batch.detach(), reduction="batchmean")
            rec_loss = sdec_loss(batch, xhat)

            loss = kl_loss + gamma * rec_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_kl    += kl_loss.item() * len(batch)
            global_step += 1

        if epoch % 5 == 0 or epoch == 1:
            log.info("  Epoch %2d/%d  KL=%.6f", epoch, epochs, epoch_kl / len(embeddings))

        if converged:
            break

    # Final latent and labels
    final_latent  = extract_latent(model, embeddings, batch_size, device)
    final_z_t     = torch.from_numpy(final_latent).to(device)
    with torch.no_grad():
        q_final   = soft_assign(final_z_t, centroids)
    final_labels  = q_final.argmax(dim=1).cpu().numpy().astype(np.int32)
    return final_labels, final_latent


def run_dec_for_model(
    embeddings:   np.ndarray,
    suffix:       str,
    source_model: str,
    latent_dim:   int,
    ae_epochs:    int,
    dec_epochs:   int,
    batch_size:   int,
    gamma:        float,
    device:       torch.device,
    k_values:     list[int],
    pretrained:   pathlib.Path | None,
) -> None:
    """Full DEC pipeline for one embedding model."""
    source_embeddings = str(NPY_GEMINI if suffix == "gemini" else NPY_MULTILINGUAL)

    weights_path = (
        pretrained if pretrained is not None
        else MODELS_DIR / f"autoencoder_{latent_dim}d_{suffix}.pt"
    )

    model = train_autoencoder(
        embeddings   = embeddings,
        latent_dim   = latent_dim,
        epochs       = ae_epochs,
        batch_size   = batch_size,
        lr           = 1e-3,
        device       = device,
        weights_path = weights_path,
    )

    log.info("Extracting pre-training latent representations [%s] …", suffix)
    latent_pretrain = extract_latent(model, embeddings, batch_size, device)

    best_sil    = -2.0
    best_labels: np.ndarray | None = None
    best_k      = k_values[0]

    for k in k_values:
        log.info("\n%s\nDEC  k=%d  [%s]\n%s", "=" * 50, k, suffix, "=" * 50)

        labels, final_latent = run_dec(
            model      = model,
            embeddings = embeddings,
            k          = k,
            latent     = latent_pretrain,
            epochs     = dec_epochs,
            batch_size = batch_size,
            device     = device,
            gamma      = gamma,
        )

        n_unique = len(np.unique(labels))
        sc       = sil_cos(embeddings, labels)
        sl       = sil_latent(final_latent, labels)
        dbi, ch  = db_ch(final_latent, labels)

        log.info("k=%d  n_clusters=%d  sil_cos=%.4f  sil_latent=%.4f  DB=%.4f  CH=%.1f",
                 k, n_unique, sc or -1, sl or -1, dbi or -1, ch or -1)

        save_result({
            "method":            f"DEC({latent_dim}D)",
            "method_type":       "deep_learning",
            "latent_dim":        latent_dim,
            "k":                 k,
            "noise_pct":         0.0,
            "sil_cos":           sc,
            "sil_latent":        sl,
            "davies_bouldin":    dbi,
            "calinski_harabasz": ch,
            "n_clusters":        n_unique,
            "source_embeddings": source_embeddings,
            "source_model":      source_model,
        })

        labels_path = DATA_DIR / f"labels_dec_{k}_{suffix}.npy"
        np.save(labels_path, labels)

        if sc is not None and sc > best_sil:
            best_sil    = sc
            best_labels = labels.copy()
            best_k      = k

    log.info("Best DEC k=%d  sil_cos=%.4f  [%s]", best_k, best_sil, suffix)
    if best_labels is not None:
        np.save(DATA_DIR / f"labels_dec_best_{suffix}.npy", best_labels)

    print("\n" + "=" * 60)
    print(f"DEC  (IDEC-style, latent_dim={latent_dim}, gamma={gamma})  [{suffix}]")
    print("=" * 60)
    print(f"  Best k:       {best_k}")
    print(f"  Best sil_cos: {best_sil:.4f}")
    print(f"  Results:      {DATA_DIR / 'deep_clustering_results.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model",      choices=["multilingual", "gemini", "both"],
                        default="both",
                        help="Which embedding model to run (default: both)")
    parser.add_argument("--k",          type=int, default=None,
                        help="Single k (default: sweep 15, 20, 25)")
    parser.add_argument("--pretrained", default=None,
                        help="Path to pretrained autoencoder weights (single model only)")
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--epochs",     type=int, default=30,
                        help="DEC fine-tuning epochs")
    parser.add_argument("--ae-epochs",  type=int, default=50,
                        help="Autoencoder pre-training epochs (if no pretrained weights)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gamma",      type=float, default=0.1,
                        help="IDEC reconstruction weight (0=pure DEC)")
    args = parser.parse_args()

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()         else
        "cpu"
    )
    log.info("Device: %s", device)

    k_values = [args.k] if args.k else [15, 20, 25]
    pretrained = pathlib.Path(args.pretrained) if args.pretrained else None

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
        log.info("Running DEC for [%s]  shape=%s", suffix, embeddings.shape)
        log.info("=" * 60)
        run_dec_for_model(
            embeddings   = embeddings,
            suffix       = suffix,
            source_model = source_model,
            latent_dim   = args.latent_dim,
            ae_epochs    = args.ae_epochs,
            dec_epochs   = args.epochs,
            batch_size   = args.batch_size,
            gamma        = args.gamma,
            device       = device,
            k_values     = k_values,
            pretrained   = pretrained,
        )


if __name__ == "__main__":
    main()
