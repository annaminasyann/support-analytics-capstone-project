#!/usr/bin/env bash
# Documents the exact commands used to produce all results in the paper.
#
# Note: this script cannot be run without access to the BigQuery dataset
# support-analytics-492410.support_analytics, which contains proprietary
# 10Web customer support data. Access is granted by the project supervisor,
# see data/raw_data/DATA_ACCESS.md for details.
#
# If you do have access, the pipeline was developed and run on Google Cloud
# Vertex AI Workbench (europe-west1), where credentials are pre-configured.
# On a local machine you would also need:
#   gcloud auth application-default login
#   conda activate capstone-nlp
#   bash reproduce.sh

set -euo pipefail

echo "Step 1: Initial clustering on base embeddings"
python scripts/chunk_clustering.py \
  --umap-components 20 \
  --min-cluster-size 2000 \
  --min-samples 15 \
  --selection-method leaf

echo "Step 2: Contrastive fine-tuning round 1"
python scripts/contrastive_finetune.py \
  --positives-per-chunk 50 \
  --hard-negatives 10 \
  --easy-negatives 10 \
  --epochs 3 \
  --re-embed

echo "Step 3: Re-clustering on round-1 embeddings"
python scripts/chunk_clustering.py \
  --embed-suffix finetuned \
  --skip-embed \
  --umap-components 20 \
  --min-cluster-size 2000 \
  --min-samples 15 \
  --selection-method leaf

echo "Step 4: Removing round-1 model before round 2"
rm -rf data/models/finetuned_embedder

echo "Step 5: Contrastive fine-tuning round 2"
python scripts/contrastive_finetune.py \
  --positives-per-chunk 50 \
  --hard-negatives 10 \
  --easy-negatives 10 \
  --epochs 3 \
  --re-embed

echo "Step 6: Final clustering on round-2 embeddings"
python scripts/chunk_clustering.py \
  --embed-suffix finetuned \
  --skip-embed \
  --umap-components 10 \
  --min-cluster-size 2000 \
  --min-samples 20 \
  --selection-method leaf \
  --confidence-threshold 0.70

echo "Step 7: Failure analysis, trends, and cluster labels"
bq query --use_legacy_sql=false < bq_pipeline/05_failure_analysis.sql
bq query --use_legacy_sql=false < bq_pipeline/06_trends.sql
bq query --use_legacy_sql=false < bq_pipeline/07_chunk_evaluation.sql

echo "Step 8: Running notebooks"
jupyter nbconvert --to notebook --execute notebooks/01_baseline_clustering.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_explore_clusters.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_cluster_deep_dive.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_failure_analysis.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_trend_analysis.ipynb
jupyter nbconvert --to notebook --execute notebooks/06_evaluation.ipynb
jupyter nbconvert --to notebook --execute notebooks/07_forecasting.ipynb

echo "Step 9: Compiling the paper"
cd paper && tectonic main.tex

echo "Done. Paper is at paper/main.pdf"
