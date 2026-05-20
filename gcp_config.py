"""GCP configuration for the support-analytics capstone project."""

PROJECT_ID  = "support-analytics-492410"
DATASET_ID  = "support_analytics"
BUCKET_NAME = "support-analytics-492410-data"

BQ_CONNECTION = f"{PROJECT_ID}.eu.vertex_ai_connection"

MODEL_TRANSLATION       = "translation_model"
MODEL_EMBEDDING         = "embedding_multilingual"
MODEL_GEMINI_EMBEDDING  = "gemini_embedding"
MODEL_GEMINI            = "gemini_model"

TABLE_CHATBOT_MESSAGES_RAW          = "chatbot_messages_raw"
TABLE_CHATBOT_MESSAGES_DEIDENTIFIED = "chatbot_messages_deidentified"
TABLE_CONVERSATION_DOCS              = "conversation_docs"
TABLE_TRANSLATION_LOG                = "translation_log"
TABLE_EMBEDDINGS_MULTILINGUAL        = "conversation_embeddings_multilingual"
TABLE_EMBEDDINGS_TRANSLATED          = "conversation_embeddings_translated"
TABLE_CLUSTER_LABELS                 = "cluster_labels"
TABLE_EVALUATION_METRICS             = "evaluation_metrics"
TABLE_MODEL_COMPARISON               = "model_comparison_results"
TABLE_FAILURE_FLAGS                  = "failure_flags"
TABLE_FAILURE_BY_CLUSTER             = "failure_by_cluster"
TABLE_FAILURE_BY_CHUNK_CLUSTER       = "failure_by_chunk_cluster"
TABLE_FAILURE_WEEKLY                 = "failure_weekly"
TABLE_WEEKLY_VOLUME                  = "weekly_volume"
TABLE_CLUSTER_TRENDS                 = "cluster_trends"
TABLE_CHUNK_CLUSTER_TRENDS           = "chunk_cluster_trends"
TABLE_LANGUAGE_TRENDS                = "language_trends"
TABLE_CLUSTER_LABEL_TABLE            = "cluster_label_table"
TABLE_CLUSTER_SIZE_COMPARISON        = "cluster_size_comparison"
TABLE_CLUSTER_PERSISTENCE            = "cluster_persistence"
TABLE_CLUSTER_STABILITY_ARI          = "cluster_stability_ari"
TABLE_HDBSCAN_SWEEP_RESULTS          = "hdbscan_sweep_results"
TABLE_CROSS_TRACK_ARI                = "cross_track_ari"

TABLE_CONVERSATION_MESSAGE_CHUNKS   = "conversation_message_chunks"
TABLE_CONVERSATION_CHUNK_EMBEDDINGS = "conversation_chunk_embeddings"
TABLE_CHUNK_CLUSTER_LABELS          = "chunk_cluster_labels"
TABLE_CONVERSATION_CHUNK_TOPICS     = "conversation_chunk_topics"
TABLE_CHUNK_CLUSTER_METRICS         = "chunk_cluster_metrics"
TABLE_CHUNK_HDBSCAN_SWEEP           = "chunk_hdbscan_sweep"
TABLE_CHUNK_CLUSTER_LABEL_TABLE     = "chunk_cluster_label_table"
TABLE_CHUNK_CLUSTER_DESCRIPTIONS    = "chunk_cluster_descriptions"


def bq_table(table_name: str) -> str:
    return f"{PROJECT_ID}.{DATASET_ID}.{table_name}"


def bq_model(model_name: str) -> str:
    return f"{PROJECT_ID}.{DATASET_ID}.{model_name}"


def gcs_path(relative_path: str) -> str:
    return f"gs://{BUCKET_NAME}/{relative_path}"
