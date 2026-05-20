-- Embeds conversation chunks. Depends: conversation_chunks | Outputs: chunk embeddings (768-D multilingual, 3072-D gemini)

-- Variant 1: text-multilingual-embedding-002 (768-D) — main pipeline
CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.conversation_chunk_embeddings`
AS
SELECT * FROM ML.GENERATE_EMBEDDING(
  MODEL `support-analytics-492410.support_analytics.embedding_multilingual`,
  (
    SELECT
      chunk_id                     AS chunk_uid,
      conversation_id,
      chunk_seq                    AS chunk_id,
      message_start_idx,
      message_end_idx,
      num_messages                 AS message_count,
      LENGTH(original_chunk_text)  AS chunk_char_count,
      original_chunk_text,
      languages                    AS detected_language,
      min_message_date,
      -- chunk_level_translation.sql was not run; text here is conversation-level from semantic_chunker.py
      TRIM(translated_chunk_text)  AS content
    FROM `support-analytics-492410.support_analytics.conversation_chunks`
    WHERE translated_chunk_text IS NOT NULL
      AND TRIM(translated_chunk_text) != ''
      AND LENGTH(TRIM(translated_chunk_text)) >= 20
  ),
  STRUCT(TRUE AS flatten_json_output)
)
WHERE LENGTH(ml_generate_embedding_status) = 0;


-- Variant 2: gemini-embedding-001, task_type=CLUSTERING (3072-D) — run separately in BQ console
CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.conversation_chunk_embeddings_gemini`
AS
SELECT * FROM ML.GENERATE_EMBEDDING(
  MODEL `support-analytics-492410.support_analytics.gemini_embedding`,
  (
    SELECT
      chunk_id                    AS chunk_uid,
      conversation_id,
      chunk_seq                   AS chunk_id,
      message_start_idx,
      message_end_idx,
      num_messages                AS message_count,
      LENGTH(original_chunk_text) AS chunk_char_count,
      original_chunk_text,
      languages                   AS detected_language,
      min_message_date,
      TRIM(translated_chunk_text) AS content
    FROM `support-analytics-492410.support_analytics.conversation_chunks`
    WHERE translated_chunk_text IS NOT NULL
      AND TRIM(translated_chunk_text) != ''
      AND LENGTH(TRIM(translated_chunk_text)) >= 20
  ),
  STRUCT(
    TRUE         AS flatten_json_output,
    'CLUSTERING' AS task_type,
    3072         AS output_dimensionality
  )
)
WHERE LENGTH(ml_generate_embedding_status) = 0;


SELECT
  COUNT(*)                                        AS total_chunks,
  COUNT(DISTINCT conversation_id)                 AS unique_conversations,
  MAX(ARRAY_LENGTH(ml_generate_embedding_result)) AS embedding_dims,
  ROUND(AVG(LENGTH(content)), 0)                  AS avg_content_chars
FROM `support-analytics-492410.support_analytics.conversation_chunk_embeddings_gemini`;


SELECT
  COUNT(*)                                        AS total_chunks,
  COUNT(DISTINCT conversation_id)                 AS unique_conversations,
  MAX(ARRAY_LENGTH(ml_generate_embedding_result)) AS embedding_dims,
  ROUND(AVG(LENGTH(content)), 0)                  AS avg_content_chars,
  ROUND(APPROX_QUANTILES(LENGTH(content), 100)[OFFSET(50)], 0) AS p50_chars,
  ROUND(APPROX_QUANTILES(LENGTH(content), 100)[OFFSET(95)], 0) AS p95_chars
FROM `support-analytics-492410.support_analytics.conversation_chunk_embeddings`;
