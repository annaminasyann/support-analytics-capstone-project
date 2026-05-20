-- Not executed. Requires chunk_level_translation.sql to run first.
-- Proposed: 0.7 × translated chunk embeddings + 0.3 × original chunk embeddings

CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.conversation_chunk_embeddings_original`
AS
SELECT * FROM ML.GENERATE_EMBEDDING(
  MODEL `support-analytics-492410.support_analytics.embedding_multilingual`,
  (
    SELECT
      -- alias must match 04_chunk_embeddings.sql so chunk_clustering.py --combine-embeddings can join on chunk_uid
      chunk_id                    AS chunk_uid,
      conversation_id,
      chunk_seq                   AS chunk_id,
      languages                   AS detected_language,
      message_start_idx,
      message_end_idx,
      num_messages                AS message_count,
      original_chunk_text         AS content
    FROM `support-analytics-492410.support_analytics.conversation_chunks`
    WHERE original_chunk_text IS NOT NULL
      AND TRIM(original_chunk_text) != ''
      AND LENGTH(TRIM(original_chunk_text)) >= 20
  ),
  STRUCT(TRUE AS flatten_json_output)
)
WHERE LENGTH(ml_generate_embedding_status) = 0;


SELECT
  COUNT(*)                                        AS total_chunks,
  COUNT(DISTINCT conversation_id)                 AS unique_conversations,
  MAX(ARRAY_LENGTH(ml_generate_embedding_result)) AS embedding_dims,
  ROUND(AVG(LENGTH(content)), 0)                  AS avg_content_chars
FROM `support-analytics-492410.support_analytics.conversation_chunk_embeddings_original`;
