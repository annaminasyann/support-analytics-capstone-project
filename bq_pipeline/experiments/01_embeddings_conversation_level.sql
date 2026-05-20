-- [EXPERIMENT] Conversation-level embeddings — not the canonical chunk pipeline.
-- Baseline results: Silhouette(cosine) = 0.077 (Track A), 0.091 (Track B), mcs=150, k=70
-- Depends: 03_prepare_docs.sql

-- Track A: multilingual client + bot text
CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.conversation_embeddings_multilingual`
AS
SELECT * FROM ML.GENERATE_EMBEDDING(
  MODEL `support-analytics-492410.support_analytics.embedding_multilingual`,
  (
    SELECT
      conversation_id,
      client_id,
      agent,
      client_name,
      conversation_start_date,
      rating,
      agreements_subscription,
      category,
      detected_language,
      client_message_count,
      total_message_count,
      was_escalated_to_agent,
      rating_score,
      ended_without_resolution,
      is_probable_loop,
      TRIM(CONCAT(
        conversation_text_multilingual,
        CASE
          WHEN conversation_text_bot_original IS NOT NULL
            AND TRIM(conversation_text_bot_original) != ''
          THEN CONCAT(
            '
',
            TRIM(REGEXP_REPLACE(
              REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
              REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
              REPLACE(REPLACE(REPLACE(
                conversation_text_bot_original,
                '&#39;',  CHR(39)), '&quot;', '"'),   '&apos;', CHR(39)),
                '&amp;',  '&'),     '&lt;',   '<'),   '&gt;',   '>'),
                '&nbsp;', ' '),     '&#160;', ' '),   '&ndash;','-'),
                '&mdash;','-'),     '&hellip;','...'), '&copy;', '(c)'),
                '&reg;',  '(R)'),
              r'\s+', ' '
            ))
          )
          ELSE ''
        END
      )) AS content
    FROM `support-analytics-492410.support_analytics.conversation_docs`
    WHERE conversation_text_multilingual IS NOT NULL
      AND TRIM(conversation_text_multilingual) != ''
  ),
  STRUCT(TRUE AS flatten_json_output)
)
WHERE LENGTH(ml_generate_embedding_status) = 0;


-- Track B: translated English client + bot text
CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.conversation_embeddings_translated`
AS
SELECT * FROM ML.GENERATE_EMBEDDING(
  MODEL `support-analytics-492410.support_analytics.embedding_multilingual`,
  (
    SELECT
      conversation_id,
      client_id,
      agent,
      client_name,
      conversation_start_date,
      rating,
      agreements_subscription,
      category,
      detected_language,
      client_message_count,
      total_message_count,
      was_escalated_to_agent,
      rating_score,
      ended_without_resolution,
      is_probable_loop,
      TRIM(CONCAT(
        conversation_text_translated,
        CASE
          WHEN conversation_text_bot_original IS NOT NULL
            AND TRIM(conversation_text_bot_original) != ''
          THEN CONCAT(
            '
',
            TRIM(REGEXP_REPLACE(
              REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
              REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
              REPLACE(REPLACE(REPLACE(
                conversation_text_bot_original,
                '&#39;',  CHR(39)), '&quot;', '"'),   '&apos;', CHR(39)),
                '&amp;',  '&'),     '&lt;',   '<'),   '&gt;',   '>'),
                '&nbsp;', ' '),     '&#160;', ' '),   '&ndash;','-'),
                '&mdash;','-'),     '&hellip;','...'), '&copy;', '(c)'),
                '&reg;',  '(R)'),
              r'\s+', ' '
            ))
          )
          ELSE ''
        END
      )) AS content
    FROM `support-analytics-492410.support_analytics.conversation_docs`
    WHERE conversation_text_translated IS NOT NULL
      AND TRIM(conversation_text_translated) != ''
  ),
  STRUCT(TRUE AS flatten_json_output)
)
WHERE LENGTH(ml_generate_embedding_status) = 0;


SELECT 'Track A (multilingual + bot)' AS track,
  COUNT(*)                                        AS row_count,
  MAX(ARRAY_LENGTH(ml_generate_embedding_result)) AS dims,
  ROUND(AVG(LENGTH(content)), 0)                  AS avg_content_chars
FROM `support-analytics-492410.support_analytics.conversation_embeddings_multilingual`

UNION ALL

SELECT 'Track B (translated + bot)',
  COUNT(*),
  MAX(ARRAY_LENGTH(ml_generate_embedding_result)),
  ROUND(AVG(LENGTH(content)), 0)
FROM `support-analytics-492410.support_analytics.conversation_embeddings_translated`;