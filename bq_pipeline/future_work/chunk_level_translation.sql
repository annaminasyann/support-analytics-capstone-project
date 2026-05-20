-- Not executed — deferred to future work.
-- 04_chunk_embeddings.sql uses conversation-level translations from semantic_chunker.py instead.
-- Depends: conversation_chunks | Outputs: conversation_chunks (overwrites translated_chunk_text for non-English chunks)

CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.conversation_chunks`
AS

WITH

english_chunks AS (
  SELECT
    *,
    translated_chunk_text AS translated_chunk_text_final
  FROM `support-analytics-492410.support_analytics.conversation_chunks`
  WHERE translation_granularity = 'message_level'
),

non_english_raw AS (
  SELECT *
  FROM `support-analytics-492410.support_analytics.conversation_chunks`
  WHERE translation_granularity = 'conversation_level'
    AND original_chunk_text IS NOT NULL
    AND TRIM(original_chunk_text) != ''
),

translated AS (
  SELECT
    chunk_id,
    REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
    REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
    REPLACE(REPLACE(REPLACE(
      JSON_EXTRACT_SCALAR(ml_translate_result.translations, '$[0].translated_text'),
      '&#39;', CHR(39)), '&quot;', '"'),   '&apos;', CHR(39)),
      '&amp;', '&'),     '&lt;',   '<'),   '&gt;',   '>'),
      '&nbsp;', ' '),    '&#160;', ' '),   '&ndash;','-'),
      '&mdash;','-'),    '&hellip;','...'), '&copy;', '(c)'),
      '&reg;', '(R)'
    ) AS translated_chunk_text_final
  FROM ML.TRANSLATE(
    MODEL `support-analytics-492410.support_analytics.translation_model`,
    (
      SELECT
        chunk_id,
        original_chunk_text AS text_content
      FROM non_english_raw
      WHERE LENGTH(TRIM(original_chunk_text)) > 0
    ),
    STRUCT('TRANSLATE_TEXT' AS translate_mode, 'en' AS target_language_code)
  )
),

non_english_chunks AS (
  SELECT
    n.*,
    COALESCE(t.translated_chunk_text_final, n.original_chunk_text)
      AS translated_chunk_text_final
  FROM non_english_raw n
  LEFT JOIN translated t USING (chunk_id)
)

SELECT
  conversation_id,
  chunk_id,
  chunk_seq,
  message_start_idx,
  message_end_idx,
  num_messages,
  chunk_char_count,
  original_chunk_text,
  translated_chunk_text_final   AS translated_chunk_text,
  CASE
    WHEN translation_granularity = 'conversation_level' THEN 'chunk_level'
    ELSE translation_granularity
  END                           AS translation_granularity,
  languages,
  min_message_date
FROM english_chunks

UNION ALL

SELECT
  conversation_id,
  chunk_id,
  chunk_seq,
  message_start_idx,
  message_end_idx,
  num_messages,
  chunk_char_count,
  original_chunk_text,
  translated_chunk_text_final   AS translated_chunk_text,
  CASE
    WHEN translation_granularity = 'conversation_level' THEN 'chunk_level'
    ELSE translation_granularity
  END                           AS translation_granularity,
  languages,
  min_message_date
FROM non_english_chunks;


SELECT
  translation_granularity,
  COUNT(*)                                        AS chunks,
  COUNT(DISTINCT conversation_id)                 AS conversations,
  ROUND(AVG(LENGTH(original_chunk_text)), 0)      AS avg_orig_chars,
  ROUND(AVG(LENGTH(translated_chunk_text)), 0)    AS avg_trans_chars,
  COUNTIF(translated_chunk_text = original_chunk_text)
                                                  AS trans_equals_orig_count
FROM `support-analytics-492410.support_analytics.conversation_chunks`
GROUP BY translation_granularity
ORDER BY translation_granularity;
