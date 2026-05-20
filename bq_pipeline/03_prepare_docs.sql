-- Aggregates messages per conversation, detects language, translates non-English client text.
-- Translation feeds TF-IDF baselines and chunk text — clustering itself runs on the multilingual model.
-- Depends: 02_deidentify.sql | Outputs: conversation_docs, translation_log

CREATE OR REPLACE TABLE `support-analytics-492410.support_analytics.conversation_docs` AS

WITH

aggregated AS (
  SELECT
    conversation_id,
    ANY_VALUE(client_id)               AS client_id,
    ANY_VALUE(agent)                   AS agent,
    ANY_VALUE(client_name)             AS client_name,
    ANY_VALUE(conversation_start_date) AS conversation_start_date,
    ANY_VALUE(rating)                  AS rating,
    ANY_VALUE(agreements_subscription) AS agreements_subscription,
    ANY_VALUE(category)                AS category,
    STRING_AGG(
      CASE WHEN agent_or_bot = 'Client' THEN content END,
      ' ' ORDER BY message_date
    )                                  AS client_text_raw,
    STRING_AGG(
      CASE WHEN agent_or_bot = 'Bot' THEN content END,
      ' ' ORDER BY message_date
    )                                  AS bot_text_raw,
    STRING_AGG(content, ' ' ORDER BY message_date)
                                       AS full_text_raw,
    COUNTIF(agent_or_bot = 'Client')   AS client_message_count,
    COUNT(*)                           AS total_message_count,
    COUNTIF(agent_or_bot = 'Agent') > 0 AS was_escalated_to_agent,
    CASE
      WHEN ANY_VALUE(rating) = '❤️' THEN  1
      WHEN ANY_VALUE(rating) = '😞' THEN -1
      ELSE 0
    END                                AS rating_score,
    ARRAY_AGG(agent_or_bot ORDER BY message_date DESC LIMIT 1)[SAFE_OFFSET(0)]
                                       AS last_author
  FROM `support-analytics-492410.support_analytics.chatbot_messages_deidentified`
  WHERE content IS NOT NULL
  GROUP BY conversation_id
  HAVING COUNTIF(agent_or_bot = 'Client') > 0
),

cleaned AS (
  SELECT
    *,
    TRIM(REGEXP_REPLACE(
      REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
      REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
      REPLACE(REPLACE(REPLACE(
        client_text_raw,
        '&#39;',  CHR(39)),  '&quot;', '"'),   '&apos;', CHR(39)),
        '&amp;',  '&'),      '&lt;',   '<'),   '&gt;',   '>'),
        '&nbsp;', ' '),      '&#160;', ' '),   '&ndash;','-'),
        '&mdash;','-'),      '&hellip;','...'), '&copy;', '(c)'),
        '&reg;',  '(R)'),
      r'\s+', ' '
    )) AS client_text_cleaned
  FROM aggregated
  WHERE client_text_raw IS NOT NULL
    AND TRIM(client_text_raw) != ''
),

-- production cache covers ~97-99% of convos; SHA-256 hash matches our conversation_id format
translation_cache AS (
  SELECT
    TO_HEX(SHA256(CAST(conversation_id AS STRING))) AS conversation_id,
    detected_language                               AS cached_language,
    conversation_text_client_translated_raw         AS cached_translation
  FROM `reporting-280507.clients_requests.conversation_docs`
),

-- drop conversations with < 10 client chars (~1.8% of data, basically empty sessions)
candidates AS (
  SELECT
    c.*,
    tc.cached_language,
    tc.cached_translation,
    CASE
      WHEN tc.cached_language     = 'en' THEN FALSE
      WHEN tc.cached_translation IS NOT NULL THEN FALSE
      ELSE TRUE
    END AS needs_detection
  FROM cleaned c
  LEFT JOIN translation_cache tc USING (conversation_id)
  WHERE LENGTH(c.client_text_cleaned) >= 10
),

language_detection AS (
  SELECT
    conversation_id,
    JSON_EXTRACT_SCALAR(ml_translate_result.translations, '$[0].detected_language_code')
      AS detected_language
  FROM ML.TRANSLATE(
    MODEL `support-analytics-492410.support_analytics.translation_model`,
    (
      SELECT
        conversation_id,
        SUBSTR(client_text_cleaned, 1, 150) AS text_content
      FROM candidates
      WHERE needs_detection
        AND TRIM(client_text_cleaned) != ''
    ),
    STRUCT('TRANSLATE_TEXT' AS translate_mode, 'en' AS target_language_code)
  )
),

fresh_translations AS (
  SELECT
    conversation_id,
    REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
    REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
    REPLACE(REPLACE(REPLACE(
      JSON_EXTRACT_SCALAR(ml_translate_result.translations, '$[0].translated_text'),
      '&#39;', CHR(39)), '&quot;', '"'),   '&apos;', CHR(39)),
      '&amp;', '&'),     '&lt;',   '<'),   '&gt;',   '>'),
      '&nbsp;', ' '),    '&#160;', ' '),   '&ndash;','-'),
      '&mdash;','-'),    '&hellip;','...'), '&copy;', '(c)'),
      '&reg;', '(R)'
    ) AS translated_text
  FROM ML.TRANSLATE(
    MODEL `support-analytics-492410.support_analytics.translation_model`,
    (
      SELECT c.conversation_id, c.client_text_cleaned AS text_content
      FROM candidates c
      JOIN language_detection ld USING (conversation_id)
      WHERE ld.detected_language != 'en'
        AND TRIM(c.client_text_cleaned) != ''
    ),
    STRUCT('TRANSLATE_TEXT' AS translate_mode, 'en' AS target_language_code)
  )
),

final AS (
  SELECT
    c.conversation_id,
    c.client_id,
    c.agent,
    c.client_name,
    c.conversation_start_date,
    c.rating,
    c.agreements_subscription,
    c.category,

    COALESCE(
      ld.detected_language,
      c.cached_language,
      'en'
    ) AS detected_language,

    -- Track A: original multilingual text
    c.client_text_cleaned AS conversation_text_multilingual,

    -- Track B: English text (kept as-is if already English, translated otherwise)
    CASE
      WHEN COALESCE(ld.detected_language, c.cached_language, 'en') = 'en'
        THEN c.client_text_cleaned
      WHEN c.cached_translation IS NOT NULL
        THEN c.cached_translation
      WHEN ft.translated_text IS NOT NULL
        THEN ft.translated_text
      ELSE c.client_text_cleaned
    END AS conversation_text_translated,

    -- stopword-filtered Track B for TF-IDF baselines
    TRIM(REGEXP_REPLACE(
      (
        SELECT STRING_AGG(token, ' ' ORDER BY pos)
        FROM UNNEST(SPLIT(LOWER(
          CASE
            WHEN COALESCE(ld.detected_language, c.cached_language, 'en') = 'en'
              THEN c.client_text_cleaned
            WHEN c.cached_translation IS NOT NULL
              THEN c.cached_translation
            WHEN ft.translated_text IS NOT NULL
              THEN ft.translated_text
            ELSE c.client_text_cleaned
          END
        ), ' ')) AS token WITH OFFSET AS pos
        WHERE TRIM(token) != ''
          AND token NOT IN (
            'i','me','my','myself','we','our','ours','ourselves',
            'you','your','yours','yourself','yourselves',
            'he','him','his','himself','she','her','hers','herself',
            'it','its','itself','they','them','their','theirs','themselves',
            'a','an','the','am','is','are','was','were','be','been','being',
            'have','has','had','having','do','does','did','doing',
            'and','but','or','if','because','as','so','than',
            'of','at','by','for','with','to','from',
            'in','on','into','onto','up','down','out','off',
            'over','under','through','during','before','after',
            'above','below','between','among','about','against',
            'across','around','within','without','until','while',
            'what','which','who','whom','when','where','why','how',
            'this','that','these','those',
            'all','any','both','each','few','more','most',
            'other','some','such','no','nor','only','own','same','too','very',
            'here','there','then','now','once','again','further',
            's','t','d','ll','m','re','ve',
            'don','doesn','didn','isn','aren','wasn','weren',
            'hasn','haven','hadn','won','wouldn','couldn','shouldn',
            'just','ok','okay','yes','yeah','sure',
            'really','actually','maybe','probably',
            'like','one','also','already','even',
            'way','well','two','left','right','long','short','big','small'
          )
      ),
      r'\s+', ' '
    )) AS conversation_text_stopword_filtered,

    c.client_text_raw AS conversation_text_client_original,
    c.bot_text_raw    AS conversation_text_bot_original,
    c.full_text_raw   AS conversation_text_full_original,

    c.client_message_count,
    c.total_message_count,
    c.was_escalated_to_agent,
    c.rating_score,
    c.last_author,
    c.last_author = 'Client'              AS ended_without_resolution,
    c.client_message_count >= 5
      AND c.rating_score <= 0
      AND NOT c.was_escalated_to_agent    AS is_probable_loop

  FROM candidates c
  LEFT JOIN language_detection ld USING (conversation_id)
  LEFT JOIN fresh_translations ft USING (conversation_id)
)

SELECT * FROM final;


-- tracks which source each conversation's English text came from
CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.translation_log`
AS
WITH cache AS (
  SELECT
    TO_HEX(SHA256(CAST(conversation_id AS STRING))) AS conversation_id,
    detected_language                               AS cached_language,
    conversation_text_client_translated_raw         AS cached_translation
  FROM `reporting-280507.clients_requests.conversation_docs`
)
SELECT
  d.conversation_id,
  d.detected_language,
  d.conversation_start_date,
  CASE
    WHEN d.detected_language = 'en'
      AND c.cached_language = 'en'                    THEN 'confirmed_english_cache'
    WHEN d.detected_language = 'en'
      AND c.cached_language IS NULL                   THEN 'api_detected_english'
    WHEN d.detected_language != 'en'
      AND c.cached_translation IS NOT NULL            THEN 'translated_from_cache'
    WHEN d.detected_language != 'en'
      AND c.cached_translation IS NULL                THEN 'translated_via_api'
    ELSE                                                   'other'
  END AS translation_source,
  LENGTH(d.conversation_text_multilingual) AS original_char_count,
  LENGTH(d.conversation_text_translated)   AS translated_char_count,
  d.conversation_text_translated != d.conversation_text_multilingual AS text_was_changed
FROM `support-analytics-492410.support_analytics.conversation_docs` d
LEFT JOIN cache c USING (conversation_id);


SELECT
  COUNT(*)                                          AS total_conversations,
  COUNT(DISTINCT detected_language)                 AS distinct_languages,
  COUNTIF(detected_language = 'en')                 AS english,
  COUNTIF(detected_language != 'en')                AS non_english,
  ROUND(100 * COUNTIF(detected_language != 'en') / COUNT(*), 1)
                                                    AS pct_non_english,
  COUNTIF(conversation_text_translated IS NOT NULL) AS has_translation,
  COUNTIF(was_escalated_to_agent)                   AS escalated,
  MIN(conversation_start_date)                      AS earliest,
  MAX(conversation_start_date)                      AS latest
FROM `support-analytics-492410.support_analytics.conversation_docs`;

SELECT
  translation_source,
  COUNT(*)                                             AS conversations,
  ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER(), 2)     AS pct,
  SUM(original_char_count)                             AS total_original_chars,
  SUM(translated_char_count)                           AS total_translated_chars,
  COUNTIF(text_was_changed)                            AS actually_translated
FROM `support-analytics-492410.support_analytics.translation_log`
GROUP BY translation_source
ORDER BY conversations DESC;
