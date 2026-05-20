-- Diagnostic and data quality queries — run each block independently in the BQ console.
-- Depends: 05_failure_analysis.sql, 06_trends.sql, 07_chunk_evaluation.sql


-- Data quality 

-- DQ1. Pipeline row count
SELECT
  COUNT(*) AS conversations_in_pipeline
FROM `support-analytics-492410.support_analytics.conversation_docs`;


-- DQ2. Translation source mix
SELECT
  translation_source,
  COUNT(*) AS conversations,
  ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM `support-analytics-492410.support_analytics.translation_log`
GROUP BY translation_source
ORDER BY conversations DESC;


-- DQ3. Linguistic agent-request signal vs actual escalation
SELECT
  COUNTIF(requested_agent AND was_escalated_to_agent)       AS ask_text_and_joined,
  COUNTIF(requested_agent AND NOT was_escalated_to_agent)   AS ask_text_no_agent,
  COUNTIF(NOT requested_agent AND was_escalated_to_agent)   AS joined_no_ask_phrase,
  COUNTIF(NOT requested_agent AND NOT was_escalated_to_agent) AS neither
FROM `support-analytics-492410.support_analytics.failure_flags`;


-- DQ4. Cross-validate Gemini descriptions: cited failure % vs actual
SELECT
  cluster_id,
  n_conversations,
  ROUND(failure_risk_proxy, 3)                                                   AS failure_risk_proxy,
  ROUND(headline_failure_rate_pct, 1)                                            AS actual_failure_pct,
  SAFE_CAST(
    REGEXP_EXTRACT(
      REGEXP_EXTRACT(cluster_description, r'### Failure diagnosis\n([^#]*)'),
      r'(\d+\.?\d*)\s*%'
    ) AS FLOAT64
  )                                                                               AS cited_pct,
  CASE
    WHEN REGEXP_EXTRACT(
           REGEXP_EXTRACT(cluster_description, r'### Failure diagnosis\n([^#]*)'),
           r'(\d+\.?\d*)\s*%'
         ) IS NULL                                                               THEN 'NO NUMBER CITED'
    WHEN ABS(
           SAFE_CAST(REGEXP_EXTRACT(
             REGEXP_EXTRACT(cluster_description, r'### Failure diagnosis\n([^#]*)'),
             r'(\d+\.?\d*)\s*%'
           ) AS FLOAT64)
           - headline_failure_rate_pct
         ) <= 10                                                                 THEN 'ACCURATE'
    WHEN SAFE_CAST(REGEXP_EXTRACT(
           REGEXP_EXTRACT(cluster_description, r'### Failure diagnosis\n([^#]*)'),
           r'(\d+\.?\d*)\s*%'
         ) AS FLOAT64) > headline_failure_rate_pct                              THEN 'OVERSTATED'
    ELSE                                                                              'UNDERSTATED'
  END                                                                             AS accuracy_flag,
  REGEXP_CONTAINS(
    REGEXP_EXTRACT(cluster_description, r'### Evidence quotes\n([^#]*)'),
    r'"[^"]+'
  )                                                                               AS has_quotes,
  REGEXP_CONTAINS(
    LOWER(REGEXP_EXTRACT(cluster_description, r'### Theme\n([^#]*)')),
    r'\btechnical issues?\b|\baccount problems?\b|\bbilling questions?\b|\bgeneral\b|\bvarious\b|\bmisc'
  )                                                                               AS theme_is_vague
FROM `support-analytics-492410.support_analytics.chunk_cluster_descriptions`
WHERE NOT is_routing_cluster
  AND cluster_description IS NOT NULL
ORDER BY failure_risk_proxy DESC;


-- Cluster and chunk diagnostics 

-- 1. Cluster size distribution (conversation level)

WITH sizes AS (
  SELECT
    primary_topic                                                        AS cluster_id,
    COUNT(*)                                                             AS n_conversations,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                   AS pct_of_total
  FROM `support-analytics-492410.support_analytics.conversation_chunk_topics`
  WHERE primary_topic != -1
  GROUP BY primary_topic
)
SELECT
  cluster_id,
  n_conversations,
  pct_of_total,
  CASE
    WHEN n_conversations > 5000     THEN 'VERY LARGE — inspect manually'
    WHEN n_conversations > 1000     THEN 'large'
    WHEN n_conversations > 200      THEN 'medium'
    WHEN n_conversations < 50       THEN 'tiny — may be fragmented'
    ELSE 'normal'
  END AS size_flag
FROM sizes
ORDER BY n_conversations DESC;


-- 2. Cluster size distribution (chunk level)

SELECT
  cluster_id,
  COUNT(*)                                                               AS n_chunks,
  COUNT(DISTINCT conversation_id)                                        AS n_conversations,
  ROUND(AVG(hdbscan_prob), 3)                                            AS avg_confidence,
  ROUND(AVG(chunk_char_count), 0)                                        AS avg_chunk_chars,
  CASE
    WHEN cluster_id = -1            THEN 'NOISE'
    WHEN COUNT(*) > 10000           THEN 'VERY LARGE — inspect manually'
    WHEN COUNT(*) > 3000            THEN 'large'
    WHEN COUNT(*) > 500             THEN 'medium'
    WHEN COUNT(*) < 100             THEN 'tiny — may be fragmented'
    ELSE 'normal'
  END AS size_flag
FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`
GROUP BY cluster_id
ORDER BY n_chunks DESC;


-- 3. Sample chunks from the 5 largest clusters

WITH top_clusters AS (
  SELECT cluster_id
  FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`
  WHERE cluster_id != -1
  GROUP BY cluster_id
  ORDER BY COUNT(*) DESC
  LIMIT 5
),
ranked AS (
  SELECT
    cl.cluster_id,
    cl.hdbscan_prob,
    cl.detected_language,
    cl.chunk_char_count,
    cl.translated_chunk_text,
    ROW_NUMBER() OVER (
      PARTITION BY cl.cluster_id
      ORDER BY cl.hdbscan_prob DESC
    ) AS rn
  FROM `support-analytics-492410.support_analytics.chunk_cluster_labels` cl
  JOIN top_clusters t USING (cluster_id)
  WHERE NOT cl.is_noise
)
SELECT
  cluster_id,
  rn AS rank_in_cluster,
  hdbscan_prob,
  detected_language,
  chunk_char_count,
  LEFT(translated_chunk_text, 400) AS text_preview
FROM ranked
WHERE rn <= 5
ORDER BY cluster_id, rn;


-- 4. Noise chunks (shortest first)

SELECT
  cl.hdbscan_prob,
  cl.detected_language,
  cl.chunk_char_count,
  d.category,
  d.was_escalated_to_agent,
  LEFT(cl.translated_chunk_text, 500) AS text_preview
FROM `support-analytics-492410.support_analytics.chunk_cluster_labels` cl
JOIN `support-analytics-492410.support_analytics.conversation_docs` d
  USING (conversation_id)
WHERE cl.is_noise
ORDER BY cl.chunk_char_count ASC
LIMIT 50;


-- 5. HDBSCAN probability distribution

SELECT
  CASE
    WHEN cluster_id = -1          THEN '[-1] noise'
    WHEN hdbscan_prob < 0.1       THEN '[0.0–0.1] boundary'
    WHEN hdbscan_prob < 0.3       THEN '[0.1–0.3]'
    WHEN hdbscan_prob < 0.5       THEN '[0.3–0.5]'
    WHEN hdbscan_prob < 0.7       THEN '[0.5–0.7]'
    WHEN hdbscan_prob < 0.9       THEN '[0.7–0.9]'
    ELSE                               '[0.9–1.0] core'
  END AS prob_bucket,
  COUNT(*)                                                               AS n_chunks,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                     AS pct
FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`
GROUP BY prob_bucket
ORDER BY prob_bucket;


-- 6. Mixed-topic conversations (chunks spanning multiple clusters)

SELECT
  ct.primary_topic                                                       AS cluster_id,
  COUNT(*)                                                               AS n_conversations,
  COUNTIF(ct.is_mixed_topic)                                             AS n_mixed,
  ROUND(100 * COUNTIF(ct.is_mixed_topic) / COUNT(*), 1)                  AS pct_mixed,
  ROUND(AVG(ct.n_chunks), 1)                                             AS avg_chunks,
  ROUND(AVG(ct.topic_confidence), 3)                                     AS avg_confidence
FROM `support-analytics-492410.support_analytics.conversation_chunk_topics` ct
WHERE ct.primary_topic != -1
GROUP BY ct.primary_topic
ORDER BY pct_mixed DESC;


-- 7. Language breakdown per cluster

SELECT
  cluster_id,
  detected_language,
  COUNT(*)                                                               AS n_chunks,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY cluster_id), 1)
                                                                         AS pct_in_cluster
FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`
WHERE cluster_id != -1
GROUP BY cluster_id, detected_language
ORDER BY cluster_id, n_chunks DESC;


-- 8. Overall chunking health

SELECT
  COUNT(DISTINCT conversation_id)                                         AS n_conversations_chunked,
  COUNT(*)                                                                AS total_chunks,
  ROUND(COUNT(*) / COUNT(DISTINCT conversation_id), 2)                    AS avg_chunks_per_conv,
  ROUND(100 * COUNTIF(is_noise) / COUNT(*), 2)                            AS noise_rate_pct,
  ROUND(AVG(hdbscan_prob), 3)                                             AS avg_hdbscan_prob,
  ROUND(AVG(chunk_char_count), 0)                                         AS avg_chunk_chars,
  ROUND(MIN(chunk_char_count), 0)                                         AS min_chunk_chars,
  ROUND(MAX(chunk_char_count), 0)                                         AS max_chunk_chars,
  COUNTIF(chunk_char_count < 200)                                         AS chunks_under_200_chars,
  COUNTIF(chunk_char_count BETWEEN 800 AND 2800)                          AS chunks_in_target_window,
  ROUND(
    100 * COUNTIF(chunk_char_count BETWEEN 800 AND 2800) / COUNT(*), 1
  )                                                                       AS pct_in_target_window
FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`;


-- 9. Chunk size histogram

SELECT
  CASE
    WHEN chunk_char_count <  200               THEN '< 200  (very short)'
    WHEN chunk_char_count <  400               THEN '200–399'
    WHEN chunk_char_count <  800               THEN '400–799'
    WHEN chunk_char_count < 1400               THEN '800–1399 ✓'
    WHEN chunk_char_count < 2000               THEN '1400–1999 ✓'
    WHEN chunk_char_count < 2800               THEN '2000–2799 ✓'
    WHEN chunk_char_count < 4000               THEN '2800–3999 (long)'
    ELSE                                            '>= 4000  (very long)'
  END AS char_bucket,
  COUNT(*)                                                                AS n_chunks,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                      AS pct
FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`
GROUP BY char_bucket
ORDER BY MIN(chunk_char_count);


-- 10. Chunks-per-conversation distribution

WITH counts AS (
  SELECT
    n_chunks,
    COUNT(*) AS n_conversations,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
  FROM `support-analytics-492410.support_analytics.conversation_chunk_topics`
  GROUP BY n_chunks
)
SELECT
  n_chunks,
  n_conversations,
  pct,
  CASE
    WHEN n_chunks = 1   THEN 'no split — short or single-topic'
    WHEN n_chunks <= 3  THEN 'light split ✓'
    WHEN n_chunks <= 6  THEN 'moderate split ✓'
    WHEN n_chunks <= 10 THEN 'heavy split'
    ELSE                     'very long conversation'
  END AS interpretation
FROM counts
ORDER BY n_chunks;


-- 11. Boilerplate contamination per cluster

SELECT
  cluster_id,
  COUNT(*)                                                                AS n_chunks,
  COUNTIF(REGEXP_CONTAINS(
    LOWER(translated_chunk_text),
    r'were (my|our) answers helpful|rate (your|this) (chat|experience)|how (would you rate|satisfied)'
  ))                                                                      AS nps_chunks,
  COUNTIF(REGEXP_CONTAINS(
    LOWER(translated_chunk_text),
    r'(chat|conversation|session) (has been |is )?closed|goodbye|take care|bye'
  ))                                                                      AS closing_chunks,
  COUNTIF(REGEXP_CONTAINS(
    LOWER(translated_chunk_text),
    r'thank you for (contacting|reaching out|using|choosing)|is there anything else'
  ))                                                                      AS filler_chunks,
  ROUND(
    100 * COUNTIF(REGEXP_CONTAINS(
      LOWER(translated_chunk_text),
      r'were (my|our) answers helpful|rate (your|this) (chat|experience)|'
      r'how (would you rate|satisfied)|(chat|conversation|session) (has been |is )?closed|'
      r'thank you for (contacting|reaching out|using|choosing)|is there anything else'
    )) / COUNT(*), 1
  )                                                                       AS boilerplate_pct
FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`
WHERE cluster_id != -1
GROUP BY cluster_id
ORDER BY boilerplate_pct DESC;


-- 12. Mixed-topic rate by conversation length

SELECT
  COUNT(*)                                                                AS total_conversations,
  COUNTIF(is_mixed_topic)                                                 AS mixed_topic_convs,
  ROUND(100 * COUNTIF(is_mixed_topic) / COUNT(*), 1)                      AS pct_mixed,
  ROUND(AVG(n_chunks), 2)                                                 AS avg_chunks,
  ROUND(AVG(IF(is_mixed_topic,     n_chunks, NULL)), 2)                   AS avg_chunks_mixed,
  ROUND(AVG(IF(NOT is_mixed_topic, n_chunks, NULL)), 2)                   AS avg_chunks_single_topic,
  ROUND(AVG(topic_confidence), 3)                                         AS avg_confidence,
  ROUND(AVG(IF(is_mixed_topic,     topic_confidence, NULL)), 3)           AS confidence_mixed,
  ROUND(AVG(IF(NOT is_mixed_topic, topic_confidence, NULL)), 3)           AS confidence_single
FROM `support-analytics-492410.support_analytics.conversation_chunk_topics`
WHERE primary_topic != -1;


-- 13. Single-chunk conversations by message count

SELECT
  ct.n_chunks,
  COUNT(*)                                                                AS n_conversations,
  ROUND(AVG(d.client_message_count), 1)                                   AS avg_client_msgs,
  ROUND(AVG(d.total_message_count),  1)                                   AS avg_total_msgs,
  COUNTIF(d.was_escalated_to_agent)                                       AS escalated,
  ROUND(100 * COUNTIF(d.was_escalated_to_agent) / COUNT(*), 1)            AS escalation_rate_pct
FROM `support-analytics-492410.support_analytics.conversation_chunk_topics` ct
JOIN `support-analytics-492410.support_analytics.conversation_docs` d
  USING (conversation_id)
WHERE ct.primary_topic != -1
  AND ct.n_chunks <= 3
GROUP BY ct.n_chunks
ORDER BY ct.n_chunks;


-- 14. Language coherence: topic-based vs language-driven clusters

WITH per_cluster_lang AS (
  SELECT
    cluster_id,
    detected_language,
    COUNT(*)                                                              AS n_chunks,
    ROUND(
      100.0 * COUNT(*)
      / SUM(COUNT(*)) OVER (PARTITION BY cluster_id), 1
    )                                                                     AS pct_in_cluster
  FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`
  WHERE cluster_id != -1
  GROUP BY cluster_id, detected_language
),
dominant AS (
  SELECT *
  FROM per_cluster_lang
  WHERE pct_in_cluster = (
    SELECT MAX(pct_in_cluster)
    FROM per_cluster_lang p2
    WHERE p2.cluster_id = per_cluster_lang.cluster_id
  )
)
SELECT
  cluster_id,
  detected_language                                                       AS dominant_language,
  pct_in_cluster                                                          AS dominant_lang_pct,
  CASE
    WHEN detected_language = 'en' AND pct_in_cluster > 80 THEN 'English-dominant (expected)'
    WHEN detected_language != 'en' AND pct_in_cluster > 60 THEN 'LANGUAGE-DRIVEN ← check this'
    ELSE 'mixed / multilingual'
  END                                                                     AS diagnosis
FROM dominant
ORDER BY pct_in_cluster DESC;


-- 15. Semantic split effectiveness: single-chunk vs multi-chunk conversations

SELECT
  CASE WHEN ct.n_chunks = 1 THEN 'single chunk' ELSE 'multi-chunk' END   AS split_type,
  COUNT(DISTINCT ct.conversation_id)                                      AS n_conversations,
  ROUND(AVG(ct.n_chunks), 2)                                              AS avg_chunks,
  ROUND(AVG(ct.topic_confidence), 3)                                      AS avg_topic_confidence,
  ROUND(100 * COUNTIF(ff.is_any_failure)       / COUNT(*), 2)             AS headline_failure_rate_pct,
  ROUND(100 * COUNTIF(ff.is_confirmed_failure) / COUNT(*), 2)             AS confirmed_failure_rate_pct,
  ROUND(100 * COUNTIF(d.was_escalated_to_agent)/ COUNT(*), 2)             AS escalation_rate_pct,
  ROUND(AVG(d.client_message_count), 1)                                   AS avg_client_messages
FROM        `support-analytics-492410.support_analytics.conversation_chunk_topics` ct
JOIN        `support-analytics-492410.support_analytics.conversation_docs`         d
       USING (conversation_id)
LEFT JOIN   `support-analytics-492410.support_analytics.failure_flags`             ff
       USING (conversation_id)
WHERE ct.primary_topic != -1
GROUP BY split_type
ORDER BY split_type;


-- 16. Artifact cluster deep-dive (clusters 0, 1, 2)

SELECT
  cl.cluster_id,
  COUNT(*)                                                                AS n_chunks,
  COUNT(DISTINCT cl.conversation_id)                                      AS n_conversations,
  ROUND(AVG(cl.hdbscan_prob), 3)                                          AS avg_confidence,
  ROUND(AVG(cl.chunk_char_count), 0)                                      AS avg_chars,
  ROUND(AVG(d.client_message_count), 1)                                   AS avg_client_msgs,
  ROUND(100 * COUNTIF(d.was_escalated_to_agent) / COUNT(DISTINCT cl.conversation_id), 1)
                                                                          AS escalation_rate_pct,
  ARRAY_AGG(
    LEFT(cl.translated_chunk_text, 200)
    ORDER BY cl.hdbscan_prob DESC
    LIMIT 3
  )                                                                       AS top_3_samples
FROM `support-analytics-492410.support_analytics.chunk_cluster_labels` cl
JOIN `support-analytics-492410.support_analytics.conversation_docs` d
  USING (conversation_id)
WHERE cl.cluster_id IN (0, 1, 2)
GROUP BY cl.cluster_id
ORDER BY cl.cluster_id;


-- 17. Noise chunk size distribution

SELECT
  CASE
    WHEN chunk_char_count <  300  THEN '< 300 (short / boilerplate)'
    WHEN chunk_char_count <  800  THEN '300–799'
    WHEN chunk_char_count < 2000  THEN '800–1999 (in-target, unique topic)'
    ELSE                               '>= 2000 (long, rare topic)'
  END AS noise_char_bucket,
  COUNT(*)                                                                AS n_noise_chunks,
  ROUND(AVG(chunk_char_count), 0)                                         AS avg_chars,
  ROUND(
    100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2
  )                                                                       AS pct_of_noise
FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`
WHERE is_noise
GROUP BY noise_char_bucket
ORDER BY MIN(chunk_char_count);


-- 18. Design verdict scorecard

WITH stats AS (
  SELECT
    COUNT(DISTINCT conversation_id)                               AS n_convs,
    COUNT(*)                                                      AS n_chunks,
    ROUND(COUNT(*) / COUNT(DISTINCT conversation_id), 2)          AS avg_chunks_per_conv,
    ROUND(100 * COUNTIF(is_noise)            / COUNT(*), 2)       AS noise_pct,
    ROUND(100 * COUNTIF(chunk_char_count < 400) / COUNT(*), 2)    AS pct_very_short,
    ROUND(100 * COUNTIF(chunk_char_count BETWEEN 800 AND 2800)
               / COUNT(*), 1)                                     AS pct_in_target
  FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`
),
mix AS (
  SELECT
    ROUND(100 * COUNTIF(is_mixed_topic) / COUNT(*), 1) AS pct_mixed
  FROM `support-analytics-492410.support_analytics.conversation_chunk_topics`
  WHERE primary_topic != -1
)
SELECT
  stats.n_convs,
  stats.n_chunks,
  stats.avg_chunks_per_conv,
  stats.noise_pct,
  stats.pct_very_short,
  stats.pct_in_target,
  mix.pct_mixed,
  CASE WHEN stats.noise_pct      > 20   THEN 'HIGH NOISE'  ELSE 'OK' END AS noise_verdict,
  CASE WHEN stats.pct_very_short > 30   THEN 'MANY SHORT'  ELSE 'OK' END AS short_verdict,
  CASE WHEN stats.pct_in_target  < 50   THEN 'FEW IN TARGET' ELSE 'OK' END AS size_verdict,
  CASE WHEN mix.pct_mixed        < 5    THEN 'LOW MIXING — chunking may be unnecessary'
       WHEN mix.pct_mixed        > 50   THEN 'HIGH MIXING — inspect noise'
       ELSE 'OK' END                                                   AS mixing_verdict
FROM stats, mix;


-- Message-level analysis 

-- A1. Character volume by role

SELECT
  agent_or_bot                                                            AS role,
  COUNT(*)                                                                AS n_messages,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)                      AS pct_of_messages,
  SUM(LENGTH(content))                                                    AS total_chars,
  ROUND(100.0 * SUM(LENGTH(content)) / SUM(SUM(LENGTH(content))) OVER (), 1)
                                                                          AS pct_of_chars,
  ROUND(AVG(LENGTH(content)), 0)                                          AS avg_chars_per_message,
  APPROX_QUANTILES(LENGTH(content), 100)[OFFSET(50)]                    AS p50_chars,
  APPROX_QUANTILES(LENGTH(content), 100)[OFFSET(90)]                    AS p90_chars,
  APPROX_QUANTILES(LENGTH(content), 100)[OFFSET(99)]                    AS p99_chars
FROM `support-analytics-492410.support_analytics.chatbot_messages_deidentified`
WHERE content IS NOT NULL
GROUP BY agent_or_bot
ORDER BY total_chars DESC;


-- A2. Message count per conversation by role

SELECT
  COUNT(DISTINCT conversation_id)                                         AS n_conversations,
  ROUND(AVG(client_msgs), 1)                                              AS avg_client_messages,
  ROUND(AVG(bot_msgs), 1)                                                 AS avg_bot_messages,
  ROUND(AVG(agent_msgs), 1)                                               AS avg_agent_messages,
  ROUND(AVG(client_chars), 0)                                             AS avg_client_chars_per_conv,
  ROUND(AVG(bot_chars), 0)                                                AS avg_bot_chars_per_conv,
  ROUND(AVG(SAFE_DIVIDE(bot_chars, client_chars + bot_chars)), 3)         AS avg_bot_char_share
FROM (
  SELECT
    conversation_id,
    COUNTIF(agent_or_bot = 'Client')         AS client_msgs,
    COUNTIF(agent_or_bot = 'Bot')            AS bot_msgs,
    COUNTIF(agent_or_bot = 'Agent')          AS agent_msgs,
    SUM(IF(agent_or_bot = 'Client', LENGTH(content), 0)) AS client_chars,
    SUM(IF(agent_or_bot = 'Bot',    LENGTH(content), 0)) AS bot_chars
  FROM `support-analytics-492410.support_analytics.chatbot_messages_deidentified`
  WHERE content IS NOT NULL
  GROUP BY conversation_id
);


-- A3. Message length distribution by role

SELECT
  agent_or_bot                                                            AS role,
  CASE
    WHEN LENGTH(content) <   30  THEN '< 30 chars (very short)'
    WHEN LENGTH(content) <  100  THEN '30–99 chars'
    WHEN LENGTH(content) <  300  THEN '100–299 chars'
    WHEN LENGTH(content) <  700  THEN '300–699 chars'
    WHEN LENGTH(content) < 1500  THEN '700–1499 chars'
    ELSE                              '>= 1500 chars (long AI response)'
  END AS length_bucket,
  COUNT(*)                                                                AS n_messages,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY agent_or_bot), 1)
                                                                          AS pct_within_role
FROM `support-analytics-492410.support_analytics.chatbot_messages_deidentified`
WHERE content IS NOT NULL
  AND agent_or_bot IN ('Client', 'Bot')
GROUP BY agent_or_bot, length_bucket
ORDER BY agent_or_bot, MIN(LENGTH(content));


-- A4. Client message boilerplate filter coverage

SELECT
  CASE
    WHEN LENGTH(TRIM(content)) < 8          THEN 'too_short (< 8 chars)'
    WHEN LOWER(TRIM(content)) IN (
      'hi','hello','hey','yes','no','ok','okay','sure','thanks','bye',
      'goodbye','please','alright','right','great','perfect','awesome',
      'cool','fine','noted','understood','agree','agreed','welcome','yep',
      'nope','np','ty','thx','yw','k','hmm','oh','ah','well','nice',
      'good','wow','thank you','thank you!','thanks!','got it','i see',
      'sounds good','no problem','no worries','of course','absolutely',
      'certainly','will do','makes sense','fair enough','i know',
      'i understand','take care','good luck','see you','cheers',
      'thank you very much','many thanks','much appreciated'
    )                                       THEN 'boilerplate_exact_match'
    WHEN LENGTH(TRIM(content)) < 50
      AND TRIM(content) LIKE '%thank%'      THEN 'likely boilerplate (short thanks)'
    ELSE                                         'substantive (survives filter)'
  END AS filter_outcome,
  COUNT(*)                                                                AS n_messages,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)                      AS pct
FROM `support-analytics-492410.support_analytics.chatbot_messages_deidentified`
WHERE content IS NOT NULL
  AND agent_or_bot = 'Client'
GROUP BY filter_outcome
ORDER BY n_messages DESC;


-- Chunk text analysis 

-- B1. [USER] vs [AI] character share inside chunk text

SELECT
  ROUND(AVG(
    ARRAY_LENGTH(REGEXP_EXTRACT_ALL(translated_chunk_text, r'\[AI\]:'))
  ), 2)                                                                   AS avg_ai_turns_per_chunk,
  ROUND(AVG(
    ARRAY_LENGTH(REGEXP_EXTRACT_ALL(translated_chunk_text, r'\[USER\]:'))
  ), 2)                                                                   AS avg_user_turns_per_chunk,
  ROUND(AVG(
    LENGTH(REGEXP_REPLACE(translated_chunk_text, r'\[USER\]: [^\[]*', ''))
    / NULLIF(LENGTH(translated_chunk_text), 0)
  ), 3)                                                                   AS approx_ai_char_share,
  ROUND(AVG(chunk_char_count), 0)                                         AS avg_chunk_chars,
  COUNT(*)                                                                AS total_chunks
FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`;


-- B2. Single-turn chunks

SELECT
  ai_turns,
  user_turns,
  COUNT(*)                                                                AS n_chunks,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                      AS pct
FROM (
  SELECT
    ARRAY_LENGTH(REGEXP_EXTRACT_ALL(translated_chunk_text, r'\[AI\]:'))   AS ai_turns,
    ARRAY_LENGTH(REGEXP_EXTRACT_ALL(translated_chunk_text, r'\[USER\]:')) AS user_turns
  FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`
  WHERE NOT is_noise
)
GROUP BY ai_turns, user_turns
ORDER BY n_chunks DESC
LIMIT 20;


-- B3. Mixed-topic rate by conversation length

SELECT
  CASE
    WHEN d.total_message_count <= 4   THEN '1–4 messages'
    WHEN d.total_message_count <= 8   THEN '5–8 messages'
    WHEN d.total_message_count <= 15  THEN '9–15 messages'
    WHEN d.total_message_count <= 25  THEN '16–25 messages'
    ELSE                                   '> 25 messages'
  END AS conv_length_bucket,
  COUNT(*)                                                                AS n_conversations,
  ROUND(AVG(ct.n_chunks), 2)                                              AS avg_chunks,
  ROUND(100 * COUNTIF(ct.is_mixed_topic) / COUNT(*), 1)                   AS pct_mixed,
  ROUND(AVG(ct.topic_confidence), 3)                                      AS avg_topic_confidence
FROM `support-analytics-492410.support_analytics.conversation_chunk_topics` ct
JOIN `support-analytics-492410.support_analytics.conversation_docs` d
  USING (conversation_id)
WHERE ct.primary_topic != -1
GROUP BY conv_length_bucket
ORDER BY MIN(d.total_message_count);


-- B4. Translation coverage (non-English chunks where translated == original)

SELECT
  CASE
    WHEN translated_chunk_text = original_chunk_text  THEN 'no translation (non-English original)'
    WHEN translated_chunk_text IS NULL                THEN 'NULL (unexpected)'
    ELSE                                                   'translated'
  END AS translation_status,
  detected_language,
  COUNT(*)                                                                AS n_chunks,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)                      AS pct
FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`
GROUP BY translation_status, detected_language
ORDER BY n_chunks DESC
LIMIT 30;


-- Cluster description checks 

-- C1. All cluster descriptions ranked by failure risk

SELECT
  cluster_id,
  n_conversations,
  is_routing_cluster,
  ROUND(confirmed_failure_rate_pct, 1)  AS confirmed_pct,
  ROUND(headline_failure_rate_pct,  1)  AS headline_pct,
  ROUND(escalation_rate_pct,        1)  AS esc_pct,
  ROUND(failure_risk_proxy,         3)  AS risk,
  gemini_status,
  cluster_description
FROM `support-analytics-492410.support_analytics.chunk_cluster_descriptions`
ORDER BY failure_risk_proxy DESC;


-- C2. Failure tier breakdown per cluster

SELECT
  cd.cluster_id,
  cd.n_conversations,
  ROUND(cd.failure_risk_proxy, 3)       AS risk,
  TRIM(REGEXP_EXTRACT(cd.cluster_description, r'### Theme\n([^\n#]+)'))   AS theme,
  lt.confirmed_failures,
  ROUND(lt.confirmed_failure_rate_pct, 1)   AS confirmed_pct,
  lt.headline_failures,
  ROUND(lt.headline_failure_rate_pct, 1)    AS headline_pct,
  ROUND(lt.suspected_failure_rate_pct, 1)   AS suspected_pct,
  lt.escalated_count,
  ROUND(lt.escalation_rate_pct, 1)          AS esc_pct,
  lt.resolved_positive
FROM `support-analytics-492410.support_analytics.chunk_cluster_descriptions` cd
JOIN `support-analytics-492410.support_analytics.chunk_cluster_label_table`  lt
  USING (cluster_id)
ORDER BY cd.failure_risk_proxy DESC;


-- C3. Routing clusters (escalated ≥ 95%, < 5% failure) vs. problem clusters

SELECT
  is_routing_cluster,
  COUNT(*)                                                                AS n_clusters,
  SUM(n_conversations)                                                     AS total_conversations,
  ROUND(AVG(failure_risk_proxy), 3)                                       AS avg_risk,
  ROUND(AVG(escalation_rate_pct), 1)                                      AS avg_esc_pct
FROM `support-analytics-492410.support_analytics.chunk_cluster_descriptions`
GROUP BY is_routing_cluster;


-- C4. Top 5 highest-risk non-routing clusters with Gemini-suggested actions

SELECT
  cd.cluster_id,
  cd.n_conversations,
  ROUND(cd.failure_risk_proxy, 3)   AS risk,
  TRIM(REGEXP_EXTRACT(cd.cluster_description, r'### Theme\n([^\n#]+)'))
                                    AS theme,
  TRIM(REGEXP_EXTRACT(cd.cluster_description, r'### Failure diagnosis\n([^\n#]+)'))
                                    AS failure_summary,
  TRIM(REGEXP_EXTRACT(cd.cluster_description, r'### Suggested action\n([^\n#]+)'))
                                    AS suggested_action
FROM `support-analytics-492410.support_analytics.chunk_cluster_descriptions` cd
WHERE NOT cd.is_routing_cluster
ORDER BY cd.failure_risk_proxy DESC
LIMIT 5;


-- C5. Gemini description quality check

SELECT
  cluster_id,
  n_conversations,
  gemini_status,
  REGEXP_CONTAINS(cluster_description, r'### Theme')            AS has_theme,
  REGEXP_CONTAINS(cluster_description, r'### Failure diagnosis') AS has_failure,
  REGEXP_CONTAINS(cluster_description, r'### Suggested action')  AS has_action,
  REGEXP_CONTAINS(cluster_description, r'### Evidence quotes')   AS has_quotes,
  LENGTH(cluster_description)                                    AS desc_length
FROM `support-analytics-492410.support_analytics.chunk_cluster_descriptions`
ORDER BY cluster_id;


-- C6. Clusters with > 30% non-English chunks

SELECT
  lt.cluster_id,
  lt.n_conversations,
  lt.languages_in_cluster,
  ROUND(lt.pct_non_english, 1)                                            AS pct_non_english,
  TRIM(REGEXP_EXTRACT(cd.cluster_description, r'### Theme\n([^\n#]+)'))   AS theme
FROM `support-analytics-492410.support_analytics.chunk_cluster_label_table`  lt
JOIN `support-analytics-492410.support_analytics.chunk_cluster_descriptions` cd
  USING (cluster_id)
WHERE lt.pct_non_english > 30
ORDER BY lt.pct_non_english DESC;
