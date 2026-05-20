-- Failure signals, flags, and cluster-level failure rates per conversation.
-- Depends: 03_prepare_docs.sql, 02_deidentify.sql, chunk_cluster_labels | Outputs: failure_signals, failure_flags, failure_by_chunk_cluster, failure_weekly

CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.failure_signals`
AS
WITH

client_pairs AS (
  SELECT
    conversation_id,
    content,
    LAG(content) OVER (
      PARTITION BY conversation_id
      ORDER BY message_date
    ) AS prev_content
  FROM `support-analytics-492410.support_analytics.chatbot_messages_deidentified`
  WHERE agent_or_bot = 'Client'
    AND content IS NOT NULL
    AND TRIM(content) != ''
),

repetition AS (
  SELECT
    conversation_id,
    COUNTIF(
      prev_content IS NOT NULL
      AND LOWER(TRIM(content)) = LOWER(TRIM(prev_content))
    )                                                                AS exact_repeat_pairs,
    -- ED < 5 on msgs >= 10 chars catches rephrasing but avoids matching short greetings
    COUNTIF(
      prev_content IS NOT NULL
      AND LENGTH(content) >= 10
      AND LENGTH(prev_content) >= 10
      AND EDIT_DISTANCE(LOWER(content), LOWER(prev_content)) < 5
    )                                                                AS near_repeat_pairs
  FROM client_pairs
  GROUP BY conversation_id
),

agent_request AS (
  SELECT
    conversation_id,
    REGEXP_CONTAINS(
      LOWER(IFNULL(conversation_text_translated, '')),
      r'\btalk[ _-]?to[ _-]?(an?[ _-])?(agent|human|person|operator|support)\b'
      r'|\bspeak[ _-]?(to|with)[ _-]?(an?[ _-])?(agent|human|person|operator|support)\b'
      r'|\bconnect[ _-]?(me[ _-]?)?(to|with)[ _-]?(an?[ _-])?(agent|human|person|operator|support)\b'
      r'|\bneed[ _-]?(an?[ _-])?(agent|human|person|operator|support)\b'
      r'|\bi[ _-]?want[ _-]?(an?[ _-])?(human|agent|person|operator|real[ _-]?person)\b'
      r'|\blive[ _-]?(agent|chat|support|person)\b'
      r'|\bhuman[ _-]?(agent|support|help)\b'
      r'|\breal[ _-]?person\b'
    )                                                                AS requested_agent
  FROM `support-analytics-492410.support_analytics.conversation_docs`
)

SELECT
  d.conversation_id,
  d.conversation_start_date,
  d.detected_language,
  d.client_message_count,
  d.total_message_count,
  d.rating_score,
  d.was_escalated_to_agent,
  d.ended_without_resolution,
  d.last_author,
  d.category,
  d.agreements_subscription,
  LENGTH(IFNULL(d.conversation_text_translated, ''))                 AS conversation_text_length,
  IFNULL(r.exact_repeat_pairs, 0)                                    AS exact_repeat_pairs,
  IFNULL(r.near_repeat_pairs,  0)                                    AS near_repeat_pairs,
  IFNULL(ar.requested_agent, FALSE)                                  AS requested_agent,
  (IFNULL(r.near_repeat_pairs, 0) >= 1)                              AS is_repetition_loop,
  (IFNULL(ar.requested_agent, FALSE) AND NOT d.was_escalated_to_agent)
                                                                     AS is_agent_request_unserved,
  (IFNULL(ar.requested_agent, FALSE) AND d.was_escalated_to_agent)
                                                                     AS is_agent_request_served

FROM        `support-analytics-492410.support_analytics.conversation_docs` d
LEFT JOIN   repetition      r   USING (conversation_id)
LEFT JOIN   agent_request   ar  USING (conversation_id);


CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.failure_flags`
AS
SELECT
  conversation_id,
  conversation_start_date,
  detected_language,
  client_message_count,
  total_message_count,
  conversation_text_length,
  rating_score,
  was_escalated_to_agent,
  ended_without_resolution,
  last_author,
  category,
  agreements_subscription,
  exact_repeat_pairs,
  near_repeat_pairs,
  requested_agent,
  is_repetition_loop,
  is_agent_request_unserved,
  is_agent_request_served,

  CASE
    WHEN rating_score = -1 AND NOT was_escalated_to_agent
         THEN 'explicit_negative_bot'
    WHEN rating_score = -1 AND was_escalated_to_agent
         THEN 'explicit_negative_escalated'
    WHEN is_repetition_loop
         AND NOT was_escalated_to_agent
         AND rating_score = 0
         THEN 'repetition_loop'
    WHEN is_agent_request_unserved
         AND rating_score = 0
         THEN 'agent_request_unserved'
    WHEN was_escalated_to_agent AND rating_score = 0
         THEN 'escalated_no_feedback'
    WHEN conversation_text_length >= 30
         AND rating_score = 0
         AND NOT was_escalated_to_agent
         AND client_message_count <= 3
         THEN 'abandoned_inquiry'
    WHEN rating_score = 1
         THEN 'resolved_positive'
    WHEN conversation_text_length < 30
         AND rating_score = 0
         AND NOT was_escalated_to_agent
         THEN 'trivial_noise'
    ELSE 'resolved_or_unknown'
  END AS failure_type,

  (rating_score = -1)                                                AS is_confirmed_failure,

  (
    rating_score = 0
    AND NOT was_escalated_to_agent
    AND (is_repetition_loop OR is_agent_request_unserved)
  )                                                                  AS is_probable_failure,

  (
    rating_score = 0
    AND (
      was_escalated_to_agent
      OR (
        conversation_text_length >= 30
        AND NOT was_escalated_to_agent
        AND client_message_count <= 3
        AND NOT is_repetition_loop
        AND NOT is_agent_request_unserved
      )
    )
  )                                                                  AS is_suspected_failure,

  (
    rating_score = -1
    OR (
      rating_score = 0
      AND NOT was_escalated_to_agent
      AND (is_repetition_loop OR is_agent_request_unserved)
    )
  )                                                                  AS is_any_failure,

  (
    rating_score = 0
    AND NOT was_escalated_to_agent
    AND is_repetition_loop
  )                                                                  AS is_probable_repetition_signal,
  (
    rating_score = 0
    AND NOT was_escalated_to_agent
    AND is_agent_request_unserved
  )                                                                  AS is_probable_agent_signal

FROM `support-analytics-492410.support_analytics.failure_signals`;


-- chunk-weighted rates (n_chunks denominator) are the primary metric — better for multi-topic convos
CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.failure_by_chunk_cluster`
AS
WITH
conv_per_cluster AS (
  SELECT DISTINCT cluster_id, conversation_id
  FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`
  WHERE NOT is_noise
),
chunk_counts AS (
  SELECT cluster_id, COUNT(*) AS n_chunks
  FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`
  WHERE NOT is_noise
  GROUP BY cluster_id
),
chunk_failure_weights AS (
  SELECT
    cl.cluster_id,
    COUNTIF(ff.is_confirmed_failure)   AS chunks_confirmed_failure,
    COUNTIF(ff.is_any_failure)         AS chunks_headline_failure,
    COUNTIF(ff.is_probable_failure)    AS chunks_probable_failure,
    COUNTIF(ff.was_escalated_to_agent) AS chunks_escalated,
    COUNTIF(ff.requested_agent)        AS chunks_agent_request
  FROM `support-analytics-492410.support_analytics.chunk_cluster_labels` cl
  LEFT JOIN `support-analytics-492410.support_analytics.failure_flags` ff
    USING (conversation_id)
  WHERE NOT cl.is_noise
  GROUP BY cl.cluster_id
)
SELECT
  cc.cluster_id,
  nc.n_chunks,
  COUNT(DISTINCT cc.conversation_id)                             AS total_conversations,
  COUNTIF(ff.is_confirmed_failure)                               AS confirmed_failures,
  COUNTIF(ff.is_probable_failure)                                AS probable_failures,
  COUNTIF(ff.is_suspected_failure)                               AS suspected_failures,
  COUNTIF(ff.is_any_failure)                                     AS headline_failures,
  COUNTIF(ff.rating_score != 0)                                  AS rated_conversations,
  COUNTIF(ff.rating_score = 1)                                   AS positive_ratings,

  ROUND(
    100 * COUNTIF(ff.is_confirmed_failure)
    / NULLIF(COUNTIF(ff.rating_score != 0), 0), 2
  )                                                              AS confirmed_failure_rate_pct,
  ROUND(100 * COUNTIF(ff.is_any_failure)
    / COUNT(DISTINCT cc.conversation_id), 2)                     AS headline_failure_rate_pct,
  ROUND(100 * COUNTIF(ff.is_suspected_failure)
    / COUNT(DISTINCT cc.conversation_id), 2)                     AS suspected_failure_rate_pct,
  ROUND(100 * COUNTIF(ff.was_escalated_to_agent)
    / COUNT(DISTINCT cc.conversation_id), 2)                     AS escalation_rate_pct,
  ROUND(100 * COUNTIF(ff.requested_agent)
    / COUNT(DISTINCT cc.conversation_id), 2)                     AS agent_request_rate_pct,

  ROUND(100 * cfw.chunks_confirmed_failure / nc.n_chunks, 2)    AS chunk_confirmed_failure_rate_pct,
  ROUND(100 * cfw.chunks_headline_failure  / nc.n_chunks, 2)    AS chunk_headline_failure_rate_pct,
  ROUND(100 * cfw.chunks_probable_failure  / nc.n_chunks, 2)    AS chunk_probable_failure_rate_pct,
  ROUND(100 * cfw.chunks_escalated         / nc.n_chunks, 2)    AS chunk_escalation_rate_pct,
  ROUND(100 * cfw.chunks_agent_request     / nc.n_chunks, 2)    AS chunk_agent_request_rate_pct,

  (COUNTIF(ff.rating_score != 0) >= 20)                          AS reliable_tier1_estimate,
  COUNTIF(ff.failure_type = 'explicit_negative_bot')             AS explicit_negative_bot,
  COUNTIF(ff.failure_type = 'explicit_negative_escalated')       AS explicit_negative_escalated,
  COUNTIF(ff.failure_type = 'repetition_loop')                   AS repetition_loop,
  COUNTIF(ff.failure_type = 'agent_request_unserved')            AS agent_request_unserved,
  COUNTIF(ff.failure_type = 'escalated_no_feedback')             AS escalated_no_feedback,
  COUNTIF(ff.failure_type = 'abandoned_inquiry')                 AS abandoned_inquiry,
  COUNTIF(ff.failure_type = 'trivial_noise')                     AS trivial_noise,
  COUNTIF(ff.failure_type = 'resolved_or_unknown')               AS resolved_or_unknown,
  COUNTIF(ff.failure_type = 'resolved_positive')                 AS resolved_positive,

  ROUND(AVG(ff.client_message_count), 2)                         AS avg_client_messages,
  ROUND(AVG(ff.rating_score), 3)                                 AS avg_rating_score,
  ROUND(nc.n_chunks / COUNT(DISTINCT cc.conversation_id), 2)     AS avg_chunks_per_conv
FROM conv_per_cluster cc
JOIN chunk_counts nc USING (cluster_id)
JOIN chunk_failure_weights cfw USING (cluster_id)
JOIN `support-analytics-492410.support_analytics.failure_flags` ff
  USING (conversation_id)
GROUP BY cc.cluster_id, nc.n_chunks,
  cfw.chunks_confirmed_failure, cfw.chunks_headline_failure,
  cfw.chunks_probable_failure, cfw.chunks_escalated, cfw.chunks_agent_request
ORDER BY chunk_headline_failure_rate_pct DESC;


CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.failure_weekly`
AS
SELECT
  DATE_TRUNC(DATE(conversation_start_date), WEEK)                  AS week_start,
  COUNT(*)                                                         AS total_conversations,
  COUNTIF(is_confirmed_failure)                                    AS confirmed_failures,
  COUNTIF(rating_score != 0)                                       AS rated_conversations,
  ROUND(
    100 * COUNTIF(is_confirmed_failure)
    / NULLIF(COUNTIF(rating_score != 0), 0), 2
  )                                                                AS confirmed_failure_rate_pct,
  COUNTIF(is_any_failure)                                          AS headline_failures,
  ROUND(100 * COUNTIF(is_any_failure) / COUNT(*), 2)               AS headline_failure_rate_pct,
  COUNTIF(is_suspected_failure)                                    AS suspected_failures,
  ROUND(100 * COUNTIF(is_suspected_failure) / COUNT(*), 2)         AS suspected_failure_rate_pct,
  COUNTIF(failure_type = 'explicit_negative_bot')                  AS explicit_negative_bot,
  COUNTIF(failure_type = 'explicit_negative_escalated')            AS explicit_negative_escalated,
  COUNTIF(failure_type = 'repetition_loop')                        AS repetition_loop,
  COUNTIF(failure_type = 'agent_request_unserved')                 AS agent_request_unserved,
  COUNTIF(failure_type = 'escalated_no_feedback')                  AS escalated_no_feedback,
  COUNTIF(failure_type = 'abandoned_inquiry')                      AS abandoned_inquiry,
  COUNTIF(failure_type = 'trivial_noise')                          AS trivial_noise,
  COUNTIF(failure_type = 'resolved_or_unknown')                    AS resolved_or_unknown,
  COUNTIF(failure_type = 'resolved_positive')                      AS resolved_positive,
  ROUND(100 * COUNTIF(was_escalated_to_agent) / COUNT(*), 2)       AS escalation_rate_pct,
  ROUND(100 * COUNTIF(requested_agent)       / COUNT(*), 2)        AS agent_request_rate_pct
FROM `support-analytics-492410.support_analytics.failure_flags`
GROUP BY week_start
ORDER BY week_start;


SELECT
  failure_type,
  COUNT(*)                                                     AS n,
  ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER(), 2)              AS pct_of_all,
  COUNTIF(is_confirmed_failure)                                AS n_confirmed,
  COUNTIF(is_probable_failure)                                 AS n_probable,
  COUNTIF(is_suspected_failure)                                AS n_suspected
FROM `support-analytics-492410.support_analytics.failure_flags`
GROUP BY failure_type
ORDER BY n DESC;

SELECT
  COUNT(*)                                                     AS total_conversations,
  COUNTIF(rating_score != 0)                                   AS rated_conversations,
  COUNTIF(is_confirmed_failure)                                AS confirmed_failures,
  ROUND(
    100 * COUNTIF(is_confirmed_failure)
    / NULLIF(COUNTIF(rating_score != 0), 0), 2
  )                                                            AS confirmed_failure_rate_pct,
  COUNTIF(is_probable_failure)                                 AS probable_failures,
  ROUND(100 * COUNTIF(is_probable_failure) / COUNT(*), 2)      AS probable_failure_rate_pct,
  COUNTIF(is_any_failure)                                      AS headline_failures,
  ROUND(100 * COUNTIF(is_any_failure) / COUNT(*), 2)           AS headline_failure_rate_pct,
  COUNTIF(is_suspected_failure)                                AS suspected_failures,
  ROUND(100 * COUNTIF(is_suspected_failure) / COUNT(*), 2)     AS suspected_failure_rate_pct
FROM `support-analytics-492410.support_analytics.failure_flags`;

SELECT
  COUNTIF(is_repetition_loop)        AS repetition_loops,
  COUNTIF(requested_agent)           AS agent_requests,
  COUNTIF(is_agent_request_unserved) AS agent_requests_unserved,
  COUNTIF(is_agent_request_served)   AS agent_requests_served,
  COUNTIF(was_escalated_to_agent)    AS escalated,
  COUNTIF(rating_score = -1)         AS neg_rating,
  COUNTIF(rating_score = 1)          AS pos_rating,
  COUNT(*)                           AS total
FROM `support-analytics-492410.support_analytics.failure_signals`;

SELECT
  COUNT(DISTINCT week_start)                        AS distinct_weeks,
  MIN(week_start)                                   AS earliest_week,
  MAX(week_start)                                   AS latest_week,
  ROUND(AVG(confirmed_failure_rate_pct), 2)         AS avg_weekly_confirmed_pct,
  ROUND(AVG(headline_failure_rate_pct),  2)         AS avg_weekly_headline_pct,
  ROUND(AVG(suspected_failure_rate_pct), 2)         AS avg_weekly_suspected_pct,
  ROUND(AVG(escalation_rate_pct),        2)         AS avg_weekly_escalation_pct,
  ROUND(AVG(agent_request_rate_pct),     2)         AS avg_weekly_agent_request_pct
FROM `support-analytics-492410.support_analytics.failure_weekly`;
