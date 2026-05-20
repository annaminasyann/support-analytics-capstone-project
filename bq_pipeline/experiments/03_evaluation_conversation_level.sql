-- [EXPERIMENT] Conversation-level cluster evaluation — not the canonical chunk pipeline.
-- Depends: cluster_labels, conversation_docs, failure_flags, gemini_model | Outputs: cluster_label_table, cluster_size_comparison, cluster_descriptions


CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.cluster_label_table`
AS
WITH

cluster_base AS (
  SELECT
    cl.pipeline_track,
    cl.cluster_id,
    COUNT(*)                                                       AS cluster_size,
    COUNTIF(d.detected_language = 'en')                            AS english_count,
    COUNTIF(d.detected_language != 'en')                           AS non_english_count,
    ROUND(100 * COUNTIF(d.detected_language != 'en') / COUNT(*), 1)
                                                                   AS pct_non_english,
    STRING_AGG(DISTINCT d.detected_language ORDER BY d.detected_language LIMIT 10)
                                                                   AS languages_in_cluster,
    COUNTIF(ff.is_confirmed_failure)                               AS confirmed_failures,
    COUNTIF(ff.is_probable_failure)                                AS probable_failures,
    COUNTIF(ff.is_suspected_failure)                               AS suspected_failures,
    COUNTIF(ff.is_any_failure)                                     AS headline_failures,
    COUNTIF(ff.rating_score != 0)                                  AS rated_conversations,
    -- Tier-1: rated convos only as denominator
    ROUND(
      100 * COUNTIF(ff.is_confirmed_failure)
      / NULLIF(COUNTIF(ff.rating_score != 0), 0), 2
    )                                                              AS confirmed_failure_rate_pct,
    ROUND(100 * COUNTIF(ff.is_any_failure)       / COUNT(*), 2)    AS headline_failure_rate_pct,
    ROUND(100 * COUNTIF(ff.is_suspected_failure) / COUNT(*), 2)    AS suspected_failure_rate_pct,
    COUNTIF(ff.failure_type = 'explicit_negative_bot')             AS explicit_negative_bot,
    COUNTIF(ff.failure_type = 'explicit_negative_escalated')       AS explicit_negative_escalated,
    COUNTIF(ff.failure_type = 'repetition_loop')                   AS repetition_loop,
    COUNTIF(ff.failure_type = 'agent_request_unserved')            AS agent_request_unserved,
    COUNTIF(ff.failure_type = 'escalated_no_feedback')             AS escalated_no_feedback,
    COUNTIF(ff.failure_type = 'abandoned_inquiry')                 AS abandoned_inquiry,
    COUNTIF(ff.failure_type = 'trivial_noise')                     AS trivial_noise,
    COUNTIF(ff.failure_type = 'resolved_positive')                 AS resolved_positive,
    COUNTIF(ff.failure_type = 'resolved_or_unknown')               AS resolved_or_unknown,
    COUNTIF(ff.was_escalated_to_agent)                             AS escalated_count,
    COUNTIF(ff.requested_agent)                                    AS agent_requests,
    ROUND(100 * COUNTIF(ff.was_escalated_to_agent) / COUNT(*), 1)  AS escalation_rate_pct,
    ROUND(100 * COUNTIF(ff.requested_agent)        / COUNT(*), 1)  AS agent_request_rate_pct,
    MIN(DATE(d.conversation_start_date))                           AS first_conversation,
    MAX(DATE(d.conversation_start_date))                           AS last_conversation,
    ROUND(AVG(d.client_message_count), 2)                          AS avg_client_messages,
    ROUND(AVG(d.total_message_count),  2)                          AS avg_total_messages
  FROM        `support-analytics-492410.support_analytics.cluster_labels` cl
  JOIN        `support-analytics-492410.support_analytics.conversation_docs` d
         USING (conversation_id)
  LEFT JOIN   `support-analytics-492410.support_analytics.failure_flags`   ff
         USING (conversation_id)
  WHERE cl.cluster_id != -1
  GROUP BY cl.pipeline_track, cl.cluster_id
),

cluster_samples_raw AS (
  SELECT
    cl.pipeline_track,
    cl.cluster_id,
    d.conversation_id,
    d.conversation_start_date,
    CASE
      WHEN cl.pipeline_track = 'multilingual'
        THEN SUBSTR(IFNULL(d.conversation_text_multilingual, ''), 1, 400)
      ELSE SUBSTR(IFNULL(d.conversation_text_translated,   ''), 1, 400)
    END AS snippet,
    CASE
      WHEN ff.failure_type IN ('explicit_negative_bot',
                               'explicit_negative_escalated')       THEN 'negative'
      WHEN ff.failure_type IN ('repetition_loop',
                               'agent_request_unserved')            THEN 'probable_failure'
      WHEN ff.failure_type = 'resolved_positive'                    THEN 'positive'
      ELSE                                                               'recent'
    END AS stratum,
    ROW_NUMBER() OVER (
      PARTITION BY
        cl.pipeline_track,
        cl.cluster_id,
        CASE
          WHEN ff.failure_type IN ('explicit_negative_bot',
                                   'explicit_negative_escalated')   THEN 'negative'
          WHEN ff.failure_type IN ('repetition_loop',
                                   'agent_request_unserved')        THEN 'probable_failure'
          WHEN ff.failure_type = 'resolved_positive'                THEN 'positive'
          ELSE                                                           'recent'
        END
      ORDER BY d.conversation_start_date DESC
    ) AS rn
  FROM        `support-analytics-492410.support_analytics.cluster_labels` cl
  JOIN        `support-analytics-492410.support_analytics.conversation_docs` d
         USING (conversation_id)
  LEFT JOIN   `support-analytics-492410.support_analytics.failure_flags`   ff
         USING (conversation_id)
  WHERE cl.cluster_id != -1
    AND (
      CASE
        WHEN cl.pipeline_track = 'multilingual'
          THEN d.conversation_text_multilingual
        ELSE d.conversation_text_translated
      END
    ) IS NOT NULL
),

cluster_samples AS (
  SELECT
    pipeline_track,
    cluster_id,
    STRING_AGG(
      CONCAT('[', stratum, '] ', snippet),
      '
'
      ORDER BY
        CASE stratum
          WHEN 'negative'         THEN 1
          WHEN 'probable_failure' THEN 2
          WHEN 'positive'         THEN 3
          ELSE                         4
        END,
        rn
    ) AS sample_texts
  FROM cluster_samples_raw
  WHERE rn <= 3
  GROUP BY pipeline_track, cluster_id
)

SELECT
  cb.*,
  cs.sample_texts,
  ROUND(
      0.8 * SAFE_DIVIDE(CAST(cb.headline_failures AS FLOAT64),   cb.cluster_size)
    + 0.1 * SAFE_DIVIDE(CAST(cb.escalated_count   AS FLOAT64),   cb.cluster_size)
    + 0.1 * SAFE_DIVIDE(CAST(cb.abandoned_inquiry AS FLOAT64),   cb.cluster_size),
    4
  ) AS failure_risk_proxy,

  -- routing clusters (escalated ≥ 95%, < 5% failure) are excluded from failure ranking
  (
    SAFE_DIVIDE(CAST(cb.escalated_count AS FLOAT64), cb.cluster_size) >= 0.95
    AND SAFE_DIVIDE(CAST(cb.headline_failures AS FLOAT64), cb.cluster_size) < 0.05
  ) AS is_routing_cluster

FROM      cluster_base    cb
LEFT JOIN cluster_samples cs USING (pipeline_track, cluster_id)
ORDER BY  cb.pipeline_track, cb.cluster_size DESC;


CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.cluster_size_comparison`
AS
WITH sized AS (
  SELECT
    pipeline_track,
    cluster_id,
    COUNT(*) AS cluster_size
  FROM `support-analytics-492410.support_analytics.cluster_labels`
  GROUP BY pipeline_track, cluster_id
),
track_n_clusters AS (
  SELECT
    pipeline_track,
    COUNT(DISTINCT IF(cluster_id != -1, cluster_id, NULL)) AS n_clusters
  FROM sized
  GROUP BY pipeline_track
)
SELECT
  s.pipeline_track,
  tc.n_clusters,
  SUM(IF(s.cluster_id = -1, s.cluster_size, 0))
    OVER (PARTITION BY s.pipeline_track)                               AS noise_points,
  s.cluster_id,
  s.cluster_size,
  ROUND(
    100 * s.cluster_size / SUM(s.cluster_size) OVER (PARTITION BY s.pipeline_track), 2
  )                                                                    AS pct_of_total
FROM sized s
JOIN track_n_clusters tc USING (pipeline_track)
ORDER BY s.pipeline_track, s.cluster_size DESC;


SELECT
  pipeline_track,
  COUNT(DISTINCT IF(cluster_id != -1, cluster_id, NULL))              AS n_clusters,
  SUM(IF(cluster_id = -1,  cluster_size, 0))                          AS noise_points,
  SUM(IF(cluster_id != -1, cluster_size, 0))                          AS clustered_points,
  ROUND(
    100 * SUM(IF(cluster_id = -1, cluster_size, 0)) / SUM(cluster_size), 2
  )                                                                    AS noise_pct,
  MIN(IF(cluster_id != -1, cluster_size, NULL))                       AS smallest_cluster,
  MAX(IF(cluster_id != -1, cluster_size, NULL))                       AS largest_cluster,
  ROUND(AVG(IF(cluster_id != -1, cluster_size, NULL)), 1)             AS avg_cluster_size
FROM (
  SELECT pipeline_track, cluster_id, COUNT(*) AS cluster_size
  FROM `support-analytics-492410.support_analytics.cluster_labels`
  GROUP BY pipeline_track, cluster_id
)
GROUP BY pipeline_track
ORDER BY pipeline_track;


CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.cluster_descriptions`
AS
SELECT
  pipeline_track,
  cluster_id,
  cluster_size,
  languages_in_cluster,
  escalation_rate_pct,
  agent_request_rate_pct,
  confirmed_failures,
  headline_failures,
  confirmed_failure_rate_pct,
  headline_failure_rate_pct,
  suspected_failure_rate_pct,
  failure_risk_proxy,
  is_routing_cluster,
  TRIM(ml_generate_text_llm_result) AS cluster_description,
  ml_generate_text_status           AS gemini_status
FROM ML.GENERATE_TEXT(
  MODEL `support-analytics-492410.support_analytics.gemini_model`,
  (
    SELECT
      pipeline_track,
      cluster_id,
      cluster_size,
      languages_in_cluster,
      escalation_rate_pct,
      agent_request_rate_pct,
      confirmed_failures,
      headline_failures,
      confirmed_failure_rate_pct,
      headline_failure_rate_pct,
      suspected_failure_rate_pct,
      failure_risk_proxy,
      is_routing_cluster,
      CONCAT(
        '## CONTEXT
',
        'You are a support operations analyst at 10Web — a cloud-based AI website builder.
',
        '10Web sells: AI-powered site creation, WordPress hosting, domain registration, ',
        'and agency/white-label plans. Customers contact the chatbot for tier-0 support ',
        'before an agent joins.

',
        'De-identification: all PII is masked. Tokens you will see:
',
        '  [URL] = any web address    [DOMAIN] = domain name    [PHONE] = phone number
',
        '  [PERSON] = personal name   [LOCATION] = place name   [CREDENTIAL] = password / key
',
        '  <image url> = an attached image or screenshot
',
        '"Talk to an Agent" / "Talk to an agent" in excerpts is a CHATBOT BUTTON CLICK, ',
        'not typed client text. Treat it as a signal that the user requested escalation.

',

        '## CLUSTER METADATA
',
        '  Pipeline track              : ', pipeline_track, '
',
        '  Cluster ID                  : ', CAST(cluster_id AS STRING), '
',
        '  Size                        : ', CAST(cluster_size AS STRING), ' conversations
',
        '  Active period               : ',
            CAST(first_conversation AS STRING), ' → ', CAST(last_conversation AS STRING), '
',
        '  Languages present           : ', COALESCE(languages_in_cluster, 'unknown'), '
',
        '  Non-English share           : ', CAST(pct_non_english AS STRING), '