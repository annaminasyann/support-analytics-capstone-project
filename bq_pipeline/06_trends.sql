-- Weekly volume, cluster trends, and language distribution.
-- Depends: 03_prepare_docs.sql, 05_failure_analysis.sql, chunk_cluster_labels | Outputs: weekly_volume, chunk_cluster_trends, language_trends
CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.weekly_volume`
AS
SELECT
  DATE_TRUNC(DATE(d.conversation_start_date), WEEK) AS week_start,
  COUNT(*)                                          AS total_conversations,
  COUNTIF(d.detected_language = 'en')               AS english_conversations,
  COUNTIF(d.detected_language != 'en')              AS non_english_conversations,
  COUNT(DISTINCT d.detected_language)               AS distinct_languages,
  COUNTIF(d.was_escalated_to_agent)                 AS escalated_count,
  COUNTIF(d.rating_score = -1)                      AS negative_ratings,
  COUNTIF(d.rating_score = 1)                       AS positive_ratings,
  COUNTIF(fs.is_repetition_loop)                    AS repetition_loops,
  COUNTIF(fs.requested_agent)                       AS agent_requests,
  COUNTIF(ff.is_confirmed_failure)                  AS confirmed_failures,
  COUNTIF(ff.is_any_failure)                        AS headline_failures,
  ROUND(100 * COUNTIF(ff.is_any_failure) / COUNT(*), 2)
                                                    AS headline_failure_rate_pct,
  ROUND(100 * COUNTIF(d.was_escalated_to_agent) / COUNT(*), 2)
                                                    AS escalation_rate_pct,
  ROUND(
    100 * COUNTIF(d.rating_score = -1)
    / NULLIF(COUNTIF(d.rating_score != 0), 0), 2
  )                                                 AS negative_rating_pct,
  ROUND(AVG(d.client_message_count), 2)             AS avg_client_messages,
  ROUND(AVG(d.total_message_count), 2)              AS avg_total_messages
FROM        `support-analytics-492410.support_analytics.conversation_docs`  d
LEFT JOIN   `support-analytics-492410.support_analytics.failure_signals`    fs
       USING (conversation_id)
LEFT JOIN   `support-analytics-492410.support_analytics.failure_flags`      ff
       USING (conversation_id)
GROUP BY week_start
ORDER BY week_start;


-- chunk_count weights by topic depth, not just conversation count
CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.chunk_cluster_trends`
AS
WITH
conv_per_cluster AS (
  SELECT DISTINCT cluster_id, conversation_id
  FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`
  WHERE NOT is_noise
),
chunk_week_counts AS (
  SELECT
    DATE_TRUNC(DATE(d.conversation_start_date), WEEK) AS week_start,
    cl.cluster_id,
    COUNT(*) AS chunk_count
  FROM `support-analytics-492410.support_analytics.chunk_cluster_labels` cl
  JOIN `support-analytics-492410.support_analytics.conversation_docs` d
    USING (conversation_id)
  WHERE NOT cl.is_noise
  GROUP BY 1, 2
)
SELECT
  DATE_TRUNC(DATE(d.conversation_start_date), WEEK) AS week_start,
  cc.cluster_id,
  COUNT(DISTINCT cc.conversation_id)                AS conversation_count,
  MAX(cwc.chunk_count)                              AS chunk_count,
  COUNTIF(d.was_escalated_to_agent)                 AS escalated_count,
  COUNTIF(d.rating_score = -1)                      AS negative_count,
  COUNTIF(ff.is_any_failure)                        AS headline_failures,
  COUNTIF(fs.is_repetition_loop)                    AS repetition_loops,
  COUNTIF(fs.requested_agent)                       AS agent_requests,
  ROUND(AVG(d.client_message_count), 2)             AS avg_client_messages
FROM conv_per_cluster cc
JOIN       `support-analytics-492410.support_analytics.conversation_docs`  d
      USING (conversation_id)
LEFT JOIN  `support-analytics-492410.support_analytics.failure_signals`    fs
      USING (conversation_id)
LEFT JOIN  `support-analytics-492410.support_analytics.failure_flags`      ff
      USING (conversation_id)
JOIN       chunk_week_counts cwc
      ON   cwc.cluster_id = cc.cluster_id
      AND  cwc.week_start = DATE_TRUNC(DATE(d.conversation_start_date), WEEK)
GROUP BY week_start, cc.cluster_id
ORDER BY cc.cluster_id, week_start;


CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.language_trends`
AS
WITH monthly AS (
  SELECT
    DATE_TRUNC(DATE(conversation_start_date), MONTH) AS month_start,
    detected_language,
    COUNT(*)                                         AS conversation_count
  FROM `support-analytics-492410.support_analytics.conversation_docs`
  GROUP BY month_start, detected_language
)
SELECT
  month_start,
  detected_language,
  conversation_count,
  ROUND(
    100 * conversation_count
    / SUM(conversation_count) OVER (PARTITION BY month_start),
    2
  ) AS pct_of_month
FROM monthly
ORDER BY month_start, conversation_count DESC;


SELECT
  MIN(week_start)             AS first_week,
  MAX(week_start)             AS last_week,
  COUNT(*)                    AS total_weeks,
  SUM(total_conversations)    AS grand_total
FROM `support-analytics-492410.support_analytics.weekly_volume`;
