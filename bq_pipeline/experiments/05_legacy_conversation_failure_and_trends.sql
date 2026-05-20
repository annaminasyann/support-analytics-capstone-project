-- [EXPERIMENT] Conversation-level failure rates and trends — not the canonical chunk pipeline.
-- Depends: cluster_labels, failure_flags, failure_signals, conversation_docs | Outputs: failure_by_cluster, cluster_trends


CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.failure_by_cluster`
AS
SELECT
  cl.pipeline_track,
  cl.cluster_id,
  COUNT(*)                                                       AS total_conversations,
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
  ROUND(100 * COUNTIF(ff.is_any_failure)       / COUNT(*), 2)    AS headline_failure_rate_pct,
  ROUND(100 * COUNTIF(ff.is_suspected_failure) / COUNT(*), 2)    AS suspected_failure_rate_pct,
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
  ROUND(100 * COUNTIF(ff.requested_agent)  / COUNT(*), 2)        AS agent_request_rate_pct,
  ROUND(100 * COUNTIF(ff.was_escalated_to_agent) / COUNT(*), 2)  AS escalation_rate_pct,
  ROUND(AVG(ff.client_message_count), 2)                         AS avg_client_messages,
  ROUND(AVG(ff.rating_score), 3)                                 AS avg_rating_score
FROM      `support-analytics-492410.support_analytics.cluster_labels` cl
JOIN      `support-analytics-492410.support_analytics.failure_flags`  ff
     USING (conversation_id)
WHERE cl.cluster_id != -1
GROUP BY cl.pipeline_track, cl.cluster_id
ORDER BY cl.pipeline_track, headline_failure_rate_pct DESC;


CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.cluster_trends`
AS
SELECT
  DATE_TRUNC(DATE(d.conversation_start_date), WEEK) AS week_start,
  cl.pipeline_track,
  cl.cluster_id,
  COUNT(*)                                          AS conversation_count,
  COUNTIF(d.was_escalated_to_agent)                 AS escalated_count,
  COUNTIF(d.rating_score = -1)                      AS negative_count,
  COUNTIF(ff.is_any_failure)                        AS headline_failures,
  COUNTIF(fs.is_repetition_loop)                    AS repetition_loops,
  COUNTIF(fs.requested_agent)                       AS agent_requests,
  ROUND(AVG(d.client_message_count), 2)             AS avg_client_messages
FROM       `support-analytics-492410.support_analytics.cluster_labels`      cl
JOIN       `support-analytics-492410.support_analytics.conversation_docs`   d
      USING (conversation_id)
LEFT JOIN  `support-analytics-492410.support_analytics.failure_signals`     fs
      USING (conversation_id)
LEFT JOIN  `support-analytics-492410.support_analytics.failure_flags`       ff
      USING (conversation_id)
WHERE cl.cluster_id != -1
GROUP BY week_start, cl.pipeline_track, cl.cluster_id
ORDER BY cl.pipeline_track, cl.cluster_id, week_start;
