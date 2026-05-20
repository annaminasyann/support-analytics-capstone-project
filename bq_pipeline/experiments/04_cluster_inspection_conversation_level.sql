-- [EXPERIMENT] Conversation-level cluster inspection — not the canonical chunk pipeline.
-- Run each block independently. Change track below to 'multilingual' (Track A) or 'translated' (Track B).
-- Depends: cluster_labels, conversation_docs

DECLARE track STRING DEFAULT 'multilingual';  -- ← change here


-- 1. Cluster size distribution

WITH sizes AS (
  SELECT
    cluster_id,
    COUNT(*) AS n_conversations,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
  FROM `support-analytics-492410.support_analytics.cluster_labels`
  WHERE pipeline_track = track
  GROUP BY cluster_id
)
SELECT
  cluster_id,
  n_conversations,
  pct_of_total,
  CASE
    WHEN cluster_id = -1            THEN 'NOISE'
    WHEN n_conversations > 5000     THEN 'VERY LARGE — inspect manually'
    WHEN n_conversations > 1000     THEN 'large'
    WHEN n_conversations > 200      THEN 'medium'
    WHEN n_conversations < 50       THEN 'tiny — may be fragmented'
    ELSE 'normal'
  END AS size_flag
FROM sizes
ORDER BY n_conversations DESC;


-- 2. Sample conversations from the 5 largest clusters (5 highest-confidence members each)

WITH top_clusters AS (
  SELECT cluster_id
  FROM `support-analytics-492410.support_analytics.cluster_labels`
  WHERE pipeline_track = track AND cluster_id != -1
  GROUP BY cluster_id
  ORDER BY COUNT(*) DESC
  LIMIT 5
),
ranked AS (
  SELECT
    cl.cluster_id,
    cl.membership_prob,
    d.detected_language,
    d.category,
    d.client_message_count,
    d.conversation_text_multilingual,
    ROW_NUMBER() OVER (
      PARTITION BY cl.cluster_id
      ORDER BY cl.membership_prob DESC
    ) AS rn
  FROM `support-analytics-492410.support_analytics.cluster_labels` cl
  JOIN `support-analytics-492410.support_analytics.conversation_docs` d
    ON cl.conversation_id = d.conversation_id
  JOIN top_clusters t ON cl.cluster_id = t.cluster_id
  WHERE cl.pipeline_track = track
)
SELECT
  cluster_id,
  rn AS rank_in_cluster,
  membership_prob,
  detected_language,
  category,
  client_message_count,
  LEFT(conversation_text_multilingual, 400) AS text_preview
FROM ranked
WHERE rn <= 5
ORDER BY cluster_id, rn;


-- 3. Noise conversations — shortest first

SELECT
  cl.membership_prob,
  d.detected_language,
  d.category,
  d.client_message_count,
  d.was_escalated_to_agent,
  LENGTH(d.conversation_text_multilingual) AS text_len,
  LEFT(d.conversation_text_multilingual, 500) AS text_preview
FROM `support-analytics-492410.support_analytics.cluster_labels` cl
JOIN `support-analytics-492410.support_analytics.conversation_docs` d
  ON cl.conversation_id = d.conversation_id
WHERE cl.pipeline_track = track
  AND cl.cluster_id = -1
ORDER BY LENGTH(d.conversation_text_multilingual) ASC
LIMIT 50;


-- 4. Noise conversations — longest first

SELECT
  d.detected_language,
  d.category,
  d.client_message_count,
  d.was_escalated_to_agent,
  LENGTH(d.conversation_text_multilingual) AS text_len,
  LEFT(d.conversation_text_multilingual, 600) AS text_preview
FROM `support-analytics-492410.support_analytics.cluster_labels` cl
JOIN `support-analytics-492410.support_analytics.conversation_docs` d
  ON cl.conversation_id = d.conversation_id
WHERE cl.pipeline_track = track
  AND cl.cluster_id = -1
ORDER BY LENGTH(d.conversation_text_multilingual) DESC
LIMIT 30;


-- 5. Tiny clusters (< 100 conversations) — 3 samples each

WITH small_clusters AS (
  SELECT cluster_id, COUNT(*) AS n
  FROM `support-analytics-492410.support_analytics.cluster_labels`
  WHERE pipeline_track = track AND cluster_id != -1
  GROUP BY cluster_id
  HAVING COUNT(*) < 100
),
ranked AS (
  SELECT
    cl.cluster_id,
    s.n AS cluster_size,
    cl.membership_prob,
    d.detected_language,
    d.category,
    d.client_message_count,
    d.conversation_text_multilingual,
    ROW_NUMBER() OVER (
      PARTITION BY cl.cluster_id ORDER BY cl.membership_prob DESC
    ) AS rn
  FROM `support-analytics-492410.support_analytics.cluster_labels` cl
  JOIN `support-analytics-492410.support_analytics.conversation_docs` d
    ON cl.conversation_id = d.conversation_id
  JOIN small_clusters s ON cl.cluster_id = s.cluster_id
  WHERE cl.pipeline_track = track
)
SELECT
  cluster_id,
  cluster_size,
  membership_prob,
  detected_language,
  category,
  client_message_count,
  LEFT(conversation_text_multilingual, 400) AS text_preview
FROM ranked
WHERE rn <= 3
ORDER BY cluster_size ASC, cluster_id, rn;


-- 6. Membership probability distribution

SELECT
  CASE
    WHEN cluster_id = -1           THEN '[-1] noise (hard excluded)'
    WHEN membership_prob < 0.1     THEN '[0.0–0.1] boundary'
    WHEN membership_prob < 0.3     THEN '[0.1–0.3]'
    WHEN membership_prob < 0.5     THEN '[0.3–0.5]'
    WHEN membership_prob < 0.7     THEN '[0.5–0.7]'
    WHEN membership_prob < 0.9     THEN '[0.7–0.9]'
    ELSE                                '[0.9–1.0] core'
  END AS prob_bucket,
  COUNT(*)                                               AS n,
  ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)    AS pct
FROM `support-analytics-492410.support_analytics.cluster_labels`
WHERE pipeline_track = track
GROUP BY prob_bucket
ORDER BY prob_bucket;