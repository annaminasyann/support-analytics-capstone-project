-- cluster IDs from NB05 OLS (slope < -0.05, p < 0.05, 31-week window)
-- Depends: 06_trends.sql, 07_chunk_evaluation.sql

WITH declining AS (
  SELECT cluster_id FROM UNNEST([1, 2, 3, 4, 5, 8, 9, 10, 11, 13, 15, 19, 20]) AS cluster_id
)

SELECT
  d.cluster_id,
  cd.n_chunks,
  cd.n_conversations,
  cd.chunk_headline_failure_rate_pct,
  cd.headline_failure_rate_pct,
  cd.cluster_description
FROM declining d
JOIN `support-analytics-492410.support_analytics.chunk_cluster_descriptions` cd
  USING (cluster_id)
ORDER BY d.cluster_id;
