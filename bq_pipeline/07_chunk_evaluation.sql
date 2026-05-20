-- Per-cluster profiles and Gemini topic labels (finetuned pass).
-- Depends: chunk_cluster_labels, 05_failure_analysis.sql | Outputs: chunk_cluster_label_table, chunk_cluster_descriptions
CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.chunk_cluster_label_table`
AS
WITH

chunk_agg AS (
  SELECT
    cluster_id,
    COUNT(*)                        AS n_chunks,
    COUNT(DISTINCT conversation_id) AS n_conversations,
    ROUND(AVG(hdbscan_prob), 3)     AS avg_hdbscan_prob
  FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`
  WHERE NOT is_noise
  GROUP BY cluster_id
),

conv_per_cluster AS (
  SELECT DISTINCT cluster_id, conversation_id
  FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`
  WHERE NOT is_noise
),

chunk_failure AS (
  SELECT
    cl.cluster_id,
    COUNTIF(ff.is_confirmed_failure)   AS chunks_confirmed_failure,
    COUNTIF(ff.is_any_failure)         AS chunks_headline_failure,
    COUNTIF(ff.was_escalated_to_agent) AS chunks_escalated,
    COUNTIF(ff.requested_agent)        AS chunks_agent_request
  FROM `support-analytics-492410.support_analytics.chunk_cluster_labels` cl
  LEFT JOIN `support-analytics-492410.support_analytics.failure_flags` ff
    USING (conversation_id)
  WHERE NOT cl.is_noise
  GROUP BY cl.cluster_id
),

cluster_base AS (
  SELECT
    ca.cluster_id,
    ca.n_chunks,
    ca.n_conversations,
    ca.avg_hdbscan_prob,
    COUNTIF(d.detected_language = 'en')                                    AS english_count,
    COUNTIF(d.detected_language != 'en')                                   AS non_english_count,
    ROUND(100 * COUNTIF(d.detected_language != 'en') / COUNT(*), 1)        AS pct_non_english,
    STRING_AGG(DISTINCT d.detected_language ORDER BY d.detected_language LIMIT 10)
                                                                           AS languages_in_cluster,
    COUNTIF(ff.is_confirmed_failure)                                       AS confirmed_failures,
    COUNTIF(ff.is_probable_failure)                                        AS probable_failures,
    COUNTIF(ff.is_suspected_failure)                                       AS suspected_failures,
    COUNTIF(ff.is_any_failure)                                             AS headline_failures,
    COUNTIF(ff.rating_score != 0)                                          AS rated_conversations,
    ROUND(
      100 * COUNTIF(ff.is_confirmed_failure)
      / NULLIF(COUNTIF(ff.rating_score != 0), 0), 2
    )                                                                      AS confirmed_failure_rate_pct,
    ROUND(100 * COUNTIF(ff.is_any_failure)
      / COUNT(DISTINCT cpc.conversation_id), 2)                            AS headline_failure_rate_pct,
    ROUND(100 * COUNTIF(ff.is_suspected_failure)
      / COUNT(DISTINCT cpc.conversation_id), 2)                            AS suspected_failure_rate_pct,
    ROUND(100 * MAX(cf.chunks_confirmed_failure) / ca.n_chunks, 2)        AS chunk_confirmed_failure_rate_pct,
    ROUND(100 * MAX(cf.chunks_headline_failure)  / ca.n_chunks, 2)        AS chunk_headline_failure_rate_pct,
    ROUND(100 * MAX(cf.chunks_escalated)         / ca.n_chunks, 2)        AS chunk_escalation_rate_pct,
    ROUND(100 * MAX(cf.chunks_agent_request)     / ca.n_chunks, 2)        AS chunk_agent_request_rate_pct,
    COUNTIF(ff.failure_type = 'explicit_negative_bot')                     AS explicit_negative_bot,
    COUNTIF(ff.failure_type = 'explicit_negative_escalated')               AS explicit_negative_escalated,
    COUNTIF(ff.failure_type = 'repetition_loop')                           AS repetition_loop,
    COUNTIF(ff.failure_type = 'agent_request_unserved')                    AS agent_request_unserved,
    COUNTIF(ff.failure_type = 'escalated_no_feedback')                     AS escalated_no_feedback,
    COUNTIF(ff.failure_type = 'abandoned_inquiry')                         AS abandoned_inquiry,
    COUNTIF(ff.failure_type = 'trivial_noise')                             AS trivial_noise,
    COUNTIF(ff.failure_type = 'resolved_positive')                         AS resolved_positive,
    COUNTIF(ff.failure_type = 'resolved_or_unknown')                       AS resolved_or_unknown,
    COUNTIF(ff.was_escalated_to_agent)                                     AS escalated_count,
    COUNTIF(ff.requested_agent)                                            AS agent_requests,
    ROUND(100 * COUNTIF(ff.was_escalated_to_agent)
      / COUNT(DISTINCT cpc.conversation_id), 1)                            AS escalation_rate_pct,
    ROUND(100 * COUNTIF(ff.requested_agent)
      / COUNT(DISTINCT cpc.conversation_id), 1)                            AS agent_request_rate_pct,
    MIN(DATE(d.conversation_start_date))                                   AS first_conversation,
    MAX(DATE(d.conversation_start_date))                                   AS last_conversation,
    ROUND(AVG(d.client_message_count), 2)                                  AS avg_client_messages,
    ROUND(AVG(d.total_message_count),  2)                                  AS avg_total_messages,
    ROUND(ca.n_chunks / ca.n_conversations, 2)                             AS avg_chunks_per_conv
  FROM        conv_per_cluster cpc
  JOIN        chunk_agg ca                                                          USING (cluster_id)
  JOIN        `support-analytics-492410.support_analytics.conversation_docs`       d
         ON  cpc.conversation_id = d.conversation_id
  LEFT JOIN   `support-analytics-492410.support_analytics.failure_flags`           ff
         ON  cpc.conversation_id = ff.conversation_id
  JOIN        chunk_failure cf                                                       USING (cluster_id)
  GROUP BY ca.cluster_id, ca.n_chunks, ca.n_conversations, ca.avg_hdbscan_prob
),

cluster_centroids AS (
  SELECT
    cluster_id,
    AVG(umap_x) AS centroid_x,
    AVG(umap_y) AS centroid_y
  FROM `support-analytics-492410.support_analytics.chunk_cluster_labels`
  WHERE NOT is_noise
  GROUP BY cluster_id
),

chunk_samples_scored AS (
  SELECT
    cl.cluster_id,
    cl.conversation_id,
    SUBSTR(cl.translated_chunk_text, 1, 900)                              AS snippet,
    cl.hdbscan_prob,
    SQRT(POW(cl.umap_x - cc.centroid_x, 2) + POW(cl.umap_y - cc.centroid_y, 2))
                                                                          AS dist_from_centroid,
    CASE
      WHEN ff.conversation_id IS NULL
        THEN 'no_failure_flags'
      WHEN ff.failure_type IN ('explicit_negative_bot',
                               'explicit_negative_escalated')
        THEN 'negative'
      WHEN ff.failure_type IN ('repetition_loop',
                               'agent_request_unserved')
        THEN 'probable_failure'
      WHEN ff.failure_type = 'resolved_positive'
        THEN 'positive'
      WHEN ff.failure_type = 'abandoned_inquiry'
        THEN 'abandoned_inquiry'
      WHEN ff.failure_type = 'escalated_no_feedback'
        THEN 'escalated_no_feedback'
      WHEN ff.failure_type = 'resolved_or_unknown'
        THEN 'resolved_or_unknown'
      WHEN ff.failure_type = 'trivial_noise'
        THEN 'trivial_noise'
      ELSE 'unmapped_failure'
    END AS stratum
  FROM        `support-analytics-492410.support_analytics.chunk_cluster_labels`  cl
  LEFT JOIN   `support-analytics-492410.support_analytics.failure_flags`         ff
         USING (conversation_id)
  JOIN        cluster_centroids                                                   cc
         USING (cluster_id)
  WHERE NOT cl.is_noise
    AND cl.translated_chunk_text IS NOT NULL
    AND LENGTH(TRIM(cl.translated_chunk_text)) >= 20
    AND NOT REGEXP_CONTAINS(
      LOWER(cl.translated_chunk_text),
      r'were my answers helpful|how would you rate the support|it was a pleasure assisting'
      r'|you.ll receive an nps survey|i.m now ending this chat session'
      r'|please don.t hesitate to start a new one'
    )
),

chunk_samples_ringed AS (
  SELECT
    *,
    NTILE(5) OVER (PARTITION BY cluster_id ORDER BY dist_from_centroid) AS dist_ring
  FROM chunk_samples_scored
),

chunk_samples_raw AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY cluster_id, stratum
      ORDER BY hdbscan_prob DESC
    ) AS rn_stratum,
    ROW_NUMBER() OVER (
      PARTITION BY cluster_id, dist_ring
      ORDER BY hdbscan_prob DESC
    ) AS rn_ring,
    ROW_NUMBER() OVER (
      PARTITION BY cluster_id
      ORDER BY hdbscan_prob DESC
    ) AS rn_global
  FROM chunk_samples_ringed
),

chunk_samples AS (
  SELECT
    cluster_id,
    STRING_AGG(
      CONCAT('[ring', CAST(dist_ring AS STRING), '|', stratum, '] ', snippet),
      '\n'
      ORDER BY
        dist_ring,
        rn_ring,
        CASE stratum
          WHEN 'negative'               THEN 1
          WHEN 'probable_failure'       THEN 2
          WHEN 'abandoned_inquiry'      THEN 3
          WHEN 'escalated_no_feedback'  THEN 4
          WHEN 'resolved_or_unknown'    THEN 5
          WHEN 'trivial_noise'          THEN 6
          WHEN 'positive'               THEN 7
          WHEN 'no_failure_flags'       THEN 8
          WHEN 'unmapped_failure'       THEN 9
          ELSE 99
        END
    ) AS sample_texts
  FROM chunk_samples_raw
  WHERE rn_ring <= 2 OR rn_stratum <= 1 OR rn_global <= 10
  GROUP BY cluster_id
)

SELECT
  cb.*,
  cs.sample_texts,
  ROUND(
      0.8 * SAFE_DIVIDE(cb.chunk_headline_failure_rate_pct, 100)
    + 0.1 * SAFE_DIVIDE(cb.chunk_escalation_rate_pct,       100)
    + 0.1 * SAFE_DIVIDE(CAST(cb.abandoned_inquiry AS FLOAT64), cb.n_conversations),
    4
  ) AS failure_risk_proxy,
  (
    SAFE_DIVIDE(CAST(cb.escalated_count AS FLOAT64), cb.n_conversations) >= 0.95
    AND SAFE_DIVIDE(CAST(cb.headline_failures AS FLOAT64), cb.n_conversations) < 0.05
  ) AS is_routing_cluster
FROM      cluster_base  cb
LEFT JOIN chunk_samples cs USING (cluster_id)
ORDER BY  cb.n_chunks DESC;


CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.chunk_cluster_descriptions`
AS
SELECT
  cluster_id,
  n_chunks,
  n_conversations,
  languages_in_cluster,
  avg_hdbscan_prob,
  avg_chunks_per_conv,
  escalation_rate_pct,
  agent_request_rate_pct,
  confirmed_failures,
  headline_failures,
  confirmed_failure_rate_pct,
  headline_failure_rate_pct,
  suspected_failure_rate_pct,
  chunk_confirmed_failure_rate_pct,
  chunk_headline_failure_rate_pct,
  chunk_escalation_rate_pct,
  chunk_agent_request_rate_pct,
  failure_risk_proxy,
  is_routing_cluster,
  TRIM(ml_generate_text_llm_result) AS cluster_description,
  ml_generate_text_status           AS gemini_status
FROM ML.GENERATE_TEXT(
  MODEL `support-analytics-492410.support_analytics.gemini_model`,
  (
    SELECT
      cluster_id,
      n_chunks,
      n_conversations,
      languages_in_cluster,
      escalation_rate_pct,
      agent_request_rate_pct,
      confirmed_failures,
      headline_failures,
      confirmed_failure_rate_pct,
      headline_failure_rate_pct,
      suspected_failure_rate_pct,
      chunk_confirmed_failure_rate_pct,
      chunk_headline_failure_rate_pct,
      chunk_escalation_rate_pct,
      chunk_agent_request_rate_pct,
      failure_risk_proxy,
      is_routing_cluster,
      first_conversation,
      last_conversation,
      pct_non_english,
      avg_client_messages,
      avg_total_messages,
      avg_hdbscan_prob,
      avg_chunks_per_conv,
      explicit_negative_bot,
      explicit_negative_escalated,
      repetition_loop,
      agent_request_unserved,
      abandoned_inquiry,
      escalated_no_feedback,
      resolved_or_unknown,
      trivial_noise,
      resolved_positive,
      escalated_count,
      sample_texts,
      CONCAT(
        '## CONTEXT\n',
        'You are a support operations analyst at 10Web — a cloud-based AI website builder.\n',
        '10Web sells: AI-powered site creation, WordPress hosting, domain registration, ',
        'and agency/white-label plans. Customers contact the chatbot for tier-0 support ',
        'before an agent joins.\n\n',
        'De-identification: all PII is masked. Tokens you will see:\n',
        '  [URL] = any web address    [DOMAIN] = domain name    [PHONE] = phone number\n',
        '  [PERSON] = personal name   [LOCATION] = place name   [CREDENTIAL] = password / key\n',
        '  <image url> = an attached image or screenshot\n',
        '"Talk to an Agent" in excerpts is a CHATBOT BUTTON CLICK, not typed client text.\n\n',

        '## IMPORTANT NOTE ON EXCERPT STYLE\n',
        'Excerpts are semantically coherent message chunks from the conversation, NOT full ',
        'conversations. Because AI responses are much longer than user messages, most chunks ',
        'contain [AI]: text describing the solution or topic. The user question that prompted ',
        'each AI response may appear briefly at the start of a chunk. Infer the user topic ',
        'from the AI response content.\n\n',

        '## CLUSTER METADATA\n',
        '  Cluster ID                  : ', CAST(cluster_id AS STRING), '\n',
        '  Chunk count (cluster size)  : ', CAST(n_chunks AS STRING), ' chunks\n',
        '  Conversations touching topic: ', CAST(n_conversations AS STRING), '\n',
        '  Avg chunks per conversation : ', CAST(ROUND(avg_chunks_per_conv, 1) AS STRING),
            IF(avg_chunks_per_conv > 2.5,
               ' ⚠ HIGH — conversations discuss multiple topics; some failure counts may reflect co-occurring issues, not only this cluster\'s topic',
               ''), '\n',
        '  Avg HDBSCAN membership prob : ', CAST(avg_hdbscan_prob AS STRING),
            IF(avg_hdbscan_prob < 0.75,
               ' ⚠ LOW — many chunks are only weakly assigned here; this may be a broad catch-all cluster covering heterogeneous content',
               ''), '\n',
        '  Active period               : ',
            CAST(first_conversation AS STRING), ' → ', CAST(last_conversation AS STRING), '\n',
        '  Languages present           : ', COALESCE(languages_in_cluster, 'unknown'), '\n',
        '  Non-English share           : ', CAST(pct_non_english AS STRING), '%\n',
        '  Avg client messages/conv    : ', CAST(ROUND(avg_client_messages, 1) AS STRING),
            ' (avg total messages: ', CAST(ROUND(avg_total_messages, 1) AS STRING), ')\n\n',

        '## FAILURE EVIDENCE\n',
        '(Failure signals are per-conversation — ratings and escalation are conversation facts.\n',
        'Chunk-weighted rates weight failures by how many chunks each conversation contributes\n',
        'to THIS cluster, making them more accurate for multi-topic conversations.)\n',
        '  CHUNK-WEIGHTED rates (primary — proportional to topic depth):\n',
        '    Tier 1 confirmed chunk rate : ', CAST(chunk_confirmed_failure_rate_pct AS STRING), '%\n',
        '    Headline failure chunk rate : ', CAST(chunk_headline_failure_rate_pct AS STRING), '%\n',
        '    Escalation chunk rate       : ', CAST(chunk_escalation_rate_pct AS STRING), '%\n',
        '    Agent request chunk rate    : ', CAST(chunk_agent_request_rate_pct AS STRING), '%\n',
        '  CONVERSATION-BASED rates (classic — for reference):\n',
        '  Tier 1 confirmed (explicit rating = -1) : ', CAST(confirmed_failures AS STRING),
            ' of rated convos (',
            COALESCE(CAST(confirmed_failure_rate_pct AS STRING), 'N/A — no rated conversations'),
            '%)\n',
        '  Tier 1+2 headline failures              : ', CAST(headline_failures AS STRING),
            ' of all convos (', CAST(headline_failure_rate_pct AS STRING), '%)\n',
        '  Tier 3 suspected failures               : ', CAST(suspected_failure_rate_pct AS STRING), '%\n',
        '  --- breakdown ---\n',
        '  Explicit negative, bot only             : ', CAST(explicit_negative_bot AS STRING), '\n',
        '  Explicit negative, after escalation     : ', CAST(explicit_negative_escalated AS STRING), '\n',
        'INTERPRETATION: if explicit_negative_escalated >> explicit_negative_bot, the primary ',
        'failure is in the AGENT tier (post-escalation), not the chatbot.\n',
        '  Repetition loop (client re-sent ≥2×)    : ', CAST(repetition_loop AS STRING), '\n',
        '  Asked for agent, none joined            : ', CAST(agent_request_unserved AS STRING), '\n',
        '  Escalated to human agent               : ', CAST(escalated_count AS STRING),
            ' (', CAST(escalation_rate_pct AS STRING), '%)\n',
        '  Abandoned after substantive question    : ', CAST(abandoned_inquiry AS STRING), '\n',
        '  Escalated, no rating yet (Tier 3 label) : ', CAST(escalated_no_feedback AS STRING), '\n',
        '  Resolved / unknown (no tier label)    : ', CAST(resolved_or_unknown AS STRING), '\n',
        '  Trivial / very short thread             : ', CAST(trivial_noise AS STRING), '\n',
        '  Resolved with positive rating (+1)      : ', CAST(resolved_positive AS STRING), '\n\n',

        '## REPRESENTATIVE CHUNK EXCERPTS\n',
        'Tag format: [ring{N}|{stratum}] where ring=distance ring (1=cluster core, 5=cluster boundary)\n',
        'and stratum=conversation outcome. Chunks ordered core→boundary (~15-25 total).\n',
        'IMPORTANT: if ring5 (boundary) excerpts describe a clearly different topic from ring1 (core),\n',
        'flag this as possible sub-topic contamination in ### Theme.\n\n',
        'Ring meaning:\n',
        '  [ring1|*] = cluster core: the most prototypical examples, highest HDBSCAN membership\n',
        '  [ring2|*] = inner zone\n',
        '  [ring3|*] = mid-zone\n',
        '  [ring4|*] = outer zone\n',
        '  [ring5|*] = cluster boundary: farthest from centroid, where this cluster borders others\n\n',
        'Stratum (outcome label from pipeline):\n',
        '  [*|negative]               = explicit_negative_bot OR explicit_negative_escalated (rated -1)\n',
        '  [*|probable_failure]       = repetition_loop OR agent_request_unserved (Tier 2 headline)\n',
        '  [*|abandoned_inquiry]      = client stopped after substantive question, no escalation (Tier 3)\n',
        '  [*|escalated_no_feedback]  = escalated to agent, rating still 0 (Tier 3)\n',
        '  [*|resolved_or_unknown]    = longer thread, no explicit rating, not another label above\n',
        '  [*|trivial_noise]          = very short / low-content thread (rating 0, not escalated)\n',
        '  [*|positive]               = resolved_positive (rated +1)\n',
        '  [*|no_failure_flags]       = conversation missing from failure_flags (treat cautiously)\n',
        '  [*|unmapped_failure]       = unexpected failure_type value — report in diagnosis\n',
        'Each excerpt is up to 900 characters of chunk text (AI responses dominate — see note above).\n\n',
        COALESCE(sample_texts, '(no excerpts available — cluster may be too small or all text filtered)'),

        '\n\n---\n\n',
        '## SPECIAL CASES — check these FIRST before writing output\n',
        'If excerpts consist mainly of: "Were my answers helpful?", rating request prompts, ',
        'chat closing messages, or NPS survey text → write ### Theme: "Conversational artifact / boilerplate" ',
        'and keep the other three sections brief.\n',
        'If excerpts are predominantly non-English, name the language in the theme.\n',
        'If n_conversations < 200, note low statistical power for failure rates.\n',
        'CONTAMINATION CHECK: If the excerpts contain two clearly unrelated topics ',
        '(e.g., billing/refund text mixed with DNS/domain text), state this explicitly in ### Theme: ',
        '"Mixed: [Topic A] + [Topic B] (possible multi-topic contamination)" and describe both topics. ',
        'Do not force a single theme if the excerpts are genuinely contradictory.\n\n',

        '## DIFFERENTIATION AND COVERAGE REQUIREMENTS\n',
        '1. FULL RANGE: Your ### Theme must describe the COMPLETE distribution of content across\n',
        '   ALL rings (ring1 = core through ring5 = boundary). Do not focus only on the most\n',
        '   distinctive or memorable examples — describe what the cluster covers FROM CORE TO EDGE.\n',
        '   If ring1 and ring5 show related-but-different sub-topics, list both with semicolons.\n',
        '2. SPECIFICITY: Each cluster MUST have a distinct theme that cannot apply to another cluster.\n',
        '   Avoid vague labels like "technical issues", "account problems", or "billing questions".\n',
        '   e.g. "SSL certificate provisioning errors" vs "SSL/HTTPS redirect configuration" are distinct.\n',
        '3. LOW-CONFIDENCE CLUSTERS: If avg_hdbscan_prob < 0.75 (flagged above), the cluster is a\n',
        '   broad catch-all. Name the dominant theme AND note it covers heterogeneous content.\n',
        '   Do NOT describe it as if it were a tight, specific cluster.\n',
        '4. HIGH-OVERLAP CLUSTERS: If avg_chunks_per_conv > 2.5 (flagged above), conversations in\n',
        '   this cluster also heavily discuss other topics. Mention this in ### Failure diagnosis —\n',
        '   failure rates may be inflated by co-occurring issues from those other topics.\n\n',

        '## OUTPUT\n',
        'Produce EXACTLY four Markdown headings in this order. No other text outside them.\n\n',
        '### Theme\n',
        'One or two sentences: name the SPECIFIC 10Web support topic with enough detail to distinguish ',
        'this cluster from all others. Include the product area, the user action, and the failure mode ',
        '(e.g. "AI site builder fails to apply custom color/font edits after regeneration" not "design issues"). ',
        'If this cluster covers multiple sub-topics, list them with semicolons.\n\n',
        '### Failure diagnosis\n',
        'One or two sentences interpreting the failure evidence. Cite specific numbers. ',
        'Distinguish between confirmed (rated -1), probable (signal-based), and suspected.\n\n',
        '### Suggested action\n',
        'One concrete, specific next step for the 10Web support team.\n\n',
        '### Evidence quotes\n',
        'Quote 3 to 5 short phrases verbatim from the excerpts above.\n'
      ) AS prompt
    FROM `support-analytics-492410.support_analytics.chunk_cluster_label_table`
  ),
  STRUCT(
    0.15  AS temperature,
    1100  AS max_output_tokens,
    0.9   AS top_p,
    TRUE  AS flatten_json_output
  )
);


-- adds a short cluster_title from the Gemini description — run after the CREATE above
ALTER TABLE `support-analytics-492410.support_analytics.chunk_cluster_descriptions`
ADD COLUMN IF NOT EXISTS cluster_title STRING;

MERGE `support-analytics-492410.support_analytics.chunk_cluster_descriptions` T
USING (
  SELECT
    cluster_id,
    TRIM(ml_generate_text_llm_result) AS cluster_title
  FROM ML.GENERATE_TEXT(
    MODEL `support-analytics-492410.support_analytics.gemini_model`,
    (
      SELECT
        cluster_id,
        CONCAT(
          'Summarise this cluster description in 3-6 words as a short topic label ',
          'suitable for a data visualization. Return only the label, ',
          'no punctuation, no quotes.\n\n',
          COALESCE(cluster_description, 'Support topic cluster')
        ) AS prompt
      FROM `support-analytics-492410.support_analytics.chunk_cluster_descriptions`
      WHERE cluster_description IS NOT NULL
    ),
    STRUCT(
      0.1  AS temperature,
      20   AS max_output_tokens,
      1.0  AS top_p,
      TRUE AS flatten_json_output
    )
  )
) S ON T.cluster_id = S.cluster_id
WHEN MATCHED THEN
  UPDATE SET T.cluster_title = S.cluster_title;

SELECT cluster_id, n_chunks, cluster_title, LEFT(cluster_description, 80) AS desc_preview
FROM `support-analytics-492410.support_analytics.chunk_cluster_descriptions`
ORDER BY n_chunks DESC;


SELECT
  cluster_id,
  n_chunks,
  n_conversations,
  is_routing_cluster,
  ROUND(confirmed_failure_rate_pct, 2)   AS confirmed_pct,
  ROUND(headline_failure_rate_pct,  2)   AS headline_pct,
  ROUND(escalation_rate_pct,        1)   AS esc_pct,
  ROUND(failure_risk_proxy,         3)   AS risk,
  gemini_status,
  cluster_description
FROM `support-analytics-492410.support_analytics.chunk_cluster_descriptions`
ORDER BY failure_risk_proxy DESC;

SELECT
  COUNT(*)                                      AS total_clusters,
  COUNTIF(cluster_description IS NULL)          AS null_descriptions,
  COUNTIF(is_routing_cluster)                   AS routing_clusters,
  COUNTIF(cluster_description IS NULL
          AND NOT is_routing_cluster)            AS unexpected_nulls,
  STRING_AGG(
    IF(cluster_description IS NULL,
       CONCAT('C', CAST(cluster_id AS STRING), ':', gemini_status),
       NULL),
    ' | ' ORDER BY cluster_id
  )                                             AS null_cluster_errors
FROM `support-analytics-492410.support_analytics.chunk_cluster_descriptions`;
