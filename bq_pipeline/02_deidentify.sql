-- SHA-256 pseudonymises ID columns; regex scrubs emails, URLs, phones, API keys from content
-- Depends: 01_copy_source_data.sql

CREATE OR REPLACE TABLE
  `support-analytics-492410.support_analytics.chatbot_messages_deidentified`
AS
SELECT
  TO_HEX(SHA256(CAST(conversation_id AS STRING)))                  AS conversation_id,
  TO_HEX(SHA256(CAST(client_id      AS STRING)))                   AS client_id,
  TO_HEX(SHA256(LOWER(TRIM(COALESCE(clients_email, 'no_email'))))) AS client_id_email_hash,
  CASE
    WHEN agent IN ('10Web AI Chatbot', 'Answer Bot', '10Web Care') THEN agent
    WHEN agent IS NULL                                             THEN NULL
    ELSE TO_HEX(SHA256(LOWER(TRIM(agent))))
  END                                                              AS agent,
  CASE
    WHEN client_name IS NULL THEN NULL
    ELSE TO_HEX(SHA256(LOWER(TRIM(client_name))))
  END                                                              AS client_name,
  conversation_start_date,
  rating,
  agreements_subscription,
  category,
  message_date,
  agent_or_bot,

  REGEXP_REPLACE(
    REGEXP_REPLACE(
      REGEXP_REPLACE(
        REGEXP_REPLACE(
          REGEXP_REPLACE(
            REGEXP_REPLACE(
              REGEXP_REPLACE(
                REGEXP_REPLACE(
                  REGEXP_REPLACE(
                    content,
                    r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}',
                    '[EMAIL]'),
                  r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
                  '[IP]'),
                r'https?://[^\s"<>()\]]+',
                '[URL]'),
              r'\bwww\.[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?',
              '[URL]'),
            r'\b[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}/wp-(?:admin|login|json|content)[^\s]*',
            '[URL]'),
          r'(?i)Bearer\s+[A-Za-z0-9\-._~+/]+=*',
          '[API_KEY]'),
        r'\b(?:sk|pk|rk|whsec|xoxb|xoxa|xoxp|xoxs|ghp|gho|ghs|ghr)[-_][A-Za-z0-9][A-Za-z0-9_-]{9,}',
        '[API_KEY]'),
      r'\+\d[\d\s\-().]{7,20}\d',
      '[PHONE]'),
    r'(?i)(?:password|passwd|pwd)\s*(?:is|:|=|was)\s*\S+',
    '[CREDENTIAL]'
  ) AS content

FROM `support-analytics-492410.support_analytics.chatbot_messages_raw`;


SELECT
  COUNT(*)                              AS total_messages,
  COUNT(DISTINCT conversation_id)       AS unique_conversations,
  MIN(LENGTH(conversation_id))          AS id_len_min,
  MAX(LENGTH(conversation_id))          AS id_len_max,
  ROUND(AVG(LENGTH(content)), 0)        AS avg_content_length,
  COUNTIF(REGEXP_CONTAINS(content, r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'))
                                        AS residual_emails
FROM `support-analytics-492410.support_analytics.chatbot_messages_deidentified`;
