-- Copy chatbot_dashboard into the capstone dataset (May–Nov 2025)

CREATE OR REPLACE TABLE `support-analytics-492410.support_analytics.chatbot_messages_raw`
AS
SELECT
  conversation_id,
  client_id,
  agent,
  client_name,
  conversation_start_date,
  rating,
  agreements_subscription,
  category,
  clients_email,
  content,
  message_date,
  agent_or_bot
FROM `reporting-280507.dataform_outputs.chatbot_dashboard`
WHERE content IS NOT NULL
  AND conversation_start_date BETWEEN '2025-05-01' AND '2025-11-30';


SELECT
  COUNT(*)                                          AS total_messages,
  COUNT(DISTINCT conversation_id)                   AS total_conversations,
  MIN(message_date)                                 AS earliest_message,
  MAX(message_date)                                 AS latest_message,
  COUNT(DISTINCT DATE_TRUNC(DATE(message_date), MONTH)) AS months_covered
FROM `support-analytics-492410.support_analytics.chatbot_messages_raw`;
