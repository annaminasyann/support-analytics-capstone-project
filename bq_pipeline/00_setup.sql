-- One-time setup: register remote ML models

CREATE OR REPLACE MODEL `support-analytics-492410.support_analytics.translation_model`
REMOTE WITH CONNECTION `support-analytics-492410.eu.vertex_ai_connection`
OPTIONS (REMOTE_SERVICE_TYPE = 'CLOUD_AI_TRANSLATE_V3');


CREATE OR REPLACE MODEL `support-analytics-492410.support_analytics.embedding_multilingual`
REMOTE WITH CONNECTION `support-analytics-492410.eu.vertex_ai_connection`
OPTIONS (ENDPOINT = 'text-multilingual-embedding-002');


CREATE OR REPLACE MODEL `support-analytics-492410.support_analytics.gemini_embedding`
REMOTE WITH CONNECTION `support-analytics-492410.eu.vertex_ai_connection`
OPTIONS (ENDPOINT = 'gemini-embedding-001');


CREATE OR REPLACE MODEL `support-analytics-492410.support_analytics.gemini_model`
REMOTE WITH CONNECTION `support-analytics-492410.eu.vertex_ai_connection`
OPTIONS (ENDPOINT = 'gemini-2.5-flash');


SELECT model_name, model_type, creation_time
FROM `support-analytics-492410.support_analytics.INFORMATION_SCHEMA.MODELS`
ORDER BY creation_time;
