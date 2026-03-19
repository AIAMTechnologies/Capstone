-- 009_add_ai_columns.sql
-- AI feature columns on leads

ALTER TABLE leads ADD COLUMN IF NOT EXISTS ai_priority VARCHAR(10);
ALTER TABLE leads ADD COLUMN IF NOT EXISTS ai_score INTEGER;
ALTER TABLE leads ADD COLUMN IF NOT EXISTS ai_reasoning TEXT;
ALTER TABLE leads ADD COLUMN IF NOT EXISTS ai_scored_at TIMESTAMP;
ALTER TABLE leads ADD COLUMN IF NOT EXISTS ai_match_explanation TEXT;
ALTER TABLE leads ADD COLUMN IF NOT EXISTS ai_conversion_likelihood FLOAT;
ALTER TABLE leads ADD COLUMN IF NOT EXISTS ai_conversion_explanation TEXT;
