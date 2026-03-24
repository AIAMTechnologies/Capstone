-- Add richer AI extraction fields to email_messages
-- These capture window-film-specific details extracted per email

ALTER TABLE email_messages
    ADD COLUMN IF NOT EXISTS ai_urgency      VARCHAR(10)  DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS ai_job_type     VARCHAR(30)  DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS ai_product      TEXT         DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS ai_window_count VARCHAR(50)  DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS ai_next_action  TEXT         DEFAULT NULL;
