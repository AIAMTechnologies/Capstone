-- Email sync configuration
CREATE TABLE IF NOT EXISTS email_sync_config (
    id SERIAL PRIMARY KEY,
    ms_tenant_id VARCHAR(255),
    ms_client_id VARCHAR(255),
    ms_client_secret TEXT,
    ms_redirect_uri VARCHAR(500),
    access_token TEXT,
    refresh_token TEXT,
    token_expires_at TIMESTAMP,
    sync_enabled BOOLEAN DEFAULT FALSE,
    sync_interval_minutes INTEGER DEFAULT 15,
    last_sync_at TIMESTAMP,
    user_email VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Synced emails
CREATE TABLE IF NOT EXISTS email_messages (
    id SERIAL PRIMARY KEY,
    ms_message_id VARCHAR(500) UNIQUE NOT NULL,
    subject TEXT,
    sender_email VARCHAR(255),
    sender_name VARCHAR(255),
    recipient_emails TEXT,  -- JSON array
    body_preview TEXT,
    body_text TEXT,
    received_at TIMESTAMP,
    is_read BOOLEAN DEFAULT FALSE,
    direction VARCHAR(10) DEFAULT 'inbound',  -- inbound or outbound
    folder VARCHAR(100),
    matched_lead_id INTEGER REFERENCES leads(id) ON DELETE SET NULL,
    match_confidence FLOAT,
    match_method VARCHAR(50),  -- 'email', 'name', 'phone', 'ai'
    ai_summary TEXT,
    ai_sentiment VARCHAR(20),  -- 'positive', 'neutral', 'negative'
    ai_action_items TEXT,  -- JSON array
    ai_ready_to_close BOOLEAN DEFAULT FALSE,
    ai_close_reasoning TEXT,
    processed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_email_messages_lead ON email_messages(matched_lead_id);
CREATE INDEX IF NOT EXISTS idx_email_messages_sender ON email_messages(sender_email);
CREATE INDEX IF NOT EXISTS idx_email_messages_received ON email_messages(received_at);

-- Review queue for leads flagged for closure
CREATE TABLE IF NOT EXISTS closure_review_queue (
    id SERIAL PRIMARY KEY,
    lead_id INTEGER REFERENCES leads(id) ON DELETE CASCADE,
    flagged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ai_reasoning TEXT,
    days_inactive INTEGER,
    last_email_at TIMESTAMP,
    email_count INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending',  -- pending, approved, dismissed
    reviewed_by VARCHAR(100),
    reviewed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_closure_review_lead ON closure_review_queue(lead_id);
CREATE INDEX IF NOT EXISTS idx_closure_review_status ON closure_review_queue(status);

-- Add email intel columns to leads
ALTER TABLE leads ADD COLUMN IF NOT EXISTS email_match_count INTEGER DEFAULT 0;
ALTER TABLE leads ADD COLUMN IF NOT EXISTS last_email_activity TIMESTAMP;
ALTER TABLE leads ADD COLUMN IF NOT EXISTS email_sentiment VARCHAR(20);
ALTER TABLE leads ADD COLUMN IF NOT EXISTS ai_closure_flagged BOOLEAN DEFAULT FALSE;
