-- 005_create_lead_logs.sql
-- Lead activity/interaction log

CREATE TABLE IF NOT EXISTS lead_logs (
    id SERIAL PRIMARY KEY,
    lead_id INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
    log_type VARCHAR(50) NOT NULL,
    message TEXT,
    dealer_id INTEGER REFERENCES dealers(id),
    created_by VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_lead_logs_lead_id ON lead_logs(lead_id);
