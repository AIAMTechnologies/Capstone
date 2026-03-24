-- 006_create_lead_assignments.sql
-- Track dealer assignments with response times

CREATE TABLE IF NOT EXISTS lead_assignments (
    id SERIAL PRIMARY KEY,
    lead_id INTEGER NOT NULL REFERENCES leads(id) ON DELETE CASCADE,
    dealer_id INTEGER NOT NULL REFERENCES dealers(id),
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    responded_at TIMESTAMP,
    response_time_hours FLOAT,
    status VARCHAR(50) DEFAULT 'pending'
);

CREATE INDEX IF NOT EXISTS idx_lead_assignments_lead_id ON lead_assignments(lead_id);
CREATE INDEX IF NOT EXISTS idx_lead_assignments_dealer_id ON lead_assignments(dealer_id);
