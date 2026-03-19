-- 008_create_notifications.sql
-- Dealer notification tracking

CREATE TABLE IF NOT EXISTS notifications (
    id SERIAL PRIMARY KEY,
    dealer_id INTEGER NOT NULL REFERENCES dealers(id) ON DELETE CASCADE,
    lead_id INTEGER REFERENCES leads(id) ON DELETE SET NULL,
    notification_type VARCHAR(50) DEFAULT 'email',
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    responded BOOLEAN DEFAULT FALSE,
    responded_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_notifications_dealer_id ON notifications(dealer_id);
