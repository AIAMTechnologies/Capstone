CREATE TABLE IF NOT EXISTS settings (
    key VARCHAR(100) PRIMARY KEY,
    value_text TEXT,
    updated_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO settings (key, value_text, updated_by)
VALUES
    ('daily_spend_limit_cad', '50', 'migration'),
    ('monthly_spend_limit_cad', '500', 'migration')
ON CONFLICT (key) DO NOTHING;

CREATE TABLE IF NOT EXISTS cost_tracking (
    id SERIAL PRIMARY KEY,
    period_type VARCHAR(20) NOT NULL,
    period_start DATE NOT NULL,
    model_used VARCHAR(100) NOT NULL,
    total_calls INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    spend_cad NUMERIC(12, 4) NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_cost_tracking_period_model
    ON cost_tracking(period_type, period_start, model_used);

CREATE INDEX IF NOT EXISTS idx_cost_tracking_period_start
    ON cost_tracking(period_type, period_start DESC);
