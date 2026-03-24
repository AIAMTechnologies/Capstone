CREATE TABLE IF NOT EXISTS lasso_sync_runs (
    id SERIAL PRIMARY KEY,
    sync_type VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'running',
    actor VARCHAR(100),
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    fetched_counts JSONB,
    materialized_counts JSONB,
    error_message TEXT,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_lasso_sync_runs_started_at ON lasso_sync_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_lasso_sync_runs_type_status ON lasso_sync_runs(sync_type, status);

CREATE TABLE IF NOT EXISTS lasso_unassigned_snapshot (
    id SERIAL PRIMARY KEY,
    sync_run_id INTEGER NOT NULL REFERENCES lasso_sync_runs(id) ON DELETE CASCADE,
    lasso_lead_id INTEGER NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    full_name VARCHAR(255),
    email VARCHAR(255),
    city VARCHAR(100),
    province VARCHAR(50),
    dealer_id INTEGER,
    dealer_name VARCHAR(255),
    current_status VARCHAR(100),
    submit_date TIMESTAMP,
    form_submit_date TIMESTAMP,
    created_date TIMESTAMP,
    last_interaction TIMESTAMP,
    raw_payload JSONB NOT NULL,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_lasso_unassigned_snapshot_run ON lasso_unassigned_snapshot(sync_run_id);
CREATE INDEX IF NOT EXISTS idx_lasso_unassigned_snapshot_lead ON lasso_unassigned_snapshot(lasso_lead_id);

CREATE TABLE IF NOT EXISTS lasso_active_snapshot (
    id SERIAL PRIMARY KEY,
    sync_run_id INTEGER NOT NULL REFERENCES lasso_sync_runs(id) ON DELETE CASCADE,
    lasso_lead_id INTEGER NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    full_name VARCHAR(255),
    email VARCHAR(255),
    city VARCHAR(100),
    province VARCHAR(50),
    dealer_id INTEGER,
    dealer_name VARCHAR(255),
    current_status VARCHAR(100),
    submit_date TIMESTAMP,
    form_submit_date TIMESTAMP,
    created_date TIMESTAMP,
    last_interaction TIMESTAMP,
    raw_payload JSONB NOT NULL,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_lasso_active_snapshot_run ON lasso_active_snapshot(sync_run_id);
CREATE INDEX IF NOT EXISTS idx_lasso_active_snapshot_lead ON lasso_active_snapshot(lasso_lead_id);
CREATE INDEX IF NOT EXISTS idx_lasso_active_snapshot_last_interaction ON lasso_active_snapshot(last_interaction DESC);

CREATE TABLE IF NOT EXISTS lasso_history_snapshot (
    id SERIAL PRIMARY KEY,
    sync_run_id INTEGER NOT NULL REFERENCES lasso_sync_runs(id) ON DELETE CASCADE,
    lasso_lead_id INTEGER NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    full_name VARCHAR(255),
    email VARCHAR(255),
    city VARCHAR(100),
    province VARCHAR(50),
    dealer_id INTEGER,
    dealer_name VARCHAR(255),
    current_status VARCHAR(100),
    submit_date TIMESTAMP,
    form_submit_date TIMESTAMP,
    created_date TIMESTAMP,
    last_interaction TIMESTAMP,
    raw_payload JSONB NOT NULL,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_lasso_history_snapshot_run ON lasso_history_snapshot(sync_run_id);
CREATE INDEX IF NOT EXISTS idx_lasso_history_snapshot_lead ON lasso_history_snapshot(lasso_lead_id);

CREATE TABLE IF NOT EXISTS lasso_lead_report_snapshot (
    id SERIAL PRIMARY KEY,
    sync_run_id INTEGER NOT NULL REFERENCES lasso_sync_runs(id) ON DELETE CASCADE,
    report_key VARCHAR(100) NOT NULL,
    raw_payload JSONB NOT NULL,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_lasso_lead_report_snapshot_run ON lasso_lead_report_snapshot(sync_run_id);

CREATE TABLE IF NOT EXISTS lasso_lead_status_report_snapshot (
    id SERIAL PRIMARY KEY,
    sync_run_id INTEGER NOT NULL REFERENCES lasso_sync_runs(id) ON DELETE CASCADE,
    dealer_id INTEGER,
    dealer_name VARCHAR(255),
    review_leads INTEGER,
    budget_leads INTEGER,
    converted_summary DECIMAL(12,2),
    lead_score DECIMAL(12,2),
    raw_payload JSONB NOT NULL,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_lasso_lead_status_report_snapshot_run ON lasso_lead_status_report_snapshot(sync_run_id);

CREATE TABLE IF NOT EXISTS lasso_lead_response_report_snapshot (
    id SERIAL PRIMARY KEY,
    sync_run_id INTEGER NOT NULL REFERENCES lasso_sync_runs(id) ON DELETE CASCADE,
    report_key VARCHAR(100),
    raw_payload JSONB NOT NULL,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_lasso_lead_response_report_snapshot_run ON lasso_lead_response_report_snapshot(sync_run_id);

CREATE TABLE IF NOT EXISTS lasso_dealer_project_report_snapshot (
    id SERIAL PRIMARY KEY,
    sync_run_id INTEGER NOT NULL REFERENCES lasso_sync_runs(id) ON DELETE CASCADE,
    report_key VARCHAR(100),
    raw_payload JSONB NOT NULL,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_lasso_dealer_project_report_snapshot_run ON lasso_dealer_project_report_snapshot(sync_run_id);

CREATE TABLE IF NOT EXISTS dashboard_unassigned_leads (
    lasso_lead_id INTEGER PRIMARY KEY,
    lead_id INTEGER,
    sync_run_id INTEGER NOT NULL REFERENCES lasso_sync_runs(id) ON DELETE CASCADE,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    name VARCHAR(255),
    email VARCHAR(255),
    city VARCHAR(100),
    province VARCHAR(50),
    location_text VARCHAR(255),
    current_status VARCHAR(100),
    record_date TIMESTAMP,
    last_interaction TIMESTAMP,
    raw_payload JSONB NOT NULL,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dashboard_unassigned_sync_run ON dashboard_unassigned_leads(sync_run_id);

CREATE TABLE IF NOT EXISTS dashboard_active_leads (
    lasso_lead_id INTEGER PRIMARY KEY,
    lead_id INTEGER,
    sync_run_id INTEGER NOT NULL REFERENCES lasso_sync_runs(id) ON DELETE CASCADE,
    dealer_id INTEGER,
    dealer_name VARCHAR(255),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    name VARCHAR(255),
    email VARCHAR(255),
    city VARCHAR(100),
    province VARCHAR(50),
    location_text VARCHAR(255),
    current_status VARCHAR(100),
    date_assigned TIMESTAMP,
    last_interaction TIMESTAMP,
    lead_details TEXT,
    raw_payload JSONB NOT NULL,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dashboard_active_sync_run ON dashboard_active_leads(sync_run_id);
CREATE INDEX IF NOT EXISTS idx_dashboard_active_last_interaction ON dashboard_active_leads(last_interaction DESC);

CREATE TABLE IF NOT EXISTS dashboard_dealer_lead_reporting (
    timeframe VARCHAR(50) PRIMARY KEY,
    converted_count INTEGER NOT NULL DEFAULT 0,
    converted_pct NUMERIC(6,2) NOT NULL DEFAULT 0,
    dead_count INTEGER NOT NULL DEFAULT 0,
    dead_pct NUMERIC(6,2) NOT NULL DEFAULT 0,
    active_count INTEGER NOT NULL DEFAULT 0,
    active_pct NUMERIC(6,2) NOT NULL DEFAULT 0,
    total_leads INTEGER NOT NULL DEFAULT 0,
    sync_run_id INTEGER NOT NULL REFERENCES lasso_sync_runs(id) ON DELETE CASCADE,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dashboard_lead_trends (
    period_start DATE PRIMARY KEY,
    total_count INTEGER NOT NULL DEFAULT 0,
    converted_count INTEGER NOT NULL DEFAULT 0,
    dead_count INTEGER NOT NULL DEFAULT 0,
    active_count INTEGER NOT NULL DEFAULT 0,
    sync_run_id INTEGER NOT NULL REFERENCES lasso_sync_runs(id) ON DELETE CASCADE,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO settings (key, value_text, updated_by, created_at, updated_at)
VALUES
    ('lasso_fast_sync_interval_minutes', '5', 'migration_016', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('lasso_full_sync_interval_minutes', '15', 'migration_016', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('lasso_active_queue_max_age_days', '110', 'migration_016', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
ON CONFLICT (key) DO NOTHING;
