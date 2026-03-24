CREATE TABLE IF NOT EXISTS dashboard_history_leads (
    lasso_lead_id INTEGER PRIMARY KEY,
    sync_run_id INTEGER NOT NULL REFERENCES lasso_sync_runs(id) ON DELETE CASCADE,
    dealer_id INTEGER,
    dealer_name VARCHAR(255),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    name VARCHAR(255),
    email VARCHAR(255),
    phone VARCHAR(50),
    address TEXT,
    city VARCHAR(100),
    province VARCHAR(50),
    postal_code VARCHAR(50),
    current_status VARCHAR(100),
    status_bucket VARCHAR(50),
    submit_date TIMESTAMP,
    form_submit_date TIMESTAMP,
    created_date TIMESTAMP,
    last_interaction TIMESTAMP,
    project_type VARCHAR(100),
    product_type VARCHAR(100),
    square_footage_text VARCHAR(100),
    square_footage_value NUMERIC(12,2),
    business_category VARCHAR(100),
    company_name VARCHAR(255),
    lead_source VARCHAR(255),
    landing_page VARCHAR(255),
    landing_page_url TEXT,
    landing_page_variant VARCHAR(50),
    utm_source VARCHAR(255),
    utm_medium VARCHAR(255),
    utm_campaign VARCHAR(255),
    utm_content VARCHAR(255),
    comments TEXT,
    value_of_order NUMERIC(12,2),
    raw_payload JSONB NOT NULL,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dashboard_history_sync_run ON dashboard_history_leads(sync_run_id);
CREATE INDEX IF NOT EXISTS idx_dashboard_history_status_bucket ON dashboard_history_leads(status_bucket);
CREATE INDEX IF NOT EXISTS idx_dashboard_history_dealer_name ON dashboard_history_leads(dealer_name);
CREATE INDEX IF NOT EXISTS idx_dashboard_history_created_date ON dashboard_history_leads(created_date DESC);

CREATE TABLE IF NOT EXISTS dashboard_dealer_performance (
    dealer_id INTEGER PRIMARY KEY,
    dealer_name VARCHAR(255) NOT NULL,
    active_leads INTEGER NOT NULL DEFAULT 0,
    converted_count INTEGER NOT NULL DEFAULT 0,
    dead_count INTEGER NOT NULL DEFAULT 0,
    total_leads INTEGER NOT NULL DEFAULT 0,
    avg_response_hours NUMERIC(12,4),
    avg_response_str VARCHAR(255),
    sync_run_id INTEGER NOT NULL REFERENCES lasso_sync_runs(id) ON DELETE CASCADE,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dashboard_dealer_project_breakdown (
    dealer_id INTEGER PRIMARY KEY,
    dealer_name VARCHAR(255) NOT NULL,
    sqft_1_499 INTEGER NOT NULL DEFAULT 0,
    sqft_500_999 INTEGER NOT NULL DEFAULT 0,
    sqft_1000_3499 INTEGER NOT NULL DEFAULT 0,
    sqft_3500_7499 INTEGER NOT NULL DEFAULT 0,
    sqft_7500_19999 INTEGER NOT NULL DEFAULT 0,
    sqft_20000_plus INTEGER NOT NULL DEFAULT 0,
    total_leads INTEGER NOT NULL DEFAULT 0,
    sync_run_id INTEGER NOT NULL REFERENCES lasso_sync_runs(id) ON DELETE CASCADE,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dashboard_dealer_status (
    dealer_id INTEGER PRIMARY KEY,
    dealer_name VARCHAR(255) NOT NULL,
    reviewing_undecided INTEGER NOT NULL DEFAULT 0,
    building_budget INTEGER NOT NULL DEFAULT 0,
    converted_total_value NUMERIC(12,2) NOT NULL DEFAULT 0,
    lead_score_pct NUMERIC(8,2) NOT NULL DEFAULT 0,
    sync_run_id INTEGER NOT NULL REFERENCES lasso_sync_runs(id) ON DELETE CASCADE,
    synced_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

DROP TABLE IF EXISTS historical_data;
DROP TABLE IF EXISTS notifications;
DROP TABLE IF EXISTS resource_permissions;
DROP TABLE IF EXISTS resource_files;
DROP TABLE IF EXISTS resource_pages;
