-- Indexes to make dashboard table reads fast.
-- These cover the ORDER BY clauses used by the read endpoints
-- and the WHERE filters used by the history page.

-- Unassigned leads: ORDER BY record_date DESC, lasso_lead_id DESC
CREATE INDEX IF NOT EXISTS idx_dashboard_unassigned_sort
    ON dashboard_unassigned_leads (record_date DESC NULLS LAST, lasso_lead_id DESC);

-- Active leads: ORDER BY last_interaction DESC, date_assigned DESC, lasso_lead_id DESC
CREATE INDEX IF NOT EXISTS idx_dashboard_active_sort
    ON dashboard_active_leads (last_interaction DESC NULLS LAST, date_assigned DESC NULLS LAST, lasso_lead_id DESC);

-- History leads: ORDER BY last_interaction DESC, created_date DESC
CREATE INDEX IF NOT EXISTS idx_dashboard_history_sort
    ON dashboard_history_leads (last_interaction DESC NULLS LAST, created_date DESC NULLS LAST);

-- History leads: WHERE province = ? AND dealer_name ILIKE ?
CREATE INDEX IF NOT EXISTS idx_dashboard_history_province
    ON dashboard_history_leads (province);

-- Sync runs: status lookups used by scheduler and status endpoint
CREATE INDEX IF NOT EXISTS idx_lasso_sync_runs_status_completed
    ON lasso_sync_runs (status, completed_at DESC NULLS LAST, id DESC);
