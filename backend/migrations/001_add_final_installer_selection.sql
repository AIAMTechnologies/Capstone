-- Adds the final_installer_selection column to the operational tables
ALTER TABLE IF EXISTS leads
    ADD COLUMN IF NOT EXISTS final_installer_selection TEXT;

ALTER TABLE IF EXISTS historical_data
    ADD COLUMN IF NOT EXISTS final_installer_selection TEXT;
