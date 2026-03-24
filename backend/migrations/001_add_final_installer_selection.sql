-- Adds the final_dealer_selection column to the operational tables
ALTER TABLE IF EXISTS leads
    ADD COLUMN IF NOT EXISTS final_dealer_selection TEXT;

ALTER TABLE IF EXISTS historical_data
    ADD COLUMN IF NOT EXISTS final_dealer_selection TEXT;
