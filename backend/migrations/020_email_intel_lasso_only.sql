-- Migration 020: Switch email_intel to Lasso-only data
--
-- The local `leads` table is now empty (all lead data lives in Lasso snapshot
-- tables). This migration:
--   1. Adds a dedicated lasso_lead_id column to email_messages
--   2. Drops the FK constraint so matched_lead_id no longer requires a valid leads.id
--   3. Clears all existing matched_lead_id values (they referenced deleted local leads)

-- 1. Add lasso_lead_id column (stores the Lasso integer lead ID after re-sync)
ALTER TABLE email_messages
    ADD COLUMN IF NOT EXISTS lasso_lead_id INTEGER;

CREATE INDEX IF NOT EXISTS idx_email_messages_lasso_lead
    ON email_messages (lasso_lead_id)
    WHERE lasso_lead_id IS NOT NULL;

-- 2. Drop the foreign-key constraint that tied matched_lead_id to leads.id
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN
        SELECT conname
        FROM pg_constraint
        WHERE conrelid = 'email_messages'::regclass
          AND contype = 'f'
          AND conname ILIKE '%lead%'
    LOOP
        EXECUTE format('ALTER TABLE email_messages DROP CONSTRAINT IF EXISTS %I', r.conname);
    END LOOP;
END
$$;

-- 3. Clear stale matched_lead_id values (referenced leads no longer exist)
UPDATE email_messages SET matched_lead_id = NULL WHERE matched_lead_id IS NOT NULL;
