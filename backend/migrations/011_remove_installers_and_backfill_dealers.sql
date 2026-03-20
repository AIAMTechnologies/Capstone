-- Hard cutover from installer terminology/schema to dealer terminology/schema.
-- This migration intentionally drops unmatched installer-only history.

ALTER TABLE leads
    ADD COLUMN IF NOT EXISTS recommended_dealer_id INTEGER REFERENCES dealers(id),
    ADD COLUMN IF NOT EXISTS final_dealer_selection TEXT,
    ADD COLUMN IF NOT EXISTS distance_to_dealer_km NUMERIC,
    ADD COLUMN IF NOT EXISTS dealer_ml_probability DOUBLE PRECISION;

ALTER TABLE historical_data
    ADD COLUMN IF NOT EXISTS final_dealer_selection TEXT;

DROP TRIGGER IF EXISTS update_installer_capacity_trigger ON leads;
DROP FUNCTION IF EXISTS update_installer_capacity();

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'leads'
      AND column_name = 'final_installer_selection'
  ) THEN
    EXECUTE $sql$
      UPDATE leads l
      SET final_dealer_selection = d.name
      FROM dealers d
      WHERE (l.final_dealer_selection IS NULL OR BTRIM(l.final_dealer_selection) = '')
        AND l.final_installer_selection IS NOT NULL
        AND BTRIM(l.final_installer_selection) <> ''
        AND LOWER(BTRIM(l.final_installer_selection)) = LOWER(BTRIM(d.name))
    $sql$;
  END IF;
END $$;

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'historical_data'
      AND column_name = 'final_installer_selection'
  ) THEN
    EXECUTE $sql$
      UPDATE historical_data h
      SET final_dealer_selection = d.name
      FROM dealers d
      WHERE (h.final_dealer_selection IS NULL OR BTRIM(h.final_dealer_selection) = '')
        AND h.final_installer_selection IS NOT NULL
        AND BTRIM(h.final_installer_selection) <> ''
        AND LOWER(BTRIM(h.final_installer_selection)) = LOWER(BTRIM(d.name))
    $sql$;
  END IF;
END $$;

UPDATE leads
SET recommended_dealer_id = assigned_dealer_id
WHERE recommended_dealer_id IS NULL
  AND assigned_dealer_id IS NOT NULL;

UPDATE leads l
SET recommended_dealer_id = d.id
FROM dealers d
WHERE l.recommended_dealer_id IS NULL
  AND l.final_dealer_selection IS NOT NULL
  AND BTRIM(l.final_dealer_selection) <> ''
  AND LOWER(BTRIM(l.final_dealer_selection)) = LOWER(BTRIM(d.name));

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'leads'
      AND column_name = 'distance_to_installer_km'
  ) THEN
    EXECUTE $sql$
      UPDATE leads
      SET distance_to_dealer_km = distance_to_installer_km
      WHERE distance_to_dealer_km IS NULL
        AND distance_to_installer_km IS NOT NULL
    $sql$;
  END IF;
END $$;

DROP VIEW IF EXISTS installer_performance CASCADE;

ALTER TABLE leads
    DROP COLUMN IF EXISTS assigned_installer_id CASCADE,
    DROP COLUMN IF EXISTS installer_override_id CASCADE,
    DROP COLUMN IF EXISTS final_installer_selection CASCADE,
    DROP COLUMN IF EXISTS distance_to_installer_km CASCADE;

ALTER TABLE historical_data
    DROP COLUMN IF EXISTS final_installer_selection CASCADE;

DROP TABLE IF EXISTS installers CASCADE;
