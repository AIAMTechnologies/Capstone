-- Backfill final_dealer_selection so legacy data displays ML assignments.
-- Safe to re-run after installer removal.
DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_name = 'installers'
  ) AND EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'leads'
      AND column_name = 'assigned_installer_id'
  ) THEN
    EXECUTE $sql$
      UPDATE leads l
      SET final_dealer_selection = i.name
      FROM installers i
      WHERE l.assigned_installer_id = i.id
        AND (l.final_dealer_selection IS NULL OR trim(l.final_dealer_selection) = '')
        AND i.name IS NOT NULL
    $sql$;
  END IF;
END $$;

DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_name = 'historical_data'
  ) THEN
    EXECUTE $sql$
      UPDATE historical_data
      SET final_dealer_selection = dealer_name
      WHERE (final_dealer_selection IS NULL OR trim(final_dealer_selection) = '')
        AND dealer_name IS NOT NULL
        AND trim(dealer_name) <> ''
    $sql$;
  END IF;
END $$;
