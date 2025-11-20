-- Backfill final_installer_selection so legacy data displays ML assignments
UPDATE leads l
SET final_installer_selection = i.name
FROM installers i
WHERE l.assigned_installer_id = i.id
  AND (l.final_installer_selection IS NULL OR trim(l.final_installer_selection) = '')
  AND i.name IS NOT NULL;

UPDATE historical_data
SET final_installer_selection = dealer_name
WHERE (final_installer_selection IS NULL OR trim(final_installer_selection) = '')
  AND dealer_name IS NOT NULL
  AND trim(dealer_name) <> '';
