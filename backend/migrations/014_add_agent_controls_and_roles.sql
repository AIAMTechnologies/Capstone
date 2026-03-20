ALTER TABLE settings
    ADD COLUMN IF NOT EXISTS value_boolean BOOLEAN;

ALTER TABLE admin_users
    ADD COLUMN IF NOT EXISTS role VARCHAR(50);

UPDATE admin_users
SET role = 'admin'
WHERE role IS NULL OR BTRIM(role) = '';

INSERT INTO settings (key, value_text, value_boolean, updated_by)
VALUES ('agent_enabled', 'true', TRUE, 'migration')
ON CONFLICT (key) DO UPDATE
SET value_text = COALESCE(settings.value_text, EXCLUDED.value_text),
    value_boolean = COALESCE(settings.value_boolean, EXCLUDED.value_boolean),
    updated_at = CURRENT_TIMESTAMP;
