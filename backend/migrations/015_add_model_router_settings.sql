INSERT INTO settings (key, value_text, updated_by)
VALUES
    (
        'model_router_mapping',
        '{"bulk_scoring":"gpt-4.1-nano","email_analysis":"gpt-4.1-mini","reasoning":"gpt-4.1-mini","realtime":"gpt-4.1-mini"}',
        'migration'
    ),
    (
        'model_router_fallback_mapping',
        '{"gpt-4.1-nano":["gpt-4o-mini"],"gpt-4.1-mini":["gpt-4o-mini"],"gpt-4.1":["gpt-4.1-mini","gpt-4o-mini"],"gpt-4o-mini":[]}',
        'migration'
    )
ON CONFLICT (key) DO NOTHING;
