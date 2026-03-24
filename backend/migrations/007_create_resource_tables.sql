-- 007_create_resource_tables.sql
-- CMS resource pages, permissions, and files

CREATE TABLE IF NOT EXISTS resource_pages (
    id SERIAL PRIMARY KEY,
    parent_id INTEGER REFERENCES resource_pages(id) ON DELETE SET NULL,
    title VARCHAR(255) NOT NULL,
    slug VARCHAR(255),
    content TEXT,
    sort_order INTEGER DEFAULT 0,
    is_published BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS resource_permissions (
    id SERIAL PRIMARY KEY,
    resource_page_id INTEGER NOT NULL REFERENCES resource_pages(id) ON DELETE CASCADE,
    dealer_id INTEGER NOT NULL REFERENCES dealers(id) ON DELETE CASCADE,
    UNIQUE(resource_page_id, dealer_id)
);

CREATE TABLE IF NOT EXISTS resource_files (
    id SERIAL PRIMARY KEY,
    resource_page_id INTEGER REFERENCES resource_pages(id) ON DELETE SET NULL,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size INTEGER,
    mime_type VARCHAR(100),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
