-- 003_create_dealers.sql
-- Create dealers table for WFC authorized dealers

CREATE TABLE IF NOT EXISTS dealers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    city VARCHAR(100),
    province VARCHAR(10),
    email VARCHAR(255),
    notification_email VARCHAR(255),
    phone VARCHAR(20),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Seed with sample dealers across provinces
INSERT INTO dealers (name, city, province, email, notification_email, phone)
VALUES
    ('SunGuard Window Films Toronto', 'Toronto', 'ON', 'info@sunguardtoronto.ca', 'leads@sunguardtoronto.ca', '416-555-0101'),
    ('Pacific Window Tinting Vancouver', 'Vancouver', 'BC', 'info@pacifictinting.ca', 'leads@pacifictinting.ca', '604-555-0102'),
    ('Rocky Mountain Solar Films', 'Calgary', 'AB', 'info@rockymountainfilms.ca', 'leads@rockymountainfilms.ca', '403-555-0103'),
    ('ClimaGuard Films Montreal', 'Montreal', 'QC', 'info@climaguardmtl.ca', 'leads@climaguardmtl.ca', '514-555-0104'),
    ('Prairie Sun Control Winnipeg', 'Winnipeg', 'MB', 'info@prairiesuncontrol.ca', 'leads@prairiesuncontrol.ca', '204-555-0105'),
    ('Northern Shield Window Films', 'Saskatoon', 'SK', 'info@northernshieldfilms.ca', 'leads@northernshieldfilms.ca', '306-555-0106'),
    ('Atlantic Window Tinting', 'Halifax', 'NS', 'info@atlantictinting.ca', 'leads@atlantictinting.ca', '902-555-0107'),
    ('Maritime Solar Films', 'Moncton', 'NB', 'info@maritimesolar.ca', 'leads@maritimesolar.ca', '506-555-0108'),
    ('Capital Region Window Films', 'Ottawa', 'ON', 'info@capitalwindowfilms.ca', 'leads@capitalwindowfilms.ca', '613-555-0109'),
    ('Fraser Valley Tinting', 'Surrey', 'BC', 'info@fraservalleytinting.ca', 'leads@fraservalleytinting.ca', '604-555-0110'),
    ('Edmonton Solar Shield', 'Edmonton', 'AB', 'info@edmontonsolarshield.ca', 'leads@edmontonsolarshield.ca', '780-555-0111'),
    ('Quebec City Film Solutions', 'Quebec City', 'QC', 'info@qcfilmsolutions.ca', 'leads@qcfilmsolutions.ca', '418-555-0112')
ON CONFLICT DO NOTHING;
