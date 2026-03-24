import React, { useState } from 'react';
import { exportLeads, changePassword } from '../../services/api';
import AIControlPanel from '../../components/admin/AIControlPanel';
import LassoSyncToolsPanel from '../../components/admin/LassoSyncToolsPanel';

const PROVINCES = [
  { value: '', label: 'All Provinces' },
  { value: 'AB', label: 'Alberta' }, { value: 'BC', label: 'British Columbia' },
  { value: 'MB', label: 'Manitoba' }, { value: 'NB', label: 'New Brunswick' },
  { value: 'NL', label: 'Newfoundland' }, { value: 'NS', label: 'Nova Scotia' },
  { value: 'NT', label: 'NWT' }, { value: 'NU', label: 'Nunavut' },
  { value: 'ON', label: 'Ontario' }, { value: 'PE', label: 'PEI' },
  { value: 'QC', label: 'Quebec' }, { value: 'SK', label: 'Saskatchewan' },
  { value: 'YT', label: 'Yukon' },
];

const cardStyle: React.CSSProperties = { background: 'white', borderRadius: 8, padding: 16, boxShadow: '0 2px 8px rgba(0,0,0,0.1)' };
const inputStyle: React.CSSProperties = { padding: '8px 12px', borderRadius: 6, border: '1px solid #ddd', fontSize: 14 };
const btnPrimary: React.CSSProperties = { padding: '8px 20px', background: '#c91414', color: 'white', border: 'none', borderRadius: 6, fontSize: 14, fontWeight: 600, cursor: 'pointer' };

const SECTIONS = [
  'AI Controls',
  'Lasso Sync & Health',
  'Lead Export',
  'Change Password',
] as const;

const Tools: React.FC = () => {
  const [openSection, setOpenSection] = useState<string | null>(null);

  const toggleSection = (section: string) => {
    setOpenSection((current) => (current === section ? null : section));
  };

  return (
    <div style={{ padding: 24 }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, color: '#1a1a2e', marginBottom: 20 }}>Tools</h1>

      {SECTIONS.map((section) => (
        <div key={section} style={{ marginBottom: 8 }}>
          <div
            onClick={() => toggleSection(section)}
            style={{
              padding: '14px 20px',
              background: '#f8f9fa',
              border: '1px solid #eee',
              cursor: 'pointer',
              fontWeight: 600,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              borderRadius: openSection === section ? '8px 8px 0 0' : 8,
            }}
          >
            <span>{section}</span>
            <span style={{ fontSize: 12 }}>{openSection === section ? '\u25B2' : '\u25BC'}</span>
          </div>
          {openSection === section && (
            <div style={{ ...cardStyle, borderRadius: '0 0 8px 8px', borderTop: 'none' }}>
              {section === 'AI Controls' && <AIControlPanel />}
              {section === 'Lasso Sync & Health' && <LassoSyncToolsPanel />}
              {section === 'Lead Export' && <LassoExportSection />}
              {section === 'Change Password' && <PasswordSection />}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

const LassoExportSection: React.FC = () => {
  const [status, setStatus] = useState('');
  const [province, setProvince] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleExport = async () => {
    setExporting(true);
    setError('');
    setSuccess('');
    try {
      const params: Record<string, string> = {};
      if (status) params.status = status;
      if (province) params.province = province;
      if (startDate) params.start_date = startDate;
      if (endDate) params.end_date = endDate;

      const res = await exportLeads(params);
      const rows = res.leads as Record<string, unknown>[];

      if (rows.length === 0) {
        setError('No Lasso-backed rows found with these filters');
        setExporting(false);
        return;
      }

      const allKeys = Object.keys(rows[0]);
      const csvRows = [allKeys.join(',')];
      rows.forEach((row) => {
        const csvRow = allKeys.map((key) => {
          const value = row[key];
          if (value == null) return '';
          const text = String(value);
          if (text.includes(',') || text.includes('"') || text.includes('\n')) {
            return `"${text.replace(/"/g, '""')}"`;
          }
          return text;
        });
        csvRows.push(csvRow.join(','));
      });

      const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `lasso_export_${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      URL.revokeObjectURL(url);
      setSuccess(`Exported ${rows.length} Lasso-backed rows`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Export failed');
    } finally {
      setExporting(false);
    }
  };

  return (
    <div>
      {error && <div style={{ color: '#c91414', marginBottom: 12, fontSize: 13 }}>{error}</div>}
      {success && <div style={{ color: '#1a7a3a', marginBottom: 12, fontSize: 13 }}>{success}</div>}

      <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 16 }}>
        <select value={status} onChange={(e) => setStatus(e.target.value)} style={inputStyle}>
          <option value="">All Statuses</option>
          <option value="active">Active</option>
          <option value="converted">Converted</option>
          <option value="dead">Dead</option>
        </select>
        <select value={province} onChange={(e) => setProvince(e.target.value)} style={inputStyle}>
          {PROVINCES.map((provinceOption) => (
            <option key={provinceOption.value} value={provinceOption.value}>
              {provinceOption.label}
            </option>
          ))}
        </select>
        <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} style={inputStyle} />
        <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} style={inputStyle} />
      </div>

      <button onClick={handleExport} disabled={exporting} style={{ ...btnPrimary, opacity: exporting ? 0.6 : 1 }}>
        {exporting ? 'Exporting...' : 'Export Live Lasso Data'}
      </button>

      <div style={{ color: '#6b7280', fontSize: 13, marginTop: 12 }}>
        Exports the current Lasso-backed dashboard and history snapshot from Azure, not the local operational lead table.
      </div>
    </div>
  );
};

const PasswordSection: React.FC = () => {
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const handleSubmit = async () => {
    setError('');
    setSuccess('');
    if (newPassword.length < 8) { setError('New password must be at least 8 characters'); return; }
    if (newPassword !== confirmPassword) { setError('Passwords do not match'); return; }
    setSaving(true);
    try {
      const res = await changePassword({ current_password: currentPassword, new_password: newPassword });
      setSuccess(res.message || 'Password changed successfully');
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to change password');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ maxWidth: 400 }}>
      {error && <div style={{ color: '#c91414', marginBottom: 12, fontSize: 13 }}>{error}</div>}
      {success && <div style={{ color: '#1a7a3a', marginBottom: 12, fontSize: 13 }}>{success}</div>}

      <div style={{ marginBottom: 12 }}>
        <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#555', marginBottom: 4 }}>Current Password</label>
        <input type="password" value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' as const }} />
      </div>

      <div style={{ marginBottom: 12 }}>
        <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#555', marginBottom: 4 }}>New Password</label>
        <input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' as const }} />
      </div>

      <div style={{ marginBottom: 16 }}>
        <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#555', marginBottom: 4 }}>Confirm New Password</label>
        <input type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' as const }} />
      </div>

      <button onClick={handleSubmit} disabled={saving} style={{ ...btnPrimary, opacity: saving ? 0.6 : 1 }}>
        {saving ? 'Changing...' : 'Change Password'}
      </button>
    </div>
  );
};

export default Tools;
