import React, { useState, useEffect, useCallback } from 'react';
import {
  uploadCSV, importCSV, sendMassEmail, getDealerOptions,
  getNotificationCheck, resendNotification, exportLeads, changePassword,
} from '../../services/api';
import type { Dealer, DealerNotificationStatus, ExtendedLead } from '../../types/types';

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

const DB_COLUMNS = [
  '', 'first_name', 'last_name', 'email', 'phone', 'address', 'city', 'province',
  'postal_code', 'job_type', 'product_type', 'square_footage', 'project_type',
  'business_category', 'lead_source', 'company_name', 'comments',
];

const inputStyle: React.CSSProperties = { padding: '8px 12px', borderRadius: 6, border: '1px solid #ddd', fontSize: 14 };
const cardStyle: React.CSSProperties = { background: 'white', borderRadius: 8, padding: 16, boxShadow: '0 2px 8px rgba(0,0,0,0.1)' };
const btnPrimary: React.CSSProperties = { padding: '8px 20px', background: '#c91414', color: 'white', border: 'none', borderRadius: 6, fontSize: 14, fontWeight: 600, cursor: 'pointer' };
const thStyle: React.CSSProperties = { background: '#f8f9fa', textAlign: 'left' as const, padding: '10px 12px', fontSize: 13, fontWeight: 600, color: '#555' };
const tdStyle: React.CSSProperties = { padding: '10px 12px', borderBottom: '1px solid #eee', fontSize: 13 };

const SECTIONS = [
  'CSV Upload & Import',
  'Mass Email',
  'Notification Check',
  'Lead Export',
  'Change Password',
] as const;

const Tools: React.FC = () => {
  const [openSection, setOpenSection] = useState<string | null>(null);

  const toggleSection = (s: string) => {
    setOpenSection(prev => prev === s ? null : s);
  };

  return (
    <div style={{ padding: 24 }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, color: '#1a1a2e', marginBottom: 20 }}>Tools</h1>

      {SECTIONS.map(section => (
        <div key={section} style={{ marginBottom: 8 }}>
          <div
            onClick={() => toggleSection(section)}
            style={{
              padding: '14px 20px', background: '#f8f9fa', border: '1px solid #eee',
              cursor: 'pointer', fontWeight: 600, display: 'flex', justifyContent: 'space-between',
              alignItems: 'center', borderRadius: openSection === section ? '8px 8px 0 0' : 8,
            }}
          >
            <span>{section}</span>
            <span style={{ fontSize: 12 }}>{openSection === section ? '\u25B2' : '\u25BC'}</span>
          </div>
          {openSection === section && (
            <div style={{ ...cardStyle, borderRadius: '0 0 8px 8px', borderTop: 'none' }}>
              {section === 'CSV Upload & Import' && <CSVSection />}
              {section === 'Mass Email' && <MassEmailSection />}
              {section === 'Notification Check' && <NotificationSection />}
              {section === 'Lead Export' && <ExportSection />}
              {section === 'Change Password' && <PasswordSection />}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

// ===================== CSV Upload & Import =====================
const CSVSection: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [headers, setHeaders] = useState<string[]>([]);
  const [preview, setPreview] = useState<Record<string, string>[]>([]);
  const [totalRows, setTotalRows] = useState(0);
  const [mappings, setMappings] = useState<Record<string, string>>({});
  const [uploading, setUploading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [result, setResult] = useState<{ inserted: number; errors: string[] } | null>(null);
  const [error, setError] = useState('');

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError('');
    setResult(null);
    try {
      const res = await uploadCSV(file);
      setHeaders(res.headers);
      setPreview(res.preview);
      setTotalRows(res.total_rows);
      const initialMappings: Record<string, string> = {};
      res.headers.forEach(h => { initialMappings[h] = ''; });
      setMappings(initialMappings);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleImport = async () => {
    setImporting(true);
    setError('');
    try {
      const mappingArr = Object.entries(mappings)
        .filter(([, db]) => db)
        .map(([csv, db]) => ({ csv_column: csv, db_column: db }));
      if (mappingArr.length === 0) {
        setError('Please map at least one column');
        setImporting(false);
        return;
      }
      const res = await importCSV({ mappings: mappingArr, data: preview });
      setResult({ inserted: res.inserted, errors: res.errors });
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Import failed');
    } finally {
      setImporting(false);
    }
  };

  return (
    <div>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 16 }}>
        <input type="file" accept=".csv" onChange={e => setFile(e.target.files?.[0] || null)} style={{ fontSize: 14 }} />
        <button onClick={handleUpload} disabled={!file || uploading} style={{ ...btnPrimary, opacity: !file || uploading ? 0.6 : 1 }}>
          {uploading ? 'Uploading...' : 'Upload'}
        </button>
      </div>

      {error && <div style={{ color: '#c91414', marginBottom: 12, fontSize: 13 }}>{error}</div>}

      {headers.length > 0 && (
        <>
          <p style={{ fontSize: 13, color: '#666', marginBottom: 12 }}>Total rows: {totalRows}</p>

          <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Column Mapping</h4>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px 16px', marginBottom: 16 }}>
            {headers.map(h => (
              <React.Fragment key={h}>
                <span style={{ fontSize: 13, fontWeight: 500 }}>{h}</span>
                <select
                  value={mappings[h] || ''}
                  onChange={e => setMappings(prev => ({ ...prev, [h]: e.target.value }))}
                  style={inputStyle}
                >
                  {DB_COLUMNS.map(c => (
                    <option key={c} value={c}>{c || '-- Skip --'}</option>
                  ))}
                </select>
              </React.Fragment>
            ))}
          </div>

          <h4 style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Preview (first {Math.min(preview.length, 5)} rows)</h4>
          <div style={{ overflowX: 'auto', marginBottom: 16 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>{headers.map(h => <th key={h} style={thStyle}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {preview.slice(0, 5).map((row, i) => (
                  <tr key={i}>{headers.map(h => <td key={h} style={tdStyle}>{row[h] || ''}</td>)}</tr>
                ))}
              </tbody>
            </table>
          </div>

          <button onClick={handleImport} disabled={importing} style={{ ...btnPrimary, opacity: importing ? 0.6 : 1 }}>
            {importing ? 'Importing...' : 'Import'}
          </button>
        </>
      )}

      {result && (
        <div style={{ marginTop: 16, padding: 12, borderRadius: 6, background: '#f0fdf4', fontSize: 13 }}>
          <strong>Inserted:</strong> {result.inserted}
          {result.errors.length > 0 && (
            <div style={{ marginTop: 8, color: '#c91414' }}>
              <strong>Errors:</strong>
              <ul style={{ margin: '4px 0', paddingLeft: 20 }}>
                {result.errors.map((e, i) => <li key={i}>{e}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ===================== Mass Email =====================
const MassEmailSection: React.FC = () => {
  const [dealers, setDealers] = useState<Dealer[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [subject, setSubject] = useState('');
  const [body, setBody] = useState('');
  const [sending, setSending] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const loadDealers = useCallback(async () => {
    if (loaded) return;
    try {
      const res = await getDealerOptions();
      setDealers(res.dealers);
      setLoaded(true);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load dealers');
    }
  }, [loaded]);

  useEffect(() => { loadDealers(); }, [loadDealers]);

  const toggleDealer = (id: number) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const selectAll = () => setSelectedIds(new Set(dealers.map(d => d.id)));
  const deselectAll = () => setSelectedIds(new Set());

  const handleSend = async () => {
    if (selectedIds.size === 0) { setError('Select at least one dealer'); return; }
    if (!subject.trim()) { setError('Subject is required'); return; }
    if (!body.trim()) { setError('Body is required'); return; }
    setSending(true);
    setError('');
    setSuccess('');
    try {
      const res = await sendMassEmail({ dealer_ids: Array.from(selectedIds), subject, body });
      setSuccess(res.message || 'Emails sent successfully');
      setSubject('');
      setBody('');
      setSelectedIds(new Set());
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to send');
    } finally {
      setSending(false);
    }
  };

  return (
    <div>
      {error && <div style={{ color: '#c91414', marginBottom: 12, fontSize: 13 }}>{error}</div>}
      {success && <div style={{ color: '#1a7a3a', marginBottom: 12, fontSize: 13 }}>{success}</div>}

      <div style={{ marginBottom: 12 }}>
        <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
          <button onClick={selectAll} style={{ ...btnPrimary, background: '#555', fontSize: 12, padding: '4px 12px' }}>Select All</button>
          <button onClick={deselectAll} style={{ ...btnPrimary, background: '#999', fontSize: 12, padding: '4px 12px' }}>Deselect All</button>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, maxHeight: 150, overflowY: 'auto' }}>
          {dealers.map(d => (
            <label key={d.id} style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 13, cursor: 'pointer' }}>
              <input type="checkbox" checked={selectedIds.has(d.id)} onChange={() => toggleDealer(d.id)} />
              {d.name}
            </label>
          ))}
        </div>
      </div>

      <div style={{ marginBottom: 12 }}>
        <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#555', marginBottom: 4 }}>Subject</label>
        <input value={subject} onChange={e => setSubject(e.target.value)} style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' as const }} placeholder="Email subject" />
      </div>

      <div style={{ marginBottom: 16 }}>
        <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#555', marginBottom: 4 }}>Body</label>
        <textarea value={body} onChange={e => setBody(e.target.value)} style={{ ...inputStyle, width: '100%', height: 200, boxSizing: 'border-box' as const, resize: 'vertical' }} placeholder="Email body..." />
      </div>

      <button onClick={handleSend} disabled={sending} style={{ ...btnPrimary, opacity: sending ? 0.6 : 1 }}>
        {sending ? 'Sending...' : 'Send'}
      </button>
    </div>
  );
};

// ===================== Notification Check =====================
const NotificationSection: React.FC = () => {
  const [dealers, setDealers] = useState<DealerNotificationStatus[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [resending, setResending] = useState<number | null>(null);
  const [success, setSuccess] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await getNotificationCheck();
      setDealers(res.dealers);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleResend = async (dealerId: number) => {
    setResending(dealerId);
    setSuccess('');
    setError('');
    try {
      const res = await resendNotification(dealerId);
      setSuccess(res.message || 'Notification resent');
      await load();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to resend');
    } finally {
      setResending(null);
    }
  };

  const fmtDate = (d?: string) => d ? new Date(d).toLocaleString('en-CA') : '-';

  return (
    <div>
      {error && <div style={{ color: '#c91414', marginBottom: 12, fontSize: 13 }}>{error}</div>}
      {success && <div style={{ color: '#1a7a3a', marginBottom: 12, fontSize: 13 }}>{success}</div>}
      {loading && <p style={{ color: '#999', fontSize: 13 }}>Loading...</p>}

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={thStyle}>Dealer</th>
              <th style={thStyle}>Email</th>
              <th style={thStyle}>Total</th>
              <th style={thStyle}>Responded</th>
              <th style={thStyle}>Last Sent</th>
              <th style={thStyle}>Last Responded</th>
              <th style={thStyle}>Action</th>
            </tr>
          </thead>
          <tbody>
            {dealers.map(d => (
              <tr key={d.id}>
                <td style={tdStyle}>{d.name}</td>
                <td style={tdStyle}>{d.email}</td>
                <td style={tdStyle}>{d.total_notifications}</td>
                <td style={tdStyle}>{d.responded}</td>
                <td style={tdStyle}>{fmtDate(d.last_sent)}</td>
                <td style={tdStyle}>{fmtDate(d.last_responded)}</td>
                <td style={tdStyle}>
                  <button
                    onClick={() => handleResend(d.id)}
                    disabled={resending === d.id}
                    style={{ ...btnPrimary, fontSize: 12, padding: '4px 12px', opacity: resending === d.id ? 0.6 : 1 }}
                  >
                    {resending === d.id ? '...' : 'Resend'}
                  </button>
                </td>
              </tr>
            ))}
            {dealers.length === 0 && !loading && (
              <tr><td colSpan={7} style={{ ...tdStyle, textAlign: 'center', color: '#999' }}>No data</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// ===================== Lead Export =====================
const ExportSection: React.FC = () => {
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
      const leads: ExtendedLead[] = res.leads;

      if (leads.length === 0) {
        setError('No leads found with these filters');
        setExporting(false);
        return;
      }

      const allKeys = Object.keys(leads[0]);
      const csvRows = [allKeys.join(',')];
      leads.forEach(lead => {
        const row = allKeys.map(k => {
          const val = (lead as any)[k];
          if (val == null) return '';
          const str = String(val);
          if (str.includes(',') || str.includes('"') || str.includes('\n')) {
            return `"${str.replace(/"/g, '""')}"`;
          }
          return str;
        });
        csvRows.push(row.join(','));
      });

      const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `leads_export_${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setSuccess(`Exported ${leads.length} leads`);
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
        <select value={status} onChange={e => setStatus(e.target.value)} style={inputStyle}>
          <option value="">All Statuses</option>
          <option value="active">Active</option>
          <option value="converted">Converted</option>
          <option value="dead">Dead</option>
          <option value="follow_up">Follow Up</option>
        </select>
        <select value={province} onChange={e => setProvince(e.target.value)} style={inputStyle}>
          {PROVINCES.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
        </select>
        <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} style={inputStyle} />
        <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} style={inputStyle} />
      </div>

      <button onClick={handleExport} disabled={exporting} style={{ ...btnPrimary, opacity: exporting ? 0.6 : 1 }}>
        {exporting ? 'Exporting...' : 'Export'}
      </button>
    </div>
  );
};

// ===================== Change Password =====================
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
        <input type="password" value={currentPassword} onChange={e => setCurrentPassword(e.target.value)} style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' as const }} />
      </div>

      <div style={{ marginBottom: 12 }}>
        <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#555', marginBottom: 4 }}>New Password</label>
        <input type="password" value={newPassword} onChange={e => setNewPassword(e.target.value)} style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' as const }} />
      </div>

      <div style={{ marginBottom: 16 }}>
        <label style={{ display: 'block', fontSize: 13, fontWeight: 600, color: '#555', marginBottom: 4 }}>Confirm New Password</label>
        <input type="password" value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' as const }} />
      </div>

      <button onClick={handleSubmit} disabled={saving} style={{ ...btnPrimary, opacity: saving ? 0.6 : 1 }}>
        {saving ? 'Changing...' : 'Change Password'}
      </button>
    </div>
  );
};

export default Tools;
