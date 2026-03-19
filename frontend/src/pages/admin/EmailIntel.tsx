import React, { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import type { EmailSyncConfig, EmailSyncStatus, EmailSyncResult, ClosureReview, EmailMessage } from '../../types/types';
import {
  getEmailSyncConfig,
  saveEmailSyncConfig,
  getOAuthAuthorizeUrl,
  triggerEmailSync,
  getEmailSyncStatus,
  getClosureReviewQueue,
  approveClosureReview,
  dismissClosureReview,
  getLeadEmails,
} from '../../services/api';

const inputStyle: React.CSSProperties = { padding: '8px 12px', borderRadius: 6, border: '1px solid #ddd', fontSize: 14 };
const cardStyle: React.CSSProperties = { background: 'white', borderRadius: 8, padding: 20, boxShadow: '0 2px 8px rgba(0,0,0,0.08)', marginBottom: 20 };
const btnPrimary: React.CSSProperties = { padding: '8px 20px', background: '#c91414', color: 'white', border: 'none', borderRadius: 6, fontSize: 14, fontWeight: 600, cursor: 'pointer' };
const btnGreen: React.CSSProperties = { ...btnPrimary, background: '#1a7a3a' };
const btnGray: React.CSSProperties = { ...btnPrimary, background: '#888' };
const thStyle: React.CSSProperties = { background: '#f8f9fa', textAlign: 'left' as const, padding: '10px 12px', fontSize: 13, fontWeight: 600, color: '#555' };
const tdStyle: React.CSSProperties = { padding: '10px 12px', borderBottom: '1px solid #eee', fontSize: 13 };
const labelStyle: React.CSSProperties = { display: 'block', fontSize: 13, fontWeight: 600, color: '#555', marginBottom: 4 };

const fmtDate = (d?: string | null) => d ? new Date(d).toLocaleString('en-CA') : '-';

// Safely extract error message string from axios errors (FastAPI validation errors are objects/arrays)
const getErrorMsg = (err: any, fallback: string): string => {
  const detail = err?.response?.data?.detail;
  if (!detail) return err?.message || fallback;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((d: any) => d.msg || JSON.stringify(d)).join('; ');
  if (typeof detail === 'object') return detail.msg || JSON.stringify(detail);
  return fallback;
};

const EmailIntel: React.FC = () => {
  const [oauthMsg, setOauthMsg] = useState('');

  useEffect(() => {
    // Handle OAuth redirect query params
    const params = new URLSearchParams(window.location.search);
    if (params.get('oauth_success') === 'true') {
      const email = params.get('email') || '';
      setOauthMsg(`Connected successfully to ${email}`);
      // Clean URL
      window.history.replaceState({}, '', window.location.pathname);
    } else if (params.get('oauth_error')) {
      setOauthMsg(`OAuth error: ${params.get('oauth_error')}`);
      window.history.replaceState({}, '', window.location.pathname);
    }
  }, []);

  return (
    <div style={{ padding: 24 }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, color: '#1a1a2e', marginBottom: 20 }}>Email Intelligence</h1>
      {oauthMsg && (
        <div style={{
          padding: '12px 16px', borderRadius: 6, marginBottom: 16,
          background: oauthMsg.includes('error') ? '#fef2f2' : '#f0fdf4',
          color: oauthMsg.includes('error') ? '#c91414' : '#166534',
          fontWeight: 600, fontSize: 14,
        }}>
          {oauthMsg}
        </div>
      )}
      <SyncConfigSection />
      <ReviewQueueSection />
      <RecentEmailsSection />
    </div>
  );
};

// ===================== Sync Configuration =====================
const SyncConfigSection: React.FC = () => {
  const [config, setConfig] = useState<Partial<EmailSyncConfig>>({
    ms_tenant_id: '',
    ms_client_id: '',
    ms_redirect_uri: '',
    sync_enabled: false,
    sync_interval_minutes: 15,
  });
  const [clientSecret, setClientSecret] = useState('');
  const [status, setStatus] = useState<EmailSyncStatus | null>(null);
  const [saving, setSaving] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [syncResult, setSyncResult] = useState<EmailSyncResult | null>(null);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const loadConfig = useCallback(async () => {
    try {
      const res = await getEmailSyncConfig();
      if (res) {
        setConfig(res);
      }
    } catch {
      // Config may not exist yet
    }
  }, []);

  const loadStatus = useCallback(async () => {
    try {
      const res = await getEmailSyncStatus();
      setStatus(res);
    } catch {
      // Status endpoint may not be ready
    }
  }, []);

  useEffect(() => {
    loadConfig();
    loadStatus();
  }, [loadConfig, loadStatus]);

  const handleSaveConfig = async () => {
    setSaving(true);
    setError('');
    setSuccess('');
    try {
      const payload: Partial<EmailSyncConfig> & { ms_client_secret?: string } = { ...config };
      if (clientSecret) {
        payload.ms_client_secret = clientSecret;
      }
      await saveEmailSyncConfig(payload);
      setSuccess('Configuration saved');
      setClientSecret('');
    } catch (err: any) {
      setError(getErrorMsg(err, 'Failed to save configuration'));
    } finally {
      setSaving(false);
    }
  };

  const handleConnectOutlook = async () => {
    setError('');
    try {
      const res = await getOAuthAuthorizeUrl();
      // Redirect in same window — Microsoft OAuth works best this way
      window.location.href = res.auth_url;
    } catch (err: any) {
      setError(getErrorMsg(err, 'Failed to get authorization URL'));
    }
  };

  const handleSyncNow = async () => {
    setSyncing(true);
    setSyncResult(null);
    setError('');
    try {
      const res = await triggerEmailSync();
      setSyncResult(res);
      await loadStatus();
    } catch (err: any) {
      setError(getErrorMsg(err, 'Sync failed'));
    } finally {
      setSyncing(false);
    }
  };

  const handleToggleSync = async () => {
    const updated = { ...config, sync_enabled: !config.sync_enabled };
    setConfig(updated);
    try {
      await saveEmailSyncConfig({ sync_enabled: updated.sync_enabled });
    } catch (err: any) {
      setConfig(prev => ({ ...prev, sync_enabled: !updated.sync_enabled }));
      setError(getErrorMsg(err, 'Failed to toggle sync'));
    }
  };

  return (
    <div style={cardStyle}>
      <h2 style={{ fontSize: 18, fontWeight: 700, color: '#1a1a2e', marginBottom: 16 }}>Email Sync Configuration</h2>

      {error && <div style={{ color: '#c91414', marginBottom: 12, fontSize: 13 }}>{error}</div>}
      {success && <div style={{ color: '#1a7a3a', marginBottom: 12, fontSize: 13 }}>{success}</div>}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
        <div>
          <label style={labelStyle}>Tenant ID</label>
          <input
            value={config.ms_tenant_id || ''}
            onChange={e => setConfig(prev => ({ ...prev, ms_tenant_id: e.target.value }))}
            style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' as const }}
            placeholder="Microsoft Tenant ID"
          />
        </div>
        <div>
          <label style={labelStyle}>Client ID</label>
          <input
            value={config.ms_client_id || ''}
            onChange={e => setConfig(prev => ({ ...prev, ms_client_id: e.target.value }))}
            style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' as const }}
            placeholder="Microsoft Client ID"
          />
        </div>
        <div>
          <label style={labelStyle}>Client Secret</label>
          <input
            type="password"
            value={clientSecret}
            onChange={e => setClientSecret(e.target.value)}
            style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' as const }}
            placeholder="Enter new secret (leave blank to keep existing)"
          />
        </div>
        <div>
          <label style={labelStyle}>Redirect URI</label>
          <input
            value={config.ms_redirect_uri || ''}
            onChange={e => setConfig(prev => ({ ...prev, ms_redirect_uri: e.target.value }))}
            style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' as const }}
            placeholder="https://yourdomain.com/api/email-intel/oauth/callback"
          />
        </div>
      </div>

      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 20 }}>
        <button onClick={handleSaveConfig} disabled={saving} style={{ ...btnPrimary, opacity: saving ? 0.6 : 1 }}>
          {saving ? 'Saving...' : 'Save Config'}
        </button>
        <button onClick={handleConnectOutlook} style={{ ...btnPrimary, background: '#0078d4' }}>
          Connect Outlook
        </button>
      </div>

      {/* Sync Status */}
      <div style={{ borderTop: '1px solid #eee', paddingTop: 16 }}>
        <h3 style={{ fontSize: 15, fontWeight: 600, color: '#1a1a2e', marginBottom: 12 }}>Sync Status</h3>
        <div style={{ display: 'flex', gap: 24, alignItems: 'center', flexWrap: 'wrap', marginBottom: 12 }}>
          <div style={{ fontSize: 13, color: '#555' }}>
            <strong>Last Sync:</strong> {fmtDate(status?.last_sync)}
          </div>
          <div style={{ fontSize: 13, color: '#555' }}>
            <strong>Total Emails:</strong> {status?.total_emails ?? 0}
          </div>
          <div style={{ fontSize: 13, color: '#555' }}>
            <strong>Matched:</strong> {status?.matched_emails ?? 0}
          </div>
          <div style={{ fontSize: 13, color: '#555' }}>
            <strong>Pending Reviews:</strong> {status?.pending_reviews ?? 0}
          </div>
        </div>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <button onClick={handleSyncNow} disabled={syncing} style={{ ...btnPrimary, opacity: syncing ? 0.6 : 1 }}>
            {syncing ? 'Syncing...' : 'Sync Now'}
          </button>
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13, cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={config.sync_enabled || false}
              onChange={handleToggleSync}
            />
            Auto-sync enabled
          </label>
          {config.user_email && (
            <span style={{ fontSize: 12, color: '#888' }}>Connected: {config.user_email}</span>
          )}
        </div>

        {syncResult && (
          <div style={{ marginTop: 12, padding: 12, borderRadius: 6, background: '#f0fdf4', fontSize: 13 }}>
            <strong>Sync Complete:</strong> {syncResult.synced} emails synced, {syncResult.matched} matched, {syncResult.flagged_for_review} flagged for review
          </div>
        )}
      </div>
    </div>
  );
};

// ===================== Review Queue =====================
const ReviewQueueSection: React.FC = () => {
  const [reviews, setReviews] = useState<ClosureReview[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [actionLoading, setActionLoading] = useState<number | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await getClosureReviewQueue();
      setReviews(res);
    } catch (err: any) {
      console.error('Review queue error:', err);
      setError(getErrorMsg(err, 'Failed to load review queue'));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleApprove = async (id: number) => {
    setActionLoading(id);
    try {
      await approveClosureReview(id);
      setReviews(prev => prev.filter(r => r.id !== id));
    } catch (err: any) {
      setError(getErrorMsg(err, 'Failed to approve'));
    } finally {
      setActionLoading(null);
    }
  };

  const handleDismiss = async (id: number) => {
    setActionLoading(id);
    try {
      await dismissClosureReview(id);
      setReviews(prev => prev.filter(r => r.id !== id));
    } catch (err: any) {
      setError(getErrorMsg(err, 'Failed to dismiss'));
    } finally {
      setActionLoading(null);
    }
  };

  return (
    <div style={cardStyle}>
      <h2 style={{ fontSize: 18, fontWeight: 700, color: '#1a1a2e', marginBottom: 16 }}>Review Queue</h2>

      {error && <div style={{ color: '#c91414', marginBottom: 12, fontSize: 13 }}>{error}</div>}
      {loading && <p style={{ color: '#999', fontSize: 13 }}>Loading...</p>}

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={thStyle}>Lead Name</th>
              <th style={thStyle}>Email</th>
              <th style={thStyle}>Dealer</th>
              <th style={thStyle}>Days Inactive</th>
              <th style={thStyle}>Email Count</th>
              <th style={thStyle}>AI Reasoning</th>
              <th style={thStyle}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {reviews.map(r => (
              <tr key={r.id}>
                <td style={tdStyle}>{r.lead_name}</td>
                <td style={tdStyle}>{r.lead_email}</td>
                <td style={tdStyle}>{r.dealer_name}</td>
                <td style={tdStyle}>{r.days_inactive}</td>
                <td style={tdStyle}>{r.email_count}</td>
                <td style={{ ...tdStyle, maxWidth: 250 }}>
                  <span title={r.ai_reasoning}>
                    {r.ai_reasoning.length > 80 ? r.ai_reasoning.slice(0, 80) + '...' : r.ai_reasoning}
                  </span>
                </td>
                <td style={tdStyle}>
                  <div style={{ display: 'flex', gap: 6 }}>
                    <button
                      onClick={() => handleApprove(r.id)}
                      disabled={actionLoading === r.id}
                      style={{ ...btnGreen, fontSize: 12, padding: '4px 12px', opacity: actionLoading === r.id ? 0.6 : 1 }}
                    >
                      {actionLoading === r.id ? '...' : 'Approve Close'}
                    </button>
                    <button
                      onClick={() => handleDismiss(r.id)}
                      disabled={actionLoading === r.id}
                      style={{ ...btnGray, fontSize: 12, padding: '4px 12px', opacity: actionLoading === r.id ? 0.6 : 1 }}
                    >
                      Dismiss
                    </button>
                  </div>
                </td>
              </tr>
            ))}
            {reviews.length === 0 && !loading && (
              <tr><td colSpan={7} style={{ ...tdStyle, textAlign: 'center', color: '#999' }}>No leads pending review</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// ===================== Recent Email Activity =====================
const RecentEmailsSection: React.FC = () => {
  const [emails, setEmails] = useState<EmailMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      // Load recent emails (leadId 0 = recent/all)
      const res = await getLeadEmails(0);
      setEmails(res);
    } catch (err: any) {
      setError(getErrorMsg(err, 'Failed to load recent emails'));
      setEmails([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const sentimentBadge = (s: EmailMessage['ai_sentiment']) => {
    const colors: Record<string, { bg: string; color: string }> = {
      positive: { bg: '#dcfce7', color: '#166534' },
      neutral: { bg: '#f3f4f6', color: '#374151' },
      negative: { bg: '#fde8e8', color: '#991b1b' },
    };
    const c = s ? colors[s] : colors.neutral;
    return (
      <span style={{
        padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
        background: c.bg, color: c.color,
      }}>
        {s || 'unknown'}
      </span>
    );
  };

  const matchBadge = (email: EmailMessage) => {
    if (email.matched_lead_id) {
      return (
        <span style={{
          padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
          background: '#dcfce7', color: '#166534',
        }}>
          Matched
        </span>
      );
    }
    return (
      <span style={{
        padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600,
        background: '#f3f4f6', color: '#6b7280',
      }}>
        Unmatched
      </span>
    );
  };

  return (
    <div style={cardStyle}>
      <h2 style={{ fontSize: 18, fontWeight: 700, color: '#1a1a2e', marginBottom: 16 }}>Recent Email Activity</h2>

      {error && <div style={{ color: '#c91414', marginBottom: 12, fontSize: 13 }}>{error}</div>}
      {loading && <p style={{ color: '#999', fontSize: 13 }}>Loading...</p>}

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={thStyle}>Sender</th>
              <th style={thStyle}>Subject</th>
              <th style={thStyle}>Received</th>
              <th style={thStyle}>Match Status</th>
              <th style={thStyle}>Matched Lead</th>
              <th style={thStyle}>Sentiment</th>
            </tr>
          </thead>
          <tbody>
            {emails.map(e => (
              <tr key={e.id}>
                <td style={tdStyle}>{e.sender_name || e.sender_email}</td>
                <td style={{ ...tdStyle, maxWidth: 250 }}>
                  <span title={e.subject}>
                    {e.subject.length > 50 ? e.subject.slice(0, 50) + '...' : e.subject}
                  </span>
                </td>
                <td style={tdStyle}>{fmtDate(e.received_at)}</td>
                <td style={tdStyle}>{matchBadge(e)}</td>
                <td style={tdStyle}>
                  {e.matched_lead_id ? (
                    <Link
                      to={`/admin/dashboard?lead=${e.matched_lead_id}`}
                      style={{ color: '#c91414', textDecoration: 'none', fontWeight: 600 }}
                      title={`View lead #${e.matched_lead_id}`}
                    >
                      {e.lead_first_name || e.lead_last_name
                        ? `${e.lead_first_name || ''} ${e.lead_last_name || ''}`.trim()
                        : `Lead #${e.matched_lead_id}`}
                    </Link>
                  ) : '-'}
                </td>
                <td style={tdStyle}>{sentimentBadge(e.ai_sentiment)}</td>
              </tr>
            ))}
            {emails.length === 0 && !loading && (
              <tr><td colSpan={6} style={{ ...tdStyle, textAlign: 'center', color: '#999' }}>No recent emails</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default EmailIntel;
