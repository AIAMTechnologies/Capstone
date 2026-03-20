import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Link } from 'react-router-dom';
import type {
  EmailSyncConfig,
  EmailSyncStatus,
  EmailSyncResult,
  ClosureReview,
  EmailMessage,
  ActiveMatchReviewItem,
  NewLeadCandidate,
} from '../../types/types';
import {
  getEmailSyncConfig,
  saveEmailSyncConfig,
  getOAuthAuthorizeUrl,
  triggerEmailSync,
  getEmailSyncStatus,
  getClosureReviewQueue,
  getActiveMatchReview,
  getNewLeadCandidates,
  createLeadFromEmailCandidate,
  approveClosureReview,
  dismissClosureReview,
  getLeadEmails,
} from '../../services/api';
import { getApiErrorMessage } from '../../utils/apiErrors';
import AssignDealerModal from '../../components/admin/AssignDealerModal';
import EmailIntelDrawer from '../../components/admin/EmailIntelDrawer';

const inputStyle: React.CSSProperties = { padding: '8px 12px', borderRadius: 6, border: '1px solid #ddd', fontSize: 14 };
const cardStyle: React.CSSProperties = { background: 'white', borderRadius: 8, padding: 20, boxShadow: '0 2px 8px rgba(0,0,0,0.08)', marginBottom: 20 };
const btnPrimary: React.CSSProperties = { padding: '8px 20px', background: '#c91414', color: 'white', border: 'none', borderRadius: 6, fontSize: 14, fontWeight: 600, cursor: 'pointer' };
const btnGreen: React.CSSProperties = { ...btnPrimary, background: '#1a7a3a' };
const btnGray: React.CSSProperties = { ...btnPrimary, background: '#888' };
const btnSecondary: React.CSSProperties = { ...btnPrimary, background: '#f8fafc', color: '#1f2937', border: '1px solid #d1d5db' };
const thStyle: React.CSSProperties = { background: '#f8f9fa', textAlign: 'left' as const, padding: '10px 12px', fontSize: 13, fontWeight: 600, color: '#555' };
const tdStyle: React.CSSProperties = { padding: '10px 12px', borderBottom: '1px solid #eee', fontSize: 13 };
const labelStyle: React.CSSProperties = { display: 'block', fontSize: 13, fontWeight: 600, color: '#555', marginBottom: 4 };
const statCardStyle: React.CSSProperties = { padding: 14, borderRadius: 8, background: '#f8fafc', border: '1px solid #e5e7eb' };

const fmtDate = (d?: string | null) => d ? new Date(d).toLocaleString('en-CA') : '-';
const truncateText = (value?: string | null, max = 80) => {
  if (!value) return '-';
  return value.length > max ? `${value.slice(0, max)}...` : value;
};

const renderSentimentBadge = (sentiment?: 'positive' | 'neutral' | 'negative' | null) => {
  const colors: Record<string, { bg: string; color: string }> = {
    positive: { bg: '#dcfce7', color: '#166534' },
    neutral: { bg: '#f3f4f6', color: '#374151' },
    negative: { bg: '#fde8e8', color: '#991b1b' },
  };
  const key = sentiment || 'neutral';
  const color = colors[key] || colors.neutral;
  return (
    <span
      style={{
        padding: '2px 8px',
        borderRadius: 10,
        fontSize: 11,
        fontWeight: 600,
        background: color.bg,
        color: color.color,
      }}
    >
      {sentiment || 'unknown'}
    </span>
  );
};

const renderPriorityBadge = (priority: ActiveMatchReviewItem['review_priority']) => {
  const colors = {
    high: { bg: '#fee2e2', color: '#b91c1c' },
    medium: { bg: '#fef3c7', color: '#92400e' },
    low: { bg: '#e0f2fe', color: '#0f4c81' },
  };
  const color = colors[priority];
  return (
    <span
      style={{
        padding: '2px 8px',
        borderRadius: 10,
        fontSize: 11,
        fontWeight: 700,
        textTransform: 'uppercase',
        background: color.bg,
        color: color.color,
      }}
    >
      {priority}
    </span>
  );
};

const EmailIntel: React.FC = () => {
  const [oauthMsg, setOauthMsg] = useState('');
  const [queueRefreshToken, setQueueRefreshToken] = useState(0);

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
      <ActiveMatchReviewSection
        refreshToken={queueRefreshToken}
        onQueueChanged={() => setQueueRefreshToken((prev) => prev + 1)}
      />
      <NewLeadCandidatesSection
        refreshToken={queueRefreshToken}
        onQueueChanged={() => setQueueRefreshToken((prev) => prev + 1)}
      />
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
    shared_mailbox_email: '',
    target_mailbox_type: 'connected',
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
  const prevSyncInProgress = useRef(false);

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
      if (res.last_sync_result) {
        setSyncResult(res.last_sync_result);
      }
      if (res.last_sync_error) {
        setError(res.last_sync_error);
      }
    } catch {
      // Status endpoint may not be ready
    }
  }, []);

  useEffect(() => {
    loadConfig();
    loadStatus();
  }, [loadConfig, loadStatus]);

  useEffect(() => {
    if (!status?.sync_in_progress) {
      setSyncing(false);
      if (prevSyncInProgress.current) {
        loadStatus();
      }
      prevSyncInProgress.current = false;
      return;
    }

    prevSyncInProgress.current = true;
    const interval = window.setInterval(() => {
      loadStatus();
    }, 3000);

    return () => window.clearInterval(interval);
  }, [status?.sync_in_progress, loadStatus]);

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
      setError(getApiErrorMessage(err, 'Failed to save configuration'));
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
      setError(getApiErrorMessage(err, 'Failed to get authorization URL'));
    }
  };

  const handleSyncNow = async () => {
    setSyncing(true);
    setError('');
    setSuccess('');
    try {
      const res = await triggerEmailSync();
      if (res.result) {
        setSyncResult(res.result);
      } else if (res.synced !== undefined || res.matched !== undefined || res.flagged_for_review !== undefined) {
        setSyncResult(res);
      }
      if (res.message) {
        setSuccess(res.message);
      }
      await loadStatus();
    } catch (err: any) {
      setError(getApiErrorMessage(err, 'Sync failed'));
      } finally {
      if (!status?.sync_in_progress) {
        setSyncing(false);
      }
    }
  };

  const syncInProgress = syncing || status?.sync_in_progress;
  const syncStatusLabel = status?.sync_message
    ? status.sync_message
    : status?.sync_in_progress
      ? 'Syncing...'
      : null;
  const targetMailboxType = config.target_mailbox_type || 'connected';
  const targetMailboxLabel = targetMailboxType === 'group'
    ? 'Microsoft 365 Group Email'
    : targetMailboxType === 'shared'
      ? 'Shared Mailbox'
      : 'Connected Mailbox';

  const handleToggleSync = async () => {
    const updated = { ...config, sync_enabled: !config.sync_enabled };
    setConfig(updated);
    try {
      await saveEmailSyncConfig({ sync_enabled: updated.sync_enabled });
    } catch (err: any) {
      setConfig(prev => ({ ...prev, sync_enabled: !updated.sync_enabled }));
      setError(getApiErrorMessage(err, 'Failed to toggle sync'));
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
        <div>
          <label style={labelStyle}>Target Mailbox Type</label>
          <select
            value={targetMailboxType}
            onChange={e => setConfig(prev => ({ ...prev, target_mailbox_type: e.target.value as EmailSyncConfig['target_mailbox_type'] }))}
            style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' as const }}
          >
            <option value="connected">Connected mailbox</option>
            <option value="shared">Shared mailbox</option>
            <option value="group">Microsoft 365 Group</option>
          </select>
        </div>
        {targetMailboxType !== 'connected' && (
          <div>
            <label style={labelStyle}>{targetMailboxLabel}</label>
            <input
              value={config.shared_mailbox_email || ''}
              onChange={e => setConfig(prev => ({ ...prev, shared_mailbox_email: e.target.value }))}
              style={{ ...inputStyle, width: '100%', boxSizing: 'border-box' as const }}
              placeholder="info@windowfilmcanada.ca"
            />
          </div>
        )}
      </div>

      <div style={{ marginBottom: 16, fontSize: 13, lineHeight: 1.5, color: '#555' }}>
        {targetMailboxType === 'group' ? (
          <span>
            Microsoft 365 Group mode uses Graph group conversations instead of `/users/{'{'}mailbox{'}'}/messages`.
            Add delegated `Group.Read.All` and `Group-Conversation.Read.All` to the same app registration,
            grant admin consent, then click <strong>Connect Outlook</strong> again so the refreshed token includes the new scopes.
          </span>
        ) : targetMailboxType === 'shared' ? (
          <span>
            Shared mailbox mode uses delegated `Mail.Read.Shared` access on the same app registration.
          </span>
        ) : (
          <span>
            Connected mailbox mode syncs the signed-in Outlook mailbox directly.
          </span>
        )}
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
          {syncStatusLabel && (
            <div style={{ fontSize: 13, color: '#555' }}>
              <strong>Sync Status:</strong>{' '}
              {syncStatusLabel}
            </div>
          )}
        </div>

        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <button onClick={handleSyncNow} disabled={syncInProgress} style={{ ...btnPrimary, opacity: syncInProgress ? 0.6 : 1 }}>
            {syncInProgress ? 'Syncing...' : 'Sync Now'}
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
            <span style={{ fontSize: 12, color: '#888' }}>
              Connected: {config.user_email}
              {config.target_mailbox_email && config.target_mailbox_email !== config.user_email
                ? ` | Syncing (${targetMailboxType}): ${config.target_mailbox_email}`
                : ` | Syncing (${targetMailboxType})`}
            </span>
          )}
        </div>

        {syncResult && (
          <div style={{ marginTop: 12, padding: 12, borderRadius: 6, background: '#f0fdf4', fontSize: 13 }}>
            <strong>Latest Sync:</strong> {syncResult.synced ?? 0} emails synced, {syncResult.matched ?? 0} matched, {syncResult.flagged_for_review ?? 0} flagged for review
          </div>
        )}

        {status?.ai_costs && status.ai_costs.calls > 0 && (
          <div style={{ marginTop: 12, padding: 12, borderRadius: 6, background: '#f5f3ff', fontSize: 13, border: '1px solid #e5e0ff' }}>
            <strong>AI Usage (this session):</strong>{' '}
            {status.ai_costs.calls} calls | {status.ai_costs.total_input_tokens.toLocaleString()} input + {status.ai_costs.total_output_tokens.toLocaleString()} output tokens |{' '}
            <span style={{ fontWeight: 700, color: (status.ai_costs.total_cost_usd * 1.44) > 0.50 ? '#c91414' : '#166534' }}>
              ${(status.ai_costs.total_cost_usd * 1.44).toFixed(4)} CAD
            </span>
            {status.ai_costs.by_model && Object.keys(status.ai_costs.by_model).length > 0 && (
              <span style={{ marginLeft: 8, color: '#888' }}>
                ({Object.entries(status.ai_costs.by_model).map(([m, d]: [string, any]) => `${m}: ${d.calls} calls $${(d.cost * 1.44).toFixed(4)} CAD`).join(' | ')})
              </span>
            )}
          </div>
        )}

        {status?.sync_in_progress && status.current_sync_counts && (
          <div style={{ marginTop: 12, padding: 12, borderRadius: 6, background: '#eff6ff', fontSize: 13 }}>
            <strong>In Progress:</strong> {status.current_sync_counts.synced} emails synced, {status.current_sync_counts.matched} matched so far
          </div>
        )}
      </div>
    </div>
  );
};

// ===================== Active Matched Lead Cleanup =====================
const ActiveMatchReviewSection: React.FC<{
  refreshToken: number;
  onQueueChanged: () => void;
}> = ({ refreshToken, onQueueChanged }) => {
  const [items, setItems] = useState<ActiveMatchReviewItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [assignLead, setAssignLead] = useState<ActiveMatchReviewItem | null>(null);
  const [drawerLead, setDrawerLead] = useState<{ leadId: number; leadName: string } | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await getActiveMatchReview();
      setItems(res);
    } catch (err: any) {
      setError(getApiErrorMessage(err, 'Failed to load active matched leads'));
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load, refreshToken]);

  const missingDealerCount = items.filter((item) => item.missing_dealer).length;
  const reviewCount = items.filter((item) => item.needs_match_review).length;

  return (
    <div style={cardStyle}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, alignItems: 'flex-start', marginBottom: 16, flexWrap: 'wrap' }}>
        <div>
          <h2 style={{ fontSize: 18, fontWeight: 700, color: '#1a1a2e', margin: 0 }}>Active Match Cleanup</h2>
          <p style={{ margin: '6px 0 0 0', fontSize: 13, color: '#6b7280', lineHeight: 1.5 }}>
            Review the active leads where matched email activity can reduce manual assignment or cleanup work.
          </p>
        </div>
        <button onClick={load} disabled={loading} style={{ ...btnSecondary, opacity: loading ? 0.6 : 1 }}>
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, marginBottom: 16 }}>
        <div style={statCardStyle}>
          <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>Matched Active Leads</div>
          <div style={{ fontSize: 24, fontWeight: 700, color: '#111827' }}>{items.length}</div>
        </div>
        <div style={statCardStyle}>
          <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>Missing Dealer</div>
          <div style={{ fontSize: 24, fontWeight: 700, color: '#b91c1c' }}>{missingDealerCount}</div>
        </div>
        <div style={statCardStyle}>
          <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>Needs Match Review</div>
          <div style={{ fontSize: 24, fontWeight: 700, color: '#92400e' }}>{reviewCount}</div>
        </div>
      </div>

      {error && <div style={{ color: '#c91414', marginBottom: 12, fontSize: 13 }}>{error}</div>}
      {loading && <p style={{ color: '#999', fontSize: 13 }}>Loading cleanup queue...</p>}

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={thStyle}>Lead</th>
              <th style={thStyle}>Assignment</th>
              <th style={thStyle}>Latest Email</th>
              <th style={thStyle}>Signals</th>
              <th style={thStyle}>Review Reason</th>
              <th style={thStyle}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.lead_id}>
                <td style={{ ...tdStyle, minWidth: 220 }}>
                  <div style={{ fontWeight: 700, color: '#1f2937', marginBottom: 4 }}>{item.lead_name || `Lead #${item.lead_id}`}</div>
                  <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>
                    {item.lead_email || 'No email'} {item.lead_phone ? `| ${item.lead_phone}` : ''}
                  </div>
                  <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 6 }}>
                    Status: {item.lead_status} | Source: {item.lead_source || 'unknown'}
                  </div>
                  <Link
                    to={`/admin/dashboard?lead=${item.lead_id}`}
                    style={{ color: '#c91414', textDecoration: 'none', fontWeight: 600, fontSize: 12 }}
                  >
                    Open lead #{item.lead_id}
                  </Link>
                </td>
                <td style={{ ...tdStyle, minWidth: 200 }}>
                  <div style={{ fontWeight: 600, color: item.missing_dealer ? '#b91c1c' : '#1f2937', marginBottom: 4 }}>
                    Dealer: {item.assigned_dealer_name || 'Missing dealer'}
                  </div>
                  {item.landing_page && (
                    <div style={{ fontSize: 12, color: '#0f4c81', marginBottom: 4 }}>
                      Dealer-site signal: {item.landing_page}
                    </div>
                  )}
                  <div style={{ fontSize: 12, color: '#6b7280' }}>
                    Lead age: {item.lead_age_days}d | Last email: {item.days_since_last_email ?? '-'}d ago
                  </div>
                </td>
                <td style={{ ...tdStyle, minWidth: 260 }}>
                  <div style={{ fontWeight: 600, color: '#1f2937', marginBottom: 4 }}>
                    {truncateText(item.latest_subject, 70)}
                  </div>
                  <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>
                    {item.latest_sender_name || item.latest_sender_email} | {fmtDate(item.latest_email_at || item.last_email_activity)}
                  </div>
                  <div style={{ fontSize: 12, color: '#374151', lineHeight: 1.5, marginBottom: 6 }}>
                    {truncateText(item.latest_ai_summary, 120)}
                  </div>
                  {renderSentimentBadge(item.latest_ai_sentiment)}
                </td>
                <td style={{ ...tdStyle, minWidth: 200 }}>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
                    {renderPriorityBadge(item.review_priority)}
                    {item.weak_match_count > 0 && (
                      <span style={{ padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600, background: '#fef3c7', color: '#92400e' }}>
                        {item.weak_match_count} weak match
                      </span>
                    )}
                    {item.active_duplicate_email_count > 1 && (
                      <span style={{ padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600, background: '#fee2e2', color: '#991b1b' }}>
                        duplicate email
                      </span>
                    )}
                    {item.max_match_confidence !== null && (
                      <span style={{ padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600, background: '#e0f2fe', color: '#0f4c81' }}>
                        {(item.max_match_confidence * 100).toFixed(0)}% confidence
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 6 }}>
                    Match methods: {item.match_methods || '-'}
                  </div>
                  <div style={{ fontSize: 12, color: '#6b7280' }}>
                    Matched emails: {item.matched_email_count} | Strong: {item.strong_match_count}
                  </div>
                </td>
                <td style={{ ...tdStyle, minWidth: 220, color: '#374151', lineHeight: 1.5 }}>
                  {item.review_reason}
                </td>
                <td style={{ ...tdStyle, minWidth: 180 }}>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <button
                      onClick={() => setDrawerLead({ leadId: item.lead_id, leadName: item.lead_name || `Lead #${item.lead_id}` })}
                      style={{ ...btnSecondary, width: '100%', fontSize: 12, padding: '6px 12px' }}
                    >
                      Review Emails
                    </button>
                    {item.missing_dealer && (
                      <button
                        onClick={() => setAssignLead(item)}
                        style={{ ...btnPrimary, width: '100%', fontSize: 12, padding: '6px 12px' }}
                      >
                        Assign Dealer
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
            {items.length === 0 && !loading && (
              <tr>
                <td colSpan={6} style={{ ...tdStyle, textAlign: 'center', color: '#999' }}>
                  No active matched leads need cleanup review yet
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {assignLead && (
        <AssignDealerModal
          leadId={assignLead.lead_id}
          isOpen={true}
          onClose={() => setAssignLead(null)}
          onAssigned={() => {
            setAssignLead(null);
            load();
            onQueueChanged();
          }}
        />
      )}

      {drawerLead && (
        <EmailIntelDrawer
          leadId={drawerLead.leadId}
          leadName={drawerLead.leadName}
          isOpen={true}
          onClose={() => setDrawerLead(null)}
        />
      )}
    </div>
  );
};

// ===================== New Lead Candidates =====================
const NewLeadCandidatesSection: React.FC<{
  refreshToken: number;
  onQueueChanged: () => void;
}> = ({ refreshToken, onQueueChanged }) => {
  const [items, setItems] = useState<NewLeadCandidate[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [actionMessage, setActionMessage] = useState('');
  const [creatingId, setCreatingId] = useState<number | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await getNewLeadCandidates();
      setItems(res);
    } catch (err: any) {
      setError(getApiErrorMessage(err, 'Failed to load new lead candidates'));
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load, refreshToken]);

  const handleCreateLead = async (emailId: number) => {
    setCreatingId(emailId);
    setError('');
    setActionMessage('');
    try {
      const res = await createLeadFromEmailCandidate(emailId);
      const actionLabel = res.action === 'created' ? 'Created lead' : 'Matched existing lead';
      setActionMessage(`${actionLabel} #${res.lead_id} from candidate email. ${res.matched_email_count} email(s) linked.`);
      await load();
      onQueueChanged();
    } catch (err: any) {
      setError(getApiErrorMessage(err, 'Failed to create lead from candidate email'));
    } finally {
      setCreatingId(null);
    }
  };

  const highSignalCount = items.filter((item) => item.candidate_score >= 6).length;
  const overlapCount = items.filter((item) => item.existing_sender_lead_count > 0).length;

  return (
    <div style={cardStyle}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, alignItems: 'flex-start', marginBottom: 16, flexWrap: 'wrap' }}>
        <div>
          <h2 style={{ fontSize: 18, fontWeight: 700, color: '#1a1a2e', margin: 0 }}>New Lead Candidates</h2>
          <p style={{ margin: '6px 0 0 0', fontSize: 13, color: '#6b7280', lineHeight: 1.5 }}>
            Unmatched inbound emails that look like real lead inquiries. This stays read-only until the create-from-email flow is ready.
          </p>
        </div>
        <button onClick={load} disabled={loading} style={{ ...btnSecondary, opacity: loading ? 0.6 : 1 }}>
          {loading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, marginBottom: 16 }}>
        <div style={statCardStyle}>
          <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>Candidate Emails</div>
          <div style={{ fontSize: 24, fontWeight: 700, color: '#111827' }}>{items.length}</div>
        </div>
        <div style={statCardStyle}>
          <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>High-Signal Candidates</div>
          <div style={{ fontSize: 24, fontWeight: 700, color: '#166534' }}>{highSignalCount}</div>
        </div>
        <div style={statCardStyle}>
          <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>Sender Already In Leads</div>
          <div style={{ fontSize: 24, fontWeight: 700, color: '#92400e' }}>{overlapCount}</div>
        </div>
      </div>

      {error && <div style={{ color: '#c91414', marginBottom: 12, fontSize: 13 }}>{error}</div>}
      {actionMessage && <div style={{ color: '#166534', marginBottom: 12, fontSize: 13 }}>{actionMessage}</div>}
      {loading && <p style={{ color: '#999', fontSize: 13 }}>Loading candidate emails...</p>}

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={thStyle}>Sender</th>
              <th style={thStyle}>Subject / Preview</th>
              <th style={thStyle}>Reason</th>
              <th style={thStyle}>Score</th>
              <th style={thStyle}>Existing Lead Overlap</th>
              <th style={thStyle}>Received</th>
              <th style={thStyle}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.id}>
                <td style={{ ...tdStyle, minWidth: 220 }}>
                  <div style={{ fontWeight: 700, color: '#1f2937', marginBottom: 4 }}>{item.sender_name || item.sender_email}</div>
                  <div style={{ fontSize: 12, color: '#6b7280' }}>{item.sender_email}</div>
                </td>
                <td style={{ ...tdStyle, minWidth: 280 }}>
                  <div style={{ fontWeight: 600, color: '#1f2937', marginBottom: 4 }}>{truncateText(item.subject, 75)}</div>
                  <div style={{ fontSize: 12, color: '#6b7280', lineHeight: 1.5 }}>{truncateText(item.body_preview, 120)}</div>
                </td>
                <td style={{ ...tdStyle, minWidth: 160, color: '#374151' }}>{item.candidate_reason}</td>
                <td style={tdStyle}>
                  <span
                    style={{
                      padding: '2px 8px',
                      borderRadius: 10,
                      fontSize: 11,
                      fontWeight: 700,
                      background: item.candidate_score >= 6 ? '#dcfce7' : '#fef3c7',
                      color: item.candidate_score >= 6 ? '#166534' : '#92400e',
                    }}
                  >
                    {item.candidate_score}
                  </span>
                </td>
                <td style={tdStyle}>
                  <span style={{ color: item.existing_sender_lead_count > 0 ? '#92400e' : '#6b7280', fontWeight: item.existing_sender_lead_count > 0 ? 600 : 500 }}>
                    {item.existing_sender_lead_count > 0 ? `${item.existing_sender_lead_count} lead(s)` : 'None'}
                  </span>
                </td>
                <td style={tdStyle}>{fmtDate(item.received_at)}</td>
                <td style={{ ...tdStyle, minWidth: 160 }}>
                  <button
                    onClick={() => handleCreateLead(item.id)}
                    disabled={creatingId === item.id}
                    style={{ ...btnPrimary, width: '100%', fontSize: 12, padding: '6px 12px', opacity: creatingId === item.id ? 0.6 : 1 }}
                  >
                    {creatingId === item.id ? 'Working...' : item.existing_sender_lead_count > 0 ? 'Create / Match' : 'Create Lead'}
                  </button>
                </td>
              </tr>
            ))}
            {items.length === 0 && !loading && (
              <tr>
                <td colSpan={7} style={{ ...tdStyle, textAlign: 'center', color: '#999' }}>
                  No unmatched inbound emails meet the current candidate threshold
                </td>
              </tr>
            )}
          </tbody>
        </table>
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
      setError(getApiErrorMessage(err, 'Failed to load review queue'));
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
      setError(getApiErrorMessage(err, 'Failed to approve'));
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
      setError(getApiErrorMessage(err, 'Failed to dismiss'));
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
      // Load recent matched emails (leadId 0 = dashboard feed)
      const res = await getLeadEmails(0);
      setEmails(res);
    } catch (err: any) {
      setError(getApiErrorMessage(err, 'Failed to load recent emails'));
      setEmails([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  return (
    <div style={cardStyle}>
      <h2 style={{ fontSize: 18, fontWeight: 700, color: '#1a1a2e', marginBottom: 16 }}>Recent Matched Email Activity</h2>

      {error && <div style={{ color: '#c91414', marginBottom: 12, fontSize: 13 }}>{error}</div>}
      {loading && <p style={{ color: '#999', fontSize: 13 }}>Loading...</p>}

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={thStyle}>Sender</th>
              <th style={thStyle}>Subject</th>
              <th style={thStyle}>Received</th>
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
                <td style={tdStyle}>
                  {e.matched_lead_id ? (
                    <Link
                      to={`/admin/dashboard?lead=${e.matched_lead_id}`}
                      style={{ color: '#c91414', textDecoration: 'none', fontWeight: 600 }}
                      title={`View lead #${e.matched_lead_id}`}
                    >
                      {e.lead_first_name || e.lead_last_name
                        ? `${e.lead_first_name || ''} ${e.lead_last_name || ''}`.trim()
                        : 'Lead'}
                      {` (#${e.matched_lead_id})`}
                    </Link>
                  ) : '-'}
                </td>
                <td style={tdStyle}>{renderSentimentBadge(e.ai_sentiment)}</td>
              </tr>
            ))}
            {emails.length === 0 && !loading && (
              <tr><td colSpan={5} style={{ ...tdStyle, textAlign: 'center', color: '#999' }}>No matched emails yet</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default EmailIntel;
