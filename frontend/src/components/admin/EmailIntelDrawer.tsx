import React, { useState, useEffect, useCallback } from 'react';
import type { EmailMessage, EmailLeadContext } from '../../types/types';
import { getLeadEmails, getLeadEmailContext } from '../../services/api';

interface EmailIntelDrawerProps {
  leadId: number;
  leadName: string;
  isOpen: boolean;
  onClose: () => void;
}

/** ai_action_items arrives as a JSON string or an array — normalize to string[] */
const parseActions = (raw: unknown): string[] => {
  if (Array.isArray(raw)) return raw;
  if (typeof raw === 'string') {
    try { const parsed = JSON.parse(raw); return Array.isArray(parsed) ? parsed : []; } catch { return []; }
  }
  return [];
};

const fmtDate = (d?: string | null) =>
  d ? new Date(d).toLocaleDateString('en-CA', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' }) : '-';

const fmtShortDate = (d?: string | null) =>
  d ? new Date(d).toLocaleDateString('en-CA', { month: 'short', day: 'numeric' }) : '-';

const getErrorMsg = (err: any, fallback: string): string => {
  const detail = err?.response?.data?.detail;
  if (!detail) return err?.message || fallback;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((d: any) => d.msg || JSON.stringify(d)).join('; ');
  return fallback;
};

// ── Badges ──────────────────────────────────────────────────────────────────

const SentimentBadge: React.FC<{ s: EmailMessage['ai_sentiment'] }> = ({ s }) => {
  const colors: Record<string, { bg: string; color: string; label: string }> = {
    positive: { bg: '#dcfce7', color: '#166534', label: 'Positive' },
    neutral:  { bg: '#f3f4f6', color: '#374151', label: 'Neutral' },
    negative: { bg: '#fde8e8', color: '#991b1b', label: 'Negative' },
  };
  const c = s ? colors[s] : colors.neutral;
  return (
    <span style={{ padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600, background: c.bg, color: c.color }}>
      {c.label}
    </span>
  );
};

const UrgencyBadge: React.FC<{ u: EmailMessage['ai_urgency'] }> = ({ u }) => {
  if (!u || u === 'low') return null;
  const colors = {
    high:   { bg: '#fee2e2', color: '#991b1b', label: '🔴 High Urgency' },
    medium: { bg: '#fef3c7', color: '#92400e', label: '🟡 Medium' },
    low:    { bg: '#f3f4f6', color: '#6b7280', label: 'Low' },
  };
  const c = colors[u] ?? colors.low;
  return (
    <span style={{ padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 700, background: c.bg, color: c.color }}>
      {c.label}
    </span>
  );
};

const JobBadge: React.FC<{ type: EmailMessage['ai_job_type'] }> = ({ type }) => {
  if (!type || type === 'unknown') return null;
  const labels: Record<string, string> = { residential: 'Residential', commercial: 'Commercial', replacement: 'Replacement' };
  return (
    <span style={{ padding: '2px 8px', borderRadius: 10, fontSize: 11, fontWeight: 600, background: '#eff6ff', color: '#1d4ed8' }}>
      {labels[type] ?? type}
    </span>
  );
};

const DealStageBadge: React.FC<{ ready: boolean }> = ({ ready }) => {
  if (!ready) return null;
  return (
    <span style={{ padding: '2px 10px', borderRadius: 10, fontSize: 11, fontWeight: 700, background: '#f0fdf4', color: '#15803d', border: '1px solid #bbf7d0' }}>
      ✓ Ready to Close
    </span>
  );
};

// ── Main component ──────────────────────────────────────────────────────────

const EmailIntelDrawer: React.FC<EmailIntelDrawerProps> = ({ leadId, leadName, isOpen, onClose }) => {
  const [context, setContext] = useState<EmailLeadContext | null>(null);
  const [emails, setEmails] = useState<EmailMessage[]>([]);
  const [loadingContext, setLoadingContext] = useState(false);
  const [loadingEmails, setLoadingEmails] = useState(false);
  const [expandedEmail, setExpandedEmail] = useState<number | null>(null);
  const [contextError, setContextError] = useState('');
  const [emailsError, setEmailsError] = useState('');
  const [checkedActions, setCheckedActions] = useState<Set<string>>(new Set());

  const loadData = useCallback(async () => {
    if (!leadId) return;
    setLoadingContext(true);
    setLoadingEmails(true);
    setContextError('');
    setEmailsError('');

    getLeadEmailContext(leadId)
      .then(setContext)
      .catch((err: any) => setContextError(getErrorMsg(err, 'Failed to load AI context')))
      .finally(() => setLoadingContext(false));

    getLeadEmails(leadId)
      .then((res) => {
        setEmails(res);
        // Auto-expand the most recent email
        if (res.length > 0) setExpandedEmail(res[0].id);
      })
      .catch((err: any) => setEmailsError(getErrorMsg(err, 'Failed to load emails')))
      .finally(() => setLoadingEmails(false));
  }, [leadId]);

  useEffect(() => {
    if (isOpen) {
      loadData();
    } else {
      setContext(null);
      setEmails([]);
      setExpandedEmail(null);
      setCheckedActions(new Set());
    }
  }, [isOpen, loadData]);

  const toggleAction = (key: string) => {
    setCheckedActions(prev => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  };

  // Most recent email for the "Next Action" banner
  const latestEmail = emails[0] ?? null;

  if (!isOpen) return null;

  return (
    <>
      <div onClick={onClose} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.45)', zIndex: 999 }} />

      <div style={{
        position: 'fixed', top: 0, right: 0, width: 520, height: '100vh',
        background: '#fff', zIndex: 1000, boxShadow: '-4px 0 24px rgba(0,0,0,0.15)',
        display: 'flex', flexDirection: 'column', animation: 'slideInRight 0.25s ease-out',
      }}>
        {/* Header */}
        <div style={{ padding: '16px 20px', borderBottom: '1px solid #e5e7eb', flexShrink: 0, background: '#fff' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: '#9ca3af', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 2 }}>
                Email Intelligence
              </div>
              <div style={{ fontSize: 17, fontWeight: 700, color: '#111827' }}>{leadName}</div>
            </div>
            <button onClick={onClose} style={{ background: 'none', border: 'none', fontSize: 20, cursor: 'pointer', color: '#9ca3af', padding: '0 4px', lineHeight: 1, marginTop: 2 }}>✕</button>
          </div>

          {/* Quick stats row */}
          {emails.length > 0 && (
            <div style={{ display: 'flex', gap: 12, marginTop: 10, flexWrap: 'wrap', fontSize: 12, color: '#6b7280' }}>
              <span>{emails.length} email{emails.length !== 1 ? 's' : ''}</span>
              {latestEmail && <span>Last: {fmtShortDate(latestEmail.received_at)}</span>}
              {latestEmail?.ai_urgency === 'high' && <UrgencyBadge u="high" />}
              {latestEmail?.ai_ready_to_close && <DealStageBadge ready={true} />}
            </div>
          )}
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px' }}>

          {/* ── Next Action Banner ───────────────────────────────────── */}
          {latestEmail?.ai_next_action && (
            <div style={{
              padding: '12px 14px', borderRadius: 8, background: '#fef3c7',
              border: '1px solid #fbbf24', marginBottom: 16,
            }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: '#92400e', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 4 }}>
                Next Action
              </div>
              <div style={{ fontSize: 13, color: '#78350f', fontWeight: 500 }}>{latestEmail.ai_next_action}</div>
            </div>
          )}

          {/* ── Job Details Card ─────────────────────────────────────── */}
          {latestEmail && (latestEmail.ai_job_type || latestEmail.ai_product || latestEmail.ai_window_count) && (
            <div style={{
              padding: '12px 14px', borderRadius: 8, background: '#f8fafc',
              border: '1px solid #e2e8f0', marginBottom: 16,
            }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8 }}>
                Job Details
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                <JobBadge type={latestEmail.ai_job_type} />
                {latestEmail.ai_window_count && (
                  <span style={{ fontSize: 12, color: '#374151', background: '#f3f4f6', padding: '2px 8px', borderRadius: 10 }}>
                    🪟 {latestEmail.ai_window_count}
                  </span>
                )}
                {latestEmail.ai_product && (
                  <span style={{ fontSize: 12, color: '#374151', background: '#f3f4f6', padding: '2px 8px', borderRadius: 10 }}>
                    {latestEmail.ai_product}
                  </span>
                )}
              </div>
            </div>
          )}

          {/* ── Action Items Checklist ───────────────────────────────── */}
          {parseActions(latestEmail?.ai_action_items).length > 0 && (
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8 }}>
                Action Items
              </div>
              {parseActions(latestEmail?.ai_action_items).map((item, i) => {
                const key = `${latestEmail.id}-${i}`;
                const checked = checkedActions.has(key);
                return (
                  <div
                    key={key}
                    onClick={() => toggleAction(key)}
                    style={{
                      display: 'flex', alignItems: 'flex-start', gap: 8, padding: '6px 0',
                      cursor: 'pointer', borderBottom: '1px solid #f3f4f6',
                    }}
                  >
                    <div style={{
                      width: 16, height: 16, borderRadius: 4, border: `2px solid ${checked ? '#c91414' : '#d1d5db'}`,
                      background: checked ? '#c91414' : 'white', flexShrink: 0, marginTop: 1,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                    }}>
                      {checked && <span style={{ color: 'white', fontSize: 10, fontWeight: 700 }}>✓</span>}
                    </div>
                    <span style={{ fontSize: 13, color: checked ? '#9ca3af' : '#374151', textDecoration: checked ? 'line-through' : 'none', lineHeight: 1.4 }}>
                      {item}
                    </span>
                  </div>
                );
              })}
            </div>
          )}

          {/* ── AI Context / Key Insights ────────────────────────────── */}
          <div style={{ marginBottom: 20 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8 }}>
              AI Summary
            </div>

            {loadingContext && <p style={{ color: '#9ca3af', fontSize: 13 }}>Loading...</p>}
            {contextError && (
              contextError.toLowerCase().includes('not configured') ? (
                <div style={{ padding: '10px 12px', borderRadius: 6, background: '#f8fafc', border: '1px solid #e2e8f0', fontSize: 12, color: '#6b7280' }}>
                  AI summary unavailable — add your <strong>OPENAI_API_KEY</strong> to the backend <code>.env</code> file and restart the server.
                </div>
              ) : (
                <p style={{ color: '#c91414', fontSize: 13 }}>{contextError}</p>
              )
            )}

            {context && !loadingContext && (
              <>
                <p style={{ fontSize: 13, color: '#374151', lineHeight: 1.6, margin: '0 0 12px 0' }}>{context.context}</p>

                {context.key_insights.length > 0 && (
                  <ul style={{ margin: '0 0 12px 0', paddingLeft: 18 }}>
                    {context.key_insights.map((insight, i) => (
                      <li key={i} style={{ fontSize: 13, color: '#374151', marginBottom: 4 }}>{insight}</li>
                    ))}
                  </ul>
                )}

                {context.recommended_action && (
                  <div style={{ padding: '10px 12px', borderRadius: 6, background: '#fef9c3', border: '1px solid #fde047', fontSize: 13, color: '#713f12' }}>
                    <strong>Recommendation: </strong>{context.recommended_action}
                  </div>
                )}
              </>
            )}
          </div>

          {/* ── Email Timeline ───────────────────────────────────────── */}
          <div>
            <div style={{ fontSize: 11, fontWeight: 700, color: '#64748b', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 10 }}>
              Emails ({emails.length})
            </div>

            {loadingEmails && <p style={{ color: '#9ca3af', fontSize: 13 }}>Loading...</p>}
            {emailsError && <p style={{ color: '#c91414', fontSize: 13 }}>{emailsError}</p>}
            {emails.length === 0 && !loadingEmails && !emailsError && (
              <p style={{ color: '#9ca3af', fontSize: 13 }}>No matched emails for this lead.</p>
            )}

            {emails.map((email, idx) => {
              const isExpanded = expandedEmail === email.id;
              const isFirst = idx === 0;
              return (
                <div key={email.id} style={{
                  borderRadius: 8, border: `1px solid ${isFirst ? '#fca5a5' : '#e5e7eb'}`,
                  marginBottom: 10, overflow: 'hidden',
                  background: isFirst ? '#fff7f7' : '#f9fafb',
                }}>
                  {/* Email header — always visible */}
                  <div
                    style={{ padding: '10px 14px', cursor: 'pointer' }}
                    onClick={() => setExpandedEmail(isExpanded ? null : email.id)}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 8 }}>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 13, fontWeight: 600, color: '#111827', marginBottom: 2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {email.subject || '(no subject)'}
                        </div>
                        <div style={{ fontSize: 11, color: '#9ca3af' }}>
                          {email.direction === 'inbound' ? '↙ ' : '↗ '}
                          {email.sender_name || email.sender_email} · {fmtDate(email.received_at)}
                        </div>
                      </div>
                      <div style={{ display: 'flex', gap: 4, flexShrink: 0, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
                        <SentimentBadge s={email.ai_sentiment} />
                        <UrgencyBadge u={email.ai_urgency} />
                        <DealStageBadge ready={email.ai_ready_to_close} />
                      </div>
                    </div>

                    {/* AI summary — always visible */}
                    {email.ai_summary && (
                      <p style={{ fontSize: 12, color: '#4b5563', margin: '6px 0 0 0', lineHeight: 1.5 }}>
                        {email.ai_summary}
                      </p>
                    )}

                    {/* Job details inline */}
                    {(email.ai_window_count || email.ai_product || email.ai_job_type) && (
                      <div style={{ display: 'flex', gap: 6, marginTop: 6, flexWrap: 'wrap' }}>
                        <JobBadge type={email.ai_job_type} />
                        {email.ai_window_count && (
                          <span style={{ fontSize: 11, color: '#6b7280', background: '#f3f4f6', padding: '1px 6px', borderRadius: 8 }}>
                            {email.ai_window_count} windows
                          </span>
                        )}
                        {email.ai_product && (
                          <span style={{ fontSize: 11, color: '#6b7280', background: '#f3f4f6', padding: '1px 6px', borderRadius: 8 }}>
                            {email.ai_product}
                          </span>
                        )}
                      </div>
                    )}

                    <div style={{ fontSize: 11, color: '#c91414', marginTop: 6, fontWeight: 500 }}>
                      {isExpanded ? '▲ Hide' : '▼ Show full email'}
                    </div>
                  </div>

                  {/* Expanded body */}
                  {isExpanded && (
                    <div style={{
                      padding: '0 14px 14px 14px',
                      borderTop: '1px solid #f3f4f6',
                    }}>
                      {/* Action items for this email */}
                      {parseActions(email.ai_action_items).length > 0 && (
                        <div style={{ padding: '8px 0', marginBottom: 8 }}>
                          <div style={{ fontSize: 11, fontWeight: 600, color: '#6b7280', marginBottom: 4 }}>Actions:</div>
                          {parseActions(email.ai_action_items).map((item, i) => {
                            const key = `${email.id}-${i}`;
                            const checked = checkedActions.has(key);
                            return (
                              <div key={key} onClick={() => toggleAction(key)} style={{ display: 'flex', gap: 6, alignItems: 'flex-start', padding: '3px 0', cursor: 'pointer' }}>
                                <div style={{
                                  width: 14, height: 14, borderRadius: 3, border: `2px solid ${checked ? '#c91414' : '#d1d5db'}`,
                                  background: checked ? '#c91414' : 'white', flexShrink: 0, marginTop: 1,
                                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                                }}>
                                  {checked && <span style={{ color: 'white', fontSize: 9, fontWeight: 700 }}>✓</span>}
                                </div>
                                <span style={{ fontSize: 12, color: checked ? '#9ca3af' : '#374151', textDecoration: checked ? 'line-through' : 'none' }}>{item}</span>
                              </div>
                            );
                          })}
                        </div>
                      )}

                      {/* Full email body */}
                      <div style={{
                        padding: '10px 12px', background: '#fff', borderRadius: 6,
                        border: '1px solid #e5e7eb', fontSize: 13, color: '#374151',
                        lineHeight: 1.6, whiteSpace: 'pre-wrap', maxHeight: 320, overflowY: 'auto',
                      }}>
                        {email.body_text || email.body_preview || 'No body content available.'}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes slideInRight {
          from { transform: translateX(100%); }
          to { transform: translateX(0); }
        }
      `}</style>
    </>
  );
};

export default EmailIntelDrawer;
