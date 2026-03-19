import React, { useState, useEffect, useCallback } from 'react';
import type { EmailMessage, EmailLeadContext } from '../../types/types';
import { getLeadEmails, getLeadEmailContext } from '../../services/api';

interface EmailIntelDrawerProps {
  leadId: number;
  leadName: string;
  isOpen: boolean;
  onClose: () => void;
}

const fmtDate = (d?: string | null) => d ? new Date(d).toLocaleString('en-CA') : '-';

const EmailIntelDrawer: React.FC<EmailIntelDrawerProps> = ({ leadId, leadName, isOpen, onClose }) => {
  const [context, setContext] = useState<EmailLeadContext | null>(null);
  const [emails, setEmails] = useState<EmailMessage[]>([]);
  const [loadingContext, setLoadingContext] = useState(false);
  const [loadingEmails, setLoadingEmails] = useState(false);
  const [expandedEmail, setExpandedEmail] = useState<number | null>(null);
  const [contextError, setContextError] = useState('');
  const [emailsError, setEmailsError] = useState('');

  const loadData = useCallback(async () => {
    if (!leadId) return;

    setLoadingContext(true);
    setLoadingEmails(true);
    setContextError('');
    setEmailsError('');

    try {
      const res = await getLeadEmailContext(leadId);
      setContext(res);
    } catch (err: any) {
      setContextError(err.response?.data?.detail || 'Failed to load AI context');
    } finally {
      setLoadingContext(false);
    }

    try {
      const res = await getLeadEmails(leadId);
      setEmails(res);
    } catch (err: any) {
      setEmailsError(err.response?.data?.detail || 'Failed to load emails');
    } finally {
      setLoadingEmails(false);
    }
  }, [leadId]);

  useEffect(() => {
    if (isOpen) {
      loadData();
    } else {
      setContext(null);
      setEmails([]);
      setExpandedEmail(null);
    }
  }, [isOpen, loadData]);

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

  const timelineSentimentDot = (sentiment: string) => {
    const colorMap: Record<string, string> = {
      positive: '#22c55e',
      neutral: '#9ca3af',
      negative: '#ef4444',
    };
    return (
      <span style={{
        display: 'inline-block',
        width: 8,
        height: 8,
        borderRadius: '50%',
        background: colorMap[sentiment] || '#9ca3af',
        marginRight: 6,
      }} />
    );
  };

  if (!isOpen) return null;

  return (
    <>
      {/* Overlay */}
      <div
        onClick={onClose}
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.5)',
          zIndex: 999,
        }}
      />

      {/* Drawer */}
      <div
        style={{
          position: 'fixed',
          top: 0,
          right: 0,
          width: 500,
          height: '100vh',
          background: '#fff',
          zIndex: 1000,
          boxShadow: '-4px 0 20px rgba(0,0,0,0.15)',
          display: 'flex',
          flexDirection: 'column',
          animation: 'slideInRight 0.3s ease-out',
        }}
      >
        {/* Header */}
        <div style={{
          padding: '16px 20px',
          borderBottom: '1px solid #eee',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexShrink: 0,
        }}>
          <div>
            <h2 style={{ fontSize: 18, fontWeight: 700, color: '#1a1a2e', margin: 0 }}>Email Intelligence</h2>
            <p style={{ fontSize: 13, color: '#888', margin: '4px 0 0 0' }}>{leadName}</p>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              fontSize: 22,
              cursor: 'pointer',
              color: '#999',
              padding: '4px 8px',
              lineHeight: 1,
            }}
          >
            X
          </button>
        </div>

        {/* Scrollable Content */}
        <div style={{ flex: 1, overflowY: 'auto', padding: 20 }}>
          {/* AI Context Summary */}
          <div style={{ marginBottom: 24 }}>
            <h3 style={{ fontSize: 15, fontWeight: 600, color: '#1a1a2e', marginBottom: 12 }}>AI Context Summary</h3>

            {loadingContext && <p style={{ color: '#999', fontSize: 13 }}>Loading context...</p>}
            {contextError && <p style={{ color: '#c91414', fontSize: 13 }}>{contextError}</p>}

            {context && !loadingContext && (
              <>
                {/* Narrative */}
                <p style={{ fontSize: 13, color: '#333', lineHeight: 1.6, marginBottom: 16 }}>{context.context}</p>

                {/* Key Insights */}
                {context.key_insights.length > 0 && (
                  <div style={{ marginBottom: 16 }}>
                    <h4 style={{ fontSize: 13, fontWeight: 600, color: '#555', marginBottom: 6 }}>Key Insights</h4>
                    <ul style={{ margin: 0, paddingLeft: 20 }}>
                      {context.key_insights.map((insight, i) => (
                        <li key={i} style={{ fontSize: 13, color: '#333', marginBottom: 4 }}>{insight}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Recommended Action */}
                {context.recommended_action && (
                  <div style={{
                    padding: 12,
                    borderRadius: 6,
                    background: '#fef3c7',
                    border: '1px solid #fbbf24',
                    marginBottom: 16,
                  }}>
                    <span style={{ fontSize: 12, fontWeight: 600, color: '#92400e', display: 'block', marginBottom: 4 }}>
                      Recommended Action
                    </span>
                    <span style={{ fontSize: 13, color: '#78350f' }}>{context.recommended_action}</span>
                  </div>
                )}

                {/* Timeline */}
                {context.timeline.length > 0 && (
                  <div style={{ marginBottom: 8 }}>
                    <h4 style={{ fontSize: 13, fontWeight: 600, color: '#555', marginBottom: 8 }}>Timeline</h4>
                    <div style={{ borderLeft: '2px solid #e5e7eb', paddingLeft: 16 }}>
                      {context.timeline.map((item, i) => (
                        <div key={i} style={{ marginBottom: 10, position: 'relative' }}>
                          <div style={{
                            position: 'absolute',
                            left: -21,
                            top: 4,
                            width: 10,
                            height: 10,
                            borderRadius: '50%',
                            background: '#fff',
                            border: '2px solid #d1d5db',
                          }} />
                          <div style={{ fontSize: 11, color: '#999', marginBottom: 2 }}>{fmtDate(item.date)}</div>
                          <div style={{ fontSize: 13, color: '#333' }}>
                            {timelineSentimentDot(item.sentiment)}
                            {item.event}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}

            {!context && !loadingContext && !contextError && (
              <p style={{ color: '#999', fontSize: 13 }}>No AI context available for this lead.</p>
            )}
          </div>

          {/* Matched Emails */}
          <div>
            <h3 style={{ fontSize: 15, fontWeight: 600, color: '#1a1a2e', marginBottom: 12 }}>Matched Emails</h3>

            {loadingEmails && <p style={{ color: '#999', fontSize: 13 }}>Loading emails...</p>}
            {emailsError && <p style={{ color: '#c91414', fontSize: 13 }}>{emailsError}</p>}

            {emails.length === 0 && !loadingEmails && !emailsError && (
              <p style={{ color: '#999', fontSize: 13 }}>No matched emails for this lead.</p>
            )}

            {emails.map(email => (
              <div
                key={email.id}
                style={{
                  background: '#f9fafb',
                  borderRadius: 8,
                  padding: 14,
                  marginBottom: 10,
                  border: '1px solid #e5e7eb',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 6 }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: 14, fontWeight: 600, color: '#1a1a2e', marginBottom: 2 }}>
                      {email.subject.length > 60 ? email.subject.slice(0, 60) + '...' : email.subject}
                    </div>
                    <div style={{ fontSize: 12, color: '#888' }}>
                      {email.sender_name || email.sender_email} &middot; {fmtDate(email.received_at)}
                    </div>
                  </div>
                  <div style={{ marginLeft: 8, flexShrink: 0 }}>
                    {sentimentBadge(email.ai_sentiment)}
                  </div>
                </div>

                {email.ai_summary && (
                  <p style={{ fontSize: 13, color: '#555', margin: '8px 0', lineHeight: 1.5 }}>
                    {email.ai_summary}
                  </p>
                )}

                <button
                  onClick={() => setExpandedEmail(expandedEmail === email.id ? null : email.id)}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: '#c91414',
                    fontSize: 12,
                    fontWeight: 600,
                    cursor: 'pointer',
                    padding: 0,
                    marginTop: 4,
                  }}
                >
                  {expandedEmail === email.id ? 'Hide Full Email' : 'Show Full Email'}
                </button>

                {expandedEmail === email.id && (
                  <div style={{
                    marginTop: 10,
                    padding: 12,
                    background: '#fff',
                    borderRadius: 6,
                    border: '1px solid #e5e7eb',
                    fontSize: 13,
                    color: '#333',
                    lineHeight: 1.6,
                    whiteSpace: 'pre-wrap',
                    maxHeight: 300,
                    overflowY: 'auto',
                  }}>
                    {email.body_text || email.body_preview || 'No body content available.'}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Inline keyframe animation */}
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
