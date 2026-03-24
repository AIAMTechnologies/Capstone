import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import type {
  ActiveMatchReviewItem,
  ClosureReview,
  EmailSyncStatus,
  NewLeadCandidate,
} from '../../types/types';
import {
  createLeadFromEmailCandidate,
  getActiveMatchReview,
  getClosureReviewQueue,
  getEmailSyncStatus,
  getNewLeadCandidates,
} from '../../services/api';
import { getApiErrorMessage } from '../../utils/apiErrors';
import AssignDealerModal from '../../components/admin/AssignDealerModal';
import EmailIntelDrawer from '../../components/admin/EmailIntelDrawer';

const pageStyle: React.CSSProperties = { padding: 24, background: '#f7f5ef', minHeight: '100%' };
const cardStyle: React.CSSProperties = { background: 'white', borderRadius: 10, padding: 20, boxShadow: '0 2px 10px rgba(15, 23, 42, 0.08)' };
const statCardStyle: React.CSSProperties = { ...cardStyle, padding: 18 };
const sectionHeaderStyle: React.CSSProperties = { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, marginBottom: 16, flexWrap: 'wrap' };
const sectionTitleStyle: React.CSSProperties = { fontSize: 18, fontWeight: 700, color: '#1f2937', margin: 0 };
const sectionTextStyle: React.CSSProperties = { margin: '6px 0 0 0', fontSize: 13, color: '#6b7280', lineHeight: 1.55 };
const btnPrimary: React.CSSProperties = { padding: '8px 14px', background: '#b45309', color: 'white', border: 'none', borderRadius: 6, fontSize: 13, fontWeight: 700, cursor: 'pointer' };
const btnSecondary: React.CSSProperties = { ...btnPrimary, background: '#f8fafc', color: '#1f2937', border: '1px solid #d1d5db' };
const btnGhost: React.CSSProperties = { ...btnPrimary, background: '#fff7ed', color: '#9a3412', border: '1px solid #fdba74' };
const thStyle: React.CSSProperties = { background: '#fafaf9', textAlign: 'left', padding: '10px 12px', fontSize: 12, fontWeight: 700, color: '#57534e', letterSpacing: '0.02em' };
const tdStyle: React.CSSProperties = { padding: '12px', borderBottom: '1px solid #f1f5f9', fontSize: 13, color: '#334155', verticalAlign: 'top' };

type OpsBucket = {
  key: string;
  label: string;
  value: number;
  tone: 'warm' | 'critical' | 'cool' | 'neutral' | 'success';
  note: string;
};

type LocalQueueState = Record<string, 'dismissed' | 'snoozed'>;

const fmtDate = (value?: string | null) => value ? new Date(value).toLocaleString('en-CA') : '-';
const truncate = (value?: string | null, max = 120) => {
  if (!value) return '-';
  return value.length > max ? `${value.slice(0, max)}...` : value;
};

const toneStyles: Record<OpsBucket['tone'], { bg: string; color: string; border: string }> = {
  warm: { bg: '#fff7ed', color: '#9a3412', border: '#fdba74' },
  critical: { bg: '#fef2f2', color: '#b91c1c', border: '#fca5a5' },
  cool: { bg: '#eff6ff', color: '#1d4ed8', border: '#93c5fd' },
  neutral: { bg: '#f8fafc', color: '#334155', border: '#cbd5e1' },
  success: { bg: '#ecfdf5', color: '#166534', border: '#86efac' },
};

const priorityBadge = (priority: ActiveMatchReviewItem['review_priority']) => {
  const styles = {
    high: { bg: '#fee2e2', color: '#b91c1c' },
    medium: { bg: '#fef3c7', color: '#92400e' },
    low: { bg: '#e0f2fe', color: '#0f4c81' },
  };
  const style = styles[priority];
  return (
    <span style={{ padding: '2px 8px', borderRadius: 999, fontSize: 11, fontWeight: 700, background: style.bg, color: style.color, textTransform: 'uppercase' }}>
      {priority}
    </span>
  );
};

const statusPill = (label: string, tone: OpsBucket['tone']) => {
  const style = toneStyles[tone];
  return (
    <span style={{ padding: '3px 10px', borderRadius: 999, fontSize: 11, fontWeight: 700, background: style.bg, color: style.color, border: `1px solid ${style.border}` }}>
      {label}
    </span>
  );
};

const emptyCard = (message: string, hint: string) => (
  <div style={{ ...cardStyle, padding: 28, textAlign: 'center', color: '#64748b' }}>
    <div style={{ fontWeight: 700, color: '#334155', marginBottom: 8 }}>{message}</div>
    <div style={{ fontSize: 13 }}>{hint}</div>
  </div>
);

const EmailOps: React.FC = () => {
  const [status, setStatus] = useState<EmailSyncStatus | null>(null);
  const [activeMatches, setActiveMatches] = useState<ActiveMatchReviewItem[]>([]);
  const [newLeadCandidates, setNewLeadCandidates] = useState<NewLeadCandidate[]>([]);
  const [closureQueue, setClosureQueue] = useState<ClosureReview[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [creatingCandidateId, setCreatingCandidateId] = useState<number | null>(null);
  const [assignmentLead, setAssignmentLead] = useState<ActiveMatchReviewItem | null>(null);
  const [drawerLead, setDrawerLead] = useState<{ leadId: number; leadName: string } | null>(null);
  const [localQueueState, setLocalQueueState] = useState<LocalQueueState>({});

  const load = useCallback(async (mode: 'initial' | 'refresh' = 'refresh') => {
    if (mode === 'initial') {
      setLoading(true);
    } else {
      setRefreshing(true);
    }
    setError('');
    try {
      const [statusRes, matchesRes, candidatesRes, closureRes] = await Promise.all([
        getEmailSyncStatus(),
        getActiveMatchReview(),
        getNewLeadCandidates(),
        getClosureReviewQueue(),
      ]);
      setStatus(statusRes);
      setActiveMatches(matchesRes);
      setNewLeadCandidates(candidatesRes);
      setClosureQueue(closureRes);
    } catch (err: any) {
      setError(getApiErrorMessage(err, 'Failed to load Email Ops'));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    load('initial');
  }, [load]);

  const markLocalQueueState = (key: string, value: 'dismissed' | 'snoozed') => {
    setLocalQueueState((prev) => ({ ...prev, [key]: value }));
  };

  const isVisible = (key: string) => !localQueueState[key];

  const assignmentNeeded = activeMatches.filter((item) => item.missing_dealer && isVisible(`assign:${item.lead_id}`));
  const matchReview = activeMatches.filter((item) => (item.needs_match_review || item.active_duplicate_email_count > 1 || item.weak_match_count > 0) && isVisible(`review:${item.lead_id}`));
  const dealerFollowUp = activeMatches.filter((item) => !item.missing_dealer && !item.needs_match_review && item.days_since_last_email !== null && item.days_since_last_email >= 3 && isVisible(`followup:${item.lead_id}`));
  const intakeQueue = newLeadCandidates.filter((item) => isVisible(`candidate:${item.id}`));

  const opsBuckets: OpsBucket[] = [
    {
      key: 'new-intake',
      label: 'New Lead Intake',
      value: intakeQueue.length,
      tone: 'warm',
      note: 'High-signal unmatched inbound messages that look like real leads.',
    },
    {
      key: 'assignment-needed',
      label: 'Assignment Needed',
      value: assignmentNeeded.length,
      tone: 'critical',
      note: 'Matched lead activity exists, but no dealer is assigned yet.',
    },
    {
      key: 'dealer-follow-up',
      label: 'Dealer Follow-Up',
      value: dealerFollowUp.length,
      tone: 'cool',
      note: 'Prototype queue derived from assigned dealer threads with stale recent activity.',
    },
    {
      key: 'match-review',
      label: 'Match Review',
      value: matchReview.length,
      tone: 'neutral',
      note: 'Weak matches, duplicate-email risk, and low-trust threads.',
    },
    {
      key: 'closure-review',
      label: 'Closure Review',
      value: closureQueue.length,
      tone: 'success',
      note: 'Still human-review only. This page does not auto-close anything.',
    },
  ];

  const lastSyncLabel = useMemo(() => {
    if (!status) return 'Loading sync status...';
    if (!status.sync_enabled) return 'Sync disabled';
    if (status.last_sync) return `Last sync ${fmtDate(status.last_sync)}`;
    return 'No sync has run yet';
  }, [status]);

  const handleCreateLead = async (candidate: NewLeadCandidate) => {
    setCreatingCandidateId(candidate.id);
    setError('');
    setSuccess('');
    try {
      const result = await createLeadFromEmailCandidate(candidate.id);
      setSuccess(`Created lead #${result.lead_id} from ${candidate.sender_email}`);
      await load();
    } catch (err: any) {
      setError(getApiErrorMessage(err, 'Failed to create lead from email candidate'));
    } finally {
      setCreatingCandidateId(null);
    }
  };

  if (loading) {
    return <div style={{ ...pageStyle, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6b7280' }}>Loading Email Ops...</div>;
  }

  return (
    <div style={pageStyle}>
      <div style={{ ...cardStyle, marginBottom: 20, background: 'linear-gradient(135deg, #111827 0%, #7c2d12 100%)', color: 'white' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 20, flexWrap: 'wrap' }}>
          <div style={{ maxWidth: 760 }}>
            <h1 style={{ fontSize: 28, fontWeight: 800, margin: 0 }}>Email Ops</h1>
            <p style={{ margin: '10px 0 0 0', fontSize: 14, lineHeight: 1.7, color: 'rgba(255,255,255,0.86)' }}>
              Ops-first view of the shared mailbox. This page treats email as an operational signal source and prioritizes
              new lead intake, dealer assignment, dealer follow-up, and match review over generic sentiment reporting.
            </p>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginTop: 16 }}>
              <Link to="/admin/email-intel" style={{ ...btnSecondary, textDecoration: 'none', background: 'rgba(255,255,255,0.08)', color: 'white', borderColor: 'rgba(255,255,255,0.25)' }}>
                Compare With Email Intel
              </Link>
              <button onClick={() => load()} disabled={refreshing} style={{ ...btnGhost, background: '#fef3c7', color: '#78350f', borderColor: '#fcd34d', opacity: refreshing ? 0.7 : 1 }}>
                {refreshing ? 'Refreshing...' : 'Refresh Ops View'}
              </button>
            </div>
          </div>
          <div style={{ minWidth: 250 }}>
            <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.04em', textTransform: 'uppercase', color: 'rgba(255,255,255,0.7)', marginBottom: 8 }}>
              Current Baseline
            </div>
            <div style={{ fontSize: 14, lineHeight: 1.8, color: 'rgba(255,255,255,0.92)' }}>
              <div>{status?.matched_emails ?? 0} matched of {status?.total_emails ?? 0} emails</div>
              <div>{activeMatches.length} matched active leads</div>
              <div>{assignmentNeeded.length} open leads still missing a dealer</div>
              <div>{intakeQueue.length} high-signal intake candidates</div>
              <div>{lastSyncLabel}</div>
            </div>
          </div>
        </div>
      </div>

      {(error || success) && (
        <div style={{ ...cardStyle, marginBottom: 20, border: error ? '1px solid #fecaca' : '1px solid #bbf7d0', background: error ? '#fef2f2' : '#f0fdf4', color: error ? '#b91c1c' : '#166534' }}>
          {error || success}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 14, marginBottom: 20 }}>
        {opsBuckets.map((bucket) => {
          const tone = toneStyles[bucket.tone];
          return (
            <div key={bucket.key} style={{ ...statCardStyle, border: `1px solid ${tone.border}`, background: tone.bg }}>
              <div style={{ fontSize: 12, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em', color: tone.color, marginBottom: 8 }}>
                {bucket.label}
              </div>
              <div style={{ fontSize: 30, fontWeight: 800, color: '#111827', marginBottom: 8 }}>{bucket.value}</div>
              <div style={{ fontSize: 12, color: '#6b7280', lineHeight: 1.5 }}>{bucket.note}</div>
            </div>
          );
        })}
      </div>

      <div style={{ ...cardStyle, marginBottom: 20, border: '1px dashed #d6d3d1', background: '#fffbeb' }}>
        <div style={{ fontSize: 13, fontWeight: 700, color: '#92400e', marginBottom: 8 }}>Prototype Notes</div>
        <div style={{ fontSize: 13, color: '#57534e', lineHeight: 1.7 }}>
          Dealer follow-up, dismiss, and snooze are prototype behaviors on this page, derived from the current Email Intel APIs.
          They let us compare the workflow shape before we add persisted `lead_email_signals` and `email_ops_queue` tables.
        </div>
      </div>

      <OpsSection
        title="New Lead Intake"
        description="High-signal unmatched inbound messages that look like form submissions or forwarded customer leads."
        count={intakeQueue.length}
        rightSlot={statusPill(status?.sync_enabled ? 'Sync Connected' : 'Sync Not Enabled', status?.sync_enabled ? 'success' : 'neutral')}
      >
        {intakeQueue.length === 0 ? emptyCard('No intake candidates right now', 'When unmatched high-signal inbound mail arrives, it will appear here.') : (
          <div style={{ display: 'grid', gap: 14 }}>
            {intakeQueue.map((candidate) => (
              <div key={candidate.id} style={{ ...cardStyle, padding: 16, border: '1px solid #fed7aa', boxShadow: 'none' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap', marginBottom: 10 }}>
                  <div>
                    <div style={{ fontSize: 15, fontWeight: 700, color: '#1f2937' }}>{candidate.sender_name || candidate.sender_email}</div>
                    <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>{candidate.sender_email} • {fmtDate(candidate.received_at)}</div>
                  </div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {statusPill(candidate.candidate_reason, candidate.candidate_score >= 10 ? 'critical' : 'warm')}
                    {statusPill(`Score ${candidate.candidate_score}`, 'neutral')}
                    {candidate.existing_sender_lead_count > 0 && statusPill(`${candidate.existing_sender_lead_count} existing sender match`, 'cool')}
                  </div>
                </div>
                <div style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 6 }}>{truncate(candidate.subject, 100)}</div>
                <div style={{ fontSize: 13, color: '#475569', lineHeight: 1.6, marginBottom: 14 }}>{truncate(candidate.body_preview, 220)}</div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <button
                    onClick={() => handleCreateLead(candidate)}
                    disabled={creatingCandidateId === candidate.id}
                    style={{ ...btnPrimary, opacity: creatingCandidateId === candidate.id ? 0.7 : 1 }}
                  >
                    {creatingCandidateId === candidate.id ? 'Creating...' : 'Create Lead'}
                  </button>
                  <Link to="/admin/email-intel" style={{ ...btnSecondary, textDecoration: 'none' }}>
                    Open Email Intel
                  </Link>
                  <button onClick={() => markLocalQueueState(`candidate:${candidate.id}`, 'snoozed')} style={btnSecondary}>Snooze</button>
                  <button onClick={() => markLocalQueueState(`candidate:${candidate.id}`, 'dismissed')} style={btnSecondary}>Dismiss</button>
                </div>
              </div>
            ))}
          </div>
        )}
      </OpsSection>

      <OpsSection
        title="Assignment Needed"
        description="Open matched leads that have real email activity but still no dealer assigned."
        count={assignmentNeeded.length}
        rightSlot={statusPill('Human Review First', 'critical')}
      >
        {assignmentNeeded.length === 0 ? emptyCard('No assignment gaps detected', 'When matched lead activity exists without a dealer assignment, it will appear here.') : (
          <OpsTable
            rows={assignmentNeeded.map((item) => (
              <tr key={item.lead_id}>
                <td style={{ ...tdStyle, minWidth: 240 }}>
                  <div style={{ fontWeight: 700, color: '#111827', marginBottom: 4 }}>{item.lead_name || `Lead #${item.lead_id}`}</div>
                  <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>{item.lead_email || 'No email'} {item.lead_phone ? `• ${item.lead_phone}` : ''}</div>
                  <div style={{ fontSize: 12, color: '#6b7280' }}>Lead #{item.lead_id} • Status {item.lead_status}</div>
                </td>
                <td style={{ ...tdStyle, minWidth: 260 }}>
                  <div style={{ fontWeight: 600, color: '#b91c1c', marginBottom: 6 }}>No dealer assigned</div>
                  {item.landing_page && <div style={{ fontSize: 12, color: '#0f766e', marginBottom: 4 }}>Dealer-site signal: {item.landing_page}</div>}
                  <div style={{ fontSize: 12, color: '#6b7280' }}>{item.review_reason}</div>
                </td>
                <td style={{ ...tdStyle, minWidth: 240 }}>
                  <div style={{ fontWeight: 600, color: '#1f2937', marginBottom: 4 }}>{truncate(item.latest_subject, 72)}</div>
                  <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>{item.latest_sender_name || item.latest_sender_email}</div>
                  <div style={{ fontSize: 12, color: '#6b7280' }}>Last email {fmtDate(item.latest_email_at || item.last_email_activity)} • {item.days_since_last_email ?? '-'}d ago</div>
                </td>
                <td style={{ ...tdStyle, minWidth: 220 }}>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
                    {priorityBadge(item.review_priority)}
                    {statusPill(`${item.matched_email_count} matched`, 'warm')}
                    {item.max_match_confidence !== null && statusPill(`${(item.max_match_confidence * 100).toFixed(0)}% confidence`, 'cool')}
                  </div>
                  <div style={{ fontSize: 12, color: '#6b7280' }}>{item.match_methods || 'No match methods recorded'}</div>
                </td>
                <td style={{ ...tdStyle, minWidth: 260 }}>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    <button onClick={() => setAssignmentLead(item)} style={btnPrimary}>Assign Dealer</button>
                    <button onClick={() => setDrawerLead({ leadId: item.lead_id, leadName: item.lead_name || `Lead #${item.lead_id}` })} style={btnSecondary}>Review Emails</button>
                    <Link to={`/admin/dashboard?lead=${item.lead_id}`} style={{ ...btnSecondary, textDecoration: 'none' }}>Open Lead</Link>
                    <button onClick={() => markLocalQueueState(`assign:${item.lead_id}`, 'snoozed')} style={btnSecondary}>Snooze</button>
                    <button onClick={() => markLocalQueueState(`assign:${item.lead_id}`, 'dismissed')} style={btnSecondary}>Dismiss</button>
                  </div>
                </td>
              </tr>
            ))}
          />
        )}
      </OpsSection>

      <OpsSection
        title="Dealer Follow-Up"
        description="Prototype queue for assigned dealer threads that have gone quiet for 3+ days after recent email activity."
        count={dealerFollowUp.length}
        rightSlot={statusPill('Derived From Current Data', 'cool')}
      >
        {dealerFollowUp.length === 0 ? emptyCard('No follow-up candidates right now', 'This queue will light up when assigned dealer threads go stale after recent activity.') : (
          <OpsTable
            rows={dealerFollowUp.map((item) => (
              <tr key={item.lead_id}>
                <td style={{ ...tdStyle, minWidth: 220 }}>
                  <div style={{ fontWeight: 700, color: '#111827', marginBottom: 4 }}>{item.lead_name || `Lead #${item.lead_id}`}</div>
                  <div style={{ fontSize: 12, color: '#6b7280' }}>{item.assigned_dealer_name || 'Assigned dealer missing'} • Lead #{item.lead_id}</div>
                </td>
                <td style={{ ...tdStyle, minWidth: 260 }}>
                  <div style={{ fontWeight: 600, color: '#0f4c81', marginBottom: 4 }}>{truncate(item.latest_subject, 72)}</div>
                  <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>{truncate(item.latest_ai_summary, 140)}</div>
                  <div style={{ fontSize: 12, color: '#6b7280' }}>Last email {item.days_since_last_email ?? '-'}d ago • Match methods {item.match_methods || '-'}</div>
                </td>
                <td style={{ ...tdStyle, minWidth: 220 }}>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
                    {statusPill('dealer_follow_up', 'cool')}
                    {priorityBadge(item.review_priority)}
                  </div>
                  <div style={{ fontSize: 12, color: '#6b7280' }}>{item.review_reason}</div>
                </td>
                <td style={{ ...tdStyle, minWidth: 240 }}>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    <button onClick={() => setDrawerLead({ leadId: item.lead_id, leadName: item.lead_name || `Lead #${item.lead_id}` })} style={btnSecondary}>Review Emails</button>
                    <Link to={`/admin/dashboard?lead=${item.lead_id}`} style={{ ...btnSecondary, textDecoration: 'none' }}>Open Lead</Link>
                    <button onClick={() => markLocalQueueState(`followup:${item.lead_id}`, 'snoozed')} style={btnSecondary}>Snooze</button>
                    <button onClick={() => markLocalQueueState(`followup:${item.lead_id}`, 'dismissed')} style={btnSecondary}>Dismiss</button>
                  </div>
                </td>
              </tr>
            ))}
          />
        )}
      </OpsSection>

      <OpsSection
        title="Match Review / Duplicates"
        description="Low-trust threads, weak fallback matches, and duplicate-email collisions that should not drive automation."
        count={matchReview.length}
        rightSlot={statusPill('Quarantine Weak Signals', 'neutral')}
      >
        {matchReview.length === 0 ? emptyCard('No risky matches detected', 'Weak matches and duplicate email conflicts will appear here for review.') : (
          <OpsTable
            rows={matchReview.map((item) => (
              <tr key={item.lead_id}>
                <td style={{ ...tdStyle, minWidth: 220 }}>
                  <div style={{ fontWeight: 700, color: '#111827', marginBottom: 4 }}>{item.lead_name || `Lead #${item.lead_id}`}</div>
                  <div style={{ fontSize: 12, color: '#6b7280' }}>{item.assigned_dealer_name || 'No assigned dealer'} • Lead #{item.lead_id}</div>
                </td>
                <td style={{ ...tdStyle, minWidth: 260 }}>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
                    {priorityBadge(item.review_priority)}
                    {item.weak_match_count > 0 && statusPill(`${item.weak_match_count} weak match`, 'warm')}
                    {item.active_duplicate_email_count > 1 && statusPill('duplicate_review', 'critical')}
                    {item.max_match_confidence !== null && statusPill(`${(item.max_match_confidence * 100).toFixed(0)}% confidence`, 'cool')}
                  </div>
                  <div style={{ fontSize: 12, color: '#6b7280' }}>{item.review_reason}</div>
                </td>
                <td style={{ ...tdStyle, minWidth: 240 }}>
                  <div style={{ fontWeight: 600, color: '#1f2937', marginBottom: 4 }}>{truncate(item.latest_subject, 72)}</div>
                  <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 4 }}>{item.latest_sender_name || item.latest_sender_email}</div>
                  <div style={{ fontSize: 12, color: '#6b7280' }}>{truncate(item.latest_ai_summary, 120)}</div>
                </td>
                <td style={{ ...tdStyle, minWidth: 240 }}>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    <button onClick={() => setDrawerLead({ leadId: item.lead_id, leadName: item.lead_name || `Lead #${item.lead_id}` })} style={btnSecondary}>Review Emails</button>
                    <Link to={`/admin/dashboard?lead=${item.lead_id}`} style={{ ...btnSecondary, textDecoration: 'none' }}>Open Lead</Link>
                    <button onClick={() => markLocalQueueState(`review:${item.lead_id}`, 'snoozed')} style={btnSecondary}>Snooze</button>
                    <button onClick={() => markLocalQueueState(`review:${item.lead_id}`, 'dismissed')} style={btnSecondary}>Dismiss</button>
                  </div>
                </td>
              </tr>
            ))}
          />
        )}
      </OpsSection>

      <OpsSection
        title="Closure Stays Review-Only"
        description="This remains separate from the ops workflow until the queues above are producing consistently useful actions."
        count={closureQueue.length}
        rightSlot={statusPill('Human Review Only', 'success')}
      >
        <div style={{ ...cardStyle, padding: 16, boxShadow: 'none', border: '1px solid #dcfce7', background: '#f0fdf4' }}>
          <div style={{ fontSize: 14, color: '#166534', fontWeight: 700, marginBottom: 6 }}>
            {closureQueue.length === 0 ? 'No pending closure reviews' : `${closureQueue.length} closure review item(s) pending`}
          </div>
          <div style={{ fontSize: 13, color: '#166534', lineHeight: 1.6 }}>
            The ops prototype intentionally leaves closure separate. Use Email Intel for the existing closure review flow until we have stronger queue signals and better match confidence.
          </div>
          <div style={{ marginTop: 12 }}>
            <Link to="/admin/email-intel" style={{ ...btnSecondary, textDecoration: 'none' }}>Open Email Intel</Link>
          </div>
        </div>
      </OpsSection>

      <AssignDealerModal
        leadId={assignmentLead?.lead_id ?? 0}
        isOpen={!!assignmentLead}
        onClose={() => setAssignmentLead(null)}
        onAssigned={() => {
          setSuccess(`Assigned dealer for lead #${assignmentLead?.lead_id}`);
          setAssignmentLead(null);
          load();
        }}
      />

      <EmailIntelDrawer
        leadId={drawerLead?.leadId ?? 0}
        leadName={drawerLead?.leadName ?? ''}
        isOpen={!!drawerLead}
        onClose={() => setDrawerLead(null)}
      />
    </div>
  );
};

const OpsSection: React.FC<{
  title: string;
  description: string;
  count: number;
  rightSlot?: React.ReactNode;
  children: React.ReactNode;
}> = ({ title, description, count, rightSlot, children }) => (
  <section style={{ marginBottom: 24 }}>
    <div style={sectionHeaderStyle}>
      <div>
        <h2 style={sectionTitleStyle}>{title}</h2>
        <p style={sectionTextStyle}>{description}</p>
      </div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
        {statusPill(`${count} items`, count === 0 ? 'neutral' : 'warm')}
        {rightSlot}
      </div>
    </div>
    {children}
  </section>
);

const OpsTable: React.FC<{ rows: React.ReactNode }> = ({ rows }) => (
  <div style={{ ...cardStyle, padding: 0, overflowX: 'auto' }}>
    <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 980 }}>
      <thead>
        <tr>
          <th style={thStyle}>Lead / Candidate</th>
          <th style={thStyle}>Ops Signal</th>
          <th style={thStyle}>Latest Evidence</th>
          <th style={thStyle}>Actions</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
);

export default EmailOps;
