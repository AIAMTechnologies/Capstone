import React, { useCallback, useEffect, useState } from 'react';
import { getDashboardActiveLeads, archiveActiveLead, reassignActiveLead, getDealerOptions } from '../../services/api';
import type { DashboardActiveLead, Dealer } from '../../types/types';
import { getApiErrorMessage } from '../../utils/apiErrors';

interface LassoActiveLeadsTableProps {
  refreshToken?: number;
  onRefresh?: () => void;
  initialLeads?: DashboardActiveLead[];
}

const tableStyle: React.CSSProperties = { width: '100%', borderCollapse: 'collapse' };

const thStyle: React.CSSProperties = {
  textAlign: 'left',
  padding: '10px 16px',
  fontSize: 12,
  fontWeight: 700,
  color: '#4b5563',
  borderBottom: '2px solid #e5e7eb',
  textTransform: 'uppercase',
  letterSpacing: 0.4,
  background: '#f9fafb',
};

const tdStyle: React.CSSProperties = {
  padding: '12px 16px',
  borderBottom: '1px solid #eef2f7',
  fontSize: 13,
  color: '#374151',
  verticalAlign: 'top',
};

const outlineBtn: React.CSSProperties = {
  padding: '5px 12px',
  borderRadius: 5,
  border: '1px solid #d1d5db',
  cursor: 'pointer',
  fontSize: 12,
  fontWeight: 500,
  background: 'white',
  color: '#374151',
};


const formatDate = (value?: string | null) => {
  if (!value) return '-';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return '-';
  return parsed.toLocaleDateString('en-CA', { year: 'numeric', month: 'short', day: 'numeric' });
};

const STATUS_COLORS: Record<string, { bg: string; color: string }> = {
  'New':       { bg: '#d1fae5', color: '#065f46' },
  'Called':    { bg: '#dbeafe', color: '#1e40af' },
  'Active':    { bg: '#d1fae5', color: '#065f46' },
  'Converted': { bg: '#fef3c7', color: '#92400e' },
  'Dead Lead': { bg: '#fee2e2', color: '#991b1b' },
};

const StatusBadge: React.FC<{ status?: string | null }> = ({ status }) => {
  if (!status) return null;
  const style = STATUS_COLORS[status] ?? { bg: '#f3f4f6', color: '#374151' };
  return (
    <span style={{
      display: 'inline-block',
      padding: '2px 8px',
      borderRadius: 4,
      fontSize: 11,
      fontWeight: 600,
      background: style.bg,
      color: style.color,
      marginBottom: 4,
    }}>
      {status}
    </span>
  );
};

// ── Reassign modal ──────────────────────────────────────────────────────────
interface ReassignModalProps {
  lead: DashboardActiveLead;
  onClose: () => void;
  onDone: () => void;
}

const ReassignModal: React.FC<ReassignModalProps> = ({ lead, onClose, onDone }) => {
  const [dealers, setDealers] = useState<Dealer[]>([]);
  const [selectedId, setSelectedId] = useState<number | ''>('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getDealerOptions()
      .then((r) => setDealers(r.dealers))
      .catch(() => setError('Could not load dealers.'));
  }, []);

  const handleSave = async () => {
    if (!selectedId) return;
    const dealer = dealers.find((d) => d.id === selectedId);
    setBusy(true);
    try {
      await reassignActiveLead(lead.lasso_lead_id, dealer?.id, dealer?.name);
      onDone();
    } catch (err: any) {
      setError(getApiErrorMessage(err, 'Reassign failed.'));
      setBusy(false);
    }
  };

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.4)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 9999,
    }}>
      <div style={{ background: 'white', borderRadius: 10, padding: 28, width: 380, boxShadow: '0 8px 32px rgba(0,0,0,0.18)' }}>
        <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 4 }}>Re-assign Lead</div>
        <div style={{ color: '#6b7280', fontSize: 13, marginBottom: 20 }}>
          {lead.name} &nbsp;·&nbsp; #{lead.lasso_lead_id}
          {lead.dealer_name && <> &nbsp;·&nbsp; currently: <strong>{lead.dealer_name}</strong></>}
        </div>

        {error && <div style={{ color: '#c0392b', marginBottom: 12, fontSize: 13 }}>{error}</div>}

        <select
          value={selectedId}
          onChange={(e) => setSelectedId(e.target.value ? Number(e.target.value) : '')}
          style={{ width: '100%', padding: '8px 10px', borderRadius: 6, border: '1px solid #d1d5db', fontSize: 13, marginBottom: 16 }}
        >
          <option value="">Select dealer…</option>
          {dealers.map((d) => (
            <option key={d.id} value={d.id}>{d.name}{d.province ? ` (${d.province})` : ''}</option>
          ))}
        </select>

        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button onClick={onClose} style={{ ...outlineBtn, padding: '8px 16px' }}>Cancel</button>
          <button
            onClick={handleSave}
            disabled={!selectedId || busy}
            style={{ padding: '8px 16px', borderRadius: 5, border: 'none', cursor: selectedId && !busy ? 'pointer' : 'default', background: '#c91414', color: 'white', fontWeight: 600, fontSize: 13, opacity: !selectedId || busy ? 0.6 : 1 }}
          >
            {busy ? 'Saving…' : 'Confirm'}
          </button>
        </div>
      </div>
    </div>
  );
};

// ── Main table ──────────────────────────────────────────────────────────────
const LassoActiveLeadsTable: React.FC<LassoActiveLeadsTableProps> = ({ refreshToken = 0, onRefresh, initialLeads }) => {
  const [leads, setLeads] = useState<DashboardActiveLead[]>(initialLeads ?? []);
  const [loading, setLoading] = useState(initialLeads === undefined);
  const [error, setError] = useState<string | null>(null);
  const [reassignLead, setReassignLead] = useState<DashboardActiveLead | null>(null);
  const [archivingId, setArchivingId] = useState<number | null>(null);

  const fetchLeads = useCallback(async () => {
    setLoading(true);
    try {
      const response = await getDashboardActiveLeads();
      setLeads(response.leads);
      setError(null);
    } catch (err: any) {
      setError(getApiErrorMessage(err, 'Failed to load active lead snapshot.'));
      setLeads([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (refreshToken === 0 && initialLeads !== undefined) return;
    fetchLeads();
  }, [fetchLeads, refreshToken]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleArchive = async (lead: DashboardActiveLead) => {
    if (!window.confirm(`Remove "${lead.name}" from the active dashboard? The lead still exists in Lasso.`)) return;
    setArchivingId(lead.lasso_lead_id);
    try {
      await archiveActiveLead(lead.lasso_lead_id);
      setLeads((prev) => prev.filter((l) => l.lasso_lead_id !== lead.lasso_lead_id));
      onRefresh?.();
    } catch (err: any) {
      window.alert(getApiErrorMessage(err, 'Archive failed.'));
    } finally {
      setArchivingId(null);
    }
  };

  const handleReassignDone = () => {
    setReassignLead(null);
    fetchLeads();
    onRefresh?.();
  };

  if (loading) return <div style={{ padding: 20, color: '#7f8c8d' }}>Loading active leads...</div>;
  if (error) return <div style={{ padding: 20, color: '#c0392b', background: '#fdecea', borderRadius: 6 }}>{error}</div>;

  return (
    <>
      {reassignLead && (
        <ReassignModal
          lead={reassignLead}
          onClose={() => setReassignLead(null)}
          onDone={handleReassignDone}
        />
      )}

      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 16px', borderBottom: '1px solid #f3f4f6' }}>
          <span style={{ color: '#6b7280', fontSize: 13 }}>{leads.length} active lead{leads.length !== 1 ? 's' : ''} in the WFC working queue</span>
          <button onClick={fetchLeads} style={outlineBtn}>Refresh</button>
        </div>

        {leads.length === 0 ? (
          <div style={{ padding: 32, color: '#95a5a6', textAlign: 'center' }}>No active leads found.</div>
        ) : (
          <table style={tableStyle}>
            <thead>
              <tr>
                <th style={thStyle}>Date Assigned</th>
                <th style={thStyle}>Lead</th>
                <th style={thStyle}>Details</th>
                <th style={thStyle}>Last Interaction</th>
                <th style={thStyle}>Options</th>
              </tr>
            </thead>
            <tbody>
              {leads.map((lead) => {
                const location = lead.location_text
                  || `${lead.city || ''}${lead.province ? `, ${lead.province}` : ''}`.trim()
                  || '';
                const dealerLine = [lead.dealer_name, location ? `(${location})` : ''].filter(Boolean).join(' ');

                return (
                  <tr key={lead.lasso_lead_id}>
                    {/* Date Assigned */}
                    <td style={{ ...tdStyle, whiteSpace: 'nowrap', color: '#6b7280' }}>
                      {formatDate(lead.date_assigned)}
                    </td>

                    {/* Lead */}
                    <td style={tdStyle}>
                      <div style={{ fontWeight: 600 }}>{lead.name}</div>
                      <div style={{ color: '#9ca3af', fontSize: 12, marginTop: 2 }}>#{lead.lasso_lead_id}</div>
                    </td>

                    {/* Details */}
                    <td style={{ ...tdStyle, maxWidth: 320 }}>
                      {dealerLine && <div style={{ color: '#374151', marginBottom: 4 }}>{dealerLine}</div>}
                      <StatusBadge status={lead.current_status} />
                      {lead.lead_details && (
                        <div style={{ color: '#6b7280', fontSize: 12, marginTop: 4, lineHeight: 1.5 }}>
                          {lead.lead_details}
                        </div>
                      )}
                    </td>

                    {/* Last Interaction */}
                    <td style={{ ...tdStyle, whiteSpace: 'nowrap', color: '#6b7280' }}>
                      {formatDate(lead.last_interaction)}
                    </td>

                    {/* Options */}
                    <td style={tdStyle}>
                      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                        <button onClick={() => setReassignLead(lead)} style={outlineBtn}>
                          Re-assign
                        </button>
                        <button
                          onClick={() => handleArchive(lead)}
                          disabled={archivingId === lead.lasso_lead_id}
                          style={{ ...outlineBtn, opacity: archivingId === lead.lasso_lead_id ? 0.5 : 1 }}
                        >
                          {archivingId === lead.lasso_lead_id ? 'Archiving…' : 'Archive'}
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
};

export default LassoActiveLeadsTable;
