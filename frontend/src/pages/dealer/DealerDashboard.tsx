import React, { useState, useEffect, useCallback } from 'react';
import { getDealerActiveLeads, getDealerHistory, submitDealerInteraction, submitDealerWinLost } from '../../services/api';
import type { ExtendedLead } from '../../types/types';

const cardStyle: React.CSSProperties = { background: 'white', borderRadius: 8, padding: 16, boxShadow: '0 2px 8px rgba(0,0,0,0.1)', marginBottom: 12 };
const btnPrimary: React.CSSProperties = { padding: '6px 14px', background: '#c91414', color: 'white', border: 'none', borderRadius: 6, fontSize: 13, fontWeight: 600, cursor: 'pointer' };
const btnSecondary: React.CSSProperties = { padding: '6px 14px', background: 'white', color: '#555', border: '1px solid #ddd', borderRadius: 6, fontSize: 13, fontWeight: 500, cursor: 'pointer' };
const thStyle: React.CSSProperties = { background: '#f8f9fa', textAlign: 'left' as const, padding: '10px 12px', fontSize: 13, fontWeight: 600, color: '#555' };
const tdStyle: React.CSSProperties = { padding: '10px 12px', borderBottom: '1px solid #eee', fontSize: 13 };

const statusBadge = (status: string): React.CSSProperties => {
  const base: React.CSSProperties = {
    display: 'inline-block', padding: '3px 10px', borderRadius: 12,
    fontSize: 12, fontWeight: 600, textTransform: 'capitalize',
  };
  switch (status) {
    case 'converted': case 'won': return { ...base, background: '#e6f9ed', color: '#1a7a3a' };
    case 'dead': case 'lost': return { ...base, background: '#fce8e8', color: '#c91414' };
    default: return { ...base, background: '#e8f0fe', color: '#1a56db' };
  }
};

const DealerDashboard: React.FC = () => {
  const [activeLeads, setActiveLeads] = useState<ExtendedLead[]>([]);
  const [historyLeads, setHistoryLeads] = useState<ExtendedLead[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Interaction form state
  const [interactionLeadId, setInteractionLeadId] = useState<number | null>(null);
  const [interactionNote, setInteractionNote] = useState('');
  const [submittingNote, setSubmittingNote] = useState(false);

  // Won/Lost prompt state
  const [wonLeadId, setWonLeadId] = useState<number | null>(null);
  const [wonValue, setWonValue] = useState('');
  const [lostLeadId, setLostLeadId] = useState<number | null>(null);
  const [lostReason, setLostReason] = useState('');
  const [submittingWinLost, setSubmittingWinLost] = useState(false);

  const [success, setSuccess] = useState('');

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [activeRes, historyRes] = await Promise.all([
        getDealerActiveLeads(),
        getDealerHistory(),
      ]);
      setActiveLeads(activeRes.leads);
      setHistoryLeads(historyRes.leads);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleSubmitNote = async () => {
    if (!interactionLeadId || !interactionNote.trim()) return;
    setSubmittingNote(true);
    setError('');
    setSuccess('');
    try {
      await submitDealerInteraction({ lead_id: interactionLeadId, message: interactionNote });
      setSuccess('Note added successfully');
      setInteractionLeadId(null);
      setInteractionNote('');
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to submit note');
    } finally {
      setSubmittingNote(false);
    }
  };

  const handleWon = async () => {
    if (!wonLeadId) return;
    setSubmittingWinLost(true);
    setError('');
    setSuccess('');
    try {
      await submitDealerWinLost({
        lead_id: wonLeadId,
        status: 'won',
        value_of_order: parseFloat(wonValue) || undefined,
      });
      setSuccess('Lead marked as won');
      setWonLeadId(null);
      setWonValue('');
      await fetchData();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to update');
    } finally {
      setSubmittingWinLost(false);
    }
  };

  const handleLost = async () => {
    if (!lostLeadId) return;
    setSubmittingWinLost(true);
    setError('');
    setSuccess('');
    try {
      await submitDealerWinLost({
        lead_id: lostLeadId,
        status: 'lost',
        reason: lostReason || undefined,
      });
      setSuccess('Lead marked as lost');
      setLostLeadId(null);
      setLostReason('');
      await fetchData();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to update');
    } finally {
      setSubmittingWinLost(false);
    }
  };

  return (
    <div style={{ padding: 24 }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, color: '#1a1a2e', marginBottom: 20 }}>Dealer Dashboard</h1>

      {error && <div style={{ color: '#c91414', marginBottom: 12, fontSize: 14 }}>{error}</div>}
      {success && <div style={{ color: '#1a7a3a', marginBottom: 12, fontSize: 14 }}>{success}</div>}
      {loading && <p style={{ color: '#999' }}>Loading...</p>}

      {/* Active Leads */}
      <h2 style={{ fontSize: 18, fontWeight: 600, color: '#1a1a2e', marginBottom: 12 }}>Active Leads</h2>
      {activeLeads.length === 0 && !loading && <p style={{ color: '#999', fontSize: 14, marginBottom: 20 }}>No active leads</p>}

      {activeLeads.map(lead => {
        const displayName = [lead.first_name, lead.last_name].filter(Boolean).join(' ') || lead.name || '-';
        return (
          <div key={lead.id} style={cardStyle}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 8 }}>
              <div>
                <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 4 }}>{displayName}</div>
                <div style={{ fontSize: 13, color: '#666' }}>
                  {lead.city}{lead.province ? `, ${lead.province}` : ''} &middot; {lead.product_type || '-'}
                </div>
                <div style={{ marginTop: 6 }}>
                  <span style={statusBadge(lead.status)}>{lead.status}</span>
                </div>
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <button
                  onClick={() => { setInteractionLeadId(interactionLeadId === lead.id ? null : lead.id); setWonLeadId(null); setLostLeadId(null); }}
                  style={btnSecondary}
                >
                  Add Note
                </button>
                <button
                  onClick={() => { setWonLeadId(lead.id); setInteractionLeadId(null); setLostLeadId(null); setWonValue(''); }}
                  style={{ ...btnPrimary, background: '#1a7a3a' }}
                >
                  Won
                </button>
                <button
                  onClick={() => { setLostLeadId(lead.id); setInteractionLeadId(null); setWonLeadId(null); setLostReason(''); }}
                  style={btnPrimary}
                >
                  Lost
                </button>
              </div>
            </div>

            {/* Interaction form */}
            {interactionLeadId === lead.id && (
              <div style={{ marginTop: 12, padding: 12, background: '#f8f9fa', borderRadius: 6 }}>
                <textarea
                  value={interactionNote}
                  onChange={e => setInteractionNote(e.target.value)}
                  placeholder="Enter your note..."
                  style={{ width: '100%', padding: '8px 12px', borderRadius: 6, border: '1px solid #ddd', fontSize: 13, height: 80, resize: 'vertical', boxSizing: 'border-box' }}
                />
                <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                  <button onClick={handleSubmitNote} disabled={submittingNote} style={{ ...btnPrimary, fontSize: 12, opacity: submittingNote ? 0.6 : 1 }}>
                    {submittingNote ? 'Submitting...' : 'Submit'}
                  </button>
                  <button onClick={() => setInteractionLeadId(null)} style={{ ...btnSecondary, fontSize: 12 }}>Cancel</button>
                </div>
              </div>
            )}

            {/* Won prompt */}
            {wonLeadId === lead.id && (
              <div style={{ marginTop: 12, padding: 12, background: '#f0fdf4', borderRadius: 6 }}>
                <label style={{ display: 'block', fontSize: 13, fontWeight: 600, marginBottom: 4, color: '#1a7a3a' }}>Value of Order ($)</label>
                <input
                  type="number"
                  value={wonValue}
                  onChange={e => setWonValue(e.target.value)}
                  placeholder="Enter value"
                  style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid #ddd', fontSize: 13, width: 200 }}
                />
                <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                  <button onClick={handleWon} disabled={submittingWinLost} style={{ ...btnPrimary, background: '#1a7a3a', fontSize: 12, opacity: submittingWinLost ? 0.6 : 1 }}>
                    {submittingWinLost ? 'Submitting...' : 'Confirm Won'}
                  </button>
                  <button onClick={() => setWonLeadId(null)} style={{ ...btnSecondary, fontSize: 12 }}>Cancel</button>
                </div>
              </div>
            )}

            {/* Lost prompt */}
            {lostLeadId === lead.id && (
              <div style={{ marginTop: 12, padding: 12, background: '#fef2f2', borderRadius: 6 }}>
                <label style={{ display: 'block', fontSize: 13, fontWeight: 600, marginBottom: 4, color: '#c91414' }}>Reason for Loss</label>
                <textarea
                  value={lostReason}
                  onChange={e => setLostReason(e.target.value)}
                  placeholder="Enter reason..."
                  style={{ width: '100%', padding: '8px 12px', borderRadius: 6, border: '1px solid #ddd', fontSize: 13, height: 60, resize: 'vertical', boxSizing: 'border-box' }}
                />
                <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                  <button onClick={handleLost} disabled={submittingWinLost} style={{ ...btnPrimary, fontSize: 12, opacity: submittingWinLost ? 0.6 : 1 }}>
                    {submittingWinLost ? 'Submitting...' : 'Confirm Lost'}
                  </button>
                  <button onClick={() => setLostLeadId(null)} style={{ ...btnSecondary, fontSize: 12 }}>Cancel</button>
                </div>
              </div>
            )}
          </div>
        );
      })}

      {/* History */}
      <h2 style={{ fontSize: 18, fontWeight: 600, color: '#1a1a2e', marginTop: 32, marginBottom: 12 }}>History</h2>
      <div style={{ ...cardStyle, overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              <th style={thStyle}>Name</th>
              <th style={thStyle}>City</th>
              <th style={thStyle}>Product</th>
              <th style={thStyle}>Status</th>
              <th style={thStyle}>Date</th>
            </tr>
          </thead>
          <tbody>
            {historyLeads.map(lead => {
              const displayName = [lead.first_name, lead.last_name].filter(Boolean).join(' ') || lead.name || '-';
              return (
                <tr key={lead.id}>
                  <td style={tdStyle}>{displayName}</td>
                  <td style={tdStyle}>{lead.city || '-'}</td>
                  <td style={tdStyle}>{lead.product_type || '-'}</td>
                  <td style={tdStyle}><span style={statusBadge(lead.status)}>{lead.status}</span></td>
                  <td style={tdStyle}>{new Date(lead.created_at).toLocaleDateString('en-CA')}</td>
                </tr>
              );
            })}
            {historyLeads.length === 0 && !loading && (
              <tr><td colSpan={5} style={{ ...tdStyle, textAlign: 'center', color: '#999' }}>No history</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default DealerDashboard;
