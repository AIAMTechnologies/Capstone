import React, { useCallback, useEffect, useState } from 'react';
import { getDashboardUnassignedLeads } from '../../services/api';
import type { DashboardUnassignedLead } from '../../types/types';
import { getApiErrorMessage } from '../../utils/apiErrors';

interface LassoUnassignedLeadsTableProps {
  refreshToken?: number;
  onLeadAssigned?: () => void;
  initialLeads?: DashboardUnassignedLead[];
}

const tableStyle: React.CSSProperties = {
  width: '100%',
  borderCollapse: 'collapse',
};

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

const btnStyle: React.CSSProperties = {
  padding: '6px 12px',
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

const LassoUnassignedLeadsTable: React.FC<LassoUnassignedLeadsTableProps> = ({ refreshToken = 0, initialLeads }) => {
  const [leads, setLeads] = useState<DashboardUnassignedLead[]>(initialLeads ?? []);
  const [loading, setLoading] = useState(initialLeads === undefined);
  const [error, setError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<number | null>(null);

  const fetchLeads = useCallback(async () => {
    setLoading(true);
    try {
      const response = await getDashboardUnassignedLeads();
      setLeads(response.leads);
      setError(null);
    } catch (err: any) {
      setError(getApiErrorMessage(err, 'Failed to load unassigned snapshot.'));
      setLeads([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (refreshToken === 0 && initialLeads !== undefined) return;
    fetchLeads();
  }, [fetchLeads, refreshToken]); // eslint-disable-line react-hooks/exhaustive-deps

  const copyLassoId = async (id: number) => {
    try {
      await navigator.clipboard.writeText(String(id));
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 1500);
    } catch {
      window.alert(`Lasso ID: ${id}`);
    }
  };

  const copyEmail = async (email: string) => {
    try {
      await navigator.clipboard.writeText(email);
    } catch {
      window.alert(`Email: ${email}`);
    }
  };

  if (loading) return <div style={{ padding: 20, color: '#7f8c8d' }}>Loading unassigned leads...</div>;
  if (error) return <div style={{ padding: 20, color: '#c0392b', background: '#fdecea', borderRadius: 6 }}>{error}</div>;

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 16px', borderBottom: '1px solid #f3f4f6' }}>
        <span style={{ color: '#6b7280', fontSize: 13 }}>{leads.length} unassigned lead{leads.length !== 1 ? 's' : ''}</span>
        <button onClick={fetchLeads} style={{ ...btnStyle }}>Refresh</button>
      </div>

      {leads.length === 0 ? (
        <div style={{ padding: 32, color: '#95a5a6', textAlign: 'center' }}>No unassigned leads found.</div>
      ) : (
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle}>Date</th>
              <th style={thStyle}>Name</th>
              <th style={thStyle}>Location</th>
              <th style={thStyle}>Options</th>
            </tr>
          </thead>
          <tbody>
            {leads.map((lead) => (
              <tr key={lead.lasso_lead_id} style={{ background: 'white' }}>
                <td style={{ ...tdStyle, whiteSpace: 'nowrap', color: '#6b7280' }}>
                  {formatDate(lead.record_date)}
                </td>
                <td style={tdStyle}>
                  <div style={{ fontWeight: 600 }}>{lead.name}</div>
                  <div style={{ color: '#9ca3af', fontSize: 12, marginTop: 2 }}>#{lead.lasso_lead_id}</div>
                  {lead.email && <div style={{ color: '#6b7280', fontSize: 12, marginTop: 2 }}>{lead.email}</div>}
                </td>
                <td style={{ ...tdStyle, color: '#6b7280' }}>
                  {lead.location_text || `${lead.city || ''}${lead.province ? `, ${lead.province}` : ''}` || '-'}
                </td>
                <td style={tdStyle}>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    <button
                      onClick={() => copyLassoId(lead.lasso_lead_id)}
                      style={{ ...btnStyle, background: copiedId === lead.lasso_lead_id ? '#d1fae5' : 'white', borderColor: copiedId === lead.lasso_lead_id ? '#6ee7b7' : '#d1d5db' }}
                    >
                      {copiedId === lead.lasso_lead_id ? 'Copied!' : 'Copy ID'}
                    </button>
                    {lead.email && (
                      <button onClick={() => copyEmail(lead.email as string)} style={btnStyle}>
                        Copy Email
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default LassoUnassignedLeadsTable;
