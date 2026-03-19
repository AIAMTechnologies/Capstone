import React, { useState, useEffect, useCallback } from 'react';
import { getUnassignedLeads, deleteLead } from '../../services/api';
import type { ExtendedLead } from '../../types';
import AssignDealerModal from './AssignDealerModal';
import { getApiErrorMessage } from '../../utils/apiErrors';

interface UnassignedLeadsListProps {
  onLeadAssigned?: () => void;
}

const priorityColors: Record<string, string> = {
  Hot: '#e74c3c',
  Warm: '#f39c12',
  Cold: '#3498db',
};

const UnassignedLeadsList: React.FC<UnassignedLeadsListProps> = ({ onLeadAssigned }) => {
  const [leads, setLeads] = useState<ExtendedLead[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [assignLeadId, setAssignLeadId] = useState<number | null>(null);
  const [visibleCount, setVisibleCount] = useState(100);

  const fetchLeads = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await getUnassignedLeads();
      setLeads(result.leads ?? result as any);
    } catch (err: any) {
      setError(getApiErrorMessage(err, 'Failed to load unassigned leads.'));
      setLeads([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchLeads();
  }, [fetchLeads]);

  useEffect(() => {
    setVisibleCount(100);
  }, [leads]);

  const handleDelete = async (leadId: number) => {
    if (!window.confirm('Are you sure you want to delete this lead? This action cannot be undone.')) {
      return;
    }
    try {
      await deleteLead(leadId);
      setLeads((prev) => prev.filter((l) => l.id !== leadId));
    } catch (err: any) {
      alert(getApiErrorMessage(err, 'Failed to delete lead.'));
    }
  };

  const handleAssigned = () => {
    setAssignLeadId(null);
    fetchLeads();
    onLeadAssigned?.();
  };

  if (loading) {
    return (
      <div style={{ padding: 40, textAlign: 'center', color: '#7f8c8d' }}>
        Loading unassigned leads...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: 20, color: '#c0392b', background: '#fdecea', borderRadius: 6, margin: 12 }}>
        {error}
      </div>
    );
  }

  return (
    <div style={{ padding: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <span style={{ color: '#7f8c8d', fontSize: 14 }}>{leads.length} lead{leads.length !== 1 ? 's' : ''} awaiting assignment</span>
        <button
          onClick={fetchLeads}
          style={{
            padding: '8px 16px',
            borderRadius: 6,
            border: 'none',
            cursor: 'pointer',
            fontSize: 13,
            fontWeight: 500,
            background: '#f0f0f0',
            color: '#333',
          }}
        >
          Refresh
        </button>
      </div>

      {leads.length === 0 ? (
        <div style={{ padding: 40, textAlign: 'center', color: '#95a5a6' }}>
          No unassigned leads found.
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 12 }}>
          {leads.slice(0, visibleCount).map((lead) => (
            <div
              key={lead.id}
              style={{
                background: 'white',
                borderRadius: 8,
                padding: 16,
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
                <div>
                  <div style={{ fontWeight: 600, fontSize: 15, color: '#2c3e50' }}>
                    {lead.first_name || ''} {lead.last_name || lead.name || 'Unnamed'}
                  </div>
                  <div style={{ fontSize: 13, color: '#7f8c8d', marginTop: 2 }}>#{lead.id}</div>
                </div>
                {lead.ai_priority && (
                  <span style={{
                    background: priorityColors[lead.ai_priority] || '#95a5a6',
                    color: 'white',
                    padding: '2px 8px',
                    borderRadius: 12,
                    fontSize: 11,
                    fontWeight: 600,
                  }}>
                    {lead.ai_priority}
                  </span>
                )}
              </div>

              <div style={{ fontSize: 13, color: '#555', lineHeight: 1.6 }}>
                {lead.email && lead.email !== '-' && <div>{lead.email}</div>}
                {lead.phone && lead.phone !== '-' && <div>{lead.phone}</div>}
                {(lead.city || lead.province) && (
                  <div>{[lead.city, lead.province].filter(Boolean).join(', ')}</div>
                )}
                {lead.product_type && <div style={{ marginTop: 4 }}>Product: {lead.product_type}</div>}
                {lead.lead_source && <div>Source: {lead.lead_source}</div>}
              </div>

              <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
                <button
                  onClick={() => setAssignLeadId(lead.id)}
                  style={{
                    padding: '8px 16px',
                    borderRadius: 6,
                    border: 'none',
                    cursor: 'pointer',
                    fontSize: 13,
                    fontWeight: 500,
                    background: '#c91414',
                    color: 'white',
                    flex: 1,
                  }}
                >
                  Assign
                </button>
                <button
                  onClick={() => handleDelete(lead.id)}
                  style={{
                    padding: '8px 16px',
                    borderRadius: 6,
                    border: '1px solid #e74c3c',
                    cursor: 'pointer',
                    fontSize: 13,
                    fontWeight: 500,
                    background: 'white',
                    color: '#e74c3c',
                  }}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {leads.length > visibleCount && (
        <div style={{ display: 'flex', justifyContent: 'center', marginTop: 16 }}>
          <button
            onClick={() => setVisibleCount((count) => count + 100)}
            style={{
              padding: '8px 16px',
              borderRadius: 6,
              border: '1px solid #ddd',
              cursor: 'pointer',
              fontSize: 13,
              fontWeight: 500,
              background: 'white',
              color: '#333',
            }}
          >
            Show 100 More
          </button>
        </div>
      )}

      {assignLeadId !== null && (
        <AssignDealerModal
          leadId={assignLeadId}
          isOpen={true}
          onClose={() => setAssignLeadId(null)}
          onAssigned={handleAssigned}
        />
      )}
    </div>
  );
};

export default UnassignedLeadsList;
