import React, { useState, useEffect, useCallback } from 'react';
import { getActiveLeads, archiveLead, getLeadDetail } from '../../services/api';
import type { ExtendedLead } from '../../types';
import LeadDetailModal from './LeadDetailModal';
import LeadEditModal from './LeadEditModal';
import LeadLogModal from './LeadLogModal';
import AssignDealerModal from './AssignDealerModal';
import { getApiErrorMessage } from '../../utils/apiErrors';

const priorityColors: Record<string, string> = {
  Hot: '#e74c3c',
  Warm: '#f39c12',
  Cold: '#3498db',
};

interface ActiveLeadsListProps {
  onRefresh?: () => void;
}

const ActiveLeadsList: React.FC<ActiveLeadsListProps> = ({ onRefresh }) => {
  const [leads, setLeads] = useState<ExtendedLead[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewLead, setViewLead] = useState<ExtendedLead | null>(null);
  const [editLead, setEditLead] = useState<ExtendedLead | null>(null);
  const [logLeadId, setLogLeadId] = useState<number | null>(null);
  const [reassignLeadId, setReassignLeadId] = useState<number | null>(null);
  const [detailLoadingId, setDetailLoadingId] = useState<number | null>(null);
  const [visibleCount, setVisibleCount] = useState(100);

  const fetchLeads = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await getActiveLeads();
      setLeads(result.leads ?? result as any);
    } catch (err: any) {
      setError(getApiErrorMessage(err, 'Failed to load active leads.'));
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

  const handleArchive = async (leadId: number) => {
    if (!window.confirm('Archive this lead?')) return;
    try {
      await archiveLead(leadId);
      setLeads((prev) => prev.filter((l) => l.id !== leadId));
    } catch (err: any) {
      alert(getApiErrorMessage(err, 'Failed to archive lead.'));
    }
  };

  const loadLeadDetail = async (leadId: number, mode: 'view' | 'edit') => {
    setDetailLoadingId(leadId);
    setError(null);
    try {
      const lead = await getLeadDetail(leadId);
      if (mode === 'view') {
        setViewLead(lead as ExtendedLead);
      } else {
        setEditLead(lead as ExtendedLead);
      }
    } catch (err: any) {
      setError(getApiErrorMessage(err, 'Failed to load lead details.'));
    } finally {
      setDetailLoadingId(null);
    }
  };

  const handleSaved = () => {
    setEditLead(null);
    fetchLeads();
    onRefresh?.();
  };

  const handleReassigned = () => {
    setReassignLeadId(null);
    fetchLeads();
    onRefresh?.();
  };

  const btnBase: React.CSSProperties = {
    padding: '6px 12px',
    borderRadius: 6,
    border: 'none',
    cursor: 'pointer',
    fontSize: 12,
    fontWeight: 500,
  };

  if (loading) {
    return (
      <div style={{ padding: 40, textAlign: 'center', color: '#7f8c8d' }}>
        Loading active leads...
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
        <span style={{ color: '#7f8c8d', fontSize: 14 }}>{leads.length} active lead{leads.length !== 1 ? 's' : ''}</span>
        <button
          onClick={fetchLeads}
          style={{ ...btnBase, background: '#f0f0f0', color: '#333' }}
        >
          Refresh
        </button>
      </div>

      {leads.length === 0 ? (
        <div style={{ padding: 40, textAlign: 'center', color: '#95a5a6' }}>
          No active leads found.
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
                {lead.dealer_name_assigned && (
                  <div>
                    <span style={{ fontWeight: 500 }}>Dealer:</span> {lead.dealer_name_assigned}
                  </div>
                )}
                <div>
                  <span style={{ fontWeight: 500 }}>Status:</span>{' '}
                  <span style={{
                    background: '#27ae60',
                    color: 'white',
                    padding: '2px 8px',
                    borderRadius: 12,
                    fontSize: 11,
                  }}>
                    {lead.status}
                  </span>
                </div>
                {lead.product_type && <div>Product: {lead.product_type}</div>}
              </div>

              <div style={{ display: 'flex', gap: 6, marginTop: 12, flexWrap: 'wrap' }}>
                <button
                  onClick={() => loadLeadDetail(lead.id, 'view')}
                  style={{ ...btnBase, background: '#3498db', color: 'white' }}
                  disabled={detailLoadingId === lead.id}
                >
                  {detailLoadingId === lead.id ? 'Loading...' : 'View'}
                </button>
                <button
                  onClick={() => loadLeadDetail(lead.id, 'edit')}
                  style={{ ...btnBase, background: '#f0f0f0', color: '#333' }}
                  disabled={detailLoadingId === lead.id}
                >
                  {detailLoadingId === lead.id ? 'Loading...' : 'Edit'}
                </button>
                <button
                  onClick={() => setLogLeadId(lead.id)}
                  style={{ ...btnBase, background: '#f0f0f0', color: '#333' }}
                >
                  Log
                </button>
                <button
                  onClick={() => setReassignLeadId(lead.id)}
                  style={{ ...btnBase, background: '#f39c12', color: 'white' }}
                >
                  Reassign
                </button>
                <button
                  onClick={() => handleArchive(lead.id)}
                  style={{ ...btnBase, background: '#e74c3c', color: 'white' }}
                >
                  Archive
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
            style={{ ...btnBase, background: 'white', color: '#333', border: '1px solid #ddd' }}
          >
            Show 100 More
          </button>
        </div>
      )}

      <LeadDetailModal
        lead={viewLead}
        isOpen={viewLead !== null}
        onClose={() => setViewLead(null)}
      />

      <LeadEditModal
        lead={editLead}
        isOpen={editLead !== null}
        onClose={() => setEditLead(null)}
        onSaved={handleSaved}
      />

      <LeadLogModal
        leadId={logLeadId}
        isOpen={logLeadId !== null}
        onClose={() => setLogLeadId(null)}
      />

      {reassignLeadId !== null && (
        <AssignDealerModal
          leadId={reassignLeadId}
          isOpen={true}
          onClose={() => setReassignLeadId(null)}
          onAssigned={handleReassigned}
        />
      )}
    </div>
  );
};

export default ActiveLeadsList;
