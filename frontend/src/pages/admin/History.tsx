import React, { useState, useEffect, useCallback } from 'react';
import { getHistoryLeads, updateValueOfOrder } from '../../services/api';
import type { ExtendedLead } from '../../types/types';

const PROVINCES = [
  { value: '', label: 'All Provinces' },
  { value: 'AB', label: 'Alberta' },
  { value: 'BC', label: 'British Columbia' },
  { value: 'MB', label: 'Manitoba' },
  { value: 'NB', label: 'New Brunswick' },
  { value: 'NL', label: 'Newfoundland and Labrador' },
  { value: 'NS', label: 'Nova Scotia' },
  { value: 'NT', label: 'Northwest Territories' },
  { value: 'NU', label: 'Nunavut' },
  { value: 'ON', label: 'Ontario' },
  { value: 'PE', label: 'Prince Edward Island' },
  { value: 'QC', label: 'Quebec' },
  { value: 'SK', label: 'Saskatchewan' },
  { value: 'YT', label: 'Yukon' },
];

const LIMIT = 50;

const statusBadge = (status: string): React.CSSProperties => {
  const base: React.CSSProperties = {
    display: 'inline-block',
    padding: '3px 10px',
    borderRadius: 12,
    fontSize: 12,
    fontWeight: 600,
    textTransform: 'capitalize',
  };
  switch (status) {
    case 'converted':
      return { ...base, background: '#e6f9ed', color: '#1a7a3a' };
    case 'dead':
      return { ...base, background: '#fce8e8', color: '#c91414' };
    case 'archived':
      return { ...base, background: '#f0f0f0', color: '#888' };
    default:
      return { ...base, background: '#e8f0fe', color: '#1a56db' };
  }
};

const History: React.FC = () => {
  const [leads, setLeads] = useState<ExtendedLead[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [province, setProvince] = useState('');
  const [dealer, setDealer] = useState('');

  const [expandedIds, setExpandedIds] = useState<Set<number>>(new Set());

  const [editingValueId, setEditingValueId] = useState<number | null>(null);
  const [editValue, setEditValue] = useState('');
  const [savingValue, setSavingValue] = useState(false);

  const fetchLeads = useCallback(async (newOffset = 0, append = false) => {
    setLoading(true);
    setError('');
    try {
      const params: Record<string, any> = { limit: LIMIT, offset: newOffset };
      if (startDate) params.start_date = startDate;
      if (endDate) params.end_date = endDate;
      if (province) params.province = province;
      if (dealer) params.dealer = dealer;
      const res = await getHistoryLeads(params);
      if (append) {
        setLeads(prev => [...prev, ...res.leads]);
      } else {
        setLeads(res.leads);
      }
      setTotal(res.total);
      setOffset(newOffset);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load leads');
    } finally {
      setLoading(false);
    }
  }, [startDate, endDate, province, dealer]);

  useEffect(() => {
    fetchLeads(0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleFilter = () => {
    setExpandedIds(new Set());
    fetchLeads(0);
  };

  const handleLoadMore = () => {
    fetchLeads(offset + LIMIT, true);
  };

  const toggleExpand = (id: number) => {
    setExpandedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleSaveValue = async (leadId: number) => {
    setSavingValue(true);
    try {
      await updateValueOfOrder(leadId, parseFloat(editValue) || 0);
      setLeads(prev =>
        prev.map(l => l.id === leadId ? { ...l, value_of_order: parseFloat(editValue) || 0 } : l)
      );
      setEditingValueId(null);
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to update value');
    } finally {
      setSavingValue(false);
    }
  };

  const formatDate = (d?: string) => {
    if (!d) return '-';
    return new Date(d).toLocaleDateString('en-CA');
  };

  return (
    <div style={{ padding: 24 }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, color: '#1a1a2e', marginBottom: 20 }}>Lead History</h1>

      {/* Filter Bar */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', padding: '16px 24px', background: 'white', borderRadius: 8, marginBottom: 16, flexWrap: 'wrap', boxShadow: '0 2px 8px rgba(0,0,0,0.1)' }}>
        <label style={{ fontSize: 13, fontWeight: 600, color: '#555' }}>From</label>
        <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid #ddd', fontSize: 14 }} />
        <label style={{ fontSize: 13, fontWeight: 600, color: '#555' }}>To</label>
        <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid #ddd', fontSize: 14 }} />
        <select value={province} onChange={e => setProvince(e.target.value)} style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid #ddd', fontSize: 14 }}>
          {PROVINCES.map(p => <option key={p.value} value={p.value}>{p.label}</option>)}
        </select>
        <input type="text" placeholder="Dealer name" value={dealer} onChange={e => setDealer(e.target.value)} style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid #ddd', fontSize: 14, minWidth: 140 }} />
        <button onClick={handleFilter} style={{ padding: '8px 20px', background: '#c91414', color: 'white', border: 'none', borderRadius: 6, fontSize: 14, fontWeight: 600, cursor: 'pointer' }}>
          Filter
        </button>
      </div>

      <p style={{ fontSize: 14, color: '#666', marginBottom: 12 }}>
        Showing {leads.length} of {total} leads
      </p>

      {error && <div style={{ color: '#c91414', marginBottom: 12 }}>{error}</div>}

      {leads.map(lead => {
        const expanded = expandedIds.has(lead.id);
        const displayName = [lead.first_name, lead.last_name].filter(Boolean).join(' ') || lead.name || '-';
        return (
          <div key={lead.id} style={{ background: 'white', borderRadius: 8, padding: 16, boxShadow: '0 2px 8px rgba(0,0,0,0.1)', marginBottom: 12 }}>
            <div
              onClick={() => toggleExpand(lead.id)}
              style={{ display: 'flex', alignItems: 'center', gap: 16, cursor: 'pointer', flexWrap: 'wrap' }}
            >
              <span style={{ fontSize: 13, color: '#999' }}>{expanded ? '\u25BC' : '\u25B6'}</span>
              <span style={{ fontWeight: 600, fontSize: 14, minWidth: 160 }}>{displayName}</span>
              <span style={{ fontSize: 13, color: '#666', minWidth: 100 }}>{lead.city || '-'}</span>
              <span style={{ fontSize: 13, color: '#666', minWidth: 40 }}>{lead.province || '-'}</span>
              <span style={statusBadge(lead.status)}>{lead.status}</span>
              <span style={{ fontSize: 13, color: '#666', minWidth: 120 }}>{lead.dealer_name_assigned || '-'}</span>
              <span style={{ fontSize: 12, color: '#999', marginLeft: 'auto' }}>{formatDate(lead.created_at)}</span>
            </div>

            {expanded && (
              <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid #eee' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px 24px', fontSize: 13 }}>
                  <div><strong>Email:</strong> {lead.email || '-'}</div>
                  <div><strong>Phone:</strong> {lead.phone || '-'}</div>
                  <div><strong>Address:</strong> {lead.address || '-'}</div>
                  <div><strong>Product Type:</strong> {lead.product_type || '-'}</div>
                  <div><strong>Square Footage:</strong> {lead.square_footage ?? '-'}</div>
                  <div><strong>Project Type:</strong> {lead.project_type || '-'}</div>
                  <div><strong>Business Category:</strong> {lead.business_category || '-'}</div>
                  <div><strong>Lead Source:</strong> {lead.lead_source || '-'}</div>
                  <div><strong>Company:</strong> {lead.company_name || '-'}</div>
                  <div>
                    <strong>Value of Order:</strong>{' '}
                    {editingValueId === lead.id ? (
                      <span style={{ display: 'inline-flex', gap: 4, alignItems: 'center' }}>
                        <input
                          type="number"
                          value={editValue}
                          onChange={e => setEditValue(e.target.value)}
                          style={{ width: 100, padding: '4px 8px', borderRadius: 4, border: '1px solid #ddd', fontSize: 13 }}
                          onClick={e => e.stopPropagation()}
                        />
                        <button
                          disabled={savingValue}
                          onClick={e => { e.stopPropagation(); handleSaveValue(lead.id); }}
                          style={{ padding: '4px 10px', background: '#c91414', color: 'white', border: 'none', borderRadius: 4, fontSize: 12, cursor: 'pointer' }}
                        >
                          {savingValue ? '...' : 'Save'}
                        </button>
                        <button
                          onClick={e => { e.stopPropagation(); setEditingValueId(null); }}
                          style={{ padding: '4px 10px', background: '#eee', color: '#333', border: 'none', borderRadius: 4, fontSize: 12, cursor: 'pointer' }}
                        >
                          Cancel
                        </button>
                      </span>
                    ) : (
                      <span>
                        {lead.value_of_order != null ? `$${Number(lead.value_of_order).toLocaleString()}` : '-'}
                        <button
                          onClick={e => { e.stopPropagation(); setEditingValueId(lead.id); setEditValue(String(lead.value_of_order || '')); }}
                          style={{ marginLeft: 6, padding: '2px 8px', background: 'none', border: '1px solid #ddd', borderRadius: 4, fontSize: 11, cursor: 'pointer', color: '#666' }}
                        >
                          Edit
                        </button>
                      </span>
                    )}
                  </div>
                  <div><strong>Comments:</strong> {lead.comments || '-'}</div>
                  <div><strong>UTM Source:</strong> {lead.utm_source || '-'}</div>
                  <div><strong>UTM Medium:</strong> {lead.utm_medium || '-'}</div>
                  <div><strong>UTM Campaign:</strong> {lead.utm_campaign || '-'}</div>
                  <div><strong>UTM Content:</strong> {lead.utm_content || '-'}</div>
                  <div><strong>UTM Term:</strong> {lead.utm_term || '-'}</div>
                  <div><strong>Landing Page:</strong> {lead.landing_page || '-'}</div>
                </div>
              </div>
            )}
          </div>
        );
      })}

      {leads.length < total && (
        <div style={{ textAlign: 'center', marginTop: 20 }}>
          <button
            onClick={handleLoadMore}
            disabled={loading}
            style={{ padding: '10px 32px', background: '#c91414', color: 'white', border: 'none', borderRadius: 6, fontSize: 14, fontWeight: 600, cursor: 'pointer', opacity: loading ? 0.6 : 1 }}
          >
            {loading ? 'Loading...' : 'Load More'}
          </button>
        </div>
      )}

      {loading && leads.length === 0 && <p style={{ textAlign: 'center', color: '#999' }}>Loading...</p>}
    </div>
  );
};

export default History;
