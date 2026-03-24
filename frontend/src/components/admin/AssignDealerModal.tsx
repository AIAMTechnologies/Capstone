import React, { useState, useEffect } from 'react';
import { getDealerOptions, assignDealerToLead } from '../../services/api';
import type { Dealer } from '../../types';

interface AssignDealerModalProps {
  leadId: number;
  isOpen: boolean;
  onClose: () => void;
  onAssigned: () => void;
}

const overlayStyle: React.CSSProperties = {
  position: 'fixed',
  top: 0,
  left: 0,
  right: 0,
  bottom: 0,
  background: 'rgba(0,0,0,0.5)',
  zIndex: 1000,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
};

const modalStyle: React.CSSProperties = {
  background: 'white',
  borderRadius: 12,
  padding: 24,
  maxWidth: 600,
  width: '90%',
  maxHeight: '80vh',
  overflowY: 'auto',
};

const AssignDealerModal: React.FC<AssignDealerModalProps> = ({ leadId, isOpen, onClose, onAssigned }) => {
  const [dealers, setDealers] = useState<Dealer[]>([]);
  const [selected, setSelected] = useState<number[]>([]);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    setError(null);
    setSelected([]);
    setSearch('');
    getDealerOptions()
      .then((result) => {
        setDealers(result.dealers ?? result as any);
      })
      .catch((err) => {
        setError(err?.response?.data?.detail || 'Failed to load dealers.');
      })
      .finally(() => setLoading(false));
  }, [isOpen]);

  const toggleDealer = (id: number) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((d) => d !== id) : [...prev, id]
    );
  };

  const handleAssign = async () => {
    if (selected.length === 0) {
      setError('Please select at least one dealer.');
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await assignDealerToLead(leadId, selected);
      onAssigned();
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to assign dealer.');
    } finally {
      setSubmitting(false);
    }
  };

  if (!isOpen) return null;

  const filtered = dealers.filter((d) =>
    d.name.toLowerCase().includes(search.toLowerCase()) ||
    d.city.toLowerCase().includes(search.toLowerCase()) ||
    d.province.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div style={overlayStyle} onClick={onClose}>
      <div style={modalStyle} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h3 style={{ margin: 0, fontSize: 18, fontWeight: 600, color: '#2c3e50' }}>
            Assign Dealer to Lead #{leadId}
          </h3>
          <button
            onClick={onClose}
            style={{
              fontSize: 24,
              border: 'none',
              background: 'transparent',
              cursor: 'pointer',
              color: '#7f8c8d',
              padding: 0,
              lineHeight: 1,
            }}
          >
            x
          </button>
        </div>

        {error && (
          <div style={{
            marginBottom: 12,
            padding: '10px 14px',
            borderRadius: 6,
            backgroundColor: '#fdecea',
            color: '#c0392b',
            fontSize: 14,
          }}>
            {error}
          </div>
        )}

        <input
          type="text"
          placeholder="Search dealers by name, city, or province..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{
            width: '100%',
            padding: '10px 12px',
            border: '1px solid #ddd',
            borderRadius: 6,
            fontSize: 14,
            marginBottom: 12,
            boxSizing: 'border-box',
          }}
        />

        {loading ? (
          <div style={{ padding: 20, textAlign: 'center', color: '#7f8c8d' }}>Loading dealers...</div>
        ) : filtered.length === 0 ? (
          <div style={{ padding: 20, textAlign: 'center', color: '#95a5a6' }}>No dealers found.</div>
        ) : (
          <div style={{ maxHeight: 320, overflowY: 'auto', border: '1px solid #eee', borderRadius: 6 }}>
            {filtered.map((dealer) => (
              <label
                key={dealer.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '10px 12px',
                  borderBottom: '1px solid #f0f0f0',
                  cursor: 'pointer',
                  background: selected.includes(dealer.id) ? '#fef5f5' : 'white',
                }}
              >
                <input
                  type="checkbox"
                  checked={selected.includes(dealer.id)}
                  onChange={() => toggleDealer(dealer.id)}
                  style={{ width: 16, height: 16 }}
                />
                <div>
                  <div style={{ fontWeight: 500, fontSize: 14, color: '#2c3e50' }}>{dealer.name}</div>
                  <div style={{ fontSize: 12, color: '#7f8c8d' }}>
                    {dealer.city}, {dealer.province} {dealer.email && `- ${dealer.email}`}
                  </div>
                </div>
              </label>
            ))}
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 16 }}>
          <button
            onClick={onClose}
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
            Cancel
          </button>
          <button
            onClick={handleAssign}
            disabled={submitting || selected.length === 0}
            style={{
              padding: '8px 16px',
              borderRadius: 6,
              border: 'none',
              cursor: submitting ? 'not-allowed' : 'pointer',
              fontSize: 13,
              fontWeight: 500,
              background: '#c91414',
              color: 'white',
              opacity: submitting || selected.length === 0 ? 0.6 : 1,
            }}
          >
            {submitting ? 'Assigning...' : `Assign ${selected.length > 0 ? `(${selected.length})` : ''}`}
          </button>
        </div>
      </div>
    </div>
  );
};

export default AssignDealerModal;
