import React, { useState, useEffect } from 'react';
import { getLeadLogs, createLeadLog } from '../../services/api';
import type { LeadLog } from '../../types';

interface LeadLogModalProps {
  leadId: number | null;
  isOpen: boolean;
  onClose: () => void;
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

const LOG_TYPES = [
  'Note',
  'Phone Call',
  'Email Sent',
  'Email Received',
  'Meeting',
  'Follow Up',
  'Status Change',
  'Assignment',
  'Other',
];

const logTypeColors: Record<string, string> = {
  Note: '#3498db',
  'Phone Call': '#27ae60',
  'Email Sent': '#8e44ad',
  'Email Received': '#2980b9',
  Meeting: '#f39c12',
  'Follow Up': '#e67e22',
  'Status Change': '#e74c3c',
  Assignment: '#1abc9c',
  Other: '#7f8c8d',
};

const LeadLogModal: React.FC<LeadLogModalProps> = ({ leadId, isOpen, onClose }) => {
  const [logs, setLogs] = useState<LeadLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [newLogType, setNewLogType] = useState('Note');
  const [newMessage, setNewMessage] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen || leadId === null) return;
    setLoading(true);
    setError(null);
    getLeadLogs(leadId)
      .then((result) => {
        setLogs(result.logs ?? result as any);
      })
      .catch((err) => {
        setError(err?.response?.data?.detail || 'Failed to load logs.');
        setLogs([]);
      })
      .finally(() => setLoading(false));
  }, [isOpen, leadId]);

  const handleSubmit = async () => {
    if (leadId === null || !newMessage.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      await createLeadLog({ lead_id: leadId, log_type: newLogType, message: newMessage.trim() });
      setNewMessage('');
      // Reload logs
      const result = await getLeadLogs(leadId);
      setLogs(result.logs ?? result as any);
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to create log entry.');
    } finally {
      setSubmitting(false);
    }
  };

  if (!isOpen || leadId === null) return null;

  return (
    <div style={overlayStyle} onClick={onClose}>
      <div style={modalStyle} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h3 style={{ margin: 0, fontSize: 18, fontWeight: 600, color: '#2c3e50' }}>
            Lead Logs #{leadId}
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

        {/* Log entries */}
        <div style={{ marginBottom: 20, maxHeight: 340, overflowY: 'auto' }}>
          {loading ? (
            <div style={{ padding: 20, textAlign: 'center', color: '#7f8c8d' }}>Loading logs...</div>
          ) : logs.length === 0 ? (
            <div style={{ padding: 20, textAlign: 'center', color: '#95a5a6' }}>No log entries yet.</div>
          ) : (
            logs.map((log) => (
              <div
                key={log.id}
                style={{
                  padding: 12,
                  borderLeft: `3px solid ${logTypeColors[log.log_type] || '#7f8c8d'}`,
                  marginBottom: 8,
                  background: '#f8f9fa',
                  borderRadius: '0 6px 6px 0',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <span style={{
                    fontSize: 12,
                    fontWeight: 600,
                    color: logTypeColors[log.log_type] || '#7f8c8d',
                  }}>
                    {log.log_type}
                  </span>
                  <span style={{ fontSize: 11, color: '#95a5a6' }}>
                    {new Date(log.created_at).toLocaleString()}
                  </span>
                </div>
                <div style={{ fontSize: 14, color: '#2c3e50', lineHeight: 1.5 }}>
                  {log.message}
                </div>
                {log.dealer_name && (
                  <div style={{ fontSize: 12, color: '#7f8c8d', marginTop: 4 }}>
                    Dealer: {log.dealer_name}
                  </div>
                )}
                {log.created_by && (
                  <div style={{ fontSize: 12, color: '#7f8c8d', marginTop: 2 }}>
                    By: {log.created_by}
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        {/* Add new log */}
        <div style={{ borderTop: '1px solid #e0e0e0', paddingTop: 16 }}>
          <div style={{ fontWeight: 600, fontSize: 14, marginBottom: 10, color: '#2c3e50' }}>Add Log Entry</div>
          <div style={{ display: 'flex', gap: 10, marginBottom: 10 }}>
            <select
              value={newLogType}
              onChange={(e) => setNewLogType(e.target.value)}
              style={{
                padding: '8px 10px',
                border: '1px solid #ddd',
                borderRadius: 6,
                fontSize: 14,
                minWidth: 140,
              }}
            >
              {LOG_TYPES.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>
          <textarea
            value={newMessage}
            onChange={(e) => setNewMessage(e.target.value)}
            placeholder="Enter log message..."
            rows={3}
            style={{
              width: '100%',
              padding: '8px 10px',
              border: '1px solid #ddd',
              borderRadius: 6,
              fontSize: 14,
              boxSizing: 'border-box',
              resize: 'vertical',
              marginBottom: 10,
            }}
          />
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
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
              Close
            </button>
            <button
              onClick={handleSubmit}
              disabled={submitting || !newMessage.trim()}
              style={{
                padding: '8px 16px',
                borderRadius: 6,
                border: 'none',
                cursor: submitting || !newMessage.trim() ? 'not-allowed' : 'pointer',
                fontSize: 13,
                fontWeight: 500,
                background: '#c91414',
                color: 'white',
                opacity: submitting || !newMessage.trim() ? 0.6 : 1,
              }}
            >
              {submitting ? 'Adding...' : 'Add Log'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LeadLogModal;
