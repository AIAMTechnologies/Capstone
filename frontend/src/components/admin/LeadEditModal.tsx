import React, { useState, useEffect } from 'react';
import { updateLead } from '../../services/api';
import type { ExtendedLead } from '../../types';

interface LeadEditModalProps {
  lead: ExtendedLead | null;
  isOpen: boolean;
  onClose: () => void;
  onSaved: () => void;
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

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '8px 10px',
  border: '1px solid #ddd',
  borderRadius: 6,
  fontSize: 14,
  boxSizing: 'border-box',
};

const labelStyle: React.CSSProperties = {
  fontSize: 12,
  color: '#7f8c8d',
  fontWeight: 600,
  marginBottom: 4,
  display: 'block',
};

type EditableFields = {
  first_name: string;
  last_name: string;
  email: string;
  phone: string;
  company_name: string;
  city: string;
  province: string;
  postal_code: string;
  product_type: string;
  project_type: string;
  project_city: string;
  business_category: string;
  lead_source: string;
  status: string;
  comments: string;
};

const FIELD_LABELS: Record<keyof EditableFields, string> = {
  first_name: 'First Name',
  last_name: 'Last Name',
  email: 'Email',
  phone: 'Phone',
  company_name: 'Company',
  city: 'City',
  province: 'Province',
  postal_code: 'Postal Code',
  product_type: 'Product Type',
  project_type: 'Project Type',
  project_city: 'Project City',
  business_category: 'Business Category',
  lead_source: 'Lead Source',
  status: 'Status',
  comments: 'Comments',
};

const buildInitial = (lead: ExtendedLead | null): EditableFields => ({
  first_name: lead?.first_name || '',
  last_name: lead?.last_name || '',
  email: lead?.email || '',
  phone: lead?.phone || '',
  company_name: lead?.company_name || '',
  city: lead?.city || '',
  province: lead?.province || '',
  postal_code: lead?.postal_code || '',
  product_type: lead?.product_type || '',
  project_type: lead?.project_type || '',
  project_city: lead?.project_city || '',
  business_category: lead?.business_category || '',
  lead_source: lead?.lead_source || '',
  status: lead?.status || 'active',
  comments: lead?.comments || '',
});

const LeadEditModal: React.FC<LeadEditModalProps> = ({ lead, isOpen, onClose, onSaved }) => {
  const [form, setForm] = useState<EditableFields>(buildInitial(lead));
  const [original, setOriginal] = useState<EditableFields>(buildInitial(lead));
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen && lead) {
      const init = buildInitial(lead);
      setForm(init);
      setOriginal(init);
      setError(null);
    }
  }, [isOpen, lead]);

  const handleChange = (field: keyof EditableFields, value: string) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const handleSave = async () => {
    if (!lead) return;
    setError(null);

    // Only send changed fields
    const changed: Record<string, unknown> = {};
    (Object.keys(form) as Array<keyof EditableFields>).forEach((key) => {
      if (form[key] !== original[key]) {
        changed[key] = form[key];
      }
    });

    if (Object.keys(changed).length === 0) {
      onClose();
      return;
    }

    setSubmitting(true);
    try {
      await updateLead(lead.id, changed);
      onSaved();
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to update lead.');
    } finally {
      setSubmitting(false);
    }
  };

  if (!isOpen || !lead) return null;

  return (
    <div style={overlayStyle} onClick={onClose}>
      <div style={modalStyle} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h3 style={{ margin: 0, fontSize: 18, fontWeight: 600, color: '#2c3e50' }}>
            Edit Lead #{lead.id}
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

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px 16px' }}>
          {(Object.keys(FIELD_LABELS) as Array<keyof EditableFields>).map((key) => (
            <div key={key} style={key === 'comments' ? { gridColumn: '1 / -1' } : undefined}>
              <label style={labelStyle}>{FIELD_LABELS[key]}</label>
              {key === 'comments' ? (
                <textarea
                  value={form[key]}
                  onChange={(e) => handleChange(key, e.target.value)}
                  rows={3}
                  style={{ ...inputStyle, resize: 'vertical' }}
                />
              ) : key === 'status' ? (
                <select
                  value={form[key]}
                  onChange={(e) => handleChange(key, e.target.value)}
                  style={inputStyle}
                >
                  <option value="active">Active</option>
                  <option value="converted">Converted</option>
                  <option value="dead">Dead</option>
                  <option value="follow_up">Follow Up</option>
                </select>
              ) : (
                <input
                  type={key === 'email' ? 'email' : 'text'}
                  value={form[key]}
                  onChange={(e) => handleChange(key, e.target.value)}
                  style={inputStyle}
                />
              )}
            </div>
          ))}
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, marginTop: 20 }}>
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
            onClick={handleSave}
            disabled={submitting}
            style={{
              padding: '8px 16px',
              borderRadius: 6,
              border: 'none',
              cursor: submitting ? 'not-allowed' : 'pointer',
              fontSize: 13,
              fontWeight: 500,
              background: '#c91414',
              color: 'white',
              opacity: submitting ? 0.6 : 1,
            }}
          >
            {submitting ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default LeadEditModal;
