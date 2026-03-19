import React from 'react';
import type { ExtendedLead } from '../../types';

interface LeadDetailModalProps {
  lead: ExtendedLead | null;
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

const labelStyle: React.CSSProperties = {
  fontSize: 12,
  color: '#7f8c8d',
  fontWeight: 600,
  marginBottom: 2,
  textTransform: 'uppercase',
  letterSpacing: '0.5px',
};

const valueStyle: React.CSSProperties = {
  fontSize: 14,
  color: '#2c3e50',
  marginBottom: 14,
};

const priorityColors: Record<string, string> = {
  Hot: '#e74c3c',
  Warm: '#f39c12',
  Cold: '#3498db',
};

const LeadDetailModal: React.FC<LeadDetailModalProps> = ({ lead, isOpen, onClose }) => {
  if (!isOpen || !lead) return null;

  const fields: Array<{ label: string; value: string | number | undefined | null }> = [
    { label: 'Name', value: [lead.first_name, lead.last_name].filter(Boolean).join(' ') || lead.name },
    { label: 'Email', value: lead.email },
    { label: 'Phone', value: lead.phone },
    { label: 'Company', value: lead.company_name },
    { label: 'Address', value: lead.address },
    { label: 'City', value: [lead.city, lead.province].filter(Boolean).join(', ') },
    { label: 'Postal Code', value: lead.postal_code },
    { label: 'Job Type', value: lead.job_type },
    { label: 'Project Type', value: lead.project_type },
    { label: 'Project City', value: lead.project_city },
    { label: 'Product Type', value: lead.product_type },
    { label: 'Product Type 2', value: lead.product_type_2 },
    { label: 'Product Type 3', value: lead.product_type_3 },
    { label: 'Square Footage', value: lead.square_footage },
    { label: 'Business Category', value: lead.business_category },
    { label: 'Lead Source', value: lead.lead_source },
    { label: 'Dealer Email', value: lead.dealer_email },
    { label: 'Assigned Dealer', value: lead.dealer_name_assigned },
    { label: 'Installer (ML)', value: lead.installer_name || 'Unassigned' },
    { label: 'Allocation Score', value: lead.allocation_score?.toFixed(2) },
    { label: 'Status', value: lead.status },
    { label: 'Landing Page', value: lead.landing_page },
    { label: 'UTM Source', value: lead.utm_source },
    { label: 'UTM Medium', value: lead.utm_medium },
    { label: 'UTM Campaign', value: lead.utm_campaign },
    { label: 'UTM Content', value: lead.utm_content },
    { label: 'Opt-In', value: lead.opt_in !== undefined ? (lead.opt_in ? 'Yes' : 'No') : undefined },
    { label: 'Created', value: lead.created_at ? new Date(lead.created_at).toLocaleDateString() : undefined },
    { label: 'Comments', value: lead.comments },
  ];

  return (
    <div style={overlayStyle} onClick={onClose}>
      <div style={modalStyle} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h3 style={{ margin: 0, fontSize: 18, fontWeight: 600, color: '#2c3e50' }}>
            Lead Details #{lead.id}
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

        {/* AI Score / Priority Badge */}
        {(lead.ai_priority || lead.ai_score !== undefined) && (
          <div style={{
            background: '#f8f9fa',
            borderRadius: 8,
            padding: 14,
            marginBottom: 16,
            border: '1px solid #e0e0e0',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
              <span style={{ fontWeight: 600, fontSize: 14, color: '#2c3e50' }}>AI Analysis</span>
              {lead.ai_priority && (
                <span style={{
                  background: priorityColors[lead.ai_priority] || '#95a5a6',
                  color: 'white',
                  padding: '2px 10px',
                  borderRadius: 12,
                  fontSize: 12,
                  fontWeight: 600,
                }}>
                  {lead.ai_priority}
                </span>
              )}
              {lead.ai_score !== undefined && (
                <span style={{ fontSize: 13, color: '#555' }}>Score: {lead.ai_score}</span>
              )}
            </div>
            {lead.ai_reasoning && (
              <div style={{ fontSize: 13, color: '#555', marginTop: 4 }}>{lead.ai_reasoning}</div>
            )}
            {lead.ai_conversion_likelihood !== undefined && (
              <div style={{ fontSize: 13, color: '#555', marginTop: 4 }}>
                Conversion Likelihood: {(lead.ai_conversion_likelihood * 100).toFixed(0)}%
              </div>
            )}
            {lead.ai_conversion_explanation && (
              <div style={{ fontSize: 13, color: '#555', marginTop: 2 }}>
                {lead.ai_conversion_explanation}
              </div>
            )}
          </div>
        )}

        {/* Why this dealer? */}
        {lead.ai_match_explanation && (
          <div style={{
            background: '#eef6ff',
            borderRadius: 8,
            padding: 14,
            marginBottom: 16,
            border: '1px solid #bdd8f5',
          }}>
            <div style={{ fontWeight: 600, fontSize: 14, color: '#2c3e50', marginBottom: 6 }}>
              Why this dealer?
            </div>
            <div style={{ fontSize: 13, color: '#444', lineHeight: 1.5 }}>
              {lead.ai_match_explanation}
            </div>
          </div>
        )}

        {/* All fields */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0 20px' }}>
          {fields.map(({ label, value }) => {
            if (!value && value !== 0) return null;
            return (
              <div key={label}>
                <div style={labelStyle}>{label}</div>
                <div style={valueStyle}>{String(value)}</div>
              </div>
            );
          })}
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 16 }}>
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
        </div>
      </div>
    </div>
  );
};

export default LeadDetailModal;
