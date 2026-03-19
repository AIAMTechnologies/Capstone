import React, { useState, useCallback, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import AIInsightsPanel from '../../components/admin/AIInsightsPanel';
import InsertLeadForm from '../../components/admin/InsertLeadForm';
import UnassignedLeadsList from '../../components/admin/UnassignedLeadsList';
import ActiveLeadsList from '../../components/admin/ActiveLeadsList';
import LeadDetailModal from '../../components/admin/LeadDetailModal';
import { getLeadDetail } from '../../services/api';
import type { ExtendedLead } from '../../types';

const bannerStyle: React.CSSProperties = {
  background: '#c91414',
  color: 'white',
  padding: '12px 20px',
  fontSize: 18,
  fontWeight: 600,
  borderRadius: '8px 8px 0 0',
  cursor: 'pointer',
  display: 'flex',
  justifyContent: 'space-between',
  alignItems: 'center',
  userSelect: 'none',
};

const sectionStyle: React.CSSProperties = {
  background: 'white',
  borderRadius: 8,
  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
  marginBottom: 20,
  overflow: 'hidden',
};

const Dashboard: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const [insertOpen, setInsertOpen] = useState(true);
  const [unassignedOpen, setUnassignedOpen] = useState(true);
  const [activeOpen, setActiveOpen] = useState(true);
  const [linkedLead, setLinkedLead] = useState<ExtendedLead | null>(null);

  // Keys to force remount / refresh child components
  const [unassignedKey, setUnassignedKey] = useState(0);
  const [activeKey, setActiveKey] = useState(0);

  const refreshUnassigned = useCallback(() => {
    setUnassignedKey((k) => k + 1);
  }, []);

  const refreshActive = useCallback(() => {
    setActiveKey((k) => k + 1);
  }, []);

  const refreshAll = useCallback(() => {
    refreshUnassigned();
    refreshActive();
  }, [refreshUnassigned, refreshActive]);

  useEffect(() => {
    const leadId = Number(searchParams.get('lead'));
    if (!leadId) {
      setLinkedLead(null);
      return;
    }

    let cancelled = false;
    const loadLead = async () => {
      try {
        const lead = await getLeadDetail(leadId);
        if (!cancelled) {
          setLinkedLead(lead as ExtendedLead);
        }
      } catch {
        if (!cancelled) {
          setLinkedLead(null);
        }
      }
    };

    loadLead();

    return () => {
      cancelled = true;
    };
  }, [searchParams]);

  const closeLinkedLead = useCallback(() => {
    const nextParams = new URLSearchParams(searchParams);
    nextParams.delete('lead');
    setSearchParams(nextParams, { replace: true });
    setLinkedLead(null);
  }, [searchParams, setSearchParams]);

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '40px 20px' }}>
      <h1 style={{ marginBottom: 24, fontSize: 28, fontWeight: 700, color: '#2c3e50' }}>
        Admin Dashboard
      </h1>

      {/* AI Insights Panel */}
      <AIInsightsPanel />

      {/* Section 1: Insert New Lead */}
      <div style={sectionStyle}>
        <div
          style={bannerStyle}
          onClick={() => setInsertOpen((o) => !o)}
        >
          <span>Insert New Lead</span>
          <span style={{ fontSize: 14 }}>{insertOpen ? '-' : '+'}</span>
        </div>
        {insertOpen && (
          <div style={{ padding: 20 }}>
            <InsertLeadForm onLeadCreated={refreshAll} />
          </div>
        )}
      </div>

      {/* Section 2: Unassigned Leads */}
      <div style={sectionStyle}>
        <div
          style={bannerStyle}
          onClick={() => setUnassignedOpen((o) => !o)}
        >
          <span>Unassigned Leads</span>
          <span style={{ fontSize: 14 }}>{unassignedOpen ? '-' : '+'}</span>
        </div>
        {unassignedOpen && (
          <UnassignedLeadsList
            key={unassignedKey}
            onLeadAssigned={refreshActive}
          />
        )}
      </div>

      {/* Section 3: Active Leads */}
      <div style={sectionStyle}>
        <div
          style={bannerStyle}
          onClick={() => setActiveOpen((o) => !o)}
        >
          <span>Active Leads</span>
          <span style={{ fontSize: 14 }}>{activeOpen ? '-' : '+'}</span>
        </div>
        {activeOpen && (
          <ActiveLeadsList
            key={activeKey}
            onRefresh={refreshUnassigned}
          />
        )}
      </div>

      <LeadDetailModal
        lead={linkedLead}
        isOpen={linkedLead !== null}
        onClose={closeLinkedLead}
      />
    </div>
  );
};

export default Dashboard;
