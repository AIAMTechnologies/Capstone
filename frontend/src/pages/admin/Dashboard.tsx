import React, { useState, useCallback, useEffect } from 'react';
import AIInsightsPanel from '../../components/admin/AIInsightsPanel';
import AISpendWidget from '../../components/admin/AISpendWidget';
import LassoSyncStatusCard from '../../components/admin/LassoSyncStatusCard';
import LassoUnassignedLeadsTable from '../../components/admin/LassoUnassignedLeadsTable';
import LassoActiveLeadsTable from '../../components/admin/LassoActiveLeadsTable';
import { getDashboardSnapshot } from '../../services/api';
import type { LassoDashboardStatus, DashboardUnassignedLead, DashboardActiveLead } from '../../types/types';

const SNAPSHOT_CACHE_KEY = 'dashboard_snapshot_v1';

interface SnapshotCache {
  sync_status: LassoDashboardStatus;
  unassigned: DashboardUnassignedLead[];
  active: DashboardActiveLead[];
  cached_at: number;
}

function readCache(): SnapshotCache | null {
  try {
    const raw = sessionStorage.getItem(SNAPSHOT_CACHE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as SnapshotCache;
  } catch {
    return null;
  }
}

function writeCache(snap: SnapshotCache): void {
  try {
    sessionStorage.setItem(SNAPSHOT_CACHE_KEY, JSON.stringify(snap));
  } catch {
    // Storage quota exceeded — not critical
  }
}

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
  const [unassignedOpen, setUnassignedOpen] = useState(true);
  const [activeOpen, setActiveOpen] = useState(true);
  const [lassoRefreshKey, setLassoRefreshKey] = useState(0);

  // Seed from cache immediately (zero wait), then replace with fresh data
  const cached = readCache();
  const [snapshotStatus, setSnapshotStatus] = useState<LassoDashboardStatus | undefined>(cached?.sync_status);
  const [snapshotUnassigned, setSnapshotUnassigned] = useState<DashboardUnassignedLead[] | undefined>(cached?.unassigned);
  const [snapshotActive, setSnapshotActive] = useState<DashboardActiveLead[] | undefined>(cached?.active);

  useEffect(() => {
    getDashboardSnapshot()
      .then((snap) => {
        setSnapshotStatus(snap.sync_status);
        setSnapshotUnassigned(snap.unassigned.leads);
        setSnapshotActive(snap.active.leads);
        writeCache({
          sync_status: snap.sync_status,
          unassigned: snap.unassigned.leads,
          active: snap.active.leads,
          cached_at: Date.now(),
        });
      })
      .catch(() => {
        // On failure, keep whatever is already shown (cache or undefined)
      });
  }, []);

  const refreshSnapshotSections = useCallback(() => {
    setLassoRefreshKey((value) => value + 1);
  }, []);

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '40px 20px' }}>
      <h1 style={{ marginBottom: 24, fontSize: 28, fontWeight: 700, color: '#2c3e50' }}>
        Admin Dashboard
      </h1>

      <AISpendWidget />

      <LassoSyncStatusCard
        onSnapshotUpdated={refreshSnapshotSections}
        initialStatus={snapshotStatus}
      />

      {/* AI Insights Panel */}
      <AIInsightsPanel />

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
          <LassoUnassignedLeadsTable
            refreshToken={lassoRefreshKey}
            onLeadAssigned={refreshSnapshotSections}
            initialLeads={snapshotUnassigned}
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
          <LassoActiveLeadsTable
            refreshToken={lassoRefreshKey}
            onRefresh={refreshSnapshotSections}
            initialLeads={snapshotActive}
          />
        )}
      </div>
    </div>
  );
};

export default Dashboard;
