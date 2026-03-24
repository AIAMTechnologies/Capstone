import React, { useCallback, useEffect, useState } from 'react';
import { getLassoDashboardStatus, triggerLassoDashboardSync } from '../../services/api';
import type { LassoDashboardStatus } from '../../types/types';
import { getApiErrorMessage } from '../../utils/apiErrors';

const cardStyle: React.CSSProperties = {
  border: '1px solid #e5e7eb',
  borderRadius: 8,
  padding: 16,
  background: '#fafafa',
};

const buttonStyle: React.CSSProperties = {
  padding: '10px 16px',
  borderRadius: 6,
  border: 'none',
  cursor: 'pointer',
  fontWeight: 600,
  fontSize: 13,
};

const metaLabelStyle: React.CSSProperties = {
  fontSize: 12,
  color: '#6b7280',
  textTransform: 'uppercase',
  letterSpacing: 0.4,
};

const metaValueStyle: React.CSSProperties = {
  fontSize: 15,
  fontWeight: 600,
  color: '#111827',
};

const statusBarTrackStyle: React.CSSProperties = {
  width: '100%',
  height: 10,
  borderRadius: 999,
  background: '#fde5e5',
  overflow: 'hidden',
};

const statusBarFillStyle: React.CSSProperties = {
  width: '100%',
  height: '100%',
  borderRadius: 999,
  background: 'linear-gradient(90deg, #c91414 0%, #ef4444 55%, #fca5a5 100%)',
};

const formatDateTime = (value?: string | null) => {
  if (!value) return 'Never';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return 'Never';
  return parsed.toLocaleString('en-CA', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
};

const formatElapsed = (value?: string | null) => {
  if (!value) return null;
  const startedAt = new Date(value);
  if (Number.isNaN(startedAt.getTime())) return null;
  const elapsedMs = Date.now() - startedAt.getTime();
  if (elapsedMs < 0) return null;
  const totalSeconds = Math.floor(elapsedMs / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
};

const LassoSyncToolsPanel: React.FC = () => {
  const [status, setStatus] = useState<LassoDashboardStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [syncingType, setSyncingType] = useState<'fast' | 'full' | null>(null);
  const [syncRequestedAt, setSyncRequestedAt] = useState<string | null>(null);
  const [nowTick, setNowTick] = useState<number>(Date.now());

  const loadStatus = useCallback(async () => {
    try {
      const result = await getLassoDashboardStatus();
      setStatus(result);
      setError(null);
    } catch (err: any) {
      setError(getApiErrorMessage(err, 'Failed to load Lasso sync status.'));
    } finally {
      setLoading(false);
      setSyncingType(null);
    }
  }, []);

  useEffect(() => {
    loadStatus();
    const interval = window.setInterval(loadStatus, 15000);
    return () => window.clearInterval(interval);
  }, [loadStatus]);

  useEffect(() => {
    const interval = window.setInterval(() => setNowTick(Date.now()), 1000);
    return () => window.clearInterval(interval);
  }, []);

  const handleSync = async (syncType: 'fast' | 'full') => {
    setSyncingType(syncType);
    setSyncRequestedAt(new Date().toISOString());
    try {
      await triggerLassoDashboardSync(syncType);
      await loadStatus();
    } catch (err: any) {
      setError(getApiErrorMessage(err, `Failed to start ${syncType} sync.`));
      setSyncingType(null);
      setSyncRequestedAt(null);
    }
  };

  const activeSyncType = status?.sync_in_progress
    ? (status.sync_type as 'fast' | 'full' | null) ?? syncingType
    : syncingType;
  const activeSyncStartedAt = status?.sync_in_progress ? status.started_at ?? syncRequestedAt : syncRequestedAt;
  const activeSyncElapsed = formatElapsed(activeSyncStartedAt);
  const showSyncStatusBar = !!activeSyncType;
  const syncStatusLabel = status?.sync_in_progress
    ? `${activeSyncType === 'full' ? 'Full' : 'Fast'} sync running`
    : activeSyncType
      ? `${activeSyncType === 'full' ? 'Full' : 'Fast'} sync starting`
      : null;

  return (
    <div>
      {error && (
        <div style={{ color: '#c91414', marginBottom: 12, fontSize: 13 }}>
          {error}
        </div>
      )}

      <div style={{ ...cardStyle, marginBottom: 16 }}>
        {loading ? (
          <div style={{ color: '#6b7280', fontSize: 13 }}>Loading Lasso sync status...</div>
        ) : (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 16, marginBottom: 16 }}>
              <div>
                <div style={metaLabelStyle}>Last Successful Sync</div>
                <div style={metaValueStyle}>{formatDateTime(status?.last_successful_sync_at)}</div>
              </div>
              <div>
                <div style={metaLabelStyle}>Last Full Sync</div>
                <div style={metaValueStyle}>{formatDateTime(status?.last_full_successful_sync_at)}</div>
              </div>
              <div>
                <div style={metaLabelStyle}>Unassigned</div>
                <div style={metaValueStyle}>{status?.unassigned_count ?? 0}</div>
              </div>
              <div>
                <div style={metaLabelStyle}>Active Queue</div>
                <div style={metaValueStyle}>{status?.active_count ?? 0}</div>
              </div>
              <div>
                <div style={metaLabelStyle}>History Rows</div>
                <div style={metaValueStyle}>{status?.history_count ?? 0}</div>
              </div>
            </div>

            <div style={{ fontSize: 13, color: '#6b7280', display: 'flex', gap: 20, flexWrap: 'wrap', marginBottom: 16 }}>
              <div>Dealer reporting snapshot: {formatDateTime(status?.dealer_lead_reporting_synced_at)}</div>
              <div>Dealer performance snapshot: {formatDateTime(status?.dealer_performance_synced_at)}</div>
              <div>History snapshot: {formatDateTime(status?.history_synced_at)}</div>
            </div>

            {status?.last_error && (
              <div style={{ padding: 12, marginBottom: 16, borderRadius: 6, background: '#fff4e5', color: '#92400e', fontSize: 13 }}>
                Last sync issue: {status.last_error}
              </div>
            )}

            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                <button
                  onClick={() => handleSync('fast')}
                  disabled={!!syncingType || status?.sync_in_progress}
                  style={{ ...buttonStyle, background: '#c91414', color: 'white', opacity: syncingType || status?.sync_in_progress ? 0.7 : 1 }}
                >
                  {syncingType === 'fast' ? 'Starting Fast Sync...' : 'Run Fast Sync'}
                </button>
                <button
                  onClick={() => handleSync('full')}
                  disabled={!!syncingType || status?.sync_in_progress}
                  style={{ ...buttonStyle, background: '#1f2937', color: 'white', opacity: syncingType || status?.sync_in_progress ? 0.7 : 1 }}
                >
                  {syncingType === 'full' ? 'Starting Full Sync...' : 'Run Full Sync'}
                </button>
              </div>

              {showSyncStatusBar && (
                <div
                  style={{
                    minWidth: 260,
                    flex: '1 1 320px',
                    border: '1px solid #f1c5c5',
                    borderRadius: 8,
                    padding: '10px 12px',
                    background: '#fff7f7',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, marginBottom: 8 }}>
                    <div style={{ fontSize: 13, fontWeight: 700, color: '#991b1b' }}>
                      {syncStatusLabel}
                    </div>
                    <div style={{ fontSize: 12, color: '#7f1d1d', whiteSpace: 'nowrap' }}>
                      {activeSyncElapsed ? `Elapsed ${activeSyncElapsed}` : 'Preparing...'}
                    </div>
                  </div>
                  <div style={statusBarTrackStyle}>
                    <div key={nowTick} style={statusBarFillStyle} />
                  </div>
                  <div style={{ marginTop: 8, fontSize: 12, color: '#7f1d1d' }}>
                    {status?.sync_in_progress
                      ? 'The latest snapshot will replace the current dashboard data when this sync completes.'
                      : 'Sync request sent. Waiting for the backend to mark the job as running.'}
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </div>

      <div style={{ color: '#6b7280', fontSize: 13, lineHeight: 1.6 }}>
        Fast sync refreshes the dashboard queues and reporting snapshot. Full sync refreshes the Lasso-backed history dataset used for history view, project breakdown, dealer performance, and ML training.
      </div>
    </div>
  );
};

export default LassoSyncToolsPanel;
