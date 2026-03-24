import React, { useCallback, useEffect, useRef, useState } from 'react';
import { getLassoDashboardStatus, triggerLassoDashboardSync } from '../../services/api';
import type { LassoDashboardStatus } from '../../types/types';
import { getApiErrorMessage } from '../../utils/apiErrors';

interface LassoSyncStatusCardProps {
  onSnapshotUpdated?: () => void;
  initialStatus?: LassoDashboardStatus;
}

const cardStyle: React.CSSProperties = {
  background: 'white',
  borderRadius: 8,
  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
  marginBottom: 20,
  overflow: 'hidden',
};

const headerStyle: React.CSSProperties = {
  background: '#1a1a2e',
  color: 'white',
  padding: '12px 20px',
  fontSize: 16,
  fontWeight: 600,
};

const metaLabelStyle: React.CSSProperties = {
  fontSize: 12,
  color: '#6b7280',
  textTransform: 'uppercase',
  letterSpacing: 0.4,
};

const metaValueStyle: React.CSSProperties = {
  fontSize: 16,
  fontWeight: 600,
  color: '#1f2937',
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

const LassoSyncStatusCard: React.FC<LassoSyncStatusCardProps> = ({ onSnapshotUpdated, initialStatus }) => {
  const [status, setStatus] = useState<LassoDashboardStatus | null>(initialStatus ?? null);
  const [loading, setLoading] = useState(initialStatus === undefined);
  const [error, setError] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);
  const lastSuccessRef = useRef<string | null>(initialStatus?.last_successful_sync_at ?? null);

  const loadStatus = useCallback(async () => {
    try {
      const response = await getLassoDashboardStatus();
      setStatus(response);
      setError(null);
      if (
        response.last_successful_sync_at &&
        response.last_successful_sync_at !== lastSuccessRef.current
      ) {
        lastSuccessRef.current = response.last_successful_sync_at;
        onSnapshotUpdated?.();
      }
    } catch (err: any) {
      setError(getApiErrorMessage(err, 'Failed to load Lasso sync status.'));
    } finally {
      setLoading(false);
      setSyncing(false);
    }
  }, [onSnapshotUpdated]);

  useEffect(() => {
    // Skip the immediate fetch if we already have initialStatus from the parent snapshot.
    // Still set up the polling interval so status stays current.
    if (initialStatus === undefined) {
      loadStatus();
    }
    const interval = window.setInterval(loadStatus, 15000);
    const onVisibilityChange = () => {
      if (document.visibilityState === 'visible') loadStatus();
    };
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => {
      window.clearInterval(interval);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, [loadStatus, initialStatus]);

  const handleSync = async () => {
    setSyncing(true);
    try {
      await triggerLassoDashboardSync('fast');
      await loadStatus();
    } catch (err: any) {
      setError(getApiErrorMessage(err, 'Failed to start Lasso sync.'));
      setSyncing(false);
    }
  };

  return (
    <div style={cardStyle}>
      <div style={headerStyle}>Lasso Snapshot Status</div>
      <div style={{ padding: 20 }}>
        {loading ? (
          <div style={{ color: '#7f8c8d' }}>Loading sync status...</div>
        ) : (
          <>
            {error && (
              <div style={{ padding: 12, marginBottom: 16, borderRadius: 6, background: '#fdecea', color: '#c0392b' }}>
                {error}
              </div>
            )}

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 16, marginBottom: 16 }}>
              <div>
                <div style={metaLabelStyle}>Last Successful Sync</div>
                <div style={metaValueStyle}>{formatDateTime(status?.last_successful_sync_at)}</div>
                <div style={{ color: '#6b7280', fontSize: 12, marginTop: 4 }}>
                  {status?.last_successful_sync_type === 'fast' ? 'Fast snapshot' : status?.last_successful_sync_type === 'full' ? 'Full snapshot' : 'No completed sync yet'}
                </div>
              </div>
              <div>
                <div style={metaLabelStyle}>Unassigned Snapshot</div>
                <div style={metaValueStyle}>{status?.unassigned_count ?? 0}</div>
              </div>
              <div>
                <div style={metaLabelStyle}>Active Queue</div>
                <div style={metaValueStyle}>{status?.active_count ?? 0}</div>
              </div>
              <div>
                <div style={metaLabelStyle}>Active Queue Window</div>
                <div style={metaValueStyle}>{status?.active_queue_max_age_days ?? 0} days</div>
              </div>
            </div>

            <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', marginBottom: 16, fontSize: 13, color: '#6b7280' }}>
              <div>Fast snapshot cadence: every {status?.fast_sync_interval_minutes ?? 5} minutes</div>
              <div>Last full backfill: {formatDateTime(status?.last_full_successful_sync_at)}</div>
            </div>

            {status?.last_error && (
              <div style={{ padding: 12, marginBottom: 16, borderRadius: 6, background: '#fff4e5', color: '#92400e' }}>
                Last sync issue: {status.last_error}
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              <div style={{ fontSize: 13, color: '#6b7280' }}>
                {status?.sync_in_progress
                  ? `Sync in progress${status.sync_type ? ` (${status.sync_type})` : ''}...`
                  : `Auto-refresh target: every ${status?.fast_sync_interval_minutes ?? 5} minutes`}
              </div>
              <button
                onClick={handleSync}
                disabled={syncing || status?.sync_in_progress}
                style={{
                  padding: '10px 16px',
                  borderRadius: 6,
                  border: 'none',
                  cursor: syncing || status?.sync_in_progress ? 'default' : 'pointer',
                  background: '#c91414',
                  color: 'white',
                  fontWeight: 600,
                  opacity: syncing || status?.sync_in_progress ? 0.7 : 1,
                }}
              >
                {syncing || status?.sync_in_progress ? 'Syncing...' : 'Sync Lasso Now'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default LassoSyncStatusCard;
