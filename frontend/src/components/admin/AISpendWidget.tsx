import React, { useCallback, useEffect, useState } from 'react';

import { getCostTracking } from '../../services/api';
import type { CostTrackingSnapshot } from '../../types';
import { getApiErrorMessage } from '../../utils/apiErrors';

const cardStyle: React.CSSProperties = {
  background: 'white',
  borderRadius: 8,
  boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
  marginBottom: 20,
  overflow: 'hidden',
};

const headerStyle: React.CSSProperties = {
  background: 'linear-gradient(135deg, #14323f, #25596b)',
  color: 'white',
  padding: '12px 20px',
  fontSize: 16,
  fontWeight: 600,
};

const metricStyle: React.CSSProperties = {
  padding: 16,
  borderRadius: 8,
  background: '#f7fafc',
  border: '1px solid #e2e8f0',
};

const AISpendWidget: React.FC = () => {
  const [snapshot, setSnapshot] = useState<CostTrackingSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchSnapshot = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setSnapshot(await getCostTracking());
    } catch (err) {
      setError(getApiErrorMessage(err, 'Failed to load AI spend.'));
      setSnapshot(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSnapshot();
  }, [fetchSnapshot]);

  return (
    <div style={cardStyle}>
      <div style={headerStyle}>AI Spend</div>
      <div style={{ padding: 16 }}>
        {loading ? (
          <div style={{ color: '#64748b' }}>Loading AI spend...</div>
        ) : error ? (
          <div style={{ color: '#c91414' }}>{error}</div>
        ) : snapshot ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
            <div style={metricStyle}>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>This Month</div>
              <div style={{ fontSize: 28, fontWeight: 700, color: '#14323f' }}>
                ${snapshot.monthly.spend_cad.toFixed(2)}
              </div>
              <div style={{ fontSize: 12, color: '#64748b' }}>
                Limit ${snapshot.monthly.limit_cad.toFixed(2)} CAD
              </div>
            </div>
            <div style={metricStyle}>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>Today</div>
              <div style={{ fontSize: 28, fontWeight: 700, color: '#14323f' }}>
                ${snapshot.daily.spend_cad.toFixed(2)}
              </div>
              <div style={{ fontSize: 12, color: '#64748b' }}>
                Remaining ${snapshot.daily.remaining_cad.toFixed(2)} CAD
              </div>
            </div>
            <div style={metricStyle}>
              <div style={{ fontSize: 12, color: '#64748b', marginBottom: 6 }}>Calls This Month</div>
              <div style={{ fontSize: 28, fontWeight: 700, color: '#14323f' }}>
                {snapshot.monthly.total_calls}
              </div>
              <div style={{ fontSize: 12, color: '#64748b' }}>
                {snapshot.monthly.by_model.length} model{snapshot.monthly.by_model.length === 1 ? '' : 's'} used
              </div>
            </div>
          </div>
        ) : (
          <div style={{ color: '#64748b' }}>No AI spend data yet.</div>
        )}
      </div>
    </div>
  );
};

export default AISpendWidget;
